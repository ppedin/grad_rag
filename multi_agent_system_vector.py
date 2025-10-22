"""
Multi-Agent VectorRAG System with AutoGen Core API.
Includes BatchOrchestratorAgent and HyperparametersVectorAgent.
"""

import json
import logging
import statistics
import numpy as np
import faiss
import asyncio
import pickle
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime
from rank_bm25 import BM25Okapi

from logging_utils import LoggingUtils, log_agent_action, log_batch_progress, log_qa_processing, log_critique_result
from prompt_response_logger import get_global_prompt_logger, initialize_prompt_logging
from step_execution_logger import get_global_step_logger, initialize_step_logging, StepStatus
from evaluation_logger import get_global_evaluation_logger, initialize_evaluation_logging
from standardized_evaluation_logger import initialize_standardized_logging, get_standardized_logger, finalize_standardized_logging, SystemType

from autogen_core import (
    AgentId, MessageContext, RoutedAgent, message_handler,
    SingleThreadedAgentRuntime, TRACE_LOGGER_NAME
)
from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# OpenAI imports for G-Eval
from openai import OpenAI

# Gemini API imports
from google import genai
from google.genai import types

from shared_state import SharedState
from datasets_schema import Document, Question
from eval_functions import evaluate_rouge_score
import llm_keys
from autogen_dataset_agent import BatchStartMessage, BatchReadyMessage


# ===== UTILITY FUNCTIONS =====

async def retry_api_call_with_backoff(api_call_func, max_retries: int = 10, initial_delay: float = 2.0, max_delay: float = 60.0):
    """
    Retry an async API call with exponential backoff.

    Args:
        api_call_func: Async function to call
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries in seconds

    Returns:
        The result of the API call

    Raises:
        The last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await api_call_func()
        except Exception as e:
            last_exception = e
            error_msg = str(e)

            # Check if it's a 503 overload error
            is_overload = "503" in error_msg or "overloaded" in error_msg.lower() or "UNAVAILABLE" in error_msg

            if is_overload and attempt < max_retries - 1:
                logging.warning(f"API overload error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                logging.warning(f"Retrying in {delay:.1f} seconds...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)  # Exponential backoff with cap
            else:
                # Non-overload error or final attempt
                raise

    # If we get here, all retries failed
    raise last_exception


# ===== MESSAGE TYPES =====

# Import shared messages from DatasetAgent to avoid duplication
# BatchStartMessage and BatchReadyMessage are imported above

# New messages for vector orchestration workflow
class HyperparametersVectorStartMessage(BaseModel):
    qa_pair_id: str
    qa_pair: Dict[str, Any]
    batch_id: int
    repetition: int
    dataset: str
    setting: str

class VectorStartMessage(BaseModel):
    batch_id: int
    repetition: int
    chunk_size: int
    dataset: str
    setting: str
    shared_state: Dict[str, Any]

class VectorReadyMessage(BaseModel):
    batch_id: int
    repetition: int
    dataset: str
    setting: str
    faiss_index_path: str
    chunk_metadata_path: str
    total_chunks: int = 0
    index_reused: bool = False

class VectorRetrievalStartMessage(BaseModel):
    batch_id: int
    repetition: int
    qa_pair_id: str
    query: str
    dataset: str
    setting: str
    k_iterations: int = 6
    shared_state: Dict[str, Any]

class VectorRetrievalReadyMessage(BaseModel):
    batch_id: int
    repetition: int
    retrieved_context: str
    dataset: str
    setting: str
    shared_state: Dict[str, Any]  # Message-passing state architecture

class HyperparametersVectorReadyMessage(BaseModel):
    qa_pair_id: str
    chunk_size: int
    batch_id: int
    repetition: int

class AnswerGenerationStartMessage(BaseModel):
    qa_pair_id: str
    question: str
    retrieved_context: str
    batch_id: int
    repetition: int
    dataset: str
    setting: str
    shared_state: Dict[str, Any]  # Message-passing state architecture

class AnswerGenerationReadyMessage(BaseModel):
    qa_pair_id: str
    generated_answer: str
    batch_id: int
    repetition: int
    shared_state: Dict[str, Any]  # Message-passing state architecture

class ResponseEvaluationStartMessage(BaseModel):
    qa_pair_id: str
    original_query: str
    generated_answer: str
    gold_answers: List[str]
    rouge_score: float
    batch_id: int
    repetition: int
    dataset: str
    setting: str
    document_text: str = ""  # Added for G-Eval computation
    retrieved_context: str = ""  # Retrieved context used to generate the answer
    shared_state: Dict[str, Any]  # Message-passing state architecture

class ResponseEvaluationReadyMessage(BaseModel):
    qa_pair_id: str
    evaluation_result: Dict[str, Any]
    rouge_score: float
    continue_optimization: bool
    batch_id: int
    repetition: int
    shared_state: Dict[str, Any]  # Message-passing state architecture

class BackwardPassStartMessage(BaseModel):
    batch_id: int
    repetition: int
    dataset: str
    setting: str
    all_qa_results: List[Dict[str, Any]]

class BackwardPassReadyMessage(BaseModel):
    batch_id: int
    repetition: int
    backward_pass_results: Dict[str, Any]
    dataset: str
    setting: str

# Response format for HyperparametersVectorAgent
class HyperparametersVectorResponse(BaseModel):
    reasoning: str
    chunk_size: int
    confidence_score: float

# Message types for SummarizerAgent
class SummarizationStartMessage(BaseModel):
    batch_id: int
    repetition: int
    retrieved_contexts: List[str]
    dataset: str
    setting: str

class SummarizationReadyMessage(BaseModel):
    batch_id: int
    repetition: int
    context_summaries: List[str]
    concatenated_summary: str


# ===== BATCH ORCHESTRATOR AGENT =====

class BatchOrchestratorAgent(RoutedAgent):
    """
    Orchestrates the processing of QA pairs in a batch through the multi-agent vector pipeline.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.batch_orchestrator")
        self.shared_state = SharedState("agent_states")

        # Initialize logging systems
        initialize_step_logging()
        initialize_evaluation_logging()

        # Track QA pairs processing
        self.current_batch_qa_pairs: Dict[str, Dict[str, Any]] = {}
        self.completed_qa_pairs: Dict[str, Dict[str, Any]] = {}
        self.current_batch_id: Optional[int] = None
        self.current_repetition: Optional[int] = None
        self.current_dataset: Optional[str] = None
        self.current_setting: Optional[str] = None

        # Standardized evaluation logging
        self.standardized_logger = None

        # Initialize QA pair and iteration tracking for multi-iteration support
        self.qa_pair_results: Dict[str, List[Dict[str, Any]]] = {}
        self.qa_pair_rouge_progression: Dict[str, List[float]] = {}

    @message_handler
    async def handle_batch_start(self, message: BatchStartMessage, ctx: MessageContext) -> BatchReadyMessage:
        """Handle BatchStart message by iterating over QA pairs."""
        self.logger.info(f"VectorRAG BatchOrchestrator received BatchStart for batch {message.batch_id}")

        # Initialize step logger
        step_logger = get_global_step_logger()
        step_logger.start_pipeline(message.dataset, message.setting, len(message.shared_state.get("batch_information", {}).get("qa_pairs", [])))

        # Initialize prompt logging with system-specific folder
        system_log_dir = f"prompt_response_logs/vector_{message.dataset}_{message.setting}"
        initialize_prompt_logging(system_log_dir)

        # Initialize standardized evaluation logging for VectorRAG
        if self.standardized_logger is None:
            self.standardized_logger = initialize_standardized_logging(
                SystemType.VECTORRAG, message.dataset, message.setting
            )

        # Validate that only test datasets are used for test-time training
        if message.setting != "test":
            error_msg = f"VectorRAG test-time training only supports 'test' setting. Got: '{message.setting}'"
            self.logger.error(error_msg)
            step_logger.complete_pipeline(success=False, error_message=error_msg)
            raise ValueError(error_msg)

        self.current_batch_id = message.batch_id
        self.current_repetition = message.repetition
        self.current_dataset = message.dataset
        self.current_setting = message.setting
        self.current_batch_qa_pairs = {}
        self.completed_qa_pairs = {}

        try:
            # Extract QA pairs from batch information
            batch_info = message.shared_state.get("batch_information", {})
            qa_pairs = batch_info.get("qa_pairs", [])

            self.logger.info(f"Processing {len(qa_pairs)} QA pairs in batch {message.batch_id}")

            # First, populate ALL QA pairs in the tracking dictionary
            for i, qa_pair in enumerate(qa_pairs):
                qa_pair_id = qa_pair.get("question_id", f"qa_{i}")
                self.current_batch_qa_pairs[qa_pair_id] = qa_pair

            self.logger.info(f"Initialized tracking for {len(self.current_batch_qa_pairs)} QA pairs")

            # Carry over continue_optimization flags from previous repetitions (cross-batch)
            # so early-stop works on repetition > 0 for already-satisfactory QA pairs
            try:
                if message.repetition > 0:
                    prev_states = self.shared_state.get_all_states(message.dataset, message.setting)
                    # Sort is already by batch_id ascending; iterate from latest to oldest
                    copied = 0
                    for qa in qa_pairs:
                        qa_pair_id = qa.get("question_id", "")
                        if not qa_pair_id:
                            continue
                        flag_key = f"continue_optimization_{qa_pair_id}"
                        # Only set if not present in current state
                        current_state = self.shared_state.load_state_fresh(message.dataset, message.setting, message.batch_id)
                        if flag_key in current_state:
                            continue
                        # Find most recent prior value
                        for ps in reversed(prev_states):
                            if ps.get("_metadata", {}).get("batch_id", 0) == message.batch_id:
                                continue
                            if flag_key in ps:
                                val = ps.get(flag_key)
                                if isinstance(val, bool):
                                    current_state[flag_key] = val
                                    copied += 1
                                    break
                    if copied:
                        self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)
                        print(f"[Orchestrator] Carried over {copied} early-stop flags from previous repetitions")
            except Exception as e:
                print(f"[Orchestrator] Early-stop flag carry-over error: {e}")

            # FIXED: Use message.repetition as current iteration instead of processing multiple iterations
            # The DatasetAgent calls us once per repetition, so we should process exactly one iteration
            current_iteration = message.repetition
            total_iterations = batch_info.get("total_iterations", 1)  # Keep for logging context
            self.logger.info(f"Processing {len(qa_pairs)} QA pairs - current iteration: {current_iteration} (DatasetAgent repetition: {message.repetition})")

            # Initialize tracking for multi-iteration processing
            for qa_pair in qa_pairs:
                qa_pair_id = qa_pair.get("question_id", f"qa_{qa_pairs.index(qa_pair)}")
                self.qa_pair_results[qa_pair_id] = []
                self.qa_pair_rouge_progression[qa_pair_id] = []

            # Process each QA pair for the current iteration only (DatasetAgent handles repetitions)
            for qa_pair in qa_pairs:
                qa_pair_id = qa_pair.get("question_id", f"qa_{qa_pairs.index(qa_pair)}")
                self.logger.info(f"ðŸ”„ Processing QA pair {qa_pair_id} for iteration {current_iteration}")

                # Log QA pair start for evaluation (matching GraphRAG pattern)
                if current_iteration == 0:  # Only log start on first iteration
                    eval_logger = get_global_evaluation_logger()
                    document_text = batch_info.get("document_text", "")
                    eval_logger.start_qa_pair_evaluation(
                        qa_pair_id=qa_pair_id,
                        question=qa_pair.get("question", ""),
                        reference_answers=qa_pair.get("answers", []),
                        document_text=document_text,
                        total_iterations=total_iterations,
                        metadata={"batch_id": message.batch_id, "dataset": message.dataset, "setting": message.setting}
                    )

                    # Also log to standardized logger
                    self.standardized_logger.start_qa_pair_evaluation(
                        qa_pair_id=qa_pair_id,
                        question=qa_pair.get("question", ""),
                        reference_answers=qa_pair.get("answers", []),
                        document_text=document_text,
                        total_iterations=total_iterations,
                        metadata={"batch_id": message.batch_id, "dataset": message.dataset, "setting": message.setting}
                    )

                # Initialize shared state for this QA pair
                current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)

                # Display document chunk count at the beginning of QA pair
                total_chunks = current_state.get("total_chunks", "unknown")
                print(f"\n{'='*80}")
                print(f"QA PAIR: {qa_pair_id} | Iteration: {current_iteration} | Document Chunks: {total_chunks}")
                print(f"{'='*80}\n")

                # Determine if this is a new QA pair or new iteration
                transition_type = self.shared_state.detect_transition_type(qa_pair_id, current_iteration)

                if transition_type == 'new_qa_pair':
                    # Complete reset for new QA pair
                    current_state = self.shared_state.reset_for_new_qa_pair(qa_pair_id, message.dataset, message.setting, message.batch_id)
                    current_state["batch_information"] = batch_info
                    current_state["full_document_text"] = batch_info.get("document_text", "")
                    # Restore document_index for fault-tolerant FAISS index naming
                    if "document_index" in message.shared_state:
                        current_state["document_index"] = message.shared_state["document_index"]
                    self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)
                elif transition_type == 'new_iteration':
                    # Partial reset for new iteration of same QA pair
                    current_state = self.shared_state.reset_for_new_iteration(qa_pair_id, current_iteration, message.dataset, message.setting, message.batch_id)
                    current_state["batch_information"] = batch_info
                    current_state["full_document_text"] = batch_info.get("document_text", "")
                    self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

                # Check if we should skip this iteration because evaluation determined response is satisfactory
                if current_iteration > 0:  # Only check after first iteration
                    # Force fresh state load to observe evaluator's latest signals
                    current_state = self.shared_state.load_state_fresh(message.dataset, message.setting, message.batch_id)

                    # Try 0: in-memory map (updated by evaluator immediately) â€” authoritative
                    mem_flag = None
                    try:
                        from shared_state import SharedState as _SS
                        mem_flag = _SS.get_continue_flag(message.dataset, message.setting, qa_pair_id)
                    except Exception:
                        mem_flag = None

                    used_source = "current_batch_evals"
                    evals = current_state.get("response_evaluations", [])
                    if mem_flag is not None:
                        should_continue = bool(mem_flag)
                        used_source = "memory_map"
                    else:
                        # Try 1: most recent eval in current batch for this QA
                        should_continue = True
                        if evals:
                            qa_evals = [e for e in evals if e.get('qa_pair_id') == qa_pair_id]
                            if qa_evals:
                                last_eval = qa_evals[-1]
                                cont = last_eval.get('continue_optimization')
                                if isinstance(cont, bool):
                                    should_continue = cont
                                    used_source = "current_batch_evals"
                        # Try 2: scan all previous states for last eval for this qa_pair_id
                        if used_source == "current_batch_evals" and evals and 'should_continue' in locals():
                            pass  # already resolved from current batch evals
                        else:
                            all_states = self.shared_state.get_all_states(message.dataset, message.setting)
                            for st in reversed(all_states):
                                evs = st.get("response_evaluations", [])
                                qa_evs = [e for e in evs if e.get('qa_pair_id') == qa_pair_id]
                                if qa_evs:
                                    last = qa_evs[-1]
                                    cont = last.get('continue_optimization')
                                    if isinstance(cont, bool):
                                        should_continue = cont
                                        used_source = "prior_batches_evals"
                                        break
                        # Try 3: dedicated flag in current batch or prior states
                        if used_source not in ("memory_map", "current_batch_evals", "prior_batches_evals"):
                            flag_key = f"continue_optimization_{qa_pair_id}"
                            if flag_key in current_state:
                                should_continue = current_state.get(flag_key, True)
                                used_source = "current_batch_flag"
                            else:
                                for st in reversed(self.shared_state.get_all_states(message.dataset, message.setting)):
                                    if flag_key in st:
                                        should_continue = st.get(flag_key, True)
                                        used_source = "prior_batches_flag"
                                        break

                    print(f"[Orchestrator] Early-stop check qa_pair_id={qa_pair_id} flag={should_continue} (source={used_source}, current_batch_evals={len(evals)})")
                    if not should_continue:
                        self.logger.info(f"â­ï¸  Skipping iteration {current_iteration} for {qa_pair_id} - evaluation indicated response was satisfactory in previous iteration")
                        print(f"â­ï¸  Early stopping: Skipping {qa_pair_id} iteration {current_iteration} (response satisfactory)")

                        # Use the last successful result
                        if qa_pair_id in self.qa_pair_results and self.qa_pair_results[qa_pair_id]:
                            last_result = self.qa_pair_results[qa_pair_id][-1]
                            self.qa_pair_results[qa_pair_id].append(last_result)
                            last_rouge = last_result.get("rouge_score", 0.0)
                            self.qa_pair_rouge_progression[qa_pair_id].append(last_rouge)
                            self.logger.info(f"âœ“ Reusing result from previous iteration for {qa_pair_id} | ROUGE: {last_rouge:.4f}")

                            # Log iteration completion for the final iteration
                            if current_iteration == total_iterations - 1:
                                current_rouge = self.qa_pair_rouge_progression[qa_pair_id][-1] if self.qa_pair_rouge_progression[qa_pair_id] else 0.0
                                eval_logger = get_global_evaluation_logger()
                                # Log to old evaluation logger for backward compatibility
                                eval_logger.complete_qa_pair_evaluation(
                                    qa_pair_id=qa_pair_id,
                                    final_rouge_score=current_rouge,
                                    rouge_progression=self.qa_pair_rouge_progression[qa_pair_id],
                                    best_iteration=current_iteration,
                                    total_iterations_completed=current_iteration + 1,
                                    improvement_gained=0.0,
                                    final_metrics={}
                                )

                                # Log to standardized evaluation logger
                                best_answer = last_result.get("generated_answer", "")
                                self.standardized_logger.complete_qa_pair_evaluation(
                                    qa_pair_id=qa_pair_id,
                                    final_rouge_score=current_rouge,
                                    rouge_progression=self.qa_pair_rouge_progression[qa_pair_id],
                                    best_iteration=current_iteration,
                                    total_iterations_completed=current_iteration + 1,
                                    best_answer=best_answer
                                )

                        # IMPORTANT: Skip to next iteration - don't process this one
                        continue

                # Process the current iteration only
                self.logger.info(f"ðŸ“Š Processing QA pair {qa_pair_id}, iteration {current_iteration}")

                try:
                    # Start batch for this iteration
                    step_logger = get_global_step_logger()
                    step_logger.start_batch(message.batch_id, qa_pair_id, current_iteration, total_iterations)

                    # Process this iteration
                    iteration_results = await self._process_qa_pair_iteration(qa_pair_id, qa_pair, current_iteration, message, ctx)

                    # Store iteration results
                    if qa_pair_id not in self.qa_pair_results:
                        self.qa_pair_results[qa_pair_id] = []
                        self.qa_pair_rouge_progression[qa_pair_id] = []

                    self.qa_pair_results[qa_pair_id].append(iteration_results)
                    rouge_score = iteration_results.get("rouge_score", 0.0)
                    self.qa_pair_rouge_progression[qa_pair_id].append(rouge_score)

                    self.logger.info(f"âœ“ Iteration {current_iteration} completed for {qa_pair_id} | ROUGE: {rouge_score:.4f}")

                    # Complete batch for this iteration
                    step_logger.complete_batch(success=True, final_rouge_score=rouge_score)

                except Exception as e:
                    self.logger.error(f"Critical error in iteration {current_iteration} for {qa_pair_id}: {e}")
                    # Store error results
                    if qa_pair_id not in self.qa_pair_results:
                        self.qa_pair_results[qa_pair_id] = []
                        self.qa_pair_rouge_progression[qa_pair_id] = []

                    error_results = {
                        "qa_pair_id": qa_pair_id,
                        "iteration": current_iteration,
                        "error": str(e),
                        "rouge_score": 0.0
                    }
                    self.qa_pair_results[qa_pair_id].append(error_results)
                    self.qa_pair_rouge_progression[qa_pair_id].append(0.0)

                    step_logger.complete_batch(success=False, error_message=str(e))

                # Log iteration completion (DatasetAgent handles overall QA pair completion tracking)
                current_rouge = self.qa_pair_rouge_progression[qa_pair_id][-1] if self.qa_pair_rouge_progression[qa_pair_id] else 0.0
                self.logger.info(f"ðŸŽ¯ QA pair {qa_pair_id} iteration {current_iteration} completed - ROUGE: {current_rouge:.4f}")

                # Add QA pair completion logging for the final iteration (matching GraphRAG)
                if current_iteration == total_iterations - 1:
                    eval_logger = get_global_evaluation_logger()
                    # Log to old evaluation logger for backward compatibility
                    eval_logger.complete_qa_pair_evaluation(
                        qa_pair_id=qa_pair_id,
                        final_rouge_score=current_rouge,
                        rouge_progression=self.qa_pair_rouge_progression[qa_pair_id],
                        best_iteration=current_iteration,  # In single-iteration mode, best is current
                        total_iterations_completed=current_iteration + 1,
                        improvement_gained=0.0,
                        final_metrics={}
                    )

                    # Log to standardized evaluation logger
                    best_answer = iteration_results.get("generated_answer", "")
                    self.standardized_logger.complete_qa_pair_evaluation(
                        qa_pair_id=qa_pair_id,
                        final_rouge_score=current_rouge,
                        rouge_progression=self.qa_pair_rouge_progression[qa_pair_id],
                        best_iteration=current_iteration,
                        total_iterations_completed=current_iteration + 1,
                        best_answer=best_answer
                    )

            # Complete pipeline
            step_logger.complete_pipeline(success=True, total_qa_pairs_processed=len(qa_pairs),
                                        total_iterations_completed=len(qa_pairs) * 1)  # One iteration per call

            # Finalize standardized evaluation logging
            if self.standardized_logger:
                summary_path = self.standardized_logger.finalize_session()
                self.logger.info(f"Standardized evaluation session finalized: {summary_path}")

            # Return BatchReady message indicating completion
            return BatchReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                status="completed",
                metrics={"qa_pairs_processed": len(qa_pairs)}
            )

        except Exception as e:
            self.logger.error(f"Error processing batch {message.batch_id}: {e}")
            return BatchReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                status="failed",
                metrics={"error": str(e)}
            )

    async def _process_qa_pair_iteration(self, qa_pair_id: str, qa_pair: Dict[str, Any], iteration: int,
                                       original_message: BatchStartMessage, ctx: MessageContext) -> Dict[str, Any]:
        """
        Process a single iteration of a QA pair through the complete vector pipeline.

        Args:
            qa_pair_id (str): QA pair identifier
            qa_pair (Dict[str, Any]): QA pair data
            iteration (int): Current iteration number
            original_message (BatchStartMessage): Original batch start message
            ctx (MessageContext): Message context

        Returns:
            Dict[str, Any]: Results from this iteration including ROUGE score
        """
        self.logger.info(f"ðŸ”„ Processing iteration {iteration} for QA pair {qa_pair_id}")

        # Track intermediate outputs for comprehensive evaluation logging (matching GraphRAG)
        intermediate_outputs = {}
        step_start_time = datetime.now()

        try:
            # Fixed chunk size (150 words with 20-30% overlap)
            FIXED_CHUNK_SIZE = 200

            # Step 1: Vector Building (only on first iteration)
            self.logger.info(f"Step 1/4: Vector building for {qa_pair_id}")

            step_agent_start = datetime.now()
            try:
                # Force fresh load to pick up indices saved by previous iteration (bypass cache)
                current_state = self.shared_state.load_state_fresh(original_message.dataset, original_message.setting, original_message.batch_id)

                vector_start_msg = VectorStartMessage(
                    batch_id=original_message.batch_id,
                    repetition=original_message.repetition,
                    chunk_size=FIXED_CHUNK_SIZE,
                    dataset=original_message.dataset,
                    setting=original_message.setting,
                    shared_state=current_state
                )

                vector_builder_agent_id = AgentId("vector_builder_agent", "default")
                vector_response = await self.send_message(vector_start_msg, vector_builder_agent_id)

                step_execution_time = (datetime.now() - step_agent_start).total_seconds() * 1000

                # Log intermediate output
                intermediate_outputs["vector_building"] = {
                    "chunk_size": FIXED_CHUNK_SIZE,
                    "faiss_index_path": getattr(vector_response, 'faiss_index_path', 'N/A'),
                    "vectors_created": True,
                    "embedding_method": "openai",
                    "processing_time_ms": step_execution_time,
                    "index_reused": getattr(vector_response, 'index_reused', False)
                }

            except Exception as e:
                step_execution_time = (datetime.now() - step_agent_start).total_seconds() * 1000
                intermediate_outputs["vector_building"] = {
                    "error": str(e),
                    "processing_time_ms": step_execution_time,
                    "vectors_created": False
                }
                raise

            # Step 2: Vector Retrieval
            self.logger.info(f"Step 2/4: Vector retrieval for {qa_pair_id}")
            # Update shared state with FAISS index paths for retrieval
            # CRITICAL: Use load_state_fresh to pick up indices saved by previous iteration (bypass stale cache)
            current_state = self.shared_state.load_state_fresh(original_message.dataset, original_message.setting, original_message.batch_id)
            current_state["faiss_index_path"] = vector_response.faiss_index_path
            current_state["chunk_metadata_path"] = vector_response.chunk_metadata_path
            current_state["total_chunks"] = vector_response.total_chunks
            self.shared_state.save_state(current_state, original_message.dataset, original_message.setting, original_message.batch_id)

            retrieval_start_msg = VectorRetrievalStartMessage(
                batch_id=vector_response.batch_id,
                repetition=vector_response.repetition,
                qa_pair_id=qa_pair_id,
                query=qa_pair.get("question", ""),
                dataset=original_message.dataset,
                setting=original_message.setting,
                k_iterations=5,
                shared_state=current_state
            )

            retrieval_agent_id = AgentId("vector_retrieval_planner_agent", "default")
            retrieval_response = await self.send_message(retrieval_start_msg, retrieval_agent_id)

            # Log intermediate output for retrieval
            intermediate_outputs["retrieval"] = {
                "query": qa_pair.get("question", ""),
                "retrieved_context": retrieval_response.retrieved_context[:200] + "..." if len(retrieval_response.retrieved_context) > 200 else retrieval_response.retrieved_context,
                "retrieved_context_length": len(retrieval_response.retrieved_context)
            }

            # Use state from retrieval message (message-passing architecture)
            current_state = retrieval_response.shared_state.copy()
            retrieved_contexts = current_state.get("retrieved_contexts", [])
            retrieved_contexts.append(retrieval_response.retrieved_context)
            current_state["retrieved_contexts"] = retrieved_contexts

            retrieval_queries = current_state.get("retrieval_queries", [])
            retrieval_queries.append(qa_pair.get("question", ""))
            current_state["retrieval_queries"] = retrieval_queries

            # Step 3: Answer Generation
            self.logger.info(f"Step 3/4: Answer generation for {qa_pair_id}")
            answer_gen_msg = AnswerGenerationStartMessage(
                qa_pair_id=qa_pair_id,
                question=qa_pair.get("question", ""),
                retrieved_context=retrieval_response.retrieved_context,
                batch_id=retrieval_response.batch_id,
                repetition=retrieval_response.repetition,
                dataset=original_message.dataset,
                setting=original_message.setting,
                shared_state=current_state  # Pass state through message
            )

            answer_gen_agent_id = AgentId("answer_generator_agent", "default")
            answer_response = await self.send_message(answer_gen_msg, answer_gen_agent_id)

            # Log intermediate output for answer generation
            intermediate_outputs["answer_generation"] = {
                "question": qa_pair.get("question", ""),
                "generated_answer": answer_response.generated_answer[:200] + "..." if len(answer_response.generated_answer) > 200 else answer_response.generated_answer,
                "generated_answer_length": len(answer_response.generated_answer)
            }

            # Step 4: Evaluation
            self.logger.info(f"Step 4/4: Evaluation for {qa_pair_id}")

            # Use state from answer message (message-passing architecture)
            current_state = answer_response.shared_state.copy()

            # Create evaluation message (ROUGE score will be computed by ResponseEvaluatorAgent)
            # Debug: check qa_pair structure
            print(f"\n[DEBUG] QA Pair available keys: {list(qa_pair.keys())}")
            self.logger.info(f"QA Pair keys: {list(qa_pair.keys())}")
            self.logger.info(f"QA Pair answer field: {qa_pair.get('answer', 'NOT_FOUND')}")

            # Get document text from shared state (where DatasetAgent stores it)
            document_text = current_state.get("full_document_text", "")
            if document_text:
                print(f"[DEBUG] Document text retrieved from shared state, length: {len(document_text)} chars")
            else:
                print(f"[DEBUG] âš ï¸ No document text in shared state!")

            # Try multiple possible answer field names
            gold_answer = qa_pair.get("answer") or qa_pair.get("answers") or qa_pair.get("expected_answer") or ""
            if isinstance(gold_answer, list):
                gold_answers = gold_answer
            else:
                gold_answers = [gold_answer] if gold_answer else []

            self.logger.info(f"Gold answers for ROUGE: {gold_answers}")

            # Get the latest retrieved context from state
            retrieved_contexts = current_state.get("retrieved_contexts", [])
            latest_retrieved_context = retrieved_contexts[-1] if retrieved_contexts else ""

            eval_start_msg = ResponseEvaluationStartMessage(
                qa_pair_id=qa_pair_id,
                original_query=qa_pair.get("question", ""),
                generated_answer=answer_response.generated_answer,
                gold_answers=gold_answers,
                rouge_score=0.0,  # Will be computed by ResponseEvaluatorAgent
                batch_id=answer_response.batch_id,
                repetition=answer_response.repetition,
                dataset=original_message.dataset,
                setting=original_message.setting,
                document_text=document_text,  # For G-Eval computation
                retrieved_context=latest_retrieved_context,  # Pass retrieved context for evaluation
                shared_state=current_state  # Pass state through message
            )

            eval_agent_id = AgentId("response_evaluator_agent", "default")
            eval_response = await self.send_message(eval_start_msg, eval_agent_id)

            # Log intermediate output for evaluation
            intermediate_outputs["evaluation"] = {
                "evaluation_feedback": eval_response.evaluation_result.get("evaluation_feedback", "N/A"),
                "rouge_score": eval_response.rouge_score,
                "gold_answers": gold_answers
            }

            # Extract ROUGE score from evaluation response (computed by ResponseEvaluatorAgent)
            rouge_score = eval_response.rouge_score

            # Collect comprehensive vector statistics
            try:
                comprehensive_vector_stats = {
                    "chunk_size": FIXED_CHUNK_SIZE,
                    "total_chunks": getattr(vector_response, 'total_chunks', 0),
                    "faiss_index_path": getattr(vector_response, 'faiss_index_path', ''),
                    "retrieved_context_length": len(retrieval_response.retrieved_context) if retrieval_response.retrieved_context else 0,
                    "embedding_dimension": 1536  # Standard OpenAI embedding dimension
                }
            except Exception as e:
                self.logger.warning(f"Could not collect detailed vector statistics: {e}")
                comprehensive_vector_stats = {"error": f"Stats collection failed: {e}"}

            # Note: ResponseEvaluatorAgent already stores evaluation results in response_evaluations
            # with the proper format needed for backward pass. No need to duplicate here.

            # Use state from evaluation message and save once (message-passing architecture)
            current_state = eval_response.shared_state.copy()
            rouge_scores = current_state.get("rouge_scores", [])
            rouge_scores.append(rouge_score)
            current_state["rouge_scores"] = rouge_scores

            # Store vector statistics for tracking
            vector_statistics = current_state.get("vector_statistics", [])
            vector_statistics.append({
                "qa_pair_id": qa_pair_id,
                "iteration": iteration,
                "statistics": comprehensive_vector_stats,
                "timestamp": __import__('datetime').datetime.now().isoformat()
            })
            current_state["vector_statistics"] = vector_statistics

            # Save state ONCE at end of iteration (message-passing architecture)
            self.shared_state.save_state(current_state, original_message.dataset, original_message.setting, original_message.batch_id)

            # Log to evaluation logger with comprehensive statistics (matching GraphRAG pattern)
            total_execution_time = (datetime.now() - step_start_time).total_seconds()

            eval_logger = get_global_evaluation_logger()
            # Log to old evaluation logger for backward compatibility
            eval_logger.log_iteration_evaluation(
                qa_pair_id=qa_pair_id,
                iteration=iteration,
                intermediate_outputs=intermediate_outputs,  # Use comprehensive intermediate outputs
                generated_answer=answer_response.generated_answer,
                rouge_scores={
                    "rouge-l": rouge_score,
                    "rouge-1": rouge_score,  # Use same score for ROUGE-1 (simplified)
                    "rouge-2": rouge_score * 0.85  # Estimate ROUGE-2 as typically lower
                },
                hyperparameters={"chunk_size": FIXED_CHUNK_SIZE},
                additional_metrics=comprehensive_vector_stats,
                retrieval_context=retrieval_response.retrieved_context,
                execution_time_seconds=total_execution_time
            )

            # Log to standardized evaluation logger
            self.standardized_logger.log_iteration_evaluation(
                qa_pair_id=qa_pair_id,
                iteration=iteration,
                generated_answer=answer_response.generated_answer,
                rouge_scores={
                    "rouge-l": rouge_score,
                    "rouge-1": rouge_score,  # Use same score for ROUGE-1 (simplified)
                    "rouge-2": rouge_score * 0.85  # Estimate ROUGE-2 as typically lower
                },
                intermediate_outputs=intermediate_outputs,
                hyperparameters={"chunk_size": FIXED_CHUNK_SIZE},
                execution_time_seconds=total_execution_time,
                system_specific_metrics=comprehensive_vector_stats
            )

            # Perform backward pass ONLY if evaluator determined answer needs refinement
            # The ResponseEvaluatorAgent sets continue_optimization=True when decision is NEEDS_REFINEMENT
            if eval_response.continue_optimization:
                print(f"[Orchestrator] Immediate backward pass trigger (continue_optimization=True) iteration={iteration}")
                self.logger.info(f"ðŸ”„ Starting backward pass for iteration {iteration} (evaluator determined answer needs refinement)")

                # Gather QA results for backward pass
                qa_result = {
                    "qa_pair_id": qa_pair_id,
                    "iteration": iteration,
                    "question": qa_pair.get("question", ""),
                    "expected_answer": qa_pair.get("answer", ""),
                    "generated_answer": answer_response.generated_answer,
                    "retrieved_context": retrieval_response.retrieved_context,
                    "evaluation": eval_response.evaluation_result,
                    "rouge_score": rouge_score,
                    "chunk_size": FIXED_CHUNK_SIZE
                }

                backward_pass_msg = BackwardPassStartMessage(
                    batch_id=original_message.batch_id,
                    repetition=original_message.repetition,
                    dataset=original_message.dataset,
                    setting=original_message.setting,
                    all_qa_results=[qa_result]  # Include current QA result
                )

                try:
                    backward_agent_id = AgentId("backward_pass_agent", "default")
                    backward_response = await self.send_message(backward_pass_msg, backward_agent_id)
                    print(f"[Orchestrator] Immediate BackwardPassReady received for iteration={iteration}")
                    self.logger.info(f"âœ“ Backward pass completed for iteration {iteration}")

                    # Process backward pass response to update QA pair prompts (like GraphRAG does)
                    await self._process_backward_pass_response(backward_response, ctx)
                except Exception as e:
                    self.logger.error(f"âŒ Backward pass failed: {e}")
                    self.logger.error(f"Make sure 'backward_pass_agent' is registered in your runtime!")
                    self.logger.warning(f"Continuing without backward pass optimization...")
            else:
                self.logger.info(f"â­ï¸  Skipping backward pass for iteration {iteration} (evaluator determined answer is satisfactory)")

            # Return iteration results
            return {
                "qa_pair_id": qa_pair_id,
                "iteration": iteration,
                "question": qa_pair.get("question", ""),
                "expected_answer": qa_pair.get("answer", ""),
                "generated_answer": answer_response.generated_answer,
                "retrieved_context": retrieval_response.retrieved_context,
                "evaluation": eval_response.evaluation_result,
                "rouge_score": rouge_score,
                "chunk_size": FIXED_CHUNK_SIZE,
                "comprehensive_vector_statistics": comprehensive_vector_stats,
                "vector_stats": {
                    "faiss_index_path": vector_response.faiss_index_path,
                    "chunk_metadata_path": vector_response.chunk_metadata_path,
                    "total_chunks": getattr(vector_response, 'total_chunks', 0)
                }
            }

        except Exception as e:
            self.logger.error(f"Error in iteration {iteration} for QA pair {qa_pair_id}: {e}")
            # Return error result
            return {
                "qa_pair_id": qa_pair_id,
                "iteration": iteration,
                "error": str(e),
                "rouge_score": 0.0
            }

    def _collect_comprehensive_vector_statistics(self, vector_response, retrieval_response, chunk_size: int) -> Dict[str, Any]:
        """
        Collect comprehensive vector statistics for evaluation logging.

        Args:
            vector_response: Response from VectorBuilderAgent
            retrieval_response: Response from VectorRetrievalAgent
            chunk_size (int): Chunk size used for vectorization

        Returns:
            Dict[str, Any]: Comprehensive vector statistics
        """
        try:
            import os
            import pickle
            import numpy as np

            stats = {
                "chunk_size": chunk_size,
                "faiss_index_path": getattr(vector_response, 'faiss_index_path', ''),
                "chunk_metadata_path": getattr(vector_response, 'chunk_metadata_path', ''),
                "total_chunks": getattr(vector_response, 'total_chunks', 0),
                "retrieved_context_length": len(getattr(retrieval_response, 'retrieved_context', '')),
                "retrieval_method": "faiss_vector_similarity"
            }

            # Try to get detailed vector statistics if FAISS index exists
            try:
                faiss_index_path = getattr(vector_response, 'faiss_index_path', '')
                chunk_metadata_path = getattr(vector_response, 'chunk_metadata_path', '')

                if os.path.exists(faiss_index_path) and os.path.exists(chunk_metadata_path):
                    # Load metadata to get chunk information
                    with open(chunk_metadata_path, 'rb') as f:
                        chunk_metadata = pickle.load(f)

                    if chunk_metadata:
                        # Calculate chunk statistics
                        chunk_lengths = [len(chunk['text']) for chunk in chunk_metadata if 'text' in chunk]
                        if chunk_lengths:
                            stats.update({
                                "total_indexed_chunks": len(chunk_metadata),
                                "avg_chunk_length": np.mean(chunk_lengths),
                                "min_chunk_length": np.min(chunk_lengths),
                                "max_chunk_length": np.max(chunk_lengths),
                                "std_chunk_length": np.std(chunk_lengths),
                                "total_text_length": sum(chunk_lengths)
                            })

                        # Calculate embedding dimension if available
                        if 'embeddings' in chunk_metadata[0]:
                            embedding_dim = len(chunk_metadata[0]['embeddings'])
                            stats["embedding_dimension"] = embedding_dim

                    # Get FAISS index statistics
                    try:
                        import faiss
                        index = faiss.read_index(faiss_index_path)
                        stats.update({
                            "faiss_index_size": index.ntotal,
                            "faiss_index_dimension": index.d,
                            "faiss_index_type": type(index).__name__
                        })
                    except Exception as e:
                        self.logger.warning(f"Could not load FAISS index for statistics: {e}")

            except Exception as e:
                self.logger.warning(f"Could not collect detailed vector statistics: {e}")

            # Add retrieval-specific statistics
            if hasattr(retrieval_response, 'retrieved_context'):
                retrieved_context = retrieval_response.retrieved_context
                if retrieved_context:
                    # Count retrieved chunks (assuming they're separated by some delimiter)
                    retrieved_chunks_count = len([chunk for chunk in retrieved_context.split('\n') if chunk.strip()])
                    stats.update({
                        "retrieved_chunks_count": retrieved_chunks_count,
                        "avg_retrieved_chunk_length": len(retrieved_context) / max(retrieved_chunks_count, 1)
                    })

            # Add vectorization efficiency metrics
            if stats.get("total_chunks", 0) > 0 and stats.get("total_text_length", 0) > 0:
                stats.update({
                    "chunks_per_1k_chars": (stats["total_chunks"] * 1000) / stats["total_text_length"],
                    "vectorization_efficiency": stats["total_chunks"] / stats.get("chunk_size", 1)
                })

            return stats

        except Exception as e:
            self.logger.error(f"Error collecting vector statistics: {e}")
            return {
                "chunk_size": chunk_size,
                "error": str(e)
            }

    async def _process_hyperparameters_response(self, hyperparams_response: HyperparametersVectorReadyMessage,
                                              original_message: BatchStartMessage, ctx: MessageContext) -> None:
        """Process the hyperparameters response and continue the pipeline."""
        self.logger.info(f"Processing hyperparameters response for QA pair {hyperparams_response.qa_pair_id}")

        # Load current shared state to pass to VectorBuilderAgent
        current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, hyperparams_response.batch_id)

        # Send VectorStart message with chunk_size
        vector_start_msg = VectorStartMessage(
            batch_id=hyperparams_response.batch_id,
            repetition=hyperparams_response.repetition,
            chunk_size=hyperparams_response.chunk_size,
            dataset=self.current_dataset,
            setting=self.current_setting,
            shared_state=current_state
        )

        self.logger.info(f"Sending VectorStart for batch {hyperparams_response.batch_id} with chunk_size {hyperparams_response.chunk_size}")

        # Send to VectorBuilderAgent and get response
        try:
            vector_builder_agent_id = AgentId("vector_builder_agent", "default")
            vector_response = await self.send_message(vector_start_msg, vector_builder_agent_id)
            self.logger.info(f"Received VectorReady response")

            # Continue processing with vector response
            await self._process_vector_response(vector_response, ctx)

        except Exception as e:
            self.logger.warning(f"VectorBuilderAgent not available, falling back to simulation: {e}")
            # Fallback to simulation if VectorBuilderAgent is not available
            await self._simulate_vector_ready(vector_start_msg, ctx)

    async def _process_vector_response(self, vector_response: VectorReadyMessage, ctx: MessageContext) -> None:
        """Process vector response and continue with retrieval."""
        self.logger.info(f"Processing vector response for batch {vector_response.batch_id}")

        # Load current shared state to get all QA pairs
        current_state = self.shared_state.load_state(vector_response.dataset, vector_response.setting, vector_response.batch_id)
        batch_info = current_state.get("batch_information", {})
        qa_pairs = batch_info.get("qa_pairs", [])

        # Process each QA pair for retrieval
        for i, qa_pair in enumerate(qa_pairs):
            qa_pair_id = qa_pair.get("question_id", f"qa_{i}")
            question = qa_pair.get("question", "")

            # Send VectorRetrievalStart message
            retrieval_start_msg = VectorRetrievalStartMessage(
                batch_id=vector_response.batch_id,
                repetition=vector_response.repetition,
                qa_pair_id=qa_pair_id,
                query=question,
                dataset=vector_response.dataset,
                setting=vector_response.setting,
                k_iterations=5,
                shared_state=current_state
            )

            self.logger.info(f"Sending VectorRetrievalStart for batch {vector_response.batch_id}, QA pair {qa_pair_id}, question: {question[:50]}...")

            # Send to VectorRetrievalPlannerAgent
            try:
                retrieval_agent_id = AgentId("vector_retrieval_planner_agent", "default")
                retrieval_response = await self.send_message(retrieval_start_msg, retrieval_agent_id)
                self.logger.info(f"Received VectorRetrievalReady response for QA pair {qa_pair_id}")

                # Continue with answer generation FOR THIS SPECIFIC QA PAIR
                await self._process_single_qa_retrieval(retrieval_response, qa_pair, qa_pair_id, ctx)

            except Exception as e:
                self.logger.warning(f"VectorRetrievalPlannerAgent not available, falling back to simulation: {e}")
                # Fallback to simulation if agent is not available
                await self._simulate_retrieval_ready(retrieval_start_msg, ctx)

    async def _process_single_qa_retrieval(self, retrieval_response: VectorRetrievalReadyMessage,
                                          qa_pair: Dict[str, Any], qa_pair_id: str, ctx: MessageContext) -> None:
        """Process retrieval, answer generation, and evaluation for a single QA pair."""
        self.logger.info(f"Processing single QA pair {qa_pair_id} for batch {retrieval_response.batch_id}")

        # Use state from message (message-passing architecture - no I/O)
        current_state = retrieval_response.shared_state.copy()

        # Store retrieved context for this QA pair
        retrieved_contexts = current_state.get("retrieved_contexts", [])
        retrieved_contexts.append(retrieval_response.retrieved_context)
        current_state["retrieved_contexts"] = retrieved_contexts

        # Store the query associated with this retrieval
        retrieval_queries = current_state.get("retrieval_queries", [])
        retrieval_queries.append(qa_pair.get("question", ""))
        current_state["retrieval_queries"] = retrieval_queries

        # Pass updated state to AnswerGenerationStart message
        answer_gen_msg = AnswerGenerationStartMessage(
            qa_pair_id=qa_pair_id,
            question=qa_pair.get("question", ""),
            retrieved_context=retrieval_response.retrieved_context,
            batch_id=retrieval_response.batch_id,
            repetition=retrieval_response.repetition,
            dataset=retrieval_response.dataset,
            setting=retrieval_response.setting,
            shared_state=current_state  # Pass state through message
        )

        self.logger.info(f"Sending AnswerGenerationStart for QA pair {qa_pair_id}")

        # Send to AnswerGeneratorAgent
        try:
            answer_gen_agent_id = AgentId("answer_generator_agent", "default")
            answer_response = await self.send_message(answer_gen_msg, answer_gen_agent_id)
            self.logger.info(f"Received AnswerGenerationReady response for QA pair {qa_pair_id}")

            # Continue with evaluation for THIS QA pair
            await self._process_answer_response(answer_response, ctx)

        except Exception as e:
            self.logger.warning(f"AnswerGeneratorAgent not available, falling back to simulation: {e}")
            await self._simulate_answer_generation_ready(answer_gen_msg, ctx)

    async def _process_answer_response(self, answer_response: AnswerGenerationReadyMessage, ctx: MessageContext) -> None:
        """Process answer generation response and continue with evaluation."""
        self.logger.info(f"Processing answer response for QA pair {answer_response.qa_pair_id}")
        print(f"[Orchestrator] _process_answer_response START qa_pair_id={answer_response.qa_pair_id}")

        qa_pair = self.current_batch_qa_pairs.get(answer_response.qa_pair_id)
        if not qa_pair:
            self.logger.error(f"QA pair {answer_response.qa_pair_id} not found")
            return

        # Compute ROUGE score using real implementation
        rouge_score = self._compute_rouge_score(qa_pair, answer_response.generated_answer)
        print(f"[Orchestrator] Computed ROUGE-L={rouge_score:.4f} for qa_pair_id={answer_response.qa_pair_id}")

        self.logger.info(f"Computed ROUGE score {rouge_score:.4f} for QA pair {answer_response.qa_pair_id}")

        # Use state from message (message-passing architecture - no I/O)
        current_state = answer_response.shared_state.copy()
        rouge_scores_list = current_state.get("rouge_scores", [])
        rouge_scores_list.append(rouge_score)
        current_state["rouge_scores"] = rouge_scores_list

        # Log to evaluation logger
        eval_logger = get_global_evaluation_logger()
        rouge_scores = {
            "rouge-l": rouge_score,
            "rouge-1": rouge_score,  # Simplified - using same score
            "rouge-2": rouge_score * 0.85  # Estimate
        }

        # Create comprehensive intermediate outputs for evaluation logging (matching GraphRAG format)
        intermediate_outputs = {
            "hyperparameters": {
                "chunk_size": current_state.get("rag_hyperparameters", {}).get("chunk_size", 150),
                "processing_time_ms": 0,  # Not available in this context
                "learned_prompt_used": False  # Not available in this context
            },
            "vector_building": {
                "chunk_size": current_state.get("rag_hyperparameters", {}).get("chunk_size", 150),
                "faiss_index_path": "vector_indexes/cached",  # Cached from previous processing
                "vectors_created": True,
                "embedding_method": "openai",
                "processing_time_ms": 0  # Not available in this context
            },
            "retrieval": {
                "query": qa_pair.get("question", ""),
                "retrieved_context": current_state.get("retrieved_contexts", [""])[-1][:200] + "..." if len(current_state.get("retrieved_contexts", [""])[-1]) > 200 else current_state.get("retrieved_contexts", [""])[-1],
                "retrieved_context_length": len(current_state.get("retrieved_contexts", [""])[-1]) if current_state.get("retrieved_contexts") else 0,
                "processing_time_ms": 0  # Not available in this context
            },
            "answer_generation": {
                "question": qa_pair.get("question", ""),
                "generated_answer": answer_response.generated_answer[:200] + "..." if len(answer_response.generated_answer) > 200 else answer_response.generated_answer,
                "generated_answer_length": len(answer_response.generated_answer),
                "processing_time_ms": 0  # Not available in this context
            },
            "evaluation": {
                "evaluation_feedback": "N/A",  # Not available in this simplified path
                "rouge_score": rouge_score,
                "gold_answers": qa_pair.get("answers", []),
                "processing_time_ms": 0  # Not available in this context
            }
        }

        # Log to old evaluation logger for backward compatibility
        eval_logger.log_iteration_evaluation(
            qa_pair_id=answer_response.qa_pair_id,
            iteration=0,  # TODO: Will be updated when multi-iteration is implemented
            intermediate_outputs=intermediate_outputs,
            generated_answer=answer_response.generated_answer,
            rouge_scores=rouge_scores,
            hyperparameters={"chunk_size": current_state.get("rag_hyperparameters", {}).get("chunk_size", 150)},
            graph_metrics={"vector_count": len(current_state.get("retrieved_contexts", [])), "retrieval_method": "vector_similarity"},
            retrieval_context=current_state.get("retrieved_contexts", [""])[-1] if current_state.get("retrieved_contexts") else "",
            additional_metrics={"qa_pair_question": qa_pair.get("question", "N/A")}
        )

        # Log to standardized evaluation logger
        if self.standardized_logger:
            self.standardized_logger.log_iteration_evaluation(
                qa_pair_id=answer_response.qa_pair_id,
                iteration=0,  # TODO: Will be updated when multi-iteration is implemented
                generated_answer=answer_response.generated_answer,
                rouge_scores=rouge_scores,
                intermediate_outputs=intermediate_outputs,
                hyperparameters={"chunk_size": current_state.get("rag_hyperparameters", {}).get("chunk_size", 150)},
                execution_time_seconds=None,  # Not available in this context
                system_specific_metrics={"vector_count": len(current_state.get("retrieved_contexts", [])), "retrieval_method": "vector_similarity"}
            )

        # Send ResponseEvaluationStart message
        # Debug: check qa_pair structure
        print(f"\n[DEBUG] QA Pair available keys: {list(qa_pair.keys())}")

        # Get document text from shared state (where DatasetAgent stores it)
        document_text = current_state.get("full_document_text", "")
        if document_text:
            print(f"[DEBUG] Document text retrieved from shared state, length: {len(document_text)} chars")
        else:
            print(f"[DEBUG] âš ï¸ No document text in shared state!")

        # Extract gold answers
        gold_answer = qa_pair.get("answer") or qa_pair.get("answers") or qa_pair.get("expected_answer") or ""
        if isinstance(gold_answer, list):
            gold_answers = gold_answer
        else:
            gold_answers = [gold_answer] if gold_answer else []

        # Get the latest retrieved context from state
        retrieved_contexts = current_state.get("retrieved_contexts", [])
        latest_retrieved_context = retrieved_contexts[-1] if retrieved_contexts else ""

        eval_start_msg = ResponseEvaluationStartMessage(
            qa_pair_id=answer_response.qa_pair_id,
            original_query=qa_pair.get("question", ""),
            generated_answer=answer_response.generated_answer,
            gold_answers=gold_answers,
            rouge_score=rouge_score,
            batch_id=answer_response.batch_id,
            repetition=answer_response.repetition,
            dataset=self.current_dataset,
            setting=self.current_setting,
            document_text=document_text,  # For G-Eval computation
            retrieved_context=latest_retrieved_context,  # Pass retrieved context for evaluation
            shared_state=current_state  # Pass state through message
        )

        self.logger.info(f"Sending ResponseEvaluationStart for QA pair {answer_response.qa_pair_id}")
        print(f"[Orchestrator] Sending ResponseEvaluationStart qa_pair_id={answer_response.qa_pair_id}")

        # Send to ResponseEvaluatorAgent
        try:
            response_eval_agent_id = AgentId("response_evaluator_agent", "default")
            eval_response = await self.send_message(eval_start_msg, response_eval_agent_id)
            print(f"[Orchestrator] Received ResponseEvaluationReady qa_pair_id={answer_response.qa_pair_id} continue={getattr(eval_response,'continue_optimization',None)}")
            self.logger.info(f"Received ResponseEvaluationReady response")

            # Continue with final processing
            await self._process_evaluation_response(eval_response, ctx)

        except Exception as e:
            self.logger.warning(f"ResponseEvaluatorAgent not available, falling back to simulation: {e}")
            # Fallback to simulation if agent is not available
            print(f"[Orchestrator] ResponseEvaluatorAgent unavailable, simulating evaluation")
            await self._simulate_response_evaluation_ready(eval_start_msg, ctx)

    async def _process_evaluation_response(self, eval_response: ResponseEvaluationReadyMessage, ctx: MessageContext) -> None:
        """Process evaluation response and track completion."""
        self.logger.info(f"Processing evaluation response for QA pair {eval_response.qa_pair_id}")
        print(f"[Orchestrator] _process_evaluation_response qa_pair_id={eval_response.qa_pair_id} continue={getattr(eval_response,'continue_optimization',None)}")

        # Note: ResponseEvaluatorAgent already stores evaluation results in response_evaluations
        # with the proper structured format needed for backward pass. No need to duplicate here.

        # Save state ONCE at the end of QA processing (message-passing architecture)
        current_state = eval_response.shared_state
        self.shared_state.save_state(current_state, self.current_dataset, self.current_setting, eval_response.batch_id)
        self.logger.info(f"Saved final state for QA pair {eval_response.qa_pair_id}")

        # Mark QA pair as completed
        self.completed_qa_pairs[eval_response.qa_pair_id] = eval_response.evaluation_result

        # Check if all QA pairs are completed
        if len(self.completed_qa_pairs) == len(self.current_batch_qa_pairs):
            self.logger.info(f"All {len(self.completed_qa_pairs)} QA pairs completed for batch {eval_response.batch_id}")
            print(f"[Orchestrator] All QA pairs completed. Starting context summary then backward pass")

            # Generate context summary before backward pass
            await self._generate_context_summary(eval_response.batch_id, eval_response.repetition, ctx)

            # Now start the backward pass
            await self._start_backward_pass(eval_response.batch_id, eval_response.repetition, ctx)

    async def _generate_context_summary(self, batch_id: int, repetition: int, ctx: MessageContext) -> None:
        """Concatenate all retrieved contexts (NO SUMMARIZATION)."""
        self.logger.info(f"Concatenating contexts for batch {batch_id}")
        print(f"[Orchestrator] _generate_context_summary START batch_id={batch_id}")

        # Load current shared state
        current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, batch_id)
        retrieved_contexts = current_state.get("retrieved_contexts", [])

        if not retrieved_contexts:
            self.logger.warning("No retrieved contexts to concatenate")
            print("[Orchestrator] No retrieved contexts to concatenate â€” skipping")
            return

        # Concatenate contexts directly without summarization
        self.logger.info(f"Concatenating {len(retrieved_contexts)} contexts")
        concatenated_contexts = "\n\n--- Context Separator ---\n\n".join(retrieved_contexts)
        current_state["context_summary"] = concatenated_contexts

        # Save state with concatenated contexts
        saved = self.shared_state.save_state(current_state, self.current_dataset, self.current_setting, batch_id)
        self.logger.info(f"Concatenated contexts saved to shared state for batch {batch_id}, length: {len(concatenated_contexts)} chars")
        print(f"[Orchestrator] Saved concatenated contexts to state. saved={saved}")

    # Continue with other methods...
    async def _start_backward_pass(self, batch_id: int, repetition: int, ctx: MessageContext) -> None:
        """Start the backward pass."""
        all_qa_results = list(self.completed_qa_pairs.values())
        print(f"[Orchestrator] _start_backward_pass START batch_id={batch_id} qa_results={len(all_qa_results)}")

        backward_pass_msg = BackwardPassStartMessage(
            batch_id=batch_id,
            repetition=repetition,
            dataset=self.current_dataset,
            setting=self.current_setting,
            all_qa_results=all_qa_results
        )

        self.logger.info(f"Starting backward pass for batch {batch_id}")
        print(f"[Orchestrator] Sending BackwardPassStart to backward_pass_agent")

        # Send to BackwardPassAgent and get response
        try:
            backward_pass_agent_id = AgentId("backward_pass_agent", "default")
            backward_response = await self.send_message(backward_pass_msg, backward_pass_agent_id)
            self.logger.info(f"Received BackwardPassReady response")
            print(f"[Orchestrator] Received BackwardPassReady")

            # Process the backward pass response and send final result to DatasetAgent
            await self._process_backward_pass_response(backward_response, ctx)

        except Exception as e:
            self.logger.error(f"BackwardPassAgent failed: {e}", exc_info=True)
            self.logger.warning(f"Falling back to backward pass simulation due to error")
            print(f"[Orchestrator] BackwardPassAgent unavailable â€” simulating backward pass")
            # Fallback to simulation if agent is not available
            await self._simulate_backward_pass_ready(backward_pass_msg, ctx)

    async def _process_backward_pass_response(self, backward_response: BackwardPassReadyMessage, ctx: MessageContext) -> None:
        """Process backward pass response and send BatchReady to DatasetAgent."""
        self.logger.info(f"Processing backward pass response for batch {backward_response.batch_id}")
        print(f"[Orchestrator] _process_backward_pass_response START batch_id={backward_response.batch_id}")

        # Update QA pair prompts with optimized versions (like GraphRAG does)
        if hasattr(backward_response, 'backward_pass_results'):
            optimized_prompts = backward_response.backward_pass_results.get("optimized_prompts", {})
            current_qa_pair_id = self.shared_state.current_qa_pair_id
            if current_qa_pair_id and optimized_prompts:
                self.shared_state.update_qa_pair_prompts(current_qa_pair_id, optimized_prompts)
                print(f"[Orchestrator] Updating prompts for qa_pair_id={current_qa_pair_id} keys={list(optimized_prompts.keys())}")

                # CRITICAL FIX: Save optimized prompts to persistent state for next DatasetAgent call
                current_state = self.shared_state.load_state(backward_response.dataset, backward_response.setting, backward_response.batch_id)
                for prompt_key, prompt_value in optimized_prompts.items():
                    current_state[prompt_key] = prompt_value
                saved = self.shared_state.save_state(current_state, backward_response.dataset, backward_response.setting, backward_response.batch_id)
                print(f"[Orchestrator] Persisted optimized prompts. saved={saved}")

                self.logger.info(f"Updated QA pair prompts for {current_qa_pair_id} - preserving learned prompts for next iteration")
                self.logger.info(f"DEBUG: BatchOrchestrator - optimized_prompts saved to persistent state: {[(k, len(v)) for k, v in optimized_prompts.items()]}")
            else:
                print(f"[Orchestrator] Skipping immediate backward pass (continue_optimization=False)")
                self.logger.warning(f"Cannot update QA pair prompts - current_qa_pair_id: {current_qa_pair_id}, optimized_prompts: {bool(optimized_prompts)}")
                print(f"[Orchestrator] No optimized prompts to apply or missing current_qa_pair_id")

        # The BatchOrchestratorAgent's handle_batch_start method already returns the BatchReady message
        # This is the final step of the pipeline

    # Simulation methods for testing
    async def _simulate_vector_ready(self, vector_start: VectorStartMessage, ctx: MessageContext) -> None:
        vector_ready_msg = VectorReadyMessage(
            batch_id=vector_start.batch_id,
            repetition=vector_start.repetition,
            dataset=vector_start.dataset,
            setting=vector_start.setting
        )
        await self._process_vector_response(vector_ready_msg, ctx)

    async def _simulate_retrieval_ready(self, retrieval_start: VectorRetrievalStartMessage, ctx: MessageContext) -> None:
        # Store simulation data in shared state for backward pass critiques
        current_state = self.shared_state.load_state(retrieval_start.dataset, retrieval_start.setting, retrieval_start.batch_id)

        # Store mock retrieval prompt and plans (simulating what VectorRetrievalPlannerAgent would store)
        current_state["retrieval_prompt"] = "Mock retrieval prompt template for vector queries"
        current_state["retrieval_plans"] = ["Mock retrieval plan 1", "Mock retrieval plan 2"]

        # Store mock retrieved contexts for backward pass
        retrieved_contexts = current_state.get("retrieved_contexts", [])
        retrieved_contexts.append("Mock retrieved context from vector database")
        current_state["retrieved_contexts"] = retrieved_contexts

        self.shared_state.save_state(current_state, retrieval_start.dataset, retrieval_start.setting, retrieval_start.batch_id)

        retrieval_ready_msg = VectorRetrievalReadyMessage(
            batch_id=retrieval_start.batch_id,
            repetition=retrieval_start.repetition,
            retrieved_context="Mock retrieved context from vector database",
            dataset=retrieval_start.dataset,
            setting=retrieval_start.setting
        )
        await self._process_retrieval_response(retrieval_ready_msg, ctx)

    async def _simulate_answer_generation_ready(self, answer_gen: AnswerGenerationStartMessage, ctx: MessageContext) -> None:
        answer_ready_msg = AnswerGenerationReadyMessage(
            qa_pair_id=answer_gen.qa_pair_id,
            generated_answer="Mock generated answer based on retrieved context",
            batch_id=answer_gen.batch_id,
            repetition=answer_gen.repetition,
            shared_state=answer_gen.shared_state  # Pass state through simulation
        )
        await self._process_answer_response(answer_ready_msg, ctx)

    async def _simulate_response_evaluation_ready(self, eval_start: ResponseEvaluationStartMessage, ctx: MessageContext) -> None:
        eval_ready_msg = ResponseEvaluationReadyMessage(
            qa_pair_id=eval_start.qa_pair_id,
            evaluation_result={
                "rouge_score": eval_start.rouge_score,
                "generated_answer": eval_start.generated_answer,
                "evaluation_metrics": {"coherence": 0.8, "relevance": 0.9}
            },
            rouge_score=eval_start.rouge_score,
            continue_optimization=True,
            batch_id=eval_start.batch_id,
            repetition=eval_start.repetition,
            shared_state=eval_start.shared_state  # Pass state through simulation
        )
        await self._process_evaluation_response(eval_ready_msg, ctx)

    async def _simulate_backward_pass_ready(self, backward_pass: BackwardPassStartMessage, ctx: MessageContext) -> None:
        backward_ready_msg = BackwardPassReadyMessage(
            batch_id=backward_pass.batch_id,
            repetition=backward_pass.repetition,
            backward_pass_results={
                "total_qa_pairs": len(backward_pass.all_qa_results),
                "avg_rouge": statistics.mean([r.get("rouge_score", 0) for r in backward_pass.all_qa_results]),
                "critiques_generated": True
            },
            dataset=backward_pass.dataset,
            setting=backward_pass.setting
        )
        await self._process_backward_pass_response(backward_ready_msg, ctx)

    def _compute_rouge_score(self, qa_pair: Dict[str, Any], generated_answer: str) -> float:
        """
        Compute ROUGE score between generated answer and reference answers.

        Args:
            qa_pair: QA pair containing question and reference answers
            generated_answer: Generated answer to evaluate

        Returns:
            float: ROUGE-L score (F1)
        """
        try:
            reference_answers = qa_pair.get("answers", [])
            if not reference_answers or not generated_answer:
                return 0.0

            # Use the first reference answer for ROUGE computation
            reference = reference_answers[0] if isinstance(reference_answers, list) else str(reference_answers)

            # Simple ROUGE-L implementation using LCS (Longest Common Subsequence)
            def lcs_length(s1: str, s2: str) -> int:
                """Compute length of longest common subsequence."""
                words1 = s1.lower().split()
                words2 = s2.lower().split()
                m, n = len(words1), len(words2)

                # Create DP table
                dp = [[0] * (n + 1) for _ in range(m + 1)]

                # Fill DP table
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if words1[i-1] == words2[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])

                return dp[m][n]

            # Compute ROUGE-L
            lcs_len = lcs_length(reference, generated_answer)
            ref_len = len(reference.split())
            gen_len = len(generated_answer.split())

            if ref_len == 0 and gen_len == 0:
                return 1.0
            elif ref_len == 0 or gen_len == 0:
                return 0.0

            # ROUGE-L F1 score
            recall = lcs_len / ref_len if ref_len > 0 else 0.0
            precision = lcs_len / gen_len if gen_len > 0 else 0.0

            if recall + precision == 0:
                return 0.0

            f1_score = 2 * (recall * precision) / (recall + precision)
            return round(f1_score, 4)

        except Exception as e:
            self.logger.error(f"Error computing ROUGE score: {e}")
            return 0.0


# ===== HYPERPARAMETERS VECTOR AGENT =====

class HyperparametersVectorAgent(RoutedAgent):
    """
    Agent that determines vector construction hyperparameters using LLM reasoning.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.hyperparameters_vector")
        self.shared_state = SharedState("agent_states")

        # Initialize Gemini model client with structured output
        self.model_client = OpenAIChatCompletionClient(
            model="gemini-2.5-flash-lite",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=llm_keys.GEMINI_KEY,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            },
            response_format=HyperparametersVectorResponse
        )



        # Base prompt for hyperparameters determination
        from parameters import base_prompt_hyperparameters_vector
        self.base_prompt_hyperparameters_vector = base_prompt_hyperparameters_vector

    @message_handler
    async def handle_hyperparameters_vector_start(
        self, message: HyperparametersVectorStartMessage, ctx: MessageContext
    ) -> HyperparametersVectorReadyMessage:
        """Handle HyperparametersVectorStart message and generate hyperparameters using LLM."""
        self.logger.info(f"HyperparametersVectorAgent processing QA pair {message.qa_pair_id}")

        try:
            # Load shared state to get learned system prompt
            current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)

            # For first repetition (repetition=0), start with empty system prompt to avoid data leakage
            if message.repetition == 0:
                learned_system_prompt = ""
                self.logger.info(f"First repetition for QA pair {message.qa_pair_id} - using empty system prompt")
            else:
                learned_system_prompt = current_state.get("learned_prompt_hyperparameters_vector", "")

            # Extract question from QA pair
            qa_pair = message.qa_pair
            question = qa_pair.get("question", "")

            # Use generic description for text sample
            text_sample = "Science-fiction short stories"

            # Prepare base prompt (without critique)
            prompt_content = self.base_prompt_hyperparameters_vector.format(
                text=text_sample,
                question=question
            ) + "\n Chunk size cannot be lower than 128."

            # Call LLM with structured output using learned system prompt
            system_message = SystemMessage(content=learned_system_prompt)
            user_message = UserMessage(content=prompt_content, source="user")

            response = await self.model_client.create(
                [system_message, user_message],
                cancellation_token=ctx.cancellation_token
            )

            # Parse structured response
            assert isinstance(response.content, str)
            hyperparams_response = HyperparametersVectorResponse.model_validate_json(response.content)

            # Log LLM interaction
            logger = get_global_prompt_logger()
            logger.log_interaction(
                agent_name="HyperparametersVectorAgent",
                interaction_type="hyperparameters_recommendation",
                system_prompt=learned_system_prompt,
                user_prompt=prompt_content,
                llm_response=response.content if isinstance(response.content, str) else str(response.content),
                batch_id=message.batch_id,
                qa_pair_id=message.qa_pair_id,
                iteration=message.repetition,
                additional_metadata={
                    "text_sample": text_sample,
                    "chunk_size": hyperparams_response.chunk_size,
                    "confidence": hyperparams_response.confidence_score,
                    "reasoning_length": len(hyperparams_response.reasoning)
                }
            )

            log_agent_action(self.logger, "HyperparametersVector", "LLM recommendation",
                            chunk_size=hyperparams_response.chunk_size,
                            confidence=f"{hyperparams_response.confidence_score:.2f}",
                            reasoning=hyperparams_response.reasoning)

            # Save chunk_size to shared state
            rag_hyperparams = current_state.get("rag_hyperparameters", {})
            rag_hyperparams["chunk_size"] = hyperparams_response.chunk_size
            rag_hyperparams["chunk_size_reasoning"] = hyperparams_response.reasoning
            rag_hyperparams["chunk_size_confidence"] = hyperparams_response.confidence_score
            current_state["rag_hyperparameters"] = rag_hyperparams

            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Return HyperparametersVectorReady message
            ready_msg = HyperparametersVectorReadyMessage(
                qa_pair_id=message.qa_pair_id,
                chunk_size=hyperparams_response.chunk_size,
                batch_id=message.batch_id,
                repetition=message.repetition
            )

            self.logger.info(f"Returning HyperparametersVectorReady for QA pair {message.qa_pair_id}")
            return ready_msg

        except Exception as e:
            self.logger.error(f"Error in LLM call: {e}")
            # Return default response on error
            return HyperparametersVectorReadyMessage(
                qa_pair_id=message.qa_pair_id,
                chunk_size=150,  # Default chunk size (with 25% overlap)
                batch_id=message.batch_id,
                repetition=message.repetition
            )

    async def close(self) -> None:
        """Close the model client."""
        await self.model_client.close()


# ===== VECTOR BUILDER AGENT =====

class VectorBuilderAgent(RoutedAgent):
    """
    Agent that builds FAISS vector index from text chunks using embeddings.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.vector_builder")
        self.shared_state = SharedState("agent_states")

        # Store the FAISS index
        self.faiss_index = None
        self.chunk_metadata = []

    @message_handler
    async def handle_vector_start(self, message: VectorStartMessage, ctx: MessageContext) -> VectorReadyMessage:
        """Handle VectorStart message by reusing or building FAISS index."""
        self.logger.info(f"VectorBuilderAgent processing batch {message.batch_id} with chunk_size {message.chunk_size} (iteration {message.repetition})")

        # Check if index already exists (reuse across iterations)
        # Use document_index from shared_state for consistent naming across QA pairs from same document
        import os
        index_dir = "vector_indexes"

        # Get document_index from shared_state (if available) for fault-tolerant index naming
        document_index = message.shared_state.get("document_index", message.batch_id)

        index_filename = os.path.join(index_dir, f"{message.dataset}_{message.setting}_batch_{document_index}.index")
        metadata_filename = os.path.join(index_dir, f"{message.dataset}_{message.setting}_batch_{document_index}_metadata.json")

        if os.path.exists(index_filename) and os.path.exists(metadata_filename):
            self.logger.info(f"Reusing existing FAISS index from {index_filename} (iteration {message.repetition})")

            # Load metadata to get total chunks
            with open(metadata_filename, 'r', encoding='utf-8') as f:
                chunk_metadata = json.load(f)

            return VectorReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                dataset=message.dataset,
                setting=message.setting,
                faiss_index_path=index_filename,
                chunk_metadata_path=metadata_filename,
                total_chunks=len(chunk_metadata),
                index_reused=True
            )

        # Index doesn't exist - build it
        self.logger.info(f"Building new FAISS index (iteration {message.repetition})")
        self._reset_vector_store()

        # Load shared state
        current_state = message.shared_state
        batch_info = current_state.get("batch_information", {})
        example_info = current_state.get("example_information", {})
        corpus = current_state.get("full_document_text", batch_info.get("document_text", example_info.get("document_text", "")))

        if not corpus:
            self.logger.error("No document text found in batch information")
            return VectorReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                dataset=message.dataset,
                setting=message.setting,
                faiss_index_path="",
                chunk_metadata_path="",
                total_chunks=0,
                index_reused=False
            )

        # Split corpus into chunks
        chunks = self._split_text_into_chunks(corpus, message.chunk_size)
        self.logger.info(f"Split corpus into {len(chunks)} chunks")

        # Process chunks concurrently for embedding (similar to GraphRAG async pattern)
        self.logger.info(f"Starting concurrent embedding generation for {len(chunks)} chunks")
        all_embeddings = []
        self.chunk_metadata = []

        # Import the async embedding function
        from llm import get_embeddings_async
        import asyncio

        # Create batches of 8 chunks each for optimal API usage
        batch_size = 8
        chunk_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

        # Define async function to process each batch
        async def process_batch_with_index(batch_idx: int, batch_chunks: List[str]) -> tuple:
            """Process a single batch of chunks concurrently."""
            try:
                self.logger.info(f"Processing embedding batch {batch_idx + 1}/{len(chunk_batches)}")
                batch_embeddings = await get_embeddings_async(batch_chunks)

                # Create metadata for this batch
                batch_metadata = []
                for j, chunk_text in enumerate(batch_chunks):
                    chunk_id = batch_idx * batch_size + j
                    batch_metadata.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "batch_id": message.batch_id,  # Keep batch_id for state reference
                        "document_index": document_index,  # Store document_index for index naming
                        "dataset": message.dataset,
                        "setting": message.setting
                    })

                self.logger.info(f"Completed embedding batch {batch_idx + 1}/{len(chunk_batches)}")
                return batch_embeddings, batch_metadata

            except Exception as e:
                self.logger.error(f"Error processing embedding batch {batch_idx + 1}: {e}")
                # Return empty results for failed batch
                return [], []

        # Execute all batch processing tasks concurrently with controlled concurrency
        self.logger.info(f"Starting concurrent processing of {len(chunk_batches)} embedding batches")

        # Limit concurrent API calls to prevent rate limiting (max 5 concurrent batches)
        semaphore = asyncio.Semaphore(5)

        async def limited_process_batch(batch_idx: int, batch_chunks: List[str]) -> tuple:
            """Process batch with concurrency limiting."""
            async with semaphore:
                return await process_batch_with_index(batch_idx, batch_chunks)

        tasks = [limited_process_batch(i, batch) for i, batch in enumerate(chunk_batches)]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred
        successful_results = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch {i+1} failed with exception: {result}")
                successful_results.append(([], []))  # Empty results for failed batch
            else:
                successful_results.append(result)

        batch_results = successful_results

        # Aggregate results from all batches
        for batch_embeddings, batch_metadata in batch_results:
            all_embeddings.extend(batch_embeddings)
            self.chunk_metadata.extend(batch_metadata)

        self.logger.info(f"Completed concurrent embedding generation for {len(all_embeddings)} embeddings")

        # Create FAISS index
        if all_embeddings:
            try:
                # Convert embeddings to numpy array
                embeddings_array = np.array(all_embeddings).astype('float32')

                # Create FAISS index
                dimension = embeddings_array.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)

                # Normalize vectors for cosine similarity
                faiss.normalize_L2(embeddings_array)

                # Add vectors to index
                self.faiss_index.add(embeddings_array)

                self.logger.info(f"Created FAISS index with {self.faiss_index.ntotal} vectors, dimension {dimension}")

                # Create BM25 index for hybrid retrieval
                self.logger.info(f"Creating BM25 index for hybrid retrieval...")
                tokenized_chunks = [chunk.lower().split() for chunk in chunks]
                bm25_index = BM25Okapi(tokenized_chunks)
                self.logger.info(f"Created BM25 index with {len(tokenized_chunks)} documents")

                # Save indexes to file for reuse (using document_index for consistent naming)
                import os
                index_dir = "vector_indexes"
                os.makedirs(index_dir, exist_ok=True)

                # Save FAISS index
                index_filename = os.path.join(index_dir, f"{message.dataset}_{message.setting}_batch_{document_index}.index")
                faiss.write_index(self.faiss_index, index_filename)

                # Save BM25 index
                bm25_filename = os.path.join(index_dir, f"{message.dataset}_{message.setting}_batch_{document_index}_bm25.pkl")
                with open(bm25_filename, 'wb') as f:
                    pickle.dump(bm25_index, f)

                # Save metadata
                metadata_filename = os.path.join(index_dir, f"{message.dataset}_{message.setting}_batch_{document_index}_metadata.json")
                with open(metadata_filename, 'w', encoding='utf-8') as f:
                    json.dump(self.chunk_metadata, f, indent=2, ensure_ascii=False)

                self.logger.info(f"Saved FAISS index, BM25 index, and metadata to {index_filename}")

            except Exception as e:
                self.logger.error(f"Error creating FAISS index: {e}")

        # Prepare paths for VectorReady message (using document_index for consistent naming)
        index_dir = "vector_indexes"
        index_filename = os.path.join(index_dir, f"{message.dataset}_{message.setting}_batch_{document_index}.index")
        metadata_filename = os.path.join(index_dir, f"{message.dataset}_{message.setting}_batch_{document_index}_metadata.json")

        # Send VectorReady message
        vector_ready_msg = VectorReadyMessage(
            batch_id=message.batch_id,
            repetition=message.repetition,
            dataset=message.dataset,
            setting=message.setting,
            faiss_index_path=index_filename,
            chunk_metadata_path=metadata_filename,
            total_chunks=len(chunks) if 'chunks' in locals() else 0,
            index_reused=False
        )

        self.logger.info(f"Returning VectorReady for batch {message.batch_id}")

        # Return the VectorReady message
        return vector_ready_msg

    def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks of approximately chunk_size words with 20-30% overlap.
        Respects sentence boundaries - chunks always start at the beginning of a sentence.

        Args:
            text: Input text to chunk
            chunk_size: Target chunk size in words (default: 150)

        Returns:
            List of text chunks with overlap
        """
        import re

        chunks = []
        overlap_ratio = 0.25  # 25% overlap (20-30% range)
        overlap_words = int(chunk_size * overlap_ratio)  # ~37-38 words for chunk_size=150

        # Split into paragraphs first (respect paragraph boundaries)
        paragraphs = re.split(r'\n\s*\n', text)

        all_sentences = []

        # Collect all sentences with their word counts
        for para in paragraphs:
            # Split paragraph into sentences (respect sentence boundaries)
            sentences = re.split(r'(?<=[.!?])\s+', para.strip())

            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    all_sentences.append(sentence)

        # Build chunks from complete sentences
        i = 0
        while i < len(all_sentences):
            current_chunk_sentences = []
            current_word_count = 0

            # Add sentences until we reach the target chunk size
            while i < len(all_sentences):
                sentence = all_sentences[i]
                sentence_word_count = len(sentence.split())

                # If this is the first sentence in the chunk, always add it (even if it exceeds chunk_size)
                if not current_chunk_sentences:
                    current_chunk_sentences.append(sentence)
                    current_word_count += sentence_word_count
                    i += 1
                # If adding this sentence would exceed chunk_size, stop here
                elif current_word_count + sentence_word_count > chunk_size:
                    break
                # Otherwise add the sentence
                else:
                    current_chunk_sentences.append(sentence)
                    current_word_count += sentence_word_count
                    i += 1

            # Create chunk from collected sentences
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))

            # Calculate overlap: go back to include sentences for overlap
            # Find how many sentences to go back to achieve ~overlap_words
            if i < len(all_sentences):  # If not at the end
                overlap_word_count = 0
                sentences_to_overlap = 0

                # Count backwards from current position to find overlap sentences
                for j in range(i - 1, -1, -1):
                    sentence_word_count = len(all_sentences[j].split())
                    if overlap_word_count + sentence_word_count <= overlap_words:
                        overlap_word_count += sentence_word_count
                        sentences_to_overlap += 1
                    else:
                        break

                # Move index back by the number of overlap sentences
                i = i - sentences_to_overlap

        return chunks

    def get_faiss_index(self):
        """Return the current FAISS index."""
        return self.faiss_index

    def get_chunk_metadata(self):
        """Return the current chunk metadata."""
        return self.chunk_metadata

    def _reset_vector_store(self):
        """Reset the vector store for reconstruction in test-time training."""
        self.faiss_index = None
        self.chunk_metadata = []
        self.logger.info("Vector store reset completed - cleared FAISS index and metadata")


# ===== VECTOR RETRIEVAL PLANNER AGENT =====

class VectorRetrievalPlannerAgent(RoutedAgent):
    """
    Agent that plans and executes vector retrieval strategies using iterative LLM calls.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.vector_retrieval_planner")
        self.shared_state = SharedState("agent_states")

        # Import response formats and prompts
        from parameters import base_prompt_vector_retrieval_planner, VectorRetrievalPlannerResponse

        # Initialize Gemini client directly (not using Autogen wrapper)
        self.gemini_client = genai.Client(api_key=llm_keys.GEMINI_KEY)
        self.gemini_model = "gemini-2.5-flash-lite"
        self.thinking_budget = 0

        # Convert Pydantic schema to dict and remove additionalProperties (Gemini doesn't support it)
        response_schema_dict = VectorRetrievalPlannerResponse.model_json_schema()
        if "additionalProperties" in response_schema_dict:
            del response_schema_dict["additionalProperties"]
        self.response_schema = response_schema_dict

        self.base_prompt_vector_retrieval_planner = base_prompt_vector_retrieval_planner
        # In-memory cache of last iteration's retrieved FAISS indices per QA
        # Keyed by dataset::setting::qa_pair_id -> List[int]
        self._last_indices_cache = {}

        # In-memory cache of query history across system iterations
        # Keyed by dataset::setting::qa_pair_id -> List[str]
        self._query_history_cache = {}

        # In-memory cache of cumulative context across system iterations
        # Keyed by dataset::setting::qa_pair_id::iterN -> str (context text)
        self._cumulative_context_cache = {}

        # In-memory cache of cumulative retrieved indices across system iterations
        # Keyed by dataset::setting::qa_pair_id::cumulative_indices -> List[int]
        self._cumulative_indices_cache = {}

    def _log_retrieval(self, qa_pair_id: str, dataset: str, setting: str, iteration: int,
                       query: str, hypothetical_doc: str, retrieved_docs: list) -> None:
        """Log retrieval query and retrieved chunks to a dedicated file."""
        import os
        from datetime import datetime

        # Create logs directory if it doesn't exist
        log_dir = "retrieval_logs"
        os.makedirs(log_dir, exist_ok=True)

        # Create log file path: retrieval_logs/{dataset}_{setting}_{qa_pair_id}.txt
        log_file = os.path.join(log_dir, f"{dataset}_{setting}_{qa_pair_id}.txt")

        # Prepare log entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"\n{'='*100}\n"
        log_entry += f"RETRIEVAL ITERATION {iteration}\n"
        log_entry += f"Timestamp: {timestamp}\n"
        log_entry += f"{'='*100}\n\n"
        log_entry += f"QUERY:\n{query}\n\n"
        log_entry += f"{'-'*100}\n"
        log_entry += f"RETRIEVED CHUNKS ({len(retrieved_docs)} total):\n"
        log_entry += f"{'-'*100}\n\n"

        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs, 1):
                chunk_text = doc.get('text', '')
                faiss_idx = doc.get('faiss_index', 'N/A')
                score = doc.get('score', 0.0)
                log_entry += f"CHUNK {i}:\n"
                log_entry += f"  FAISS Index: {faiss_idx}\n"
                log_entry += f"  Score: {score:.4f}\n"
                log_entry += f"  Content:\n{chunk_text}\n\n"
                log_entry += f"{'-'*100}\n\n"
        else:
            log_entry += "No chunks retrieved.\n\n"

        # Write to file (append mode)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)

    @message_handler
    async def handle_vector_retrieval_start(self, message: VectorRetrievalStartMessage, ctx: MessageContext) -> VectorRetrievalReadyMessage:
        """Handle VectorRetrievalStart message and execute iterative retrieval."""
        self.logger.info(f"VectorRetrievalPlannerAgent processing batch {message.batch_id} for query: {message.query}")

        # Use state from message (message-passing architecture - no I/O)
        current_state = message.shared_state.copy()

        # Get document_index from shared_state for fault-tolerant index naming
        document_index = current_state.get("document_index", message.batch_id)

        # Load the correct FAISS and BM25 indexes using document_index (not batch_id)
        faiss_index, chunk_metadata, bm25_index = self._load_faiss_index(message.dataset, message.setting, document_index)

        if faiss_index is None:
            self.logger.error(f"Could not load FAISS index for batch {message.batch_id}")
            return VectorRetrievalReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                retrieved_context="Error: Could not load vector index",
                dataset=message.dataset,
                setting=message.setting
            )

        # Use learned prompt as USER prefix; system prompt must remain empty
        learned_user_prompt = current_state.get("learned_prompt_vector_retrieval_planner", "")
        if not learned_user_prompt:
            # Fallback to a very basic base user prompt
            from parameters import base_prompt_vector_retrieval_planner
            learned_user_prompt = base_prompt_vector_retrieval_planner
            self.logger.info("Retrieval planner: using basic base user prompt")
        else:
            self.logger.info("Retrieval planner: using learned user prompt (system prompt kept empty)")

        # Create base user prompt template and save to shared state
        prompt_template = self.base_prompt_vector_retrieval_planner.format(
            message.query, "{RETRIEVED_CONTEXT}", "{DECISION_HISTORY}"
        )
        current_state["retrieval_prompt"] = prompt_template

        # Initialize retrieval context and plan responses
        retrieved_context = ""
        retrieval_plan_responses = []

        # Track history of queries made - initialize from cache if available
        cache_key = message.dataset + '::' + message.setting + '::' + message.qa_pair_id
        query_history = []

        # Load previous system iteration's query history from cache
        if message.repetition > 0:
            cached_query_history = self._query_history_cache.get(cache_key, [])
            if cached_query_history:
                query_history = list(cached_query_history)  # Make a copy
                print(f'[QUERY-HISTORY] Loaded {len(query_history)} queries from previous system iteration for QA {message.qa_pair_id}')
                self.logger.info(f"Loaded {len(query_history)} queries from previous system iteration")
            else:
                print(f'[QUERY-HISTORY] No cached query history found for QA {message.qa_pair_id}')
        else:
            print(f'[QUERY-HISTORY] Repetition 0 - starting with empty query history for QA {message.qa_pair_id}')

        all_selected_documents = []  # Track all selected documents across iterations
        used_document_indices = set()  # Will be initialized below
        query_content_pairs = []  # Track query -> retrieved content pairs for backward pass

        # ========== LOAD CUMULATIVE INDICES (Initialize exclusion set) ==========
        # Initialize used_document_indices with ALL previous iterations' indices
        # This will be passed to _execute_vector_search which filters based on these indices
        cumulative_indices_key = f"{cache_key}::cumulative_indices"
        if message.repetition > 0:
            cached_cumulative_indices = self._cumulative_indices_cache.get(cumulative_indices_key, [])
            used_document_indices = set(cached_cumulative_indices)
            print(f"[CUMULATIVE-CONTEXT] Initialized used_document_indices with {len(used_document_indices)} indices from previous iterations")
        else:
            used_document_indices = set()
            print(f"[CUMULATIVE-CONTEXT] Iteration 0: Initialized empty used_document_indices")

        # Pre-populate with first two chunks as starting point (only in iteration 0)
        if message.repetition == 0:
            print(f"\n{'='*80}")
            print(f"PRE-POPULATING CONTEXT WITH FIRST TWO CHUNKS - QA {message.qa_pair_id}")
            print(f"{'='*80}")

            if chunk_metadata and len(chunk_metadata) >= 2:
                initial_chunks = []
                for i in range(2):
                    chunk_doc = {
                        'index': i,
                        'faiss_index': i,
                        'score': 1.0,  # Initial chunks have max score
                        'text': chunk_metadata[i]["text"],
                        'initial_chunk': True
                    }
                    initial_chunks.append(chunk_doc)
                    used_document_indices.add(i)
                    all_selected_documents.append(chunk_doc)

                    # Print initial chunks
                    chunk_text = chunk_metadata[i]["text"]
                    display_text = chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text
                    print(f"Initial Chunk {i+1} (FAISS index: {i}):")
                    print(f"{display_text}")
                    print(f"{'-'*80}")

                # Format initial chunks for context
                chunk_parts = []
                for i, doc in enumerate(initial_chunks, 1):
                    chunk_text = doc.get('text', '')
                    chunk_parts.append(f"Document {i}:\n{chunk_text}")
                retrieved_context = "Initial context:\n" + "\n\n".join(chunk_parts)

                self.logger.info(f"Pre-populated context with first 2 chunks (indices 0, 1)")
                print(f"{'='*80}\n")
            else:
                self.logger.warning(f"Not enough chunks available for pre-population (found {len(chunk_metadata) if chunk_metadata else 0})")
        else:
            # In iterations > 0, load cumulative context from cache
            print(f"\n{'='*80}")
            print(f"LOADING CUMULATIVE CONTEXT FROM PREVIOUS ITERATIONS - QA {message.qa_pair_id}")
            print(f"{'='*80}")

            # Load all contexts from previous iterations
            all_previous_contexts = []
            for prev_iter in range(message.repetition):
                context_key = f"{cache_key}::iter{prev_iter}"
                prev_context = self._cumulative_context_cache.get(context_key, "")
                if prev_context:
                    all_previous_contexts.append(prev_context)
                    print(f"[CUMULATIVE-CONTEXT] Loaded context from iteration {prev_iter} ({len(prev_context)} chars)")

            retrieved_context = "\n\n".join(all_previous_contexts)
            print(f"[CUMULATIVE-CONTEXT] Total cumulative context: {len(retrieved_context)} chars from {len(all_previous_contexts)} iterations")
            print(f"{'='*80}\n")

        # Load previous system iteration's retrieved indices for overlap tracking
        previous_system_iteration_indices = None
        # Use in-memory cache first to obtain previous iteration indices
        cache_key = message.dataset + '::' + message.setting + '::' + message.qa_pair_id
        if message.repetition > 0:
            try:
                cached_prev = self._last_indices_cache.get(cache_key)
                if cached_prev is not None:
                    previous_system_iteration_indices = list(cached_prev)
                    print('[OVERLAP DEBUG - QA ' + message.qa_pair_id + '] Using in-memory cached indices from previous iteration: ' + str(len(cached_prev)))
            except Exception:
                pass

        print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Current repetition: {message.repetition}")

        if message.repetition > 0:
            # Load from persistent storage to get previous iteration's data
            prev_indices_key = f"retrieved_indices_qa_{message.qa_pair_id}_rep_{message.repetition - 1}"

            print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Looking for key: {prev_indices_key}")

            # Only query current state if cache didn't have it (don't overwrite cache result!)
            if previous_system_iteration_indices is None:
                previous_system_iteration_indices = current_state.get(prev_indices_key, None)
                print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Current state lookup result: {previous_system_iteration_indices is not None}")
            else:
                print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Skipping current state lookup (already found in cache)")

            if previous_system_iteration_indices:
                print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Found previous indices: {len(previous_system_iteration_indices)} indices")

            # If not found, try loading from all previous states
            if previous_system_iteration_indices is None:
                print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Not found in current state, searching persistent storage...")
                try:
                    all_states = self.shared_state.get_all_states(message.dataset, message.setting)
                    print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Found {len(all_states)} states in persistent storage")

                    # Search through all previous batches for this QA pair's indices
                    for i, batch_state in enumerate(all_states):
                        print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Checking state {i}, keys: {list(batch_state.keys())[:10]}...")  # Show first 10 keys
                        if prev_indices_key in batch_state:
                            previous_system_iteration_indices = batch_state[prev_indices_key]
                            print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] FOUND in state {i}! Indices: {previous_system_iteration_indices}")
                            self.logger.info(f"Loaded {len(previous_system_iteration_indices)} indices from persistent storage for previous iteration {message.repetition - 1}")
                            break

                    if previous_system_iteration_indices is None:
                        print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Key '{prev_indices_key}' not found in any of the {len(all_states)} states")

                except Exception as e:
                    print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] ERROR loading from persistent storage: {e}")
                    self.logger.warning(f"Could not load previous iteration indices: {e}")

            if previous_system_iteration_indices:
                print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] FINAL: Found previous iteration data: {len(previous_system_iteration_indices)} indices from iteration {message.repetition - 1}")
                self.logger.info(f"Found previous iteration data: {len(previous_system_iteration_indices)} indices from iteration {message.repetition - 1}")
            else:
                print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] FINAL: No previous iteration data found")
                self.logger.warning(f"No previous iteration data found for QA {message.qa_pair_id}, iteration {message.repetition - 1}")
        else:
            print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Repetition 0 - skipping overlap check")

        # Execute retrieval iterations (fixed number)
        # TODO: Make this a hyperparameter
        MAX_ITERATIONS = 6
        CHUNKS_PER_ITERATION = 1

        for iteration in range(MAX_ITERATIONS):
            self.logger.info(f"Retrieval iteration {iteration + 1}/{MAX_ITERATIONS}")

            try:
                # Format query history for the prompt
                if not query_history:
                    history_text = "No previous queries in this session."
                else:
                    history_parts = []
                    for i, query in enumerate(query_history, 1):
                        history_parts.append(f"{i}. {query}")
                    history_text = "\n".join(history_parts)

                # Format current context (summaries only)
                if not retrieved_context:
                    context_text = "No summaries retrieved yet."
                else:
                    context_text = retrieved_context

                # Build user prompt: learned prompt + required sections with same variables
                # Add explicit instruction about hypothetical documents
                hyde_instruction = """
IMPORTANT: You must generate BOTH a retrieval query AND a hypothetical document.

For the hypothetical document (~150 words):
- Write it as if it were an actual excerpt from a document
- REPRODUCE THE NARRATIVE STYLE of the story/text you are retrieving from
- Match the tone, voice, and writing style of the original text
- Take inspiration from the text already retrieved (if any)
- Make it concrete and specific (not abstract or generic)
- Focus on information that would help answer the question
- Generate approximately 150 words
- This hypothetical document will be embedded for semantic search
"""

                user_prompt_content = (
                    f"{learned_user_prompt}\n\n"
                    f"{hyde_instruction}\n\n"
                    f"Query to answer:\n{message.query}\n\n"
                    f"Retrieved summaries so far:\n{context_text}\n\n"
                    f"Previous queries you made:\n{history_text}"
                )

                # Call Gemini API directly with thinking budget
                from parameters import VectorRetrievalPlannerResponse

                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=user_prompt_content,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=self.thinking_budget
                        ),
                        response_mime_type="application/json",
                        response_schema=self.response_schema
                    )
                )

                # Parse structured response from Gemini
                response_text = response.text
                retrieval_response = VectorRetrievalPlannerResponse.model_validate_json(response_text)

                # Log LLM interaction
                logger = get_global_prompt_logger()
                logger.log_interaction(
                    agent_name="VectorRetrievalPlannerAgent",
                    interaction_type="retrieval_planning",
                    system_prompt="",
                    user_prompt=user_prompt_content,
                    llm_response=response_text,
                    qa_pair_id=message.qa_pair_id,
                    iteration=message.repetition,
                    additional_metadata={
                        "iteration": iteration + 1,
                        "query": retrieval_response.query,
                        "query_history_length": len(query_history)
                    }
                )

                # Store the retrieval query
                query_summary = f"query='{retrieval_response.query}'"
                retrieval_plan_responses.append(query_summary)
                query_history.append(retrieval_response.query)

                # Print retrieval information
                print(f"\n{'='*80}")
                print(f"ITERATION {iteration + 1}/{MAX_ITERATIONS} - QA {message.qa_pair_id}")
                print(f"{'='*80}")
                print(f"Query: {retrieval_response.query}")
                print(f"{'='*80}\n")

                # Execute hybrid search
                # Pass used_document_indices to get only new documents
                retrieved_documents = await self._execute_vector_search(
                    retrieval_response.query,
                    faiss_index,
                    chunk_metadata,
                    k=CHUNKS_PER_ITERATION,
                    used_document_indices=used_document_indices,
                    bm25_index=bm25_index
                )

                # Log the retrieval to dedicated file
                self._log_retrieval(
                    qa_pair_id=message.qa_pair_id,
                    dataset=message.dataset,
                    setting=message.setting,
                    iteration=iteration + 1,
                    query=retrieval_response.query,
                    hypothetical_doc="",  # No longer used
                    retrieved_docs=retrieved_documents
                )

                # Format raw document chunks to add to context (NO SUMMARIZATION)
                # Both planner and answer generator see the raw chunks
                if retrieved_documents:
                    chunk_parts = []
                    for i, doc in enumerate(retrieved_documents, 1):
                        # Use raw chunk text directly
                        chunk_text = doc.get('text', '')
                        chunk_parts.append(f"Document {i}:\n{chunk_text}")
                    formatted_chunks = "\n\n".join(chunk_parts)
                else:
                    formatted_chunks = "No documents retrieved"

                # Print retrieved documents
                print(f"Retrieved Documents ({len(retrieved_documents)} total):")
                print(f"{'-'*80}")
                if retrieved_documents:
                    for i, doc in enumerate(retrieved_documents, 1):
                        chunk_text = doc.get('text', '')
                        # Truncate if too long for display
                        display_text = chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text
                        print(f"Document {i} (FAISS index: {doc['faiss_index']}, score: {doc['score']:.4f}):")
                        print(f"{display_text}")
                        print(f"{'-'*80}")
                else:
                    print("No documents retrieved")
                    print(f"{'-'*80}")
                print()

                # Track documents (no summaries stored)
                all_selected_documents.extend(retrieved_documents)

                # Add document indices to used set to prevent re-retrieval
                faiss_indices_this_iteration = []
                for doc in retrieved_documents:
                    used_document_indices.add(doc['faiss_index'])
                    faiss_indices_this_iteration.append(doc['faiss_index'])

                # Add formatted chunks to retrieved context
                if formatted_chunks:
                    retrieved_context += f"\n\nIteration {iteration + 1} results:\n{formatted_chunks}"

                # Store query â†’ content pair for backward pass critique
                query_content_pairs.append({
                    "query": retrieval_response.query,
                    "content": formatted_chunks
                })

                self.logger.info(f"Completed iteration {iteration + 1}, retrieved {len(retrieved_documents)} raw chunks, "
                               f"FAISS indices: {faiss_indices_this_iteration}, total unique so far: {len(used_document_indices)}, "
                               f"context length: {len(retrieved_context)}")

            except Exception as e:
                self.logger.error(f"Error in retrieval iteration {iteration + 1}: {e}")
                continue

        # Update state with retrieval plans and query-content pairs (in-memory only)
        current_state["retrieval_plans"] = retrieval_plan_responses
        current_state["query_content_pairs"] = query_content_pairs

        self.logger.info(f"Updated state with {len(query_content_pairs)} query-content pairs")

        # Store current system iteration's retrieved indices and calculate overlap with previous iteration
        current_iteration_indices = list(used_document_indices)
        current_indices_key = f"retrieved_indices_qa_{message.qa_pair_id}_rep_{message.repetition}"
        current_state[current_indices_key] = current_iteration_indices

        print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Storing current iteration indices: key='{current_indices_key}', count={len(current_iteration_indices)}")
        print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Current iteration indices: {current_iteration_indices}")
        # Update in-memory cache so the next iteration can compute overlap without relying on disk
        try:
            self._last_indices_cache[cache_key] = list(current_iteration_indices)
            print('[OVERLAP DEBUG - QA ' + message.qa_pair_id + '] Updated in-memory cache for next iteration (' + str(len(current_iteration_indices)) + ' indices)')
        except Exception:
            pass

        # Explicitly save to persistent storage to ensure it's available for next iteration
        # IMPORTANT: Load existing state first, then merge, then save
        try:
            # Load the current persistent state for this batch
            existing_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
            print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Loaded existing state, has {len(existing_state)} keys")

            # Merge our new key into the existing state
            existing_state[current_indices_key] = current_iteration_indices
            print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Merged new key into existing state")

            # Save the merged state back
            saved = self.shared_state.save_state(existing_state, message.dataset, message.setting, message.batch_id)
            print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Save to persistent storage result: {saved}")
            print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Verification - can we load it back?")

            # Verify it was saved
            verify_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
            if current_indices_key in verify_state:
                print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] VERIFIED: Key found in reloaded state with {len(verify_state[current_indices_key])} indices")
            else:
                print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] WARNING: Key NOT found in reloaded state!")

            self.logger.info(f"Saved {len(current_iteration_indices)} indices to persistent storage for iteration {message.repetition}")
        except Exception as e:
            print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] ERROR saving to persistent storage: {e}")
            import traceback
            traceback.print_exc()
            self.logger.error(f"Failed to save indices to persistent storage: {e}")

        # Calculate and display overlap with previous system iteration
        print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] About to calculate overlap, previous_system_iteration_indices is: {previous_system_iteration_indices is not None}")

        if previous_system_iteration_indices is not None:
            print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Calculating overlap...")
            current_set = set(current_iteration_indices)
            previous_set = set(previous_system_iteration_indices)

            print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Current set: {current_set}")
            print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] Previous set: {previous_set}")

            intersection = current_set.intersection(previous_set)
            union = current_set.union(previous_set)

            overlap_count = len(intersection)
            current_count = len(current_set)
            jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
            percent_overlap = (overlap_count / current_count * 100) if current_count > 0 else 0.0

            overlap_msg = (f"System Iteration {message.repetition}: Retrieval overlap with iteration {message.repetition - 1}: "
                          f"{overlap_count}/{current_count} documents ({percent_overlap:.1f}%, Jaccard: {jaccard:.3f})")

            # Print to console
            print(f"[VectorRAG - QA {message.qa_pair_id}] {overlap_msg}")

            self.logger.info(overlap_msg)
        else:
            # First system iteration - no overlap to calculate
            print(f"[OVERLAP DEBUG - QA {message.qa_pair_id}] No previous iteration data available")
            overlap_msg = f"System Iteration {message.repetition}: Retrieval overlap: N/A (first iteration or no previous data)"
            print(f"[VectorRAG - QA {message.qa_pair_id}] {overlap_msg}")
            self.logger.info(overlap_msg)

        # ========== SAVE CUMULATIVE CONTEXT AND INDICES ==========
        # Save current iteration's context to cumulative context cache
        context_key = f"{cache_key}::iter{message.repetition}"
        self._cumulative_context_cache[context_key] = retrieved_context
        print(f"[CUMULATIVE-CONTEXT] Saved context for iteration {message.repetition} ({len(retrieved_context)} chars)")
        self.logger.info(f"Saved cumulative context for iteration {message.repetition}")

        # Save cumulative indices (all indices from all iterations up to now)
        cumulative_indices_key = f"{cache_key}::cumulative_indices"
        self._cumulative_indices_cache[cumulative_indices_key] = list(used_document_indices)
        print(f"[CUMULATIVE-CONTEXT] Saved cumulative indices: {len(used_document_indices)} total indices")
        self.logger.info(f"Saved {len(used_document_indices)} cumulative indices")

        # Return VectorRetrievalReady message with modified state
        retrieval_ready_msg = VectorRetrievalReadyMessage(
            batch_id=message.batch_id,
            repetition=message.repetition,
            retrieved_context=retrieved_context,
            dataset=message.dataset,
            setting=message.setting,
            shared_state=current_state  # Pass modified state back
        )

        self.logger.info(f"Returning VectorRetrievalReady for batch {message.batch_id}")

        # Save query history to cache for next system iteration
        self._query_history_cache[cache_key] = list(query_history)
        print(f'[QUERY-HISTORY] Saved {len(query_history)} queries to cache for QA {message.qa_pair_id}')
        self.logger.info(f"Saved {len(query_history)} queries to cache for next iteration")

        # Return the retrieval ready message
        return retrieval_ready_msg

    def _load_faiss_index(self, dataset: str, setting: str, batch_id: int):
        """Load FAISS index, BM25 index, and metadata from disk."""
        try:
            import os

            index_dir = "vector_indexes"
            index_filename = os.path.join(index_dir, f"{dataset}_{setting}_batch_{batch_id}.index")
            bm25_filename = os.path.join(index_dir, f"{dataset}_{setting}_batch_{batch_id}_bm25.pkl")
            metadata_filename = os.path.join(index_dir, f"{dataset}_{setting}_batch_{batch_id}_metadata.json")

            if not os.path.exists(index_filename) or not os.path.exists(metadata_filename):
                self.logger.error(f"Index files not found: {index_filename}")
                return None, None, None

            # Load FAISS index
            faiss_index = faiss.read_index(index_filename)

            # Load BM25 index (if it exists)
            bm25_index = None
            if os.path.exists(bm25_filename):
                with open(bm25_filename, 'rb') as f:
                    bm25_index = pickle.load(f)
                self.logger.info(f"Loaded BM25 index for hybrid retrieval")
            else:
                self.logger.warning(f"BM25 index not found, using FAISS only: {bm25_filename}")

            # Load metadata
            with open(metadata_filename, 'r', encoding='utf-8') as f:
                chunk_metadata = json.load(f)

            self.logger.info(f"Loaded FAISS index with {faiss_index.ntotal} vectors")
            return faiss_index, chunk_metadata, bm25_index

        except Exception as e:
            self.logger.error(f"Error loading indexes: {e}")
            return None, None, None

    async def _execute_vector_search(self, query: str, faiss_index, chunk_metadata, k: int = 5, used_document_indices: set = None, bm25_index=None, hypothetical_document: str = None) -> List[Dict[str, Any]]:
        """
        Execute hybrid retrieval with MMR re-ranking:
        1. Retrieve 50 candidates using BM25 + FAISS hybrid search
        2. Apply MMR (Maximal Marginal Relevance) to diversify results
        3. Return top k documents

        Args:
            query: Query string (used for BM25 keyword search and embedding)
            faiss_index: FAISS index
            chunk_metadata: Metadata for chunks
            k: Number of final documents to retrieve (default: 5)
            used_document_indices: Set of FAISS indices already in context
            bm25_index: BM25 index for keyword search (optional)
            hypothetical_document: (Deprecated - no longer used)

        Returns:
            List of k unique documents (or fewer if not enough exist)
        """
        if used_document_indices is None:
            used_document_indices = set()

        try:
            # If no BM25 index, fall back to FAISS-only retrieval
            if bm25_index is None:
                self.logger.info("BM25 index not available, using FAISS-only retrieval")
                return await self._faiss_only_search(query, faiss_index, chunk_metadata, k, used_document_indices, hypothetical_document)

            # Stage 1: Hybrid retrieval to get initial candidate set (50 chunks)
            initial_k = 50  # Retrieve more candidates for re-ranking
            self.logger.info(f"Stage 1: Hybrid retrieval (BM25 + FAISS) retrieving {initial_k} candidates for query: {query[:50]}...")

            # 1. Get FAISS scores for all documents using query
            from llm import get_embeddings_async
            text_to_embed = query
            self.logger.info(f"Embedding query for retrieval")
            query_embeddings = await get_embeddings_async([text_to_embed])
            query_vector = np.array(query_embeddings[0]).astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_vector)

            # Search all documents
            num_docs = faiss_index.ntotal
            faiss_scores, faiss_indices = faiss_index.search(query_vector, num_docs)
            faiss_scores = faiss_scores[0]  # Flatten

            # 2. Get BM25 scores for all documents
            tokenized_query = query.lower().split()
            bm25_scores = bm25_index.get_scores(tokenized_query)

            # 3. Normalize scores to [0, 1] range
            # Normalize FAISS scores (cosine similarity is already in [-1, 1], shift to [0, 1])
            faiss_scores_norm = (faiss_scores + 1) / 2.0

            # Normalize BM25 scores
            bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
            bm25_scores_norm = bm25_scores / bm25_max

            # 4. Combine scores with equal weighting (0.5 FAISS + 0.5 BM25)
            alpha = 0.5  # Weight for FAISS (semantic)
            beta = 0.5   # Weight for BM25 (keyword)
            combined_scores = alpha * faiss_scores_norm + beta * bm25_scores_norm

            # 5. Create scored documents list with embeddings for MMR
            scored_docs = []
            for i, (faiss_idx, combined_score) in enumerate(zip(faiss_indices[0], combined_scores)):
                if faiss_idx < len(chunk_metadata):
                    scored_docs.append({
                        'faiss_index': int(faiss_idx),
                        'combined_score': float(combined_score),
                        'faiss_score': float(faiss_scores_norm[i]),
                        'bm25_score': float(bm25_scores_norm[faiss_idx]),
                        'text': chunk_metadata[faiss_idx]["text"],
                        'embedding_idx': i  # Store index for embedding lookup
                    })

            # 6. Sort by combined score (descending)
            scored_docs.sort(key=lambda x: x['combined_score'], reverse=True)

            # 7. Filter out already-used documents and get initial_k candidates
            candidates = []
            for doc in scored_docs:
                if doc['faiss_index'] not in used_document_indices:
                    candidates.append(doc)
                    if len(candidates) >= initial_k:
                        break

            if len(candidates) == 0:
                self.logger.warning(f"No candidates available after filtering")
                return []

            self.logger.info(f"Stage 1: Retrieved {len(candidates)} candidates")

            # Stage 2: Apply MMR re-ranking for diversity
            self.logger.info(f"Stage 2: Applying MMR re-ranking to select top {k} diverse documents")

            # Get embeddings for all candidates
            candidate_embeddings = []
            for doc in candidates:
                # Get embedding from FAISS index
                candidate_idx = doc['faiss_index']
                # Reconstruct embedding from FAISS index
                embedding = faiss_index.reconstruct(int(candidate_idx))
                candidate_embeddings.append(embedding)

            candidate_embeddings = np.array(candidate_embeddings)

            # Apply MMR
            selected_docs = self._mmr_rerank(
                query_embedding=query_vector[0],
                candidate_docs=candidates,
                candidate_embeddings=candidate_embeddings,
                k=k,
                lambda_param=0.3  # Balance between relevance (0.7) and diversity (0.3)
            )

            # Format final results
            retrieved_documents = []
            for i, doc in enumerate(selected_docs):
                retrieved_documents.append({
                    'index': i,
                    'faiss_index': doc['faiss_index'],
                    'score': doc['combined_score'],
                    'text': doc['text'],
                    'faiss_score': doc['faiss_score'],
                    'bm25_score': doc['bm25_score'],
                    'mmr_selected': True
                })

            # Log retrieval details
            if len(retrieved_documents) < k:
                self.logger.warning(f"MMR retrieval: Only retrieved {len(retrieved_documents)}/{k} documents. "
                                  f"Candidates available: {len(candidates)}")
            else:
                avg_faiss = sum(d['faiss_score'] for d in retrieved_documents) / len(retrieved_documents)
                avg_bm25 = sum(d['bm25_score'] for d in retrieved_documents) / len(retrieved_documents)
                self.logger.info(f"MMR retrieval: Retrieved {len(retrieved_documents)} docs from {len(candidates)} candidates "
                               f"(avg FAISS: {avg_faiss:.3f}, avg BM25: {avg_bm25:.3f})")

            return retrieved_documents

        except Exception as e:
            self.logger.error(f"Error executing hybrid search with MMR: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _mmr_rerank(self, query_embedding: np.ndarray, candidate_docs: List[Dict],
                    candidate_embeddings: np.ndarray, k: int, lambda_param: float = 0.7) -> List[Dict]:
        """
        Apply Maximal Marginal Relevance (MMR) to re-rank candidates.

        MMR balances relevance to query with diversity from already-selected documents.

        Args:
            query_embedding: Query embedding vector
            candidate_docs: List of candidate documents
            candidate_embeddings: Embeddings for candidate documents
            k: Number of documents to select
            lambda_param: Trade-off between relevance (higher) and diversity (lower)
                         Default 0.7 = 70% relevance, 30% diversity

        Returns:
            List of k selected documents
        """
        selected = []
        selected_indices = []
        remaining_indices = list(range(len(candidate_docs)))

        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        candidates_norm = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-9)

        # Compute relevance scores (cosine similarity with query)
        relevance_scores = np.dot(candidates_norm, query_norm)

        for _ in range(min(k, len(candidate_docs))):
            if not remaining_indices:
                break

            mmr_scores = []
            for idx in remaining_indices:
                # Relevance component
                relevance = relevance_scores[idx]

                # Diversity component (max similarity to already selected)
                if selected_indices:
                    similarities = np.dot(candidates_norm[idx], candidates_norm[selected_indices].T)
                    max_similarity = np.max(similarities)
                else:
                    max_similarity = 0.0

                # MMR formula: λ * relevance - (1-λ) * max_similarity
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append((idx, mmr_score))

            # Select document with highest MMR score
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected.append(candidate_docs[best_idx])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return selected

    async def _faiss_only_search(self, query: str, faiss_index, chunk_metadata, k: int, used_document_indices: set, hypothetical_document: str = None) -> List[Dict[str, Any]]:
        """Fallback to FAISS-only search when BM25 is not available."""
        try:
            from llm import get_embeddings_async
            text_to_embed = query
            self.logger.info(f"FAISS-only: Embedding query for retrieval")
            query_embeddings = await get_embeddings_async([text_to_embed])
            query_vector = np.array(query_embeddings[0]).astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_vector)

            max_retrieve = min(k * 10, faiss_index.ntotal)
            scores, indices = faiss_index.search(query_vector, max_retrieve)

            retrieved_documents = []
            for i, idx in enumerate(indices[0]):
                if idx < len(chunk_metadata) and int(idx) not in used_document_indices:
                    retrieved_documents.append({
                        'index': len(retrieved_documents),
                        'faiss_index': int(idx),
                        'score': float(scores[0][i]),
                        'text': chunk_metadata[idx]["text"]
                    })
                    if len(retrieved_documents) >= k:
                        break

            return retrieved_documents

        except Exception as e:
            self.logger.error(f"Error executing FAISS-only search: {e}")
            return []

    async def close(self) -> None:
        """Close method - no cleanup needed for direct Gemini API."""
        pass


# ===== RETRIEVAL SUMMARIZER AGENT =====

class RetrievalSummarizerAgent:
    """
    Agent that summarizes retrieved documents for context building.
    """

    def __init__(self, shared_state: SharedState):
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.retrieval_summarizer")
        self.shared_state = shared_state

        # Import response format and prompts
        from parameters import base_prompt_retrieval_summarizer_vector, RetrievalSummarizerResponse

        # Initialize Gemini model client with structured output
        self.model_client = OpenAIChatCompletionClient(
            model="gemini-2.5-flash-lite",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=llm_keys.GEMINI_KEY,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            },
            response_format=RetrievalSummarizerResponse
        )

        self.base_prompt_retrieval_summarizer_vector = base_prompt_retrieval_summarizer_vector

    async def summarize_document(self, query: str, document: Dict[str, Any], current_state: Dict[str, Any], ctx: MessageContext, qa_pair_id: str = None, iteration: int = None) -> str:
        """
        Summarize a single document using simple static prompt (like Self-Refine).

        NOTE: Simplified to use query-agnostic, static summarization without optimization.
        This matches Self-Refine's approach which showed better stability.

        Args:
            query: The original query (not used in simplified version)
            document: Dict with keys 'index', 'score', 'text'
            current_state: Shared state (not used for summarization prompt anymore)
            ctx: Message context for cancellation
            qa_pair_id: QA pair ID for logging
            iteration: Iteration number for logging

        Returns:
            Summary of the document
        """
        # Use simple static prompt (no query, no learned prompt, no structured output)
        simple_prompt = f"""Summarize the following text concisely, preserving key information:

{document['text']}

Summary:"""

        try:
            # Create single user message (no system prompt)
            user_message = UserMessage(content=simple_prompt, source="user")

            response = await self.model_client.create(
                [user_message],
                cancellation_token=ctx.cancellation_token
            )

            # Get direct text response (no structured parsing)
            assert isinstance(response.content, str)
            summary = response.content.strip()

            # Log interaction (optional - keep for debugging)
            logger = get_global_prompt_logger()
            logger.log_interaction(
                agent_name="RetrievalSummarizerAgent",
                interaction_type="document_summarization",
                system_prompt="",  # No system prompt in simplified version
                user_prompt=simple_prompt,
                llm_response=summary,
                qa_pair_id=qa_pair_id,
                iteration=iteration,
                additional_metadata={
                    "document_index": document['index'],
                    "summary_length": len(summary),
                    "simplified_mode": True
                }
            )

            return summary

        except Exception as e:
            self.logger.error(f"Error summarizing document {document['index']}: {e}")
            # Fallback: return original text (truncated)
            return document['text'][:500] + "..."

    async def close(self) -> None:
        """Close the model client."""
        await self.model_client.close()


# ===== ANSWER GENERATOR AGENT =====

class AnswerGeneratorAgent(RoutedAgent):
    """
    Agent that generates answers using retrieved context and LLM reasoning.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.answer_generator")
        self.shared_state = SharedState("agent_states")

        # Import prompts
        from parameters import answer_generator_initial_prompt, answer_generator_refinement_prompt

        # Initialize Gemini API client using native SDK
        self.gemini_client = genai.Client(api_key=llm_keys.GEMINI_KEY)
        self.model_name = "gemini-2.5-flash"

        self.initial_prompt = answer_generator_initial_prompt
        self.refinement_prompt = answer_generator_refinement_prompt

    @message_handler
    async def handle_answer_generation_start(self, message: AnswerGenerationStartMessage, ctx: MessageContext) -> AnswerGenerationReadyMessage:
        """Handle AnswerGenerationStart message and generate answer using LLM."""
        self.logger.info(f"AnswerGeneratorAgent processing QA pair {message.qa_pair_id}")

        # Use state from message (message-passing architecture - no I/O)
        current_state = message.shared_state.copy()
        learned_user_prompt = current_state.get("learned_prompt_answer_generator_vector", "")
        if not learned_user_prompt:
            from parameters import base_prompt_answer_generator_vector
            learned_user_prompt = base_prompt_answer_generator_vector
            self.logger.info(f"Answer generator: using basic base user prompt for QA pair {message.qa_pair_id}")
        else:
            self.logger.info(f"Answer generator: using learned user prompt (system prompt kept empty) for QA pair {message.qa_pair_id}")

        # Get previous responses and evaluations for this QA pair
        all_evaluation_responses = current_state.get("response_evaluations", [])

        # Filter evaluations for this specific QA pair
        qa_pair_evals = [
            eval_resp for eval_resp in all_evaluation_responses
            if eval_resp.get('qa_pair_id') == message.qa_pair_id
        ]

        # Format previous responses and evaluations
        if qa_pair_evals:
            previous_attempts_text = "\n\n" + "=" * 80 + "\n"
            previous_attempts_text += "PREVIOUS ATTEMPTS FOR THIS QUESTION:\n"
            previous_attempts_text += "=" * 80 + "\n\n"

            for eval_resp in qa_pair_evals:
                iter_num = eval_resp.get('repetition', 0)
                generated_ans = eval_resp.get('generated_answer', 'N/A')
                reasoning = eval_resp.get('evaluation_reasoning', 'N/A')
                critique = eval_resp.get('evaluation_feedback', 'N/A')

                previous_attempts_text += f"--- Iteration {iter_num} ---\n\n"
                previous_attempts_text += f"Generated Answer:\n{generated_ans}\n\n"
                previous_attempts_text += f"Evaluation Reasoning:\n{reasoning}\n\n"
                previous_attempts_text += f"Critique and Suggestions:\n{critique}\n\n"
                previous_attempts_text += "-" * 80 + "\n\n"

            previous_attempts_text += "=" * 80 + "\n"
            previous_attempts_text += "END OF PREVIOUS ATTEMPTS\n"
            previous_attempts_text += "=" * 80 + "\n\n"
        else:
            previous_attempts_text = ""

        # Build user prompt: learned prompt + required sections (no previous attempts/critique)
        user_prompt_content = (
            f"{learned_user_prompt}\n\n"
            f"Context:\n{message.retrieved_context}\n\n"
            f"Question: {message.question}\n"
        )

        try:
            # Use Gemini API directly with system instruction and thinking disabled
            # Build config - only include system_instruction if it's not empty
            config_dict = {
                "thinking_config": types.ThinkingConfig(thinking_budget=0)
            }
            # Do not set system_instruction (system must remain empty)

            # Run synchronous Gemini call in executor to not block event loop
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt_content,
                    config=types.GenerateContentConfig(**config_dict)
                )
            )

            # Get generated answer
            generated_answer = response.text

            # Log LLM interaction
            logger = get_global_prompt_logger()
            logger.log_interaction(
                agent_name="AnswerGeneratorAgent",
                interaction_type="answer_generation",
                system_prompt="",
                user_prompt=user_prompt_content,
                llm_response=generated_answer,
                qa_pair_id=message.qa_pair_id,
                iteration=message.repetition,
                additional_metadata={
                    "question_length": len(message.question),
                    "context_length": len(message.retrieved_context),
                    "answer_length": len(generated_answer)
                }
            )

            log_qa_processing(self.logger, message.qa_pair_id, "Generated answer", generated_answer)

            # Store conversation in shared state
            conversation_entry = {
                "qa_pair_id": message.qa_pair_id,
                "question": message.question,
                "retrieved_context": message.retrieved_context,
                "system_prompt": "",
                "user_prompt": user_prompt_content,
                "generated_answer": generated_answer,
                "repetition": message.repetition
            }

            conversations = current_state.get("conversations_answer_generation", [])
            conversations.append(conversation_entry)
            current_state["conversations_answer_generation"] = conversations

            # Return AnswerGenerationReady message with modified state
            answer_ready_msg = AnswerGenerationReadyMessage(
                qa_pair_id=message.qa_pair_id,
                generated_answer=generated_answer,
                batch_id=message.batch_id,
                repetition=message.repetition,
                shared_state=current_state  # Pass modified state back
            )

            self.logger.info(f"Returning AnswerGenerationReady for QA pair {message.qa_pair_id}")

            # Return the answer ready message
            return answer_ready_msg

        except Exception as e:
            self.logger.error(f"Error in answer generation: {e}")
            # Return default response on error with unmodified state
            return AnswerGenerationReadyMessage(
                qa_pair_id=message.qa_pair_id,
                generated_answer="Error generating answer",
                batch_id=message.batch_id,
                repetition=message.repetition,
                shared_state=current_state  # Pass state back even on error
            )

    async def close(self) -> None:
        """Close the model client."""
        # Gemini client doesn't require explicit closing
        pass


# ===== RESPONSE EVALUATOR AGENT =====

class ResponseEvaluatorAgent(RoutedAgent):
    """
    Agent that evaluates generated responses against gold answers using LLM.
    """

    def __init__(self, name: str, dataset_name: str = None) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.response_evaluator")
        self.shared_state = SharedState("agent_states")
        self.dataset_name = dataset_name

        # Import prompts
        from parameters import response_evaluator_prompt

        # Load learned gold answer patterns if available
        self.satisfactory_criteria = self._load_gold_patterns()

        # Initialize Gemini API client using native SDK
        self.gemini_client = genai.Client(api_key=llm_keys.GEMINI_KEY)
        self.model_name = "gemini-2.5-flash"

        self.response_evaluator_prompt = response_evaluator_prompt

        # Initialize OpenAI client for G-Eval
        self.openai_client = OpenAI(api_key=llm_keys.OPENAI_KEY)

        # G-Eval prompts
        self.coherence_prompt = """You will be given one summary written for a source document.
Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic."

Evaluation Steps:
1. Read the source document carefully and identify the main topic and key points.
2. Read the summary and compare it to the source document. Check if the summary covers the main topic and key points of the source document, and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.

Example:
Source Document:
{document}

Summary:
{summary}

Evaluation Form (scores ONLY):
- Coherence:"""

        self.relevance_prompt = """You will be given one summary written for a source document.
Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Relevance (1-5) - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.

Evaluation Steps:
1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the source document.
3. Assess how well the summary covers the main points of the source document, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.

Example:
Source Document:
{document}

Summary:
{summary}

Evaluation Form (scores ONLY):
- Relevance:"""

        self.reference_guided_prompt = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer.

Evaluation Steps:
1. Compare the assistant's answer with the reference answer.
2. Identify what information is correct, missing, or incorrect in the assistant's answer.
3. Assess the overall correctness and helpfulness of the assistant's answer.
4. Assign a rating from 1 to 5, where 1 is the lowest (completely incorrect/unhelpful) and 5 is the highest (fully correct and helpful).

User Question:
{question}

Reference Answer:
{reference_answer}

Assistant's Answer:
{assistant_answer}

Evaluation Form (scores ONLY):
- Rating:"""

    def _load_gold_patterns(self) -> str:
        """Load learned gold answer patterns from file. Raises error if not found."""
        from pathlib import Path

        if not self.dataset_name:
            raise ValueError("dataset_name is required for ResponseEvaluatorAgent")

        patterns_file = Path("learned_patterns") / f"{self.dataset_name}_gold_patterns.txt"
        print("Patterns file: ", patterns_file)

        if not patterns_file.exists():
            raise FileNotFoundError(
                f"Gold patterns file not found: {patterns_file}\n"
                f"Please run: python learn_gold_answer_patterns.py <training_data>.json "
                f"to generate the required gold patterns file."
            )

        with open(patterns_file, 'r', encoding='utf-8') as f:
            patterns = f.read()

        self.logger.info(f"Loaded gold patterns from {patterns_file} ({len(patterns)} characters)")

        # Return the patterns directly - no additional formatting needed
        # The patterns file already contains complete description with headers
        return patterns

    def _compute_rouge_score(self, qa_pair: Dict[str, Any], generated_answer: str) -> float:
        """
        Compute ROUGE score between generated answer and reference answers.

        Args:
            qa_pair: QA pair containing question and reference answers
            generated_answer: Generated answer to evaluate

        Returns:
            float: ROUGE-L score (F1)
        """
        try:
            # Import the ROUGE function from eval_functions
            from eval_functions import compute_rouge_l

            reference_answers = qa_pair.get("answers", [])
            if not reference_answers or not generated_answer:
                return 0.0

            # Use the first reference answer for ROUGE computation
            reference = reference_answers[0] if isinstance(reference_answers, list) else str(reference_answers)

            # Use the official ROUGE-L implementation
            rouge_score = compute_rouge_l(generated_answer, reference)
            return rouge_score

        except Exception as e:
            self.logger.error(f"Error computing ROUGE score: {e}")
            return 0.0

    def _compute_geval_scores(self, question: str, document: str, generated_answer: str, reference_answer: str) -> Dict[str, float]:
        """
        Compute G-Eval scores (Coherence, Relevance, Reference-Guided) using OpenAI API.

        Args:
            question: The question being answered
            document: The source document
            generated_answer: Generated answer to evaluate
            reference_answer: Gold/reference answer

        Returns:
            Dict with keys: 'coherence', 'relevance', 'reference_guided' (all floats)
        """
        try:
            # Compute all three G-Eval metrics
            coherence_score = self._compute_geval_metric(
                self.coherence_prompt.format(document=document, summary=generated_answer),
                "Coherence"
            )

            relevance_score = self._compute_geval_metric(
                self.relevance_prompt.format(document=document, summary=generated_answer),
                "Relevance"
            )

            reference_guided_score = self._compute_geval_metric(
                self.reference_guided_prompt.format(
                    question=question,
                    reference_answer=reference_answer,
                    assistant_answer=generated_answer
                ),
                "Reference-Guided"
            )

            return {
                'coherence': coherence_score,
                'relevance': relevance_score,
                'reference_guided': reference_guided_score
            }

        except Exception as e:
            self.logger.error(f"Error computing G-Eval scores: {e}")
            return {'coherence': 0.0, 'relevance': 0.0, 'reference_guided': 0.0}

    def _compute_geval_metric(self, prompt: str, metric_name: str, model: str = "gpt-4o-mini", top_logprobs: int = 5) -> float:
        """
        Compute a single G-Eval metric using logprobs.

        Args:
            prompt: The evaluation prompt
            metric_name: Name of the metric for logging
            model: OpenAI model to use
            top_logprobs: Number of top logprobs to request

        Returns:
            float: G-Eval score (expected value from probability distribution)
        """
        try:
            # Call OpenAI API with logprobs enabled
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=top_logprobs,
                timeout=60.0
            )

            # Extract logprobs
            logprobs_content = response.choices[0].logprobs.content[0]
            top_logprobs_list = logprobs_content.top_logprobs

            # Extract probabilities for scores 1-5
            score_probs = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

            for item in top_logprobs_list:
                token = item.token.strip()
                if token in ['1', '2', '3', '4', '5']:
                    score_probs[int(token)] = np.exp(item.logprob)

            # Calculate G-Eval score (expected value)
            scores_found = {k: v for k, v in score_probs.items() if v > 0}

            if not scores_found:
                self.logger.warning(f"No numeric scores found in logprobs for {metric_name}")
                return 0.0

            # Normalize probabilities and calculate weighted score
            total_prob = sum(scores_found.values())
            normalized_scores = {k: v/total_prob for k, v in scores_found.items()}
            g_score = sum(score * prob for score, prob in normalized_scores.items())

            return g_score

        except Exception as e:
            self.logger.error(f"Error computing G-Eval {metric_name}: {e}")
            return 0.0

    @message_handler
    async def handle_response_evaluation_start(self, message: ResponseEvaluationStartMessage, ctx: MessageContext) -> ResponseEvaluationReadyMessage:
        """Handle ResponseEvaluationStart message and evaluate response using LLM."""
        print(f"\nðŸ” ResponseEvaluatorAgent: Starting evaluation for QA pair {message.qa_pair_id}")
        print(f"   Query: {message.original_query[:100]}...")
        print(f"   Answer length: {len(message.generated_answer)} chars")

        self.logger.info(f"ResponseEvaluatorAgent evaluating QA pair {message.qa_pair_id}")

        # Compute actual ROUGE score between generated answer and gold answers
        rouge_score = 0.0
        if message.gold_answers and message.generated_answer:
            valid_gold_answers = [ans for ans in message.gold_answers if ans.strip()]
            if valid_gold_answers:
                qa_pair_for_rouge = {"answers": valid_gold_answers}
                rouge_score = self._compute_rouge_score(qa_pair_for_rouge, message.generated_answer)
                self.logger.info(f"Computed ROUGE score {rouge_score:.4f} for QA pair {message.qa_pair_id}")
        print(f"[ResponseEvaluator] ROUGE-L={rouge_score:.4f}")

        # Compute G-Eval scores (Coherence, Relevance, Reference-Guided)
        geval_scores = {'coherence': 0.0, 'relevance': 0.0, 'reference_guided': 0.0}
        if message.gold_answers and message.generated_answer and message.document_text:
            valid_gold_answers = [ans for ans in message.gold_answers if ans.strip()]
            if valid_gold_answers:
                reference_answer = valid_gold_answers[0]
                # Truncate answers for console display
                answer_preview = message.generated_answer[:80] + "..." if len(message.generated_answer) > 80 else message.generated_answer
                ref_preview = reference_answer[:80] + "..." if len(reference_answer) > 80 else reference_answer
                print(f"[ResponseEvaluator] Computing G-Eval scores...")
                print(f"   Generated: {answer_preview}")
                print(f"   Reference: {ref_preview}")

                geval_scores = self._compute_geval_scores(
                    question=message.original_query,
                    document=message.document_text,
                    generated_answer=message.generated_answer,
                    reference_answer=reference_answer
                )
                self.logger.info(f"Computed G-Eval scores for {message.qa_pair_id}: {geval_scores}")
                print(f"[ResponseEvaluator] G-Eval Coherence={geval_scores['coherence']:.4f} | Relevance={geval_scores['relevance']:.4f} | Reference-Guided={geval_scores['reference_guided']:.4f}")
        else:
            print(f"[ResponseEvaluator] Skipping G-Eval (missing data)")

        # Use state from message (message-passing architecture - no I/O)
        current_state = message.shared_state.copy()
        all_evaluation_responses = current_state.get("response_evaluations", [])
        print(f"[ResponseEvaluator] Loaded state. prev_evals_total={len(all_evaluation_responses)}")

        # Filter previous evaluations for this specific QA pair only
        qa_pair_evals = [
            eval_resp for eval_resp in all_evaluation_responses
            if eval_resp.get('qa_pair_id') == message.qa_pair_id
        ]
        print(f"[ResponseEvaluator] prev_evals_for_qa={len(qa_pair_evals)}")

        # Don't include previous evaluations in the prompt (to match Self-Refine behavior)
        # Each iteration is evaluated independently

        # Format unfound keywords history for display
        if hasattr(message, 'unfound_keywords_history') and message.unfound_keywords_history:
            unfound_keywords_text = "\n\n**KEYWORDS ALREADY TRIED (DO NOT REPEAT):**\nThe following keywords were suggested in previous iterations but were NOT found in the document:\n" + ", ".join(message.unfound_keywords_history)
        else:
            unfound_keywords_text = ""

        # Prepare prompt with query, generated response, retrieved context, and satisfactory criteria
        prompt_content = self.response_evaluator_prompt.format(
            original_query=message.original_query,
            generated_answer=message.generated_answer,
            retrieved_context=message.retrieved_context,
            #  satisfactory_criteria=self.satisfactory_criteria,
            #  unfound_keywords_history=unfound_keywords_text
        )

        print(f"   Prompt prepared ({len(prompt_content)} chars)")
        print(f"   â„¹ï¸  Evaluating answer independently (not showing previous iterations to match Self-Refine)")

        try:

            # Use retry logic to handle API overload errors
            # Use Gemini API directly with system instruction and thinking disabled
            async def api_call():
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self.gemini_client.models.generate_content(
                        model=self.model_name,
                        contents="Please evaluate the response.",
                        config=types.GenerateContentConfig(
                            system_instruction=prompt_content,
                            thinking_config=types.ThinkingConfig(thinking_budget=1024)  # Disable thinking
                        )
                    )
                )

            print("[ResponseEvaluator] Calling Gemini API for evaluation...")
            response = await retry_api_call_with_backoff(api_call)
            print("[ResponseEvaluator] LLM response received. length={}".format(len(response.text or "")))

            # Parse text response (like Self-Refine)
            response_text = response.text.strip()

            # Parse DECISION and CRITIQUE from text
            decision = "NEEDS_REFINEMENT"  # Default
            critique = ""

            lines = response_text.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("DECISION:"):
                    decision_text = line.replace("DECISION:", "").strip()
                    if "SATISFACTORY" in decision_text.upper():
                        decision = "SATISFACTORY"
                    else:
                        decision = "NEEDS_REFINEMENT"
                elif line.startswith("CRITIQUE:"):
                    critique = line.replace("CRITIQUE:", "").strip()
                    # Get rest of lines as part of critique
                    if i + 1 < len(lines):
                        critique += "\n" + "\n".join(lines[i+1:])
                    break

            print(f"[ResponseEvaluator] Parsed decision={decision}")
            # Map to internal format
            # SATISFACTORY -> continue_optimization=False
            # NEEDS_REFINEMENT -> continue_optimization=True
            continue_optimization = (decision == "NEEDS_REFINEMENT")

            # Create a simple object to hold the parsed values
            class EvalResponse:
                def __init__(self, decision, critique, continue_opt):
                    self.decision = decision
                    self.critique = critique
                    self.continue_optimization = continue_opt
                    self.reasoning = f"Decision: {decision}"  # For backward compatibility
                    self.missing_keywords = []  # No longer used

            eval_response = EvalResponse(decision, critique, continue_optimization)

            # Log LLM interaction
            logger = get_global_prompt_logger()
            logger.log_interaction(
                agent_name="ResponseEvaluatorAgent",
                interaction_type="response_evaluation",
                system_prompt=prompt_content,
                user_prompt="Please evaluate the response. Remember: if the answer states that the context doesn't contain the information, then it is not satisfactory.",
                llm_response=response.text,
                batch_id=message.batch_id,
                qa_pair_id=message.qa_pair_id,
                iteration=message.repetition,
                additional_metadata={
                    "original_query": message.original_query,
                    "generated_answer_length": len(message.generated_answer),
                    "continue_optimization": eval_response.continue_optimization,
                    "critique_length": len(eval_response.critique),
                    "rouge_score": rouge_score
                }
            )

            log_qa_processing(self.logger, message.qa_pair_id,
                            f"Evaluation completed - continue: {eval_response.continue_optimization}",
                            f"Reasoning: {eval_response.reasoning}\nCritique: {eval_response.critique}\nMissing Keywords: {eval_response.missing_keywords}")
            print(f"[ResponseEvaluator] continue_optimization={eval_response.continue_optimization}")

            # Create evaluation result dictionary (excluding gold answers to prevent data leakage)
            evaluation_data = {
                "qa_pair_id": message.qa_pair_id,
                "original_query": message.original_query,
                "generated_answer": message.generated_answer,
                "evaluation_reasoning": eval_response.reasoning,
                "evaluation_feedback": eval_response.critique,
                "missing_keywords": eval_response.missing_keywords,
                "continue_optimization": eval_response.continue_optimization,
                "repetition": message.repetition,
                "timestamp": datetime.now().isoformat(),
                "rouge_score": rouge_score
            }

            # Update state with evaluation result (in-memory only)
            response_evaluations = current_state.get("response_evaluations", [])
            response_evaluations.append(evaluation_data)
            current_state["response_evaluations"] = response_evaluations

            # Store continue_optimization flag for early stopping
            current_state[f"continue_optimization_{message.qa_pair_id}"] = eval_response.continue_optimization

            # Update in-memory early-stop map to avoid cross-batch timing issues
            try:
                from shared_state import SharedState as _SS
                _SS.set_continue_flag(message.dataset, message.setting, message.qa_pair_id, eval_response.continue_optimization)
                print(f"[ResponseEvaluator] Updated in-memory continue flag for {message.qa_pair_id} -> {eval_response.continue_optimization}")
            except Exception as _:
                pass

            # Return ResponseEvaluationReady message with modified state
            eval_ready_msg = ResponseEvaluationReadyMessage(
                qa_pair_id=message.qa_pair_id,
                evaluation_result=evaluation_data,
                rouge_score=rouge_score,
                continue_optimization=eval_response.continue_optimization,
                batch_id=message.batch_id,
                repetition=message.repetition,
                shared_state=current_state  # Pass modified state back
            )

            self.logger.info(f"Returning ResponseEvaluationReady for QA pair {message.qa_pair_id}")
            print(f"[ResponseEvaluator] RETURN qa_pair_id={message.qa_pair_id}")

            # Return the evaluation ready message
            return eval_ready_msg

        except Exception as e:
            print("=" * 80)
            print("[ResponseEvaluator] ERROR after retries")
            print("=" * 80)
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            import traceback
            print(f"Traceback:")
            print(traceback.format_exc())
            print("=" * 80)

            self.logger.error("=" * 80)
            self.logger.error("RESPONSE EVALUATOR AGENT - ERROR AFTER ALL RETRIES")
            self.logger.error("=" * 80)
            self.logger.error(f"Error Type: {type(e).__name__}")
            self.logger.error(f"Error Message: {e}")
            self.logger.error(f"Traceback:")
            self.logger.error(traceback.format_exc())
            self.logger.error("=" * 80)

            # Raise exception to stop pipeline - don't continue with bad data
            raise RuntimeError(f"Failed to evaluate response for QA pair {message.qa_pair_id} after multiple retries: {e}")

    async def close(self) -> None:
        """Close the model client."""
        # Gemini client doesn't require explicit closing
        pass


# ===== BACKWARD PASS AGENT =====

class BackwardPassAgent(RoutedAgent):
    """
    Agent that performs backward pass through all agent critiques for vector system improvement.
    """

    def __init__(self, name: str, critique_token_limit: int = 512) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.backward_pass")
        self.shared_state = SharedState("agent_states")

        # Configuration for critique token limits
        self.critique_token_limit = critique_token_limit

        # Import vector-specific gradient prompts and optimizer prompts
        from parameters import (
            generation_prompt_gradient_prompt_vector,
            retrieved_content_gradient_prompt_vector,
            retrieval_summarizer_prompt_gradient_vector,
            retrieval_plan_gradient_prompt_vector,
            retrieval_planning_prompt_gradient_vector,
            rag_hyperparameters_agent_gradient_vector,
            answer_generation_prompt_optimizer_vector,
            retrieval_summarizer_prompt_optimizer_vector,
            retrieval_planner_prompt_optimizer_vector,
            hyperparameters_vector_agent_prompt_optimizer
        )

        # Initialize Gemini model client for simple text response
        self.model_client = OpenAIChatCompletionClient(
            model="gemini-2.5-flash-lite",
            max_tokens=4096,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=llm_keys.GEMINI_KEY,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            }
        )

        # Store all vector-specific gradient prompts
        self.generation_prompt_gradient_prompt_vector = generation_prompt_gradient_prompt_vector
        self.retrieved_content_gradient_prompt_vector = retrieved_content_gradient_prompt_vector
        self.retrieval_summarizer_prompt_gradient_vector = retrieval_summarizer_prompt_gradient_vector
        self.retrieval_plan_gradient_prompt_vector = retrieval_plan_gradient_prompt_vector
        self.retrieval_planning_prompt_gradient_vector = retrieval_planning_prompt_gradient_vector
        self.rag_hyperparameters_agent_gradient_vector = rag_hyperparameters_agent_gradient_vector

        # Store all vector-specific optimizer prompts
        self.answer_generation_prompt_optimizer_vector = answer_generation_prompt_optimizer_vector
        self.retrieval_summarizer_prompt_optimizer_vector = retrieval_summarizer_prompt_optimizer_vector
        self.retrieval_planner_prompt_optimizer_vector = retrieval_planner_prompt_optimizer_vector
        self.hyperparameters_vector_agent_prompt_optimizer = hyperparameters_vector_agent_prompt_optimizer

    @message_handler
    async def handle_backward_pass_start(self, message: BackwardPassStartMessage, ctx: MessageContext) -> BackwardPassReadyMessage:
        """Handle BackwardPassStart message and perform complete backward pass critique generation."""
        self.logger.info(f"BackwardPassAgent processing backward pass for batch {message.batch_id}")
        print(f"[BackwardPass] START batch_id={message.batch_id} dataset={message.dataset} setting={message.setting} repetition={message.repetition} qa_results={len(message.all_qa_results)}")

        # Load shared state with correct dataset and setting parameters (force fresh to avoid stale cache)
        print(f"[BackwardPass] ===== LOADING INITIAL STATE =====")
        print(f"[BackwardPass] Loading state for: batch_id={message.batch_id}, dataset={message.dataset}, setting={message.setting}")
        current_state = self.shared_state.load_state_fresh(message.dataset, message.setting, message.batch_id)
        print(f"[BackwardPass] Initial state keys: {list(current_state.keys())}")
        print(f"[BackwardPass] Initial has query_content_pairs: {('query_content_pairs' in current_state)}")
        if 'query_content_pairs' in current_state:
            print(f"[BackwardPass] Initial query_content_pairs length: {len(current_state.get('query_content_pairs', []))}")
        print(f"[BackwardPass] ===== INITIAL STATE LOADED =====")

        # HYDRATE STATE FROM MESSAGE if critical fields are missing (handles immediate backward pass timing)
        try:
            hydrated = False
            if message.all_qa_results:
                # Seed response_evaluations if empty
                if not current_state.get("response_evaluations"):
                    evals = []
                    for r in message.all_qa_results:
                        ev = r.get("evaluation", {}) or {}
                        evals.append({
                            "qa_pair_id": r.get("qa_pair_id", ""),
                            "original_query": r.get("question", ""),
                            "generated_answer": r.get("generated_answer", ""),
                            "evaluation_reasoning": ev.get("evaluation_reasoning", ev.get("reasoning", "")),
                            "evaluation_feedback": ev.get("evaluation_feedback", ev.get("critique", "")),
                            "missing_keywords": ev.get("missing_keywords", []),
                            "continue_optimization": ev.get("continue_optimization", True),
                            "repetition": message.repetition,
                            "timestamp": datetime.now().isoformat(),
                            "rouge_score": r.get("rouge_score", 0.0)
                        })
                    current_state["response_evaluations"] = evals
                    print(f"[BackwardPass] Hydrated response_evaluations from message: {len(evals)}")
                    hydrated = True

                # Seed conversations_answer_generation if empty
                if not current_state.get("conversations_answer_generation"):
                    convs = []
                    for r in message.all_qa_results:
                        convs.append({
                            "qa_pair_id": r.get("qa_pair_id", ""),
                            "question": r.get("question", ""),
                            "retrieved_context": r.get("retrieved_context", ""),
                            "system_prompt": "",
                            "user_prompt": "",
                            "generated_answer": r.get("generated_answer", ""),
                            "repetition": message.repetition
                        })
                    current_state["conversations_answer_generation"] = convs
                    print(f"[BackwardPass] Hydrated conversations_answer_generation: {len(convs)}")
                    hydrated = True

                # Seed retrieved_contexts if empty
                if not current_state.get("retrieved_contexts"):
                    ctxs = []
                    for r in message.all_qa_results:
                        rc = r.get("retrieved_context", "")
                        if rc:
                            ctxs.append(rc)
                    if ctxs:
                        current_state["retrieved_contexts"] = ctxs
                        print(f"[BackwardPass] Hydrated retrieved_contexts: {len(ctxs)}")
                        hydrated = True

                # Synthesize context_summary if missing but we have contexts
                if not current_state.get("context_summary"):
                    rc_list = current_state.get("retrieved_contexts", [])
                    if rc_list:
                        current_state["context_summary"] = "\n\n--- Context Separator ---\n\n".join(rc_list)
                        print("[BackwardPass] Synthesized context_summary from retrieved_contexts")
                        hydrated = True

                # Seed retrieval_plans if empty (fallback to QA questions)
                if not current_state.get("retrieval_plans"):
                    plans = []
                    for r in message.all_qa_results:
                        q = r.get("question", "")
                        if q:
                            plans.append(f"query='{q}'")
                    if plans:
                        current_state["retrieval_plans"] = plans
                        print(f"[BackwardPass] Hydrated retrieval_plans from questions: {len(plans)}")
                        hydrated = True

                # Seed retrieval_prompt if empty (fallback to base prompt)
                if not current_state.get("retrieval_prompt"):
                    try:
                        from parameters import base_prompt_vector_retrieval_planner
                        # Store a generic template since true prompt may be template-based
                        current_state["retrieval_prompt"] = base_prompt_vector_retrieval_planner
                        print("[BackwardPass] Hydrated retrieval_prompt with base template")
                        hydrated = True
                    except Exception as _:
                        pass

                # Seed query_content_pairs if empty (fallback - should not normally happen)
                if not current_state.get("query_content_pairs"):
                    print("[BackwardPass] ===== FALLBACK: query_content_pairs missing =====")
                    print(f"[BackwardPass] Repetition: {message.repetition}")
                    print(f"[BackwardPass] Current state keys: {list(current_state.keys())}")
                    print(f"[BackwardPass] Has retrieval_plans: {('retrieval_plans' in current_state)}")

                    # Try to construct from retrieval_plans if available
                    plans = current_state.get("retrieval_plans", [])
                    print(f"[BackwardPass] Number of retrieval_plans: {len(plans)}")

                    if plans:
                        fallback_pairs = []
                        for plan in plans:
                            fallback_pairs.append({"query": plan, "content": "Content not available"})
                        current_state["query_content_pairs"] = fallback_pairs
                        print(f"[BackwardPass] Created fallback query_content_pairs from retrieval_plans: {len(fallback_pairs)} pairs")
                        hydrated = True
                    else:
                        print("[BackwardPass] No retrieval_plans available - cannot create fallback")
                    print("[BackwardPass] ===== FALLBACK COMPLETE =====")

            if hydrated:
                self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)
                print("[BackwardPass] State hydrated and saved prior to critiques")
        except Exception as e:
            print(f"[BackwardPass] Hydration error: {e}")

        try:
            # Step 1: Generate answer generation critique
            print("[BackwardPass] Step1: answer generation critique -> BEGIN")
            await self._generate_answer_generation_critique(current_state, ctx)
            print("[BackwardPass] Step1: answer generation critique -> DONE")
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)
            print("[BackwardPass] State saved after Step1")

            # Step 2: Generate retrieved content critique
            print("[BackwardPass] Step2: retrieved content critique -> BEGIN")
            await self._generate_retrieved_content_critique(current_state, ctx)
            print("[BackwardPass] Step2: retrieved content critique -> DONE")
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)
            print("[BackwardPass] State saved after Step2")

            # Step 2.5: Generate retrieval summarizer prompt critique (with skip logic)
            print("[BackwardPass] Step2.5: retrieval summarizer prompt critique -> BEGIN")
            await self._generate_retrieval_summarizer_prompt_critique(current_state, ctx)
            print("[BackwardPass] Step2.5: retrieval summarizer prompt critique -> DONE")
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)
            print("[BackwardPass] State saved after Step2.5")

            # Step 3: Generate retrieval plan critique
            print("[BackwardPass] Step3: retrieval plan critique -> BEGIN")
            await self._generate_retrieval_plan_critique(current_state, ctx)
            print("[BackwardPass] Step3: retrieval plan critique -> DONE")
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)
            print("[BackwardPass] State saved after Step3")

            # Step 4: Generate retrieval planning prompt critique
            print("[BackwardPass] Step4: retrieval planning prompt critique -> BEGIN")
            await self._generate_retrieval_planning_prompt_critique(current_state, ctx)
            print("[BackwardPass] Step4: retrieval planning prompt critique -> DONE")
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)
            print("[BackwardPass] State saved after Step4")

            # Final save to ensure everything is persisted
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)
            print("[BackwardPass] Final state save complete")

            # Prepare optimized prompts to return to BatchOrchestrator (no hyperparameters - fixed chunk size)
            optimized_prompts = {
                "learned_prompt_answer_generator_vector": current_state.get("learned_prompt_answer_generator_vector", ""),
                "learned_prompt_retrieval_summarizer_vector": "",  # Not used - now using static summarization
                "learned_prompt_vector_retrieval_planner": current_state.get("learned_prompt_vector_retrieval_planner", "")
            }

            self.logger.info(f"DEBUG: BackwardPass - optimized_prompts prepared for BatchOrchestrator")
            self.logger.info(f"DEBUG: BackwardPass - optimized_prompts values (lengths): {[(k, len(v)) for k, v in optimized_prompts.items()]}")

            # Send BackwardPassReady message with optimized prompts (like GraphRAG)
            backward_ready_msg = BackwardPassReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                backward_pass_results={
                    "critiques_generated": True,
                    "total_qa_pairs": len(message.all_qa_results),
                    "optimized_prompts": optimized_prompts,
                    "critiques_updated": [
                        "answer_generation_critique",
                        "retrieved_content_critique",
                        "retrieval_summarizer_agent_critique",
                        "retrieval_plan_critique",
                        "retrieval_planner_agent_critique"
                    ]
                },
                dataset=message.dataset,
                setting=message.setting
            )

            self.logger.info(f"Returning BackwardPassReady for batch {message.batch_id}")
            print("[BackwardPass] RETURN BackwardPassReady")

            # Return the backward pass ready message
            return backward_ready_msg

        except Exception as e:
            self.logger.error(f"Error in backward pass: {e}")
            print(f"[BackwardPass] ERROR: {e}")
            # Return error response
            return BackwardPassReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                backward_pass_results={
                    "error": f"Backward pass failed: {e}",
                    "critiques_generated": False
                },
                dataset=message.dataset,
                setting=message.setting
            )

    def _format_all_evaluation_responses(self, all_evaluation_responses: list) -> str:
        """Format all evaluation responses into a single string."""
        if not all_evaluation_responses:
            return "No evaluation responses available"

        formatted_evals = []
        for eval_resp in all_evaluation_responses:
            iter_num = eval_resp.get('repetition', 0)
            eval_text = f"Iteration {iter_num}:\n"
            eval_text += f"  Reasoning: {eval_resp.get('evaluation_reasoning', 'N/A')}\n"
            eval_text += f"  Critique: {eval_resp.get('evaluation_feedback', 'N/A')}\n"
            eval_text += f"  Continue: {eval_resp.get('continue_optimization', False)}"
            formatted_evals.append(eval_text)

        return "\n\n".join(formatted_evals)

    def _format_all_evaluation_responses(self, all_evaluation_responses: list) -> str:
        """Format all evaluation responses into a single string (matching GraphRAG format)."""
        if not all_evaluation_responses:
            return "No evaluation responses available"

        formatted_evals = []
        for eval_resp in all_evaluation_responses:
            iter_num = eval_resp.get('repetition', 0)
            eval_text = f"""
--- ITERATION {iter_num} EVALUATION ---

Reasoning:
{eval_resp.get('evaluation_reasoning', 'N/A')}

Critique:
{eval_resp.get('evaluation_feedback', 'N/A')}

Continue Optimization: {eval_resp.get('continue_optimization', False)}
"""
            formatted_evals.append(eval_text)

        return "\n".join(formatted_evals)

    def _format_response_critique_history(self, conversations: list, evaluations: list) -> str:
        """Format combined history of generated responses and their critiques (for retrieval components)."""
        if not conversations and not evaluations:
            return "No history available"

        # Create a mapping of repetition -> data
        history_by_iter = {}

        # Add conversations
        for conv in conversations:
            rep = conv.get('repetition', 0)
            if rep not in history_by_iter:
                history_by_iter[rep] = {}
            history_by_iter[rep]['question'] = conv.get('question', 'N/A')
            history_by_iter[rep]['retrieved_context'] = conv.get('retrieved_context', 'N/A')
            history_by_iter[rep]['generated_answer'] = conv.get('generated_answer', 'N/A')

        # Add evaluations
        for eval_resp in evaluations:
            rep = eval_resp.get('repetition', 0)
            if rep not in history_by_iter:
                history_by_iter[rep] = {}
            history_by_iter[rep]['evaluation_reasoning'] = eval_resp.get('evaluation_reasoning', 'N/A')
            history_by_iter[rep]['evaluation_feedback'] = eval_resp.get('evaluation_feedback', 'N/A')
            history_by_iter[rep]['continue_optimization'] = eval_resp.get('continue_optimization', False)

        # Format output
        formatted_history = []
        for rep in sorted(history_by_iter.keys()):
            data = history_by_iter[rep]

            # Truncate retrieved context if too long
            retrieved_ctx = data.get('retrieved_context', 'N/A')
            if len(retrieved_ctx) > 500:
                retrieved_ctx = retrieved_ctx[:500] + "... [truncated]"

            iter_text = f"""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
â•‘ ITERATION {rep}
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

Question:
{data.get('question', 'N/A')}

Retrieved Context:
{retrieved_ctx}

Generated Answer:
{data.get('generated_answer', 'N/A')}

Evaluation Reasoning:
{data.get('evaluation_reasoning', 'N/A')}

Evaluation Critique:
{data.get('evaluation_feedback', 'N/A')}

Continue Optimization: {data.get('continue_optimization', False)}
"""
            formatted_history.append(iter_text)

        return "\n".join(formatted_history)

    async def _generate_answer_generation_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for answer generation prompt (with skip logic)."""
        self.logger.info("Generating answer generation critique")
        print("[BackwardPass] _generate_answer_generation_critique START")

        all_evaluation_responses = current_state.get("response_evaluations", [])

        if not all_evaluation_responses:
            self.logger.warning("No evaluation data available - skipping answer generation critique")
            current_state["answer_generation_critique"] = "No critique provided"
            print("[BackwardPass] No eval data  skipping answer generation critique")
            return

        # Get the current answer generation prompt
        current_answer_prompt = current_state.get("learned_prompt_answer_generator_vector", "")
        if not current_answer_prompt:
            from parameters import base_prompt_answer_generator_vector
            current_answer_prompt = base_prompt_answer_generator_vector

        # Get previous critique (empty string for first component in backward pass)
        previous_critique = ""

        # Get response evaluator output (all iterations)
        response_evaluator_output = self._format_all_evaluation_responses(all_evaluation_responses)

        # Format prompt with new structure
        prompt_content = self.generation_prompt_gradient_prompt_vector.format(
            current_prompt=current_answer_prompt,
            previous_critique=previous_critique,
            response_evaluator_output=response_evaluator_output
        )

        # Call LLM with structured output
        from parameters import PromptCritiqueResponse
        critique_response = await self._call_llm_structured(prompt_content, ctx, PromptCritiqueResponse)
        print(f"[BackwardPass] AnswerGen critique structured: problem={getattr(critique_response,'problem_in_this_component',None)}")

        # Implement skip logic
        if not critique_response.problem_in_this_component:
            # No problem in this component - pass "No critique provided" to next component
            current_state["answer_generation_critique"] = "No critique provided"
            self.logger.info("Answer generation prompt: No problem detected, skipping optimization")
            return

        # Problem detected - store critique and optimize
        critique = critique_response.critique
        current_state["answer_generation_critique"] = critique

        # Generate optimized prompt using the critique
        optimizer_prompt = self.answer_generation_prompt_optimizer_vector.format(critique)
        optimized_prompt = await self._call_llm(optimizer_prompt, ctx, "Please provide the new prompt", add_token_limit=False)
        print(f"[BackwardPass] AnswerGen optimized prompt length={len(optimized_prompt)}")

        # Limit prompt length to prevent truncation
        MAX_PROMPT_LENGTH = 4000
        if len(optimized_prompt) > MAX_PROMPT_LENGTH:
            self.logger.warning(f"Optimized answer generator prompt too long ({len(optimized_prompt)} chars), truncating to {MAX_PROMPT_LENGTH}")
            optimized_prompt = optimized_prompt[:MAX_PROMPT_LENGTH] + "\n\n[Prompt truncated to prevent errors]"

        # Only update if not frozen
        is_frozen = self._is_prompt_frozen("answer_generator_vector", current_state)
        if not is_frozen:
            current_state["learned_prompt_answer_generator_vector"] = optimized_prompt
            self.logger.info(f"Stored optimized answer generator prompt ({len(optimized_prompt)} chars)")
            print("[BackwardPass] Stored optimized answer generator prompt")

        log_critique_result(self.logger, "answer_generator_vector", critique, is_frozen)

    async def _generate_retrieved_content_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """
        DISABLED: Retrieved content critique.

        Skipping the retrieved content evaluation step as requested.
        """
        self.logger.info("Retrieved content critique: SKIPPED (disabled by user)")
        print("[BackwardPass] Retrieved content critique: SKIPPED")
        current_state["retrieved_content_critique"] = "Skipped - retrieved content evaluation disabled"
        return

    async def _generate_retrieval_summarizer_prompt_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """
        DISABLED: Retrieval summarizer critique and optimization.

        Now using simple static summarization (like Self-Refine) for better stability.
        No need to optimize the summarization prompt via backward pass.
        """
        self.logger.info("Retrieval summarizer prompt critique: SKIPPED (using static summarization)")
        current_state["retrieval_summarizer_agent_critique"] = "Skipped - using static summarization"
        return

    async def _generate_retrieval_plan_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """
        DISABLED: Retrieval plan critique.

        Skipping the retrieval plan critique as requested. Query critiques will be
        used directly by the prompt optimizer instead.
        """
        self.logger.info("Retrieval plan critique: SKIPPED (disabled by user)")
        print("[BackwardPass] Retrieval plan critique: SKIPPED")
        current_state["retrieval_plan_critique"] = "Skipped - retrieval plan critique disabled"
        return

    async def _generate_retrieval_planning_prompt_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for retrieval planning prompt based on response evaluator feedback."""
        self.logger.info("Generating retrieval planning prompt critique from response evaluator feedback")
        print("[BackwardPass] _generate_retrieval_planning_prompt_critique START")

        query_content_pairs = current_state.get("query_content_pairs", [])
        all_evaluation_responses = current_state.get("response_evaluations", [])
        conversations = current_state.get("conversations_answer_generation", [])

        if not query_content_pairs or not all_evaluation_responses:
            self.logger.warning("Missing query_content_pairs or evaluation for critique")
            current_state["retrieval_planner_agent_critique"] = "No critique provided"
            print(f"[BackwardPass] SKIP retrieval planning prompt critique â€” missing query_content_pairs={len(query_content_pairs)} or evals={len(all_evaluation_responses)}")
            return

        # Get the question from conversations
        question = conversations[0].get('question', 'Unknown question') if conversations else 'Unknown question'

        # Determine overall performance based on evaluations
        avg_quality = 0.0
        if all_evaluation_responses:
            # Check if responses were satisfactory
            satisfactory_count = sum(1 for eval_resp in all_evaluation_responses
                                   if not eval_resp.get('continue_optimization', True))
            avg_quality = satisfactory_count / len(all_evaluation_responses)

        # Get the response evaluator's critique
        print(f"[BackwardPass] Generating retrieval plan critique based on response evaluator feedback...")

        response_evaluator_critique = ""
        if all_evaluation_responses:
            # Get the most recent evaluation critique
            latest_eval = all_evaluation_responses[-1]
            response_evaluator_critique = latest_eval.get('evaluation_feedback', latest_eval.get('critique', ''))

        # Format queries
        queries_list = []
        for i, pair in enumerate(query_content_pairs, 1):
            query_text = pair.get("query", "Unknown query")
            queries_list.append(f"{i}. {query_text}")

        queries_formatted = "\n".join(queries_list)

        # Simple prompt focused on response evaluator feedback
        analysis_prompt = f"""You are evaluating the queries made by an iterative content retriever in a RAG system.
Your goal is to provide an accurate evaluation of the queries based on the following feedback.

The original question of the user was:
{question}

QUERIES MADE:
{queries_formatted}

RESPONSE EVALUATOR FEEDBACK:
{response_evaluator_critique}

Based on the response evaluator's feedback, provide:

1. **TYPES OF QUERIES TO FOCUS ON**:
   - What types of queries would address the issues mentioned in the feedback?
   - What aspects should future queries target to improve the answer?
   - Remember that each query must target a specific part of the text. A good query targets specific content that can be found in a single chunk.

Your task is not to rate the queries, but to identify what types of new queries would best address the issues raised in the feedback
Relate your suggestions directly to the user’s original question and the retrieved content, specifying what missing details or narrative connections new queries should aim to uncover
Provide a clear, actionable analysis."""

        combined_analysis = await self._call_llm(analysis_prompt, ctx)
        print(f"[BackwardPass] Retrieval plan analysis complete")

        # Format the information for the optimizer
        optimizer_input = f"""OVERALL QUESTION: {question}

RESPONSE EVALUATOR FEEDBACK:
{response_evaluator_critique}

QUERIES MADE:
{queries_formatted}

RETRIEVAL PLAN ANALYSIS:
{combined_analysis}
"""

        # Generate optimized prompt using the analysis
        print("[BackwardPass] Generating optimized retrieval planner prompt based on response evaluator feedback...")
        optimizer_prompt = self.retrieval_planner_prompt_optimizer_vector.format(optimizer_input)
        optimized_prompt = await self._call_llm(optimizer_prompt, ctx, "Please provide the new prompt", add_token_limit=False)
        print(f"[BackwardPass] RetrievalPlanner optimized prompt length={len(optimized_prompt)}")

        # Store the simplified analysis as critique
        current_state["retrieval_planner_agent_critique"] = f"Response evaluator-based analysis:\n{optimizer_input}"

        # Only update if not frozen
        is_frozen = self._is_prompt_frozen("vector_retrieval_planner", current_state)
        if not is_frozen:
            current_state["learned_prompt_vector_retrieval_planner"] = optimized_prompt
            print("[BackwardPass] Stored optimized retrieval planner prompt")

        log_critique_result(self.logger, "vector_retrieval_planner", "Response evaluator-based optimization applied", is_frozen)

    async def _generate_hyperparameters_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for hyperparameters agent (with skip logic)."""
        print("[BackwardPass] _generate_hyperparameters_critique START")
        all_evaluation_responses = current_state.get("response_evaluations", [])
        conversations = current_state.get("conversations_answer_generation", [])

        if not all_evaluation_responses:
            current_state["hyperparameters_vector_agent_critique"] = "No critique provided"
            print("[BackwardPass] No evals â€” skipping hyperparameters critique")
            return

        # Get current hyperparameters
        rag_hyperparams = current_state.get("rag_hyperparameters", {})
        chunk_size = rag_hyperparams.get("chunk_size", "Not specified")

        # Get previous critique from retrieval plan or retrieved content
        previous_critique = current_state.get("retrieval_plan_critique", "")
        if not previous_critique:
            previous_critique = current_state.get("retrieved_content_critique", "")

        # Get combined history of responses and critiques
        response_critique_history = self._format_response_critique_history(conversations, all_evaluation_responses)

        # Get batch information
        batch_info = current_state.get("batch_information", {})
        qa_pairs = batch_info.get("qa_pairs", [])
        retrieval_prompt = current_state.get("retrieval_prompt", "")
        retrieval_plans = current_state.get("retrieval_plans", [])

        if not qa_pairs or not retrieval_prompt or not retrieval_plans:
            current_state["hyperparameters_vector_agent_critique"] = "No critique provided"
            self.logger.warning("Missing data for hyperparameters critique")
            return

        # Create triplets: query + retrieval_prompt + retrieval_plan
        triplets = []
        for i, qa_pair in enumerate(qa_pairs):
            question = qa_pair.get("question", "")
            plan = retrieval_plans[i] if i < len(retrieval_plans) else "No plan available"
            triplet = f"Query: {question}\nRetrieval Prompt: {retrieval_prompt}\nRetrieval Plan: {plan}"
            triplets.append(triplet)

        concatenated_triplets = "\n\n".join(triplets)

        # Format prompt with new structure
        prompt_content = self.rag_hyperparameters_agent_gradient_vector.format(
            chunk_size=chunk_size,
            concatenated_triplets=concatenated_triplets,
            previous_critique=previous_critique,
            #  response_critique_history=response_critique_history
        )

        # Call LLM with structured output
        from parameters import PromptCritiqueResponse
        critique_response = await self._call_llm_structured(prompt_content, ctx, PromptCritiqueResponse)
        print(f"[BackwardPass] Hyperparameters critique structured: problem={getattr(critique_response,'problem_in_this_component',None)}")

        # Implement skip logic
        if not critique_response.problem_in_this_component:
            current_state["hyperparameters_vector_agent_critique"] = "No critique provided"
            self.logger.info("Hyperparameters: No problem detected, skipping optimization")
            print("[BackwardPass] Hyperparameters: no problem â€” skipping")
            return

        # Problem detected - store critique and optimize
        critique = critique_response.critique
        current_state["hyperparameters_vector_agent_critique"] = critique
        optimizer_prompt = self.hyperparameters_vector_agent_prompt_optimizer.format(critique)
        optimized_prompt = await self._call_llm(optimizer_prompt, ctx, "Please provide the new prompt", add_token_limit=False)
        print(f"[BackwardPass] Hyperparameters optimized prompt length={len(optimized_prompt)}")

        # Limit prompt length
        MAX_PROMPT_LENGTH = 4000
        if len(optimized_prompt) > MAX_PROMPT_LENGTH:
            optimized_prompt = optimized_prompt[:MAX_PROMPT_LENGTH] + "\n\n[Prompt truncated to prevent errors]"

        is_frozen = self._is_prompt_frozen("hyperparameters_vector", current_state)
        if not is_frozen:
            current_state["learned_prompt_hyperparameters_vector"] = optimized_prompt
            print("[BackwardPass] Stored optimized hyperparameters prompt")

        log_critique_result(self.logger, "hyperparameters_vector", critique, is_frozen)

    def _is_prompt_frozen(self, prompt_type: str, current_state: Dict[str, Any]) -> bool:
        """Check if a prompt type is frozen."""
        frozen_prompts = current_state.get("frozen_prompts", [])
        return prompt_type in frozen_prompts

    async def _call_llm(self, prompt_content: str, ctx: MessageContext, user_message_content: str = "Please provide your critique and feedback.", add_token_limit: bool = True) -> str:
        """Helper method to call LLM with given prompt and optional token limit."""
        try:
            print(f"[BackwardPass] _call_llm START add_token_limit={add_token_limit} prompt_len={len(prompt_content)}")
            # Add token limit instruction only for critiques, not for prompt optimization
            if add_token_limit:
                enhanced_prompt = f"{prompt_content}\n\nIMPORTANT: Please limit your critique to approximately {self.critique_token_limit} tokens to ensure efficient inference processing. Focus on the most critical points and be concise."
            else:
                enhanced_prompt = prompt_content

            system_message = SystemMessage(content=enhanced_prompt)
            user_message = UserMessage(content=user_message_content, source="system")

            response = await self.model_client.create(
                [system_message, user_message],
                cancellation_token=ctx.cancellation_token
            )

            result = response.content if isinstance(response.content, str) else str(response.content)
            print(f"[BackwardPass] _call_llm DONE response_len={len(result)}")

            # Log LLM interaction
            logger = get_global_prompt_logger()
            logger.log_interaction(
                agent_name="BackwardPassAgent",
                interaction_type="critique_generation",
                system_prompt=enhanced_prompt,
                user_prompt=user_message_content,
                llm_response=result,
                additional_metadata={
                    "critique_token_limit": self.critique_token_limit,
                    "prompt_length": len(prompt_content),
                    "response_length": len(result)
                }
            )

            return result

        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            print(f"[BackwardPass] _call_llm ERROR: {e}")
            return f"Error generating critique: {e}"

    async def _call_llm_structured(self, prompt_content: str, ctx: MessageContext, response_format, interaction_type: str = "critique", batch_id: int = None, user_prompt: str = "Please provide your critique."):
        """Helper method to call LLM with structured output (Pydantic response format)."""
        try:
            print(f"[BackwardPass] _call_llm_structured START prompt_len={len(prompt_content)} format={getattr(response_format,'__name__',str(response_format))}")
            # Create a temporary client with the response format
            structured_client = OpenAIChatCompletionClient(
                model="gemini-2.5-flash-lite",
                max_tokens=4096,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=llm_keys.GEMINI_KEY,
                model_info={
                    "vision": False,
                    "function_calling": True,
                    "json_output": True,
                    "family": "unknown",
                    "structured_output": True,
                },
                response_format=response_format
            )

            system_message = SystemMessage(content=prompt_content)
            user_message = UserMessage(content=user_prompt, source="system")

            response = await structured_client.create(
                [system_message, user_message],
                cancellation_token=ctx.cancellation_token
            )

            # Parse structured response
            assert isinstance(response.content, str)
            parsed_response = response_format.model_validate_json(response.content)

            # Log LLM interaction
            logger = get_global_prompt_logger()
            logger.log_interaction(
                agent_name="BackwardPassAgent",
                interaction_type=interaction_type,
                system_prompt=prompt_content,
                user_prompt=user_prompt,
                llm_response=response.content,
                batch_id=batch_id,
                additional_metadata={
                    "response_format": response_format.__name__,
                    "prompt_length": len(prompt_content)
                }
            )

            return parsed_response

        except Exception as e:
            self.logger.error(f"Error calling LLM with structured output: {e}")
            raise

    async def close(self) -> None:
        """Close the model client."""
        await self.model_client.close()


# ===== FACTORY FUNCTIONS =====

def create_batch_orchestrator_agent() -> BatchOrchestratorAgent:
    """Factory function to create BatchOrchestratorAgent instances."""
    return BatchOrchestratorAgent("batch_orchestrator_agent")

def create_hyperparameters_vector_agent() -> HyperparametersVectorAgent:
    """Factory function to create HyperparametersVectorAgent instances."""
    return HyperparametersVectorAgent("hyperparameters_vector_agent")

def create_vector_builder_agent() -> VectorBuilderAgent:
    """Factory function to create VectorBuilderAgent instances."""
    return VectorBuilderAgent("vector_builder_agent")

def create_vector_retrieval_planner_agent() -> VectorRetrievalPlannerAgent:
    """Factory function to create VectorRetrievalPlannerAgent instances."""
    return VectorRetrievalPlannerAgent("vector_retrieval_planner_agent")

def create_answer_generator_agent() -> AnswerGeneratorAgent:
    """Factory function to create AnswerGeneratorAgent instances."""
    return AnswerGeneratorAgent("answer_generator_agent")

def create_response_evaluator_agent(dataset_name: str = "qmsum_test") -> ResponseEvaluatorAgent:
    """Factory function to create ResponseEvaluatorAgent instances."""
    return ResponseEvaluatorAgent("response_evaluator_agent", dataset_name=dataset_name)

# ===== SUMMARIZER AGENT =====

class SummarizerAgent(RoutedAgent):
    """
    Agent that summarizes retrieved contexts to avoid overly long backward pass prompts.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.summarizer")
        self.shared_state = SharedState("agent_states")

        # Initialize Gemini model client for simple text response
        self.model_client = OpenAIChatCompletionClient(
            model="gemini-2.5-flash",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=llm_keys.GEMINI_KEY,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            }
        )

        # Base prompt for context summarization
        self.base_summarization_prompt = """
You are a context summarization expert. Your task is to create concise but comprehensive summaries of retrieved contexts.

Please summarize the following retrieved context, preserving all key information, entities, relationships, and facts that would be relevant for question answering and system improvement:

Context to summarize:
{context}

Provide a concise summary that captures all essential information while being significantly shorter than the original context.
"""

    @message_handler
    async def handle_summarization_start(self, message: SummarizationStartMessage, ctx: MessageContext) -> SummarizationReadyMessage:
        """Handle SummarizationStart message and generate summaries of retrieved contexts."""
        self.logger.info(f"SummarizerAgent processing {len(message.retrieved_contexts)} contexts for batch {message.batch_id}")

        context_summaries = []

        try:
            # Summarize each retrieved context
            for i, context in enumerate(message.retrieved_contexts):
                if not context.strip():
                    context_summaries.append("Empty context")
                    continue

                self.logger.info(f"Summarizing context {i+1}/{len(message.retrieved_contexts)} (length: {len(context)} chars)")

                # Prepare prompt for summarization
                prompt_content = self.base_summarization_prompt.format(context=context)

                # Call LLM for summarization
                system_message = SystemMessage(content="You are a helpful assistant that creates concise summaries.")
                user_message = UserMessage(content=prompt_content, source="user")

                response = await self.model_client.create(
                    [system_message, user_message],
                    cancellation_token=ctx.cancellation_token
                )

                summary = response.content if isinstance(response.content, str) else str(response.content)
                context_summaries.append(summary)

                self.logger.info(f"Context {i+1} summarized (original: {len(context)} â†’ summary: {len(summary)} chars)")

            # Create concatenated summary
            concatenated_summary = "\n\n--- Context Summary ---\n".join([
                f"Context {i+1}: {summary}"
                for i, summary in enumerate(context_summaries)
            ])

            self.logger.info(f"Generated {len(context_summaries)} summaries, concatenated length: {len(concatenated_summary)} chars")

            # Save summaries to shared state
            current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
            current_state["retrieved_context_summaries"] = context_summaries
            current_state["concatenated_context_summary"] = concatenated_summary
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Return SummarizationReady message
            return SummarizationReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                context_summaries=context_summaries,
                concatenated_summary=concatenated_summary
            )

        except Exception as e:
            self.logger.error(f"Error in context summarization: {e}")
            # Return empty summaries on error
            return SummarizationReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                context_summaries=["Error in summarization"],
                concatenated_summary="Error in summarization"
            )

    async def close(self) -> None:
        """Close the model client."""
        await self.model_client.close()


def create_backward_pass_agent(critique_token_limit: int = 512) -> BackwardPassAgent:
    """Factory function to create BackwardPassAgent instances."""
    return BackwardPassAgent("backward_pass_agent", critique_token_limit)

def create_summarizer_agent() -> SummarizerAgent:
    """Factory function to create SummarizerAgent instances."""
    return SummarizerAgent("summarizer_agent")



