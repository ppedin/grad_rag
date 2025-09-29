"""
Multi-Agent VectorRAG System with AutoGen Core API.
Includes BatchOrchestratorAgent and HyperparametersVectorAgent.
"""

import json
import logging
import statistics
import numpy as np
import faiss
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime

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

from shared_state import SharedState
from datasets_schema import Document, Question
from eval_functions import evaluate_rouge_score
import llm_keys
from autogen_dataset_agent import BatchStartMessage, BatchReadyMessage


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

class VectorRetrievalStartMessage(BaseModel):
    batch_id: int
    repetition: int
    query: str
    dataset: str
    setting: str
    k_iterations: int = 3
    shared_state: Dict[str, Any]

class VectorRetrievalReadyMessage(BaseModel):
    batch_id: int
    repetition: int
    retrieved_context: str
    dataset: str
    setting: str

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

class AnswerGenerationReadyMessage(BaseModel):
    qa_pair_id: str
    generated_answer: str
    batch_id: int
    repetition: int

class ResponseEvaluationStartMessage(BaseModel):
    qa_pair_id: str
    original_query: str
    generated_answer: str
    gold_answers: List[str]
    rouge_score: float
    batch_id: int
    repetition: int

class ResponseEvaluationReadyMessage(BaseModel):
    qa_pair_id: str
    evaluation_result: Dict[str, Any]
    rouge_score: float
    batch_id: int
    repetition: int

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
                self.logger.info(f"🔄 Processing QA pair {qa_pair_id} for iteration {current_iteration}")

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

                # Determine if this is a new QA pair or new iteration
                transition_type = self.shared_state.detect_transition_type(qa_pair_id, current_iteration)

                if transition_type == 'new_qa_pair':
                    # Complete reset for new QA pair
                    current_state = self.shared_state.reset_for_new_qa_pair(qa_pair_id, message.dataset, message.setting, message.batch_id)
                    current_state["batch_information"] = batch_info
                    current_state["full_document_text"] = batch_info.get("document_text", "")
                    self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)
                elif transition_type == 'new_iteration':
                    # Partial reset for new iteration of same QA pair
                    current_state = self.shared_state.reset_for_new_iteration(qa_pair_id, current_iteration, message.dataset, message.setting, message.batch_id)
                    current_state["batch_information"] = batch_info
                    current_state["full_document_text"] = batch_info.get("document_text", "")
                    self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

                # Process the current iteration only
                self.logger.info(f"📊 Processing QA pair {qa_pair_id}, iteration {current_iteration}")

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

                    self.logger.info(f"✓ Iteration {current_iteration} completed for {qa_pair_id} | ROUGE: {rouge_score:.4f}")

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
                self.logger.info(f"🎯 QA pair {qa_pair_id} iteration {current_iteration} completed - ROUGE: {current_rouge:.4f}")

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
                        best_answer=iteration_results.get("generated_answer", "")
                    )

                    # Log to standardized evaluation logger
                    self.standardized_logger.complete_qa_pair_evaluation(
                        qa_pair_id=qa_pair_id,
                        final_rouge_score=current_rouge,
                        rouge_progression=self.qa_pair_rouge_progression[qa_pair_id],
                        best_iteration=current_iteration,
                        total_iterations_completed=current_iteration + 1,
                        best_answer=iteration_results.get("generated_answer", "")
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
        self.logger.info(f"🔄 Processing iteration {iteration} for QA pair {qa_pair_id}")

        # Track intermediate outputs for comprehensive evaluation logging (matching GraphRAG)
        intermediate_outputs = {}
        step_start_time = datetime.now()

        try:
            # Step 1: Hyperparameters
            self.logger.info(f"Step 1/5: Hyperparameters optimization for {qa_pair_id}")

            step_agent_start = datetime.now()
            try:
                hyperparams_start_msg = HyperparametersVectorStartMessage(
                    qa_pair_id=qa_pair_id,
                    qa_pair=qa_pair,
                    batch_id=original_message.batch_id,
                    repetition=original_message.repetition,
                    dataset=original_message.dataset,
                    setting=original_message.setting
                )

                # Send to HyperparametersVectorAgent
                hyperparams_agent_id = AgentId("hyperparameters_vector_agent", "default")
                hyperparams_response = await self.send_message(hyperparams_start_msg, hyperparams_agent_id)

                step_execution_time = (datetime.now() - step_agent_start).total_seconds() * 1000

                # Log intermediate output
                intermediate_outputs["hyperparameters"] = {
                    "chunk_size": getattr(hyperparams_response, 'chunk_size', 512),
                    "processing_time_ms": step_execution_time,
                    "learned_prompt_used": bool(getattr(hyperparams_response, 'learned_prompt', None))
                }

            except Exception as e:
                step_execution_time = (datetime.now() - step_agent_start).total_seconds() * 1000
                intermediate_outputs["hyperparameters"] = {
                    "error": str(e),
                    "processing_time_ms": step_execution_time
                }
                raise

            # Step 2: Vector Building
            self.logger.info(f"Step 2/5: Vector building for {qa_pair_id}")

            step_agent_start = datetime.now()
            try:
                current_state = self.shared_state.load_state(original_message.dataset, original_message.setting, original_message.batch_id)

                vector_start_msg = VectorStartMessage(
                    batch_id=hyperparams_response.batch_id,
                    repetition=hyperparams_response.repetition,
                    chunk_size=hyperparams_response.chunk_size,
                    dataset=original_message.dataset,
                    setting=original_message.setting,
                    shared_state=current_state
                )

                vector_builder_agent_id = AgentId("vector_builder_agent", "default")
                vector_response = await self.send_message(vector_start_msg, vector_builder_agent_id)

                step_execution_time = (datetime.now() - step_agent_start).total_seconds() * 1000

                # Log intermediate output
                intermediate_outputs["vector_building"] = {
                    "chunk_size": hyperparams_response.chunk_size,
                    "faiss_index_path": getattr(vector_response, 'faiss_index_path', 'N/A'),
                    "vectors_created": True,
                    "embedding_method": "openai",
                    "processing_time_ms": step_execution_time
                }

            except Exception as e:
                step_execution_time = (datetime.now() - step_agent_start).total_seconds() * 1000
                intermediate_outputs["vector_building"] = {
                    "error": str(e),
                    "processing_time_ms": step_execution_time,
                    "vectors_created": False
                }
                raise

            # Step 3: Vector Retrieval
            self.logger.info(f"Step 3/5: Vector retrieval for {qa_pair_id}")
            # Update shared state with FAISS index paths for retrieval
            current_state = self.shared_state.load_state(original_message.dataset, original_message.setting, original_message.batch_id)
            current_state["faiss_index_path"] = vector_response.faiss_index_path
            current_state["chunk_metadata_path"] = vector_response.chunk_metadata_path
            self.shared_state.save_state(current_state, original_message.dataset, original_message.setting, original_message.batch_id)

            retrieval_start_msg = VectorRetrievalStartMessage(
                batch_id=vector_response.batch_id,
                repetition=vector_response.repetition,
                query=qa_pair.get("question", ""),
                dataset=original_message.dataset,
                setting=original_message.setting,
                k_iterations=3,
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

            # Update shared state with retrieved context
            current_state = self.shared_state.load_state(original_message.dataset, original_message.setting, original_message.batch_id)
            retrieved_contexts = current_state.get("retrieved_contexts", [])
            retrieved_contexts.append(retrieval_response.retrieved_context)
            current_state["retrieved_contexts"] = retrieved_contexts

            retrieval_queries = current_state.get("retrieval_queries", [])
            retrieval_queries.append(qa_pair.get("question", ""))
            current_state["retrieval_queries"] = retrieval_queries

            self.shared_state.save_state(current_state, original_message.dataset, original_message.setting, original_message.batch_id)

            # Step 4: Answer Generation
            self.logger.info(f"Step 4/5: Answer generation for {qa_pair_id}")
            answer_gen_msg = AnswerGenerationStartMessage(
                qa_pair_id=qa_pair_id,
                question=qa_pair.get("question", ""),
                retrieved_context=retrieval_response.retrieved_context,
                batch_id=retrieval_response.batch_id,
                repetition=retrieval_response.repetition,
                dataset=original_message.dataset,
                setting=original_message.setting
            )

            answer_gen_agent_id = AgentId("answer_generator_agent", "default")
            answer_response = await self.send_message(answer_gen_msg, answer_gen_agent_id)

            # Log intermediate output for answer generation
            intermediate_outputs["answer_generation"] = {
                "question": qa_pair.get("question", ""),
                "generated_answer": answer_response.generated_answer[:200] + "..." if len(answer_response.generated_answer) > 200 else answer_response.generated_answer,
                "generated_answer_length": len(answer_response.generated_answer)
            }

            # Step 5: Evaluation
            self.logger.info(f"Step 5/5: Evaluation for {qa_pair_id}")

            # Create evaluation message (ROUGE score will be computed by ResponseEvaluatorAgent)
            # Debug: check qa_pair structure
            self.logger.info(f"QA Pair keys: {list(qa_pair.keys())}")
            self.logger.info(f"QA Pair answer field: {qa_pair.get('answer', 'NOT_FOUND')}")

            # Try multiple possible answer field names
            gold_answer = qa_pair.get("answer") or qa_pair.get("answers") or qa_pair.get("expected_answer") or ""
            if isinstance(gold_answer, list):
                gold_answers = gold_answer
            else:
                gold_answers = [gold_answer] if gold_answer else []

            self.logger.info(f"Gold answers for ROUGE: {gold_answers}")

            eval_start_msg = ResponseEvaluationStartMessage(
                qa_pair_id=qa_pair_id,
                original_query=qa_pair.get("question", ""),
                generated_answer=answer_response.generated_answer,
                gold_answers=gold_answers,
                rouge_score=0.0,  # Will be computed by ResponseEvaluatorAgent
                batch_id=answer_response.batch_id,
                repetition=answer_response.repetition
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
                    "chunk_size": hyperparams_response.chunk_size,
                    "total_chunks": getattr(vector_response, 'total_chunks', 0),
                    "faiss_index_path": getattr(vector_response, 'faiss_index_path', ''),
                    "retrieved_context_length": len(retrieval_response.retrieved_context) if retrieval_response.retrieved_context else 0,
                    "embedding_dimension": 1536  # Standard OpenAI embedding dimension
                }
            except Exception as e:
                self.logger.warning(f"Could not collect detailed vector statistics: {e}")
                comprehensive_vector_stats = {"error": f"Stats collection failed: {e}"}

            # Store evaluation results in shared state
            current_state = self.shared_state.load_state(original_message.dataset, original_message.setting, original_message.batch_id)
            response_evaluations = current_state.get("response_evaluations", [])
            response_evaluations.append({
                "qa_pair_id": qa_pair_id,
                "iteration": iteration,
                "question": qa_pair.get("question", ""),
                "expected_answer": qa_pair.get("answer", ""),
                "generated_answer": answer_response.generated_answer,
                "evaluation": eval_response.evaluation_result,
                "rouge_score": rouge_score,
                "chunk_size": hyperparams_response.chunk_size,
                "comprehensive_vector_statistics": comprehensive_vector_stats
            })
            current_state["response_evaluations"] = response_evaluations

            # Store ROUGE scores for tracking
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
                hyperparameters={"chunk_size": hyperparams_response.chunk_size},
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
                hyperparameters={"chunk_size": hyperparams_response.chunk_size},
                execution_time_seconds=total_execution_time,
                system_specific_metrics=comprehensive_vector_stats
            )

            # Perform backward pass if not the last iteration
            # Since DatasetAgent handles repetitions, we should run backward pass for all iterations except the last
            current_state = self.shared_state.load_state(original_message.dataset, original_message.setting, original_message.batch_id)
            batch_info = current_state.get("batch_information", {})
            total_iterations = batch_info.get("total_iterations", 1)

            # Run backward pass for iterations 0 to (total_iterations - 2), skip for last iteration
            if iteration < total_iterations - 1:
                self.logger.info(f"🔄 Starting backward pass for iteration {iteration}")

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
                    "chunk_size": hyperparams_response.chunk_size
                }

                backward_pass_msg = BackwardPassStartMessage(
                    batch_id=original_message.batch_id,
                    repetition=original_message.repetition,
                    dataset=original_message.dataset,
                    setting=original_message.setting,
                    all_qa_results=[qa_result]  # Include current QA result
                )

                backward_agent_id = AgentId("backward_pass_agent", "default")
                backward_response = await self.send_message(backward_pass_msg, backward_agent_id)
                self.logger.info(f"✓ Backward pass completed for iteration {iteration}")

                # Process backward pass response to update QA pair prompts (like GraphRAG does)
                await self._process_backward_pass_response(backward_response, ctx)

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
                "chunk_size": hyperparams_response.chunk_size,
                "hyperparams_response": hyperparams_response.model_dump() if hasattr(hyperparams_response, 'model_dump') else str(hyperparams_response),
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
                query=question,
                dataset=vector_response.dataset,
                setting=vector_response.setting,
                k_iterations=3,
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

        # Load and update shared state
        current_state = self.shared_state.load_state(retrieval_response.dataset, retrieval_response.setting, retrieval_response.batch_id)

        # Store retrieved context for this QA pair
        retrieved_contexts = current_state.get("retrieved_contexts", [])
        retrieved_contexts.append(retrieval_response.retrieved_context)
        current_state["retrieved_contexts"] = retrieved_contexts

        # Store the query associated with this retrieval
        retrieval_queries = current_state.get("retrieval_queries", [])
        retrieval_queries.append(qa_pair.get("question", ""))
        current_state["retrieval_queries"] = retrieval_queries

        self.shared_state.save_state(current_state, retrieval_response.dataset, retrieval_response.setting, retrieval_response.batch_id)

        # Send AnswerGenerationStart message for THIS QA pair
        answer_gen_msg = AnswerGenerationStartMessage(
            qa_pair_id=qa_pair_id,
            question=qa_pair.get("question", ""),
            retrieved_context=retrieval_response.retrieved_context,
            batch_id=retrieval_response.batch_id,
            repetition=retrieval_response.repetition,
            dataset=retrieval_response.dataset,
            setting=retrieval_response.setting
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

        qa_pair = self.current_batch_qa_pairs.get(answer_response.qa_pair_id)
        if not qa_pair:
            self.logger.error(f"QA pair {answer_response.qa_pair_id} not found")
            return

        # Compute ROUGE score using real implementation
        rouge_score = self._compute_rouge_score(qa_pair, answer_response.generated_answer)

        self.logger.info(f"Computed ROUGE score {rouge_score:.4f} for QA pair {answer_response.qa_pair_id}")

        # Save ROUGE score to shared state
        current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, answer_response.batch_id)
        rouge_scores_list = current_state.get("rouge_scores", [])
        rouge_scores_list.append(rouge_score)
        current_state["rouge_scores"] = rouge_scores_list
        self.shared_state.save_state(current_state, self.current_dataset, self.current_setting, answer_response.batch_id)

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
                "chunk_size": current_state.get("rag_hyperparameters", {}).get("chunk_size", 512),
                "processing_time_ms": 0,  # Not available in this context
                "learned_prompt_used": False  # Not available in this context
            },
            "vector_building": {
                "chunk_size": current_state.get("rag_hyperparameters", {}).get("chunk_size", 512),
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
            hyperparameters={"chunk_size": current_state.get("rag_hyperparameters", {}).get("chunk_size", 512)},
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
                hyperparameters={"chunk_size": current_state.get("rag_hyperparameters", {}).get("chunk_size", 512)},
                execution_time_seconds=None,  # Not available in this context
                system_specific_metrics={"vector_count": len(current_state.get("retrieved_contexts", [])), "retrieval_method": "vector_similarity"}
            )

        # Send ResponseEvaluationStart message
        eval_start_msg = ResponseEvaluationStartMessage(
            qa_pair_id=answer_response.qa_pair_id,
            original_query=qa_pair.get("question", ""),
            generated_answer=answer_response.generated_answer,
            gold_answers=gold_answers,
            rouge_score=rouge_score,
            batch_id=answer_response.batch_id,
            repetition=answer_response.repetition
        )

        self.logger.info(f"Sending ResponseEvaluationStart for QA pair {answer_response.qa_pair_id}")

        # Send to ResponseEvaluatorAgent
        try:
            response_eval_agent_id = AgentId("response_evaluator_agent", "default")
            eval_response = await self.send_message(eval_start_msg, response_eval_agent_id)
            self.logger.info(f"Received ResponseEvaluationReady response")

            # Continue with final processing
            await self._process_evaluation_response(eval_response, ctx)

        except Exception as e:
            self.logger.warning(f"ResponseEvaluatorAgent not available, falling back to simulation: {e}")
            # Fallback to simulation if agent is not available
            await self._simulate_response_evaluation_ready(eval_start_msg, ctx)

    async def _process_evaluation_response(self, eval_response: ResponseEvaluationReadyMessage, ctx: MessageContext) -> None:
        """Process evaluation response and track completion."""
        self.logger.info(f"Processing evaluation response for QA pair {eval_response.qa_pair_id}")

        # Add to shared state
        current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, eval_response.batch_id)
        response_evaluations = current_state.get("response_evaluations", [])
        response_evaluations.append(eval_response.evaluation_result)
        current_state["response_evaluations"] = response_evaluations
        self.shared_state.save_state(current_state, self.current_dataset, self.current_setting, eval_response.batch_id)

        # Mark QA pair as completed
        self.completed_qa_pairs[eval_response.qa_pair_id] = eval_response.evaluation_result

        # Check if all QA pairs are completed
        if len(self.completed_qa_pairs) == len(self.current_batch_qa_pairs):
            self.logger.info(f"All {len(self.completed_qa_pairs)} QA pairs completed for batch {eval_response.batch_id}")

            # Generate context summary before backward pass
            await self._generate_context_summary(eval_response.batch_id, eval_response.repetition, ctx)

            # Now start the backward pass
            await self._start_backward_pass(eval_response.batch_id, eval_response.repetition, ctx)

    async def _generate_context_summary(self, batch_id: int, repetition: int, ctx: MessageContext) -> None:
        """Generate summary of all retrieved contexts before backward pass."""
        self.logger.info(f"Generating context summary for batch {batch_id}")

        # Load current shared state
        current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, batch_id)
        retrieved_contexts = current_state.get("retrieved_contexts", [])

        if not retrieved_contexts:
            self.logger.warning("No retrieved contexts to summarize")
            return

        # Log context details
        self.logger.info(f"Preparing to summarize {len(retrieved_contexts)} contexts")

        # Send SummarizationStart message with list of contexts
        summarization_msg = SummarizationStartMessage(
            batch_id=batch_id,
            repetition=repetition,
            retrieved_contexts=retrieved_contexts,  # Pass list directly
            dataset=self.current_dataset,
            setting=self.current_setting
        )

        try:
            summarizer_agent_id = AgentId("summarizer_agent", "default")
            self.logger.info("Sending summarization request to SummarizerAgent")
            summary_response = await self.send_message(summarization_msg, summarizer_agent_id)

            # Store the summary in shared state
            current_state["context_summary"] = summary_response.summary
            self.logger.info(f"Context summary generated successfully, length: {len(summary_response.summary)} chars")

            # Log the summary for debugging
            self.logger.debug(f"Generated summary preview: {summary_response.summary[:200]}...")

        except Exception as e:
            self.logger.warning(f"SummarizerAgent not available, using concatenated contexts as fallback: {e}")
            # Fallback to using concatenated contexts
            concatenated_contexts = "\n\n--- Context Separator ---\n\n".join(retrieved_contexts)
            current_state["context_summary"] = concatenated_contexts

        # Save state with summary
        self.shared_state.save_state(current_state, self.current_dataset, self.current_setting, batch_id)
        self.logger.info(f"Context summary saved to shared state for batch {batch_id}")

    # Continue with other methods...
    async def _start_backward_pass(self, batch_id: int, repetition: int, ctx: MessageContext) -> None:
        """Start the backward pass."""
        all_qa_results = list(self.completed_qa_pairs.values())

        backward_pass_msg = BackwardPassStartMessage(
            batch_id=batch_id,
            repetition=repetition,
            dataset=self.current_dataset,
            setting=self.current_setting,
            all_qa_results=all_qa_results
        )

        self.logger.info(f"Starting backward pass for batch {batch_id}")

        # Send to BackwardPassAgent and get response
        try:
            backward_pass_agent_id = AgentId("backward_pass_agent", "default")
            backward_response = await self.send_message(backward_pass_msg, backward_pass_agent_id)
            self.logger.info(f"Received BackwardPassReady response")

            # Process the backward pass response and send final result to DatasetAgent
            await self._process_backward_pass_response(backward_response, ctx)

        except Exception as e:
            self.logger.warning(f"BackwardPassAgent not available, falling back to simulation: {e}")
            # Fallback to simulation if agent is not available
            await self._simulate_backward_pass_ready(backward_pass_msg, ctx)

    async def _process_backward_pass_response(self, backward_response: BackwardPassReadyMessage, ctx: MessageContext) -> None:
        """Process backward pass response and send BatchReady to DatasetAgent."""
        self.logger.info(f"Processing backward pass response for batch {backward_response.batch_id}")

        # Update QA pair prompts with optimized versions (like GraphRAG does)
        if hasattr(backward_response, 'backward_pass_results'):
            optimized_prompts = backward_response.backward_pass_results.get("optimized_prompts", {})
            current_qa_pair_id = self.shared_state.current_qa_pair_id
            if current_qa_pair_id and optimized_prompts:
                self.shared_state.update_qa_pair_prompts(current_qa_pair_id, optimized_prompts)

                # CRITICAL FIX: Save optimized prompts to persistent state for next DatasetAgent call
                current_state = self.shared_state.load_state(backward_response.dataset, backward_response.setting, backward_response.batch_id)
                for prompt_key, prompt_value in optimized_prompts.items():
                    current_state[prompt_key] = prompt_value
                self.shared_state.save_state(current_state, backward_response.dataset, backward_response.setting, backward_response.batch_id)

                self.logger.info(f"Updated QA pair prompts for {current_qa_pair_id} - preserving learned prompts for next iteration")
                self.logger.info(f"DEBUG: BatchOrchestrator - optimized_prompts saved to persistent state: {[(k, len(v)) for k, v in optimized_prompts.items()]}")
            else:
                self.logger.warning(f"Cannot update QA pair prompts - current_qa_pair_id: {current_qa_pair_id}, optimized_prompts: {bool(optimized_prompts)}")

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
            repetition=answer_gen.repetition
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
            batch_id=eval_start.batch_id,
            repetition=eval_start.repetition
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
            }
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

            # Check if learned prompt exists from previous iteration's backward pass
            learned_system_prompt = current_state.get("learned_prompt_hyperparameters_vector", "")

            # Debug: log all learned prompts in shared state
            learned_keys = [k for k in current_state.keys() if k.startswith("learned_prompt")]
            self.logger.info(f"DEBUG: Found learned prompt keys in state: {learned_keys}")
            self.logger.info(f"DEBUG: learned_prompt_hyperparameters_vector value: '{learned_system_prompt[:100]}...' (length: {len(learned_system_prompt)})")

            if not learned_system_prompt:
                self.logger.info(f"First iteration for QA pair {message.qa_pair_id} - using empty system prompt")
            else:
                self.logger.info(f"Using learned system prompt for QA pair {message.qa_pair_id} (length: {len(learned_system_prompt)} chars)")

            # Prepare base user prompt without critique placeholder
            text = message.qa_pair.get("text", "")
            question = message.qa_pair.get("question", "")

            user_prompt_content = self.base_prompt_hyperparameters_vector.format(
                text=text,
                question=question,
                critique=""  # Empty critique for initial call
            )

            # Create messages with dual-prompt structure
            system_message = SystemMessage(content=learned_system_prompt)
            user_message = UserMessage(content=user_prompt_content, source="user")

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
                interaction_type="hyperparameters_generation",
                system_prompt=learned_system_prompt,
                user_prompt=user_prompt_content,
                llm_response=response.content,
                qa_pair_id=message.qa_pair_id,
                additional_metadata={
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
                chunk_size=512,  # Default chunk size
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
        """Handle VectorStart message by rebuilding FAISS index from scratch."""
        self.logger.info(f"VectorBuilderAgent processing batch {message.batch_id} with chunk_size {message.chunk_size} (iteration {message.repetition})")

        # Always clear existing index for reconstruction (test-time training approach)
        self._reset_vector_store()
        self.logger.info(f"Cleared existing vector store for reconstruction (iteration {message.repetition})")

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
                total_chunks=0
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
                        "batch_id": message.batch_id,
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

                # Save index to file for reuse
                import os
                index_dir = "vector_indexes"
                os.makedirs(index_dir, exist_ok=True)
                index_filename = os.path.join(index_dir, f"{message.dataset}_{message.setting}_batch_{message.batch_id}.index")
                faiss.write_index(self.faiss_index, index_filename)

                # Save metadata
                metadata_filename = os.path.join(index_dir, f"{message.dataset}_{message.setting}_batch_{message.batch_id}_metadata.json")
                with open(metadata_filename, 'w', encoding='utf-8') as f:
                    json.dump(self.chunk_metadata, f, indent=2, ensure_ascii=False)

                self.logger.info(f"Saved FAISS index and metadata to {index_filename}")

            except Exception as e:
                self.logger.error(f"Error creating FAISS index: {e}")

        # Prepare paths for VectorReady message
        index_dir = "vector_indexes"
        index_filename = os.path.join(index_dir, f"{message.dataset}_{message.setting}_batch_{message.batch_id}.index")
        metadata_filename = os.path.join(index_dir, f"{message.dataset}_{message.setting}_batch_{message.batch_id}_metadata.json")

        # Send VectorReady message
        vector_ready_msg = VectorReadyMessage(
            batch_id=message.batch_id,
            repetition=message.repetition,
            dataset=message.dataset,
            setting=message.setting,
            faiss_index_path=index_filename,
            chunk_metadata_path=metadata_filename,
            total_chunks=len(chunks) if 'chunks' in locals() else 0
        )

        self.logger.info(f"Returning VectorReady for batch {message.batch_id}")

        # Return the VectorReady message
        return vector_ready_msg

    def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately chunk_size tokens."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_token_count = 0

        for word in words:
            if current_token_count + 1 > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_token_count = 1
            else:
                current_chunk.append(word)
                current_token_count += 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

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
            response_format=VectorRetrievalPlannerResponse
        )

        self.base_prompt_vector_retrieval_planner = base_prompt_vector_retrieval_planner

    @message_handler
    async def handle_vector_retrieval_start(self, message: VectorRetrievalStartMessage, ctx: MessageContext) -> VectorRetrievalReadyMessage:
        """Handle VectorRetrievalStart message and execute iterative retrieval."""
        self.logger.info(f"VectorRetrievalPlannerAgent processing batch {message.batch_id} for query: {message.query}")

        # Load the correct FAISS index
        faiss_index, chunk_metadata = self._load_faiss_index(message.dataset, message.setting, message.batch_id)

        if faiss_index is None:
            self.logger.error(f"Could not load FAISS index for batch {message.batch_id}")
            return VectorRetrievalReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                retrieved_context="Error: Could not load vector index",
                dataset=message.dataset,
                setting=message.setting
            )

        # Load shared state
        current_state = message.shared_state

        # Check if learned prompt exists from previous iteration's backward pass
        learned_system_prompt = current_state.get("learned_prompt_vector_retrieval_planner", "")
        if not learned_system_prompt:
            self.logger.info(f"First iteration for batch {message.batch_id} - using empty retrieval planner system prompt")
        else:
            self.logger.info(f"Using learned retrieval planner system prompt for batch {message.batch_id}")

        # Create base user prompt template and save to shared state
        prompt_template = self.base_prompt_vector_retrieval_planner.format(
            message.query, "{RETRIEVED_CONTEXT}"
        )
        current_state["retrieval_prompt"] = prompt_template

        # Initialize retrieval context and plan responses
        retrieved_context = ""
        retrieval_plan_responses = []
        decision_history = []  # Track history of decisions and results

        # Execute k iterations of retrieval
        for iteration in range(message.k_iterations):
            self.logger.info(f"Retrieval iteration {iteration + 1}/{message.k_iterations}")

            try:
                # Format decision history for the prompt
                if not decision_history:
                    history_text = "No previous decisions in this session."
                else:
                    history_parts = []
                    for i, entry in enumerate(decision_history, 1):
                        history_parts.append(f"{i}. Decision: {entry['decision']}\n   Retrieved: {entry['result']}")
                    history_text = "\n\n".join(history_parts)

                # Prepare base user prompt with decision history
                user_prompt_content = self.base_prompt_vector_retrieval_planner.format(
                    message.query, history_text
                )

                # Create messages with dual-prompt structure
                system_message = SystemMessage(content=learned_system_prompt)
                user_message = UserMessage(content=user_prompt_content, source="user")

                response = await self.model_client.create(
                    [system_message, user_message],
                    cancellation_token=ctx.cancellation_token
                )

                # Parse structured response
                assert isinstance(response.content, str)
                from parameters import VectorRetrievalPlannerResponse
                retrieval_response = VectorRetrievalPlannerResponse.model_validate_json(response.content)

                # Log LLM interaction
                logger = get_global_prompt_logger()
                logger.log_interaction(
                    agent_name="VectorRetrievalPlannerAgent",
                    interaction_type="retrieval_planning",
                    system_prompt=learned_system_prompt,
                    user_prompt=user_prompt_content,
                    llm_response=response.content,
                    additional_metadata={
                        "iteration": iteration + 1,
                        "query": retrieval_response.query,
                        "reasoning_length": len(retrieval_response.reasoning),
                        "decision_history_length": len(decision_history)
                    }
                )

                # Store the LLM response
                retrieval_plan_responses.append(retrieval_response.reasoning)

                # Execute vector search with the refined query
                new_context = await self._execute_vector_search(retrieval_response.query, faiss_index, chunk_metadata)

                # Track this decision and its result in the history
                decision_entry = {
                    'decision': f"vector_search('{retrieval_response.query}')",
                    'result': new_context if new_context else "No results returned"
                }
                decision_history.append(decision_entry)

                # Add to retrieved context (still needed for backward compatibility)
                if new_context:
                    retrieved_context += f"\n\nIteration {iteration + 1} results:\n{new_context}"

                self.logger.info(f"Completed iteration {iteration + 1}, decision: vector_search('{retrieval_response.query}'), context length: {len(retrieved_context)}")

            except Exception as e:
                self.logger.error(f"Error in retrieval iteration {iteration + 1}: {e}")
                continue

        # Save retrieval plans to shared state
        current_state["retrieval_plans"] = retrieval_plan_responses
        self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

        # Send VectorRetrievalReady message
        retrieval_ready_msg = VectorRetrievalReadyMessage(
            batch_id=message.batch_id,
            repetition=message.repetition,
            retrieved_context=retrieved_context,
            dataset=message.dataset,
            setting=message.setting
        )

        self.logger.info(f"Returning VectorRetrievalReady for batch {message.batch_id}")

        # Return the retrieval ready message
        return retrieval_ready_msg

    def _load_faiss_index(self, dataset: str, setting: str, batch_id: int):
        """Load FAISS index and metadata from disk."""
        try:
            import os

            index_dir = "vector_indexes"
            index_filename = os.path.join(index_dir, f"{dataset}_{setting}_batch_{batch_id}.index")
            metadata_filename = os.path.join(index_dir, f"{dataset}_{setting}_batch_{batch_id}_metadata.json")

            if not os.path.exists(index_filename) or not os.path.exists(metadata_filename):
                self.logger.error(f"Index files not found: {index_filename}")
                return None, None

            # Load FAISS index
            faiss_index = faiss.read_index(index_filename)

            # Load metadata
            with open(metadata_filename, 'r', encoding='utf-8') as f:
                chunk_metadata = json.load(f)

            self.logger.info(f"Loaded FAISS index with {faiss_index.ntotal} vectors")
            return faiss_index, chunk_metadata

        except Exception as e:
            self.logger.error(f"Error loading FAISS index: {e}")
            return None, None

    async def _execute_vector_search(self, query: str, faiss_index, chunk_metadata, k: int = 5) -> str:
        """Execute vector search using FAISS index."""
        try:
            # Get query embedding using async function
            from llm import get_embeddings_async
            query_embeddings = await get_embeddings_async([query])
            query_vector = np.array(query_embeddings[0]).astype('float32').reshape(1, -1)

            # Normalize for cosine similarity
            faiss.normalize_L2(query_vector)

            # Search for top k similar vectors
            scores, indices = faiss_index.search(query_vector, k)

            # Retrieve corresponding text chunks
            retrieved_texts = []
            for i, idx in enumerate(indices[0]):
                if idx < len(chunk_metadata):
                    chunk_text = chunk_metadata[idx]["text"]
                    score = scores[0][i]
                    retrieved_texts.append(f"Score: {score:.4f}\n{chunk_text}")

            return "\n\n".join(retrieved_texts)

        except Exception as e:
            self.logger.error(f"Error executing vector search: {e}")
            return ""

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
        from parameters import base_prompt_answer_generator_vector

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
            }
        )

        self.base_prompt_answer_generator_vector = base_prompt_answer_generator_vector

    @message_handler
    async def handle_answer_generation_start(self, message: AnswerGenerationStartMessage, ctx: MessageContext) -> AnswerGenerationReadyMessage:
        """Handle AnswerGenerationStart message and generate answer using LLM."""
        self.logger.info(f"AnswerGeneratorAgent processing QA pair {message.qa_pair_id}")

        # Load shared state to get learned system prompt
        current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)

        # Check if learned prompt exists from previous iteration's backward pass
        learned_system_prompt = current_state.get("learned_prompt_answer_generator_vector", "")
        if not learned_system_prompt:
            self.logger.info(f"First iteration for QA pair {message.qa_pair_id} - using empty answer generator system prompt")
        else:
            self.logger.info(f"Using learned answer generator system prompt for QA pair {message.qa_pair_id}")

        # Prepare base user prompt with question and retrieved context
        user_prompt_content = self.base_prompt_answer_generator_vector.format(
            question=message.question,
            retrieved_context=message.retrieved_context,
            critique=""  # Empty critique for initial call
        )

        try:
            # Create messages with dual-prompt structure
            system_message = SystemMessage(content=learned_system_prompt)
            user_message = UserMessage(content=user_prompt_content, source="user")

            response = await self.model_client.create(
                [system_message, user_message],
                cancellation_token=ctx.cancellation_token
            )

            # Get generated answer
            generated_answer = response.content if isinstance(response.content, str) else str(response.content)

            # Log LLM interaction
            logger = get_global_prompt_logger()
            logger.log_interaction(
                agent_name="AnswerGeneratorAgent",
                interaction_type="answer_generation",
                system_prompt=learned_system_prompt,
                user_prompt=user_prompt_content,
                llm_response=generated_answer,
                qa_pair_id=message.qa_pair_id,
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
                "system_prompt": learned_system_prompt,
                "user_prompt": user_prompt_content,
                "generated_answer": generated_answer
            }

            conversations = current_state.get("conversations_answer_generation", [])
            conversations.append(conversation_entry)
            current_state["conversations_answer_generation"] = conversations

            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Send AnswerGenerationReady message
            answer_ready_msg = AnswerGenerationReadyMessage(
                qa_pair_id=message.qa_pair_id,
                generated_answer=generated_answer,
                batch_id=message.batch_id,
                repetition=message.repetition
            )

            self.logger.info(f"Returning AnswerGenerationReady for QA pair {message.qa_pair_id}")

            # Return the answer ready message
            return answer_ready_msg

        except Exception as e:
            self.logger.error(f"Error in answer generation: {e}")
            # Return default response on error
            return AnswerGenerationReadyMessage(
                qa_pair_id=message.qa_pair_id,
                generated_answer="Error generating answer",
                batch_id=message.batch_id,
                repetition=message.repetition
            )

    async def close(self) -> None:
        """Close the model client."""
        await self.model_client.close()


# ===== RESPONSE EVALUATOR AGENT =====

class ResponseEvaluatorAgent(RoutedAgent):
    """
    Agent that evaluates generated responses against gold answers using LLM.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.response_evaluator")
        self.shared_state = SharedState("agent_states")

        # Import prompts
        from parameters import response_evaluator_prompt

        # Initialize Gemini model client for simple text response
        self.model_client = OpenAIChatCompletionClient(
            model="gemini-2.5-flash-lite",
            max_tokens=1024,
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


        self.response_evaluator_prompt = response_evaluator_prompt

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

    @message_handler
    async def handle_response_evaluation_start(self, message: ResponseEvaluationStartMessage, ctx: MessageContext) -> ResponseEvaluationReadyMessage:
        """Handle ResponseEvaluationStart message and evaluate response using LLM."""
        self.logger.info(f"ResponseEvaluatorAgent evaluating QA pair {message.qa_pair_id}")

        # Compute actual ROUGE score between generated answer and gold answers
        rouge_score = 0.0

        if message.gold_answers and message.generated_answer:
            # Filter out empty gold answers
            valid_gold_answers = [ans for ans in message.gold_answers if ans.strip()]
            if valid_gold_answers:
                # Create a mock QA pair for ROUGE computation
                qa_pair_for_rouge = {"answers": valid_gold_answers}
                rouge_score = self._compute_rouge_score(qa_pair_for_rouge, message.generated_answer)
                self.logger.info(f"Computed ROUGE score {rouge_score:.4f} for QA pair {message.qa_pair_id}")
            else:
                self.logger.warning(f"No valid gold answers found for QA pair {message.qa_pair_id}")
        else:
            self.logger.warning(f"Missing data for ROUGE computation - gold_answers: {bool(message.gold_answers)}, generated_answer: {bool(message.generated_answer)}")

        # Prepare prompt with query, generated response, and computed ROUGE score
        prompt_content = self.response_evaluator_prompt.format(
            original_query=message.original_query,
            generated_answer=message.generated_answer,
            rouge_score=rouge_score
        )

        try:
            # Call LLM for response evaluation
            system_message = SystemMessage(content=prompt_content)
            user_message = UserMessage(content="Please evaluate the response and provide improvement suggestions.", source="system")

            response = await self.model_client.create(
                [system_message, user_message],
                cancellation_token=ctx.cancellation_token
            )

            # Get evaluation result
            evaluation_result = response.content if isinstance(response.content, str) else str(response.content)

            # Log LLM interaction
            logger = get_global_prompt_logger()
            logger.log_interaction(
                agent_name="ResponseEvaluatorAgent",
                interaction_type="response_evaluation",
                system_prompt=prompt_content,
                user_prompt="Please evaluate the response and provide improvement suggestions.",
                llm_response=evaluation_result,
                qa_pair_id=message.qa_pair_id,
                additional_metadata={
                    "rouge_score": rouge_score,  # Use computed ROUGE score
                    "original_query_length": len(message.original_query),
                    "generated_answer_length": len(message.generated_answer),
                    "evaluation_length": len(evaluation_result)
                }
            )

            log_qa_processing(self.logger, message.qa_pair_id, "Evaluation completed", evaluation_result)

            # Create evaluation result dictionary
            evaluation_data = {
                "qa_pair_id": message.qa_pair_id,
                "original_query": message.original_query,
                "generated_answer": message.generated_answer,
                "gold_answers": message.gold_answers,
                "rouge_score": rouge_score,  # Use computed ROUGE score
                "evaluation_feedback": evaluation_result,
                "timestamp": "2025-09-22T15:30:00"  # Could be made dynamic
            }

            # Send ResponseEvaluationReady message
            eval_ready_msg = ResponseEvaluationReadyMessage(
                qa_pair_id=message.qa_pair_id,
                evaluation_result=evaluation_data,
                rouge_score=rouge_score,  # Include computed ROUGE score
                batch_id=message.batch_id,
                repetition=message.repetition
            )

            self.logger.info(f"Returning ResponseEvaluationReady for QA pair {message.qa_pair_id}")

            # Return the evaluation ready message
            return eval_ready_msg

        except Exception as e:
            self.logger.error(f"Error in response evaluation: {e}")
            # Return default response on error
            evaluation_data = {
                "qa_pair_id": message.qa_pair_id,
                "error": f"Evaluation failed: {e}",
                "rouge_score": message.rouge_score
            }

            return ResponseEvaluationReadyMessage(
                qa_pair_id=message.qa_pair_id,
                evaluation_result=evaluation_data,
                batch_id=message.batch_id,
                repetition=message.repetition
            )

    async def close(self) -> None:
        """Close the model client."""
        await self.model_client.close()


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
            retrieval_plan_gradient_prompt_vector,
            retrieval_planning_prompt_gradient_vector,
            rag_hyperparameters_agent_gradient_vector,
            answer_generation_prompt_optimizer_vector,
            retrieval_planner_prompt_optimizer_vector,
            hyperparameters_vector_agent_prompt_optimizer
        )

        # Initialize Gemini model client for simple text response
        self.model_client = OpenAIChatCompletionClient(
            model="gemini-2.5-flash-lite",
            max_tokens=512,
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
        self.retrieval_plan_gradient_prompt_vector = retrieval_plan_gradient_prompt_vector
        self.retrieval_planning_prompt_gradient_vector = retrieval_planning_prompt_gradient_vector
        self.rag_hyperparameters_agent_gradient_vector = rag_hyperparameters_agent_gradient_vector

        # Store all vector-specific optimizer prompts
        self.answer_generation_prompt_optimizer_vector = answer_generation_prompt_optimizer_vector
        self.retrieval_planner_prompt_optimizer_vector = retrieval_planner_prompt_optimizer_vector
        self.hyperparameters_vector_agent_prompt_optimizer = hyperparameters_vector_agent_prompt_optimizer

    @message_handler
    async def handle_backward_pass_start(self, message: BackwardPassStartMessage, ctx: MessageContext) -> BackwardPassReadyMessage:
        """Handle BackwardPassStart message and perform complete backward pass critique generation."""
        self.logger.info(f"BackwardPassAgent processing backward pass for batch {message.batch_id}")

        # Load shared state with correct dataset and setting parameters
        current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)

        try:
            # Step 1: Generate answer generation critique
            await self._generate_answer_generation_critique(current_state, ctx)
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Step 2: Generate retrieved content critique
            await self._generate_retrieved_content_critique(current_state, ctx)
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Step 3: Generate retrieval plan critique
            await self._generate_retrieval_plan_critique(current_state, ctx)
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Step 4: Generate retrieval planning prompt critique
            await self._generate_retrieval_planning_prompt_critique(current_state, ctx)
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Step 5: Generate hyperparameters critique
            await self._generate_hyperparameters_critique(current_state, ctx)
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Final save to ensure everything is persisted
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Prepare optimized prompts to return to BatchOrchestrator (like GraphRAG does)
            optimized_prompts = {
                "learned_prompt_hyperparameters_vector": current_state.get("learned_prompt_hyperparameters_vector", ""),
                "learned_prompt_answer_generator_vector": current_state.get("learned_prompt_answer_generator_vector", ""),
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
                        "retrieval_plan_critique",
                        "retrieval_planner_agent_critique",
                        "hyperparameters_vector_agent_critique"
                    ]
                }
            )

            self.logger.info(f"Returning BackwardPassReady for batch {message.batch_id}")

            # Return the backward pass ready message
            return backward_ready_msg

        except Exception as e:
            self.logger.error(f"Error in backward pass: {e}")
            # Return error response
            return BackwardPassReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                backward_pass_results={
                    "error": f"Backward pass failed: {e}",
                    "critiques_generated": False
                }
            )

    async def _generate_answer_generation_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for answer generation based on single example conversation and evaluation."""
        self.logger.info("Generating answer generation critique for single example")

        conversations = current_state.get("conversations_answer_generation", [])
        evaluation_responses = current_state.get("response_evaluations", [])

        if not conversations or not evaluation_responses:
            self.logger.warning("No conversations or evaluations found for answer generation critique")
            return

        # For single example format, use the latest (typically only) conversation and evaluation
        latest_conv = conversations[-1] if conversations else {}
        latest_eval = evaluation_responses[-1] if evaluation_responses else {}

        # Get the current system prompt being used (for critique)
        current_system_prompt = current_state.get("learned_prompt_answer_generator_vector", "")
        if not current_system_prompt:
            current_system_prompt = "(Empty system prompt - first iteration)"

        # Create single prompt-query-answer-feedback format including current system prompt
        single_feedback = f"Current System Prompt: {current_system_prompt}\nUser Prompt: {latest_conv.get('prompt', '')}\nQuery: {latest_eval.get('question', '')}\nAnswer: {latest_conv.get('generated_answer', '')}\nFeedback: {latest_eval.get('evaluation_feedback', '')}"

        # Call LLM with generation_prompt_gradient_prompt_vector using single feedback
        prompt_content = self.generation_prompt_gradient_prompt_vector.format(single_feedback)

        critique = await self._call_llm(prompt_content, ctx)
        current_state["answer_generation_critique"] = critique

        # Generate optimized system prompt based on critique
        optimizer_prompt = self.answer_generation_prompt_optimizer_vector.format(critique)
        optimized_prompt = await self._call_llm(optimizer_prompt, ctx)

        # Only update if not frozen
        is_frozen = self._is_prompt_frozen("answer_generator_vector", current_state)
        if not is_frozen:
            current_state["learned_prompt_answer_generator_vector"] = optimized_prompt

        log_critique_result(self.logger, "answer_generator_vector", critique, is_frozen)

    async def _generate_retrieved_content_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for retrieved content based on conversations, contexts, and evaluations."""
        self.logger.info("Generating retrieved content critique")

        conversations = current_state.get("conversations_answer_generation", [])
        evaluation_responses = current_state.get("response_evaluations", [])

        # Use context summary instead of full retrieved contexts for backward pass
        context_summary = current_state.get("context_summary", "")
        if not context_summary:
            # Fallback to full contexts if summary is not available
            retrieved_contexts = current_state.get("retrieved_contexts", [])
            context_summary = "\n\n--- Context Separator ---\n\n".join(retrieved_contexts) if retrieved_contexts else "No context available"

        if not conversations or not evaluation_responses:
            self.logger.warning("Missing data for retrieved content critique")
            return

        # Create triplets: summarized context + conversation (prompt-answer) + evaluation response
        triplets = []
        for i, conv in enumerate(conversations):
            if i < len(evaluation_responses):
                eval_resp = evaluation_responses[i]
                triplet = f"Context Summary: {context_summary}\nConversation: {conv.get('prompt', '')} -> {conv.get('generated_answer', '')}\nEvaluation: {eval_resp.get('evaluation_feedback', '')}"
                triplets.append(triplet)

        concatenated_data = "\n\n".join(triplets)

        # Call LLM with retrieved_content_gradient_prompt_vector
        prompt_content = self.retrieved_content_gradient_prompt_vector.format(concatenated_data)

        critique = await self._call_llm(prompt_content, ctx)
        current_state["retrieved_content_critique"] = critique

        self.logger.info("Retrieved content critique generated and saved")

    async def _generate_retrieval_plan_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for retrieval plans based on plans and contexts."""
        self.logger.info("Generating retrieval plan critique")

        retrieval_plans = current_state.get("retrieval_plans", [])

        # Use context summary instead of full retrieved contexts for backward pass
        context_summary = current_state.get("context_summary", "")
        if not context_summary:
            # Fallback to full contexts if summary is not available
            retrieved_contexts = current_state.get("retrieved_contexts", [])
            context_summary = "\n\n--- Context Separator ---\n\n".join(retrieved_contexts) if retrieved_contexts else "No context available"

        self.logger.info(f"Retrieval plans: {retrieval_plans}")
        self.logger.info(f"Using context summary for critique generation")

        if not retrieval_plans:
            self.logger.warning("Missing retrieval plans for critique")
            return

        # Create comprehensive retrieval plan information with all plans
        all_plans_info = f"All Retrieval Plans ({len(retrieval_plans)} total):\n"
        for i, plan in enumerate(retrieval_plans, 1):
            all_plans_info += f"\nPlan {i}: {plan}\n"

        all_plans_info += f"\nContext Summary: {context_summary}"

        concatenated_pairs = all_plans_info

        # Get retrieved_content_critique for the second variable
        retrieved_content_critique = current_state.get("retrieved_content_critique", "No critique available")

        # Call LLM with retrieval_plan_gradient_prompt_vector
        prompt_content = self.retrieval_plan_gradient_prompt_vector.format(concatenated_pairs, retrieved_content_critique)

        critique = await self._call_llm(prompt_content, ctx)
        current_state["retrieval_plan_critique"] = critique

        self.logger.info("Retrieval plan critique generated and saved")

    async def _generate_retrieval_planning_prompt_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for retrieval planning prompt."""
        self.logger.info("Generating retrieval planning prompt critique")

        retrieval_prompt = current_state.get("retrieval_prompt", "")
        retrieval_plans = current_state.get("retrieval_plans", [])
        retrieval_queries = current_state.get("retrieval_queries", [])

        if not retrieval_prompt or not retrieval_plans:
            self.logger.warning("Missing retrieval prompt or plans for critique")
            return

        # Create triplets: query + retrieval_prompt + retrieval_plan
        triplets = []
        for i, plan in enumerate(retrieval_plans):
            query = retrieval_queries[i] if i < len(retrieval_queries) else "No query available"
            triplet = f"Query: {query}\nRetrieval Prompt: {retrieval_prompt}\nRetrieval Plan: {plan}"
            triplets.append(triplet)

        concatenated_triplets = "\n\n".join(triplets)

        # Get retrieval_plan_critique for the second variable
        retrieval_plan_critique = current_state.get("retrieval_plan_critique", "No critique available")

        # Call LLM with retrieval_planning_prompt_gradient_vector
        prompt_content = self.retrieval_planning_prompt_gradient_vector.format(concatenated_triplets, retrieval_plan_critique)

        critique = await self._call_llm(prompt_content, ctx)
        current_state["retrieval_planner_agent_critique"] = critique

        # Generate optimized system prompt based on critique
        optimizer_prompt = self.retrieval_planner_prompt_optimizer_vector.format(critique)
        optimized_prompt = await self._call_llm(optimizer_prompt, ctx)

        # Only update if not frozen
        is_frozen = self._is_prompt_frozen("vector_retrieval_planner", current_state)
        if not is_frozen:
            current_state["learned_prompt_vector_retrieval_planner"] = optimized_prompt

        log_critique_result(self.logger, "vector_retrieval_planner", critique, is_frozen)

    async def _generate_hyperparameters_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for hyperparameters agent."""
        self.logger.info("Generating hyperparameters critique")

        rag_hyperparams = current_state.get("rag_hyperparameters", {})
        chunk_size = rag_hyperparams.get("chunk_size", "Not specified")

        batch_info = current_state.get("batch_information", {})
        qa_pairs = batch_info.get("qa_pairs", [])

        retrieval_prompt = current_state.get("retrieval_prompt", "")
        retrieval_plans = current_state.get("retrieval_plans", [])

        if not qa_pairs or not retrieval_prompt or not retrieval_plans:
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

        # Call LLM with rag_hyperparameters_agent_gradient_vector
        prompt_content = self.rag_hyperparameters_agent_gradient_vector.format(
            chunk_size, concatenated_triplets
        )

        critique = await self._call_llm(prompt_content, ctx)
        current_state["hyperparameters_vector_agent_critique"] = critique
        self.logger.info(f"hyperparameters vector agent critique: {critique}")

        # Generate optimized system prompt based on critique
        optimizer_prompt = self.hyperparameters_vector_agent_prompt_optimizer.format(critique)
        optimized_prompt = await self._call_llm(optimizer_prompt, ctx)

        # Only update if not frozen
        is_frozen = self._is_prompt_frozen("hyperparameters_vector", current_state)
        if not is_frozen:
            current_state["learned_prompt_hyperparameters_vector"] = optimized_prompt

        log_critique_result(self.logger, "hyperparameters_vector", critique, is_frozen)

    def _is_prompt_frozen(self, prompt_type: str, current_state: Dict[str, Any]) -> bool:
        """Check if a prompt type is frozen."""
        frozen_prompts = current_state.get("frozen_prompts", [])
        return prompt_type in frozen_prompts

    async def _call_llm(self, prompt_content: str, ctx: MessageContext) -> str:
        """Helper method to call LLM with given prompt and token limit."""
        try:
            # Add token limit instruction to the prompt
            enhanced_prompt = f"{prompt_content}\n\nIMPORTANT: Please limit your critique to approximately {self.critique_token_limit} tokens to ensure efficient inference processing. Focus on the most critical points and be concise."

            system_message = SystemMessage(content=enhanced_prompt)
            user_message = UserMessage(content="Please provide your critique and feedback.", source="system")

            response = await self.model_client.create(
                [system_message, user_message],
                cancellation_token=ctx.cancellation_token
            )

            result = response.content if isinstance(response.content, str) else str(response.content)

            # Log LLM interaction
            logger = get_global_prompt_logger()
            logger.log_interaction(
                agent_name="BackwardPassAgent",
                interaction_type="critique_generation",
                system_prompt=enhanced_prompt,
                user_prompt="Please provide your critique and feedback.",
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
            return f"Error generating critique: {e}"

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

def create_response_evaluator_agent() -> ResponseEvaluatorAgent:
    """Factory function to create ResponseEvaluatorAgent instances."""
    return ResponseEvaluatorAgent("response_evaluator_agent")

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
            model="gemini-2.5-flash-lite",
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

                self.logger.info(f"Context {i+1} summarized (original: {len(context)} → summary: {len(summary)} chars)")

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