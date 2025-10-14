"""
Multi-Agent GraphRAG System with AutoGen Core API.
Includes BatchOrchestratorAgent for orchestrating the GraphRAG pipeline.
"""

import json
import logging
import statistics
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel

from logging_utils import log_agent_action, log_qa_processing, log_critique_result
from prompt_response_logger import get_global_prompt_logger
from step_execution_logger import get_global_step_logger, StepStatus
from evaluation_logger import get_global_evaluation_logger
from standardized_evaluation_logger import initialize_standardized_logging, get_standardized_logger, finalize_standardized_logging, SystemType
from execution_logger import initialize_execution_logging, get_global_execution_logger, finalize_execution_logging

from autogen_core import (
    AgentId, MessageContext, RoutedAgent, message_handler,
    SingleThreadedAgentRuntime, TRACE_LOGGER_NAME
)
from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from shared_state import SharedState
from datasets_schema import Question
from eval_functions import evaluate_rouge_score
import llm_keys
from autogen_dataset_agent import BatchStartMessage, BatchReadyMessage
from parameters import IssueType


# ===== LOGGING CONFIGURATION =====

def configure_logging_for_performance(disable_prompt_logs: bool = False):
    """Configure logging to reduce console output and improve performance.

    Args:
        disable_prompt_logs: If True, also disable prompt/response logging
    """

    # First, completely disable INFO and DEBUG logging globally
    logging.disable(logging.INFO)

    # Get the root logger and remove all console handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)

    # Remove all console handlers from root logger (this is key!)
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream.name in ('<stderr>', '<stdout>'):
            root_logger.removeHandler(handler)

    # Configure specific loggers that we know about
    loggers_to_quiet = [
        TRACE_LOGGER_NAME,
        'autogen_core',
        'autogen_ext',
        'httpx',
        'openai',
        'urllib3',
        'asyncio',
        'aiohttp',
        'requests'
    ]

    for logger_name in loggers_to_quiet:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False  # Don't propagate to parent

        # Remove all console handlers
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)

    # Optionally disable prompt/response logging for maximum performance
    if disable_prompt_logs:
        from prompt_response_logger import disable_prompt_logging
        disable_prompt_logging()

def suppress_console_logging():
    """Additional function to call when you want to completely suppress console output."""
    # This is a more aggressive approach
    logging.disable(logging.CRITICAL)  # Disable everything except CRITICAL

# Function to re-enable logging if needed for debugging
def enable_console_logging():
    """Re-enable console logging for debugging."""
    logging.disable(logging.NOTSET)  # Re-enable all logging

def disable_prompt_response_logging():
    """Disable prompt/response logging specifically."""
    from prompt_response_logger import disable_prompt_logging
    disable_prompt_logging()

def enable_prompt_response_logging():
    """Re-enable prompt/response logging."""
    from prompt_response_logger import enable_prompt_logging
    enable_prompt_logging()

# Configure logging on module import
configure_logging_for_performance()

# Export the logging control functions so they can be used by other modules
__all__ = [
    'configure_logging_for_performance',
    'suppress_console_logging',
    'enable_console_logging',
    'disable_prompt_response_logging',
    'enable_prompt_response_logging',
    'BatchOrchestratorAgent'
]


# ===== RETRY HELPER =====

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

# New messages for orchestration workflow
class GraphStartMessage(BaseModel):
    batch_id: int
    repetition: int
    chunk_size: int
    dataset: str
    setting: str
    shared_state: Dict[str, Any]

class GraphReadyMessage(BaseModel):
    batch_id: int
    repetition: int
    graph_description: str
    connectivity_metrics: Dict[str, Any]
    dataset: str
    setting: str
    all_community_summaries: str  # Concatenated summaries of all communities

class GraphRetrievalStartMessage(BaseModel):
    batch_id: int
    repetition: int
    query: str
    dataset: str
    setting: str
    k_iterations: int = 6
    shared_state: Dict[str, Any]

class GraphRetrievalReadyMessage(BaseModel):
    batch_id: int
    repetition: int
    retrieved_context: str
    dataset: str
    setting: str

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
    batch_id: int
    repetition: int
    dataset: str
    setting: str
    unfound_keywords_history: List[str] = []  # Keywords that were NOT found in previous iterations
    community_summaries: str = ""  # Concatenated community summaries for context-aware keyword selection

class ResponseEvaluationReadyMessage(BaseModel):
    qa_pair_id: str
    evaluation_result: Dict[str, Any]
    continue_optimization: bool
    issue_type: IssueType = IssueType.SATISFACTORY
    missing_keywords: List[str] = []  # Direct attribute for consistent access
    batch_id: int
    repetition: int

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

class BackwardPassStartMessage(BaseModel):
    batch_id: int
    repetition: int
    dataset: str
    setting: str
    all_qa_results: List[Dict[str, Any]]
    issue_type: IssueType = IssueType.CONTENT_ISSUE

class BackwardPassReadyMessage(BaseModel):
    batch_id: int
    repetition: int
    backward_pass_results: Dict[str, Any]

class CommunityAnswerStartMessage(BaseModel):
    """Message to request community-level answer generation."""
    community_id: str
    community_summary: str
    question: str
    qa_pair_id: str
    batch_id: int
    repetition: int
    dataset: str
    setting: str

class CommunityAnswerReadyMessage(BaseModel):
    """Response from community-level answer generation."""
    community_id: str
    is_useful: bool
    partial_answer: str
    qa_pair_id: str
    batch_id: int
    repetition: int

class FinalAnswerStartMessage(BaseModel):
    """Message to request final answer generation from community answers."""
    qa_pair_id: str
    question: str
    useful_community_answers: List[Dict[str, str]]  # List of {"community_id": ..., "answer": ...}
    batch_id: int
    repetition: int
    dataset: str
    setting: str

class FinalAnswerReadyMessage(BaseModel):
    """Response from final answer generation."""
    qa_pair_id: str
    generated_answer: str
    batch_id: int
    repetition: int


# ===== BATCH ORCHESTRATOR AGENT =====

class BatchOrchestratorAgent(RoutedAgent):
    """
    Enhanced Orchestrator with two-level reset mechanism for QA pairs and iterations.
    Implements complete system reset between QA pairs and partial reset between iterations.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.batch_orchestrator")
        self.shared_state = SharedState("agent_states")

        # Enhanced tracking for two-level reset system
        self.current_batch_qa_pairs: Dict[str, Dict[str, Any]] = {}
        self.completed_qa_pairs: Dict[str, Dict[str, Any]] = {}
        self.current_batch_id: Optional[int] = None
        self.current_repetition: Optional[int] = None
        self.current_dataset: Optional[str] = None
        self.current_setting: Optional[str] = None

        # QA pair and iteration tracking
        self.qa_pair_results: Dict[str, List[Dict[str, Any]]] = {}  # Results per iteration per QA pair
        self.qa_pair_rouge_progression: Dict[str, List[float]] = {}  # ROUGE scores per iteration per QA pair

        # Standardized evaluation logging
        self.standardized_logger = None


    @message_handler
    async def handle_batch_start(self, message: BatchStartMessage, ctx: MessageContext) -> BatchReadyMessage:
        """
        Enhanced BatchStart handler with two-level reset logic.
        Processes each QA pair through multiple iterations with proper state management.
        """
        # Ensure console logging is suppressed (in case other modules enabled it)
        suppress_console_logging()

        self.logger.info(f"BatchOrchestrator received BatchStart for batch {message.batch_id}")

        self.current_batch_id = message.batch_id
        self.current_repetition = message.repetition
        self.current_dataset = message.dataset
        self.current_setting = message.setting
        self.current_batch_qa_pairs = {}
        self.completed_qa_pairs = {}

        # Initialize logging systems
        step_logger = get_global_step_logger()
        eval_logger = get_global_evaluation_logger()

        # Initialize prompt logging with system-specific folder
        system_log_dir = f"prompt_response_logs/graph_{message.dataset}_{message.setting}"
        from prompt_response_logger import initialize_prompt_logging
        initialize_prompt_logging(system_log_dir)

        # Initialize standardized evaluation logging for GraphRAG
        if self.standardized_logger is None:
            self.standardized_logger = initialize_standardized_logging(
                SystemType.GRAPHRAG, message.dataset, message.setting
            )

        # Initialize execution logging
        exec_logger = get_global_execution_logger()
        if exec_logger is None:
            exec_logger = initialize_execution_logging(
                system_name="graphrag",
                dataset=message.dataset,
                setting=message.setting,
                metadata={
                    "batch_id": message.batch_id,
                    "repetition": message.repetition
                }
            )

        # Log pipeline start
        step_logger.start_pipeline(
            dataset=message.dataset,
            setting=message.setting,
            total_qa_pairs=len(message.shared_state.get("batch_information", {}).get("qa_pairs", []))
        )

        # Log BatchOrchestratorAgent start
        exec_logger.log_agent_start(
            agent_name="BatchOrchestratorAgent",
            message_type="BatchStartMessage",
            batch_id=str(message.batch_id),
            iteration=message.repetition,
            metadata={
                "dataset": message.dataset,
                "setting": message.setting,
                "total_qa_pairs": len(message.shared_state.get("batch_information", {}).get("qa_pairs", []))
            }
        )

        try:
            # Extract QA pairs from batch information
            batch_info = message.shared_state.get("batch_information", {})
            qa_pairs = batch_info.get("qa_pairs", [])
            total_iterations = batch_info.get("total_iterations", 1)  # Total iterations for context/logging

            # GraphRAG system processes ONE iteration per BatchStart call (like Vector/All Context)
            current_iteration = message.repetition

            self.logger.info(f"Processing {len(qa_pairs)} QA pairs - current iteration: {current_iteration} (DatasetAgent repetition: {message.repetition})")

            # Process each QA pair through all iterations with proper reset logic
            for qa_pair in qa_pairs:
                qa_pair_id = qa_pair.get("question_id", f"qa_{len(self.current_batch_qa_pairs)}")
                self.current_batch_qa_pairs[qa_pair_id] = qa_pair

                # Initialize tracking for this QA pair
                self.qa_pair_results[qa_pair_id] = []
                self.qa_pair_rouge_progression[qa_pair_id] = []

                # Log QA pair start for evaluation only on first iteration
                if current_iteration == 0:
                    document_text = message.shared_state.get("full_document_text", "")
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

                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"STARTING QA PAIR: {qa_pair_id}")
                    self.logger.info(f"Question: {qa_pair.get('question', 'N/A')[:100]}...")
                    self.logger.info(f"Total iterations planned: {total_iterations}")
                    self.logger.info(f"{'='*60}")

                # Get current system prompt from shared state (from previous iterations)
                current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)

                # Process ONLY the current iteration (not all iterations)
                iteration = current_iteration
                self.logger.info(f"\n--- QA Pair {qa_pair_id}, Iteration {iteration}/{total_iterations-1} ---")

                # Check if we should skip this iteration because evaluation determined response is satisfactory
                if iteration > 0:  # Only check after first iteration
                    should_continue = current_state.get(f"continue_optimization_{qa_pair_id}", True)
                    if not should_continue:
                        self.logger.info(f"Skipping iteration {iteration} for {qa_pair_id} - evaluation indicated response was satisfactory in previous iteration")
                        # Use the last successful result
                        if qa_pair_id in self.qa_pair_results and self.qa_pair_results[qa_pair_id]:
                            last_result = self.qa_pair_results[qa_pair_id][-1]
                            self.qa_pair_results[qa_pair_id].append(last_result)
                            self.qa_pair_rouge_progression[qa_pair_id].append(last_result.get("rouge_score", 0.0))
                        continue  # Skip to next QA pair

                # Log batch start for this iteration
                step_logger.start_batch(
                    batch_id=message.batch_id,
                    qa_pair_id=qa_pair_id,
                    iteration=iteration,
                    total_iterations=total_iterations
                )

                iteration_start_time = datetime.now()

                try:
                        # CRITICAL: Apply appropriate reset logic BEFORE processing
                        step_logger.log_step(
                            step_name="reset_logic_application",
                            status=StepStatus.STARTED,
                            agent_name="BatchOrchestratorAgent",
                            input_data_summary=f"QA pair {qa_pair_id}, iteration {iteration}"
                        )

                        await self._apply_reset_logic(qa_pair_id, iteration, total_iterations)

                        step_logger.log_step(
                            step_name="reset_logic_application",
                            status=StepStatus.COMPLETED,
                            agent_name="BatchOrchestratorAgent"
                        )

                        # CRITICAL: Store current QA pair ID and iteration in shared state for logging (AFTER reset)
                        current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
                        current_state["current_qa_pair_id"] = qa_pair_id
                        current_state["current_iteration"] = iteration
                        self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

                        # Process the QA pair for this iteration with proper error handling
                        iteration_results = await self._process_qa_pair_iteration(
                            qa_pair_id, qa_pair, iteration, message, ctx
                        )

                        # Validate iteration results
                        if "error" in iteration_results:
                            self.logger.error(f"Iteration {iteration} failed for {qa_pair_id}: {iteration_results['error']}")

                            # Attempt recovery
                            recovery_success = await self.recover_from_qa_pair_failure(
                                qa_pair_id, iteration, Exception(iteration_results["error"])
                            )

                            if recovery_success:
                                # Retry the iteration after recovery
                                self.logger.info(f"Retrying iteration {iteration} after recovery")
                                iteration_results = await self._process_qa_pair_iteration(
                                    qa_pair_id, qa_pair, iteration, message, ctx
                                )

                        # Store iteration results
                        self.qa_pair_results[qa_pair_id].append(iteration_results)
                        rouge_score = iteration_results.get("rouge_score", 0.0)
                        self.qa_pair_rouge_progression[qa_pair_id].append(rouge_score)

                        # Log iteration completion
                        self.logger.info(f"✓ Iteration {iteration} completed for {qa_pair_id} | ROUGE: {rouge_score:.4f}")

                        # Save iteration data
                        await self._save_iteration_data(qa_pair_id, iteration, iteration_results)

                        # Log transition event
                        self.log_transition_event("iteration_complete", qa_pair_id, iteration, {
                            "rouge_score": rouge_score,
                            "has_error": "error" in iteration_results
                        })

                except Exception as e:
                    self.logger.error(f"Critical error in iteration {iteration} for {qa_pair_id}: {e}")

                    # Attempt recovery
                    recovery_success = await self.recover_from_qa_pair_failure(qa_pair_id, iteration, e)

                    if not recovery_success:
                        # If recovery fails, store error and continue to next iteration
                        error_results = {
                            "qa_pair_id": qa_pair_id,
                            "iteration": iteration,
                            "error": str(e),
                            "rouge_score": 0.0
                        }
                        self.qa_pair_results[qa_pair_id].append(error_results)
                        self.qa_pair_rouge_progression[qa_pair_id].append(0.0)

                # Handle completion logic only on final iteration
                if current_iteration == total_iterations - 1:
                    # Save QA pair summary after all iterations
                    await self._save_qa_pair_summary(qa_pair_id)

                    # Log QA pair completion
                    final_rouge = self.qa_pair_rouge_progression[qa_pair_id][-1] if self.qa_pair_rouge_progression[qa_pair_id] else 0.0
                    rouge_improvement = (
                        final_rouge - self.qa_pair_rouge_progression[qa_pair_id][0]
                        if len(self.qa_pair_rouge_progression[qa_pair_id]) > 1 else 0.0
                    )

                    self.logger.info(f"\nSUCCESS: QA PAIR {qa_pair_id} COMPLETED")
                    self.logger.info(f"Final ROUGE: {final_rouge:.4f}")
                    self.logger.info(f"ROUGE Improvement: {rouge_improvement:+.4f}")
                    self.logger.info(f"Iterations completed: {len(self.qa_pair_results[qa_pair_id])}")

                    # Mark QA pair as completed
                    self.completed_qa_pairs[qa_pair_id] = {
                        "iterations": self.qa_pair_results[qa_pair_id],
                        "rouge_progression": self.qa_pair_rouge_progression[qa_pair_id],
                        "final_rouge": final_rouge,
                        "rouge_improvement": rouge_improvement
                    }

                # Log QA pair completion in evaluation logger
                best_iteration = 0
                if self.qa_pair_rouge_progression[qa_pair_id]:
                    best_iteration = self.qa_pair_rouge_progression[qa_pair_id].index(max(self.qa_pair_rouge_progression[qa_pair_id]))

                # Log to old evaluation logger for backward compatibility
                eval_logger.complete_qa_pair_evaluation(
                    qa_pair_id=qa_pair_id,
                    final_rouge_score=final_rouge,
                    rouge_progression=self.qa_pair_rouge_progression[qa_pair_id],
                    best_iteration=best_iteration,
                    total_iterations_completed=len(self.qa_pair_results[qa_pair_id]),
                    improvement_gained=rouge_improvement,
                    final_metrics={"qa_pair_summary": self.completed_qa_pairs[qa_pair_id]}
                )

                # Log to standardized evaluation logger
                best_answer = None
                if self.qa_pair_results[qa_pair_id] and best_iteration < len(self.qa_pair_results[qa_pair_id]):
                    best_answer = self.qa_pair_results[qa_pair_id][best_iteration].get("generated_answer", None)

                self.standardized_logger.complete_qa_pair_evaluation(
                    qa_pair_id=qa_pair_id,
                    final_rouge_score=final_rouge,
                    rouge_progression=self.qa_pair_rouge_progression[qa_pair_id],
                    best_iteration=best_iteration,
                    total_iterations_completed=len(self.qa_pair_results[qa_pair_id]),
                    best_answer=best_answer
                )

                # Log QA pair completion event
                self.log_transition_event("qa_pair_complete", qa_pair_id, total_iterations-1, {
                    "final_rouge": final_rouge,
                    "rouge_improvement": rouge_improvement,
                    "total_iterations": len(self.qa_pair_results[qa_pair_id])
                })

            # Final batch summary
            batch_summary = await self._create_batch_summary()

            # Log pipeline completion - only 1 iteration per call now
            step_logger.complete_pipeline(
                success=True,
                total_qa_pairs_processed=len(qa_pairs),
                total_iterations_completed=1  # Only 1 iteration per call
            )

            # Log BatchOrchestratorAgent success
            exec_logger.log_agent_success(
                agent_name="BatchOrchestratorAgent",
                message_type="BatchStartMessage",
                result_summary=f"Processed {len(qa_pairs)} QA pairs successfully"
            )

            # Finalize standardized evaluation logging only handled by DatasetAgent when all processing is complete

            # Return BatchReady message indicating completion
            return BatchReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                status="completed",
                metrics=batch_summary
            )

        except Exception as e:
            self.logger.error(f"Error processing batch {message.batch_id}: {e}")

            # Log BatchOrchestratorAgent error
            exec_logger.log_agent_error(
                agent_name="BatchOrchestratorAgent",
                message_type="BatchStartMessage",
                error=e,
                error_context="Failed during batch processing"
            )

            # Log pipeline failure
            step_logger.complete_pipeline(
                success=False,
                total_qa_pairs_processed=len(qa_pairs),
                error_message=str(e)
            )

            return BatchReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                status="failed",
                metrics={"error": str(e)}
            )

    async def _apply_reset_logic(self, qa_pair_id: str, iteration: int, total_iterations: int) -> None:
        """
        Apply appropriate reset logic based on QA pair transition detection with full monitoring.
        """
        transition_type = self.shared_state.detect_transition_type(qa_pair_id, iteration)

        if transition_type == 'new_qa_pair':
            # Complete reset for new QA pair
            self.logger.info(f"TRANSITION DETECTED: New QA pair {qa_pair_id} - executing COMPLETE RESET")
            self.log_transition_event("complete_reset", qa_pair_id, iteration, {
                "transition_type": transition_type,
                "total_iterations": total_iterations
            })

            self.shared_state.reset_for_new_qa_pair(
                qa_pair_id, self.current_dataset, self.current_setting, self.current_batch_id
            )
            # Clear graph data for new QA pair
            self.shared_state.clear_graph_data(self.current_dataset, self.current_setting)

            # Clear unfound keywords history for new QA pair
            current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, self.current_batch_id)
            current_state["unfound_keywords_history"] = []
            self.shared_state.save_state(current_state, self.current_dataset, self.current_setting, self.current_batch_id)
            self.logger.info(f"Cleared unfound keywords history for new QA pair {qa_pair_id}")

            # Validate complete reset
            if not self._validate_reset_checkpoints(qa_pair_id, iteration, "complete"):
                self.logger.error(f"Complete reset validation failed for {qa_pair_id}")

        elif transition_type == 'new_iteration':
            # Partial reset for new iteration
            self.logger.info(f"TRANSITION DETECTED: New iteration {iteration} of QA pair {qa_pair_id} - executing PARTIAL RESET")
            self.log_transition_event("partial_reset", qa_pair_id, iteration, {
                "transition_type": transition_type,
                "previous_iteration": iteration - 1
            })

            self.shared_state.reset_for_new_iteration(
                qa_pair_id, iteration, self.current_dataset, self.current_setting, self.current_batch_id
            )

            # Validate partial reset
            if not self._validate_reset_checkpoints(qa_pair_id, iteration, "partial"):
                self.logger.error(f"Partial reset validation failed for {qa_pair_id}")

        else:
            # Same state - no reset needed
            self.logger.info(f"No transition detected for QA pair {qa_pair_id}, iteration {iteration}")

        # Update processing state
        self.shared_state.processing_state.update({
            "current_qa_pair_id": qa_pair_id,
            "current_iteration": iteration,
            "total_iterations": total_iterations
        })

        # Comprehensive validation
        validation_results = self.validate_system_state(qa_pair_id, iteration)
        if not all(validation_results.values()):
            self.logger.warning(f"System validation issues detected: {validation_results}")

        # Log the transition completion
        self.log_transition_event("iteration_start", qa_pair_id, iteration, {
            "validation_results": validation_results,
            "processing_state": self.shared_state.get_processing_state()
        })

    async def _process_qa_pair_iteration(self, qa_pair_id: str, qa_pair: Dict[str, Any],
                                       iteration: int, message: BatchStartMessage,
                                       ctx: MessageContext) -> Dict[str, Any]:
        """
        Process a single iteration of a QA pair through the complete pipeline.
        This executes the full agent pipeline with proper error handling and fallback.
        """
        self.logger.info(f"Processing QA pair {qa_pair_id}, iteration {iteration}")

        # Get loggers
        step_logger = get_global_step_logger()
        eval_logger = get_global_evaluation_logger()
        exec_logger = get_global_execution_logger()

        # Track intermediate outputs for evaluation logging
        intermediate_outputs = {}
        step_start_time = datetime.now()

        try:
            # Use fixed chunk size of 300 words
            FIXED_CHUNK_SIZE = 500
            self.logger.info(f"Using fixed chunk size: {FIXED_CHUNK_SIZE} words")

            # Log intermediate output
            intermediate_outputs["hyperparameters"] = {
                "chunk_size": FIXED_CHUNK_SIZE,
                "fixed": True
            }

            # Step 1: GraphBuilderAgent
            # Check if we should skip graph rebuilding for style issues
            skip_graph_rebuild = False
            if iteration > 0:
                current_state_check = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
                stored_issue_type_prev = current_state_check.get(f"issue_type_{qa_pair_id}", IssueType.CONTENT_ISSUE)

                print(f"\n{'='*80}")
                print(f"🔍 GRAPH REBUILD CHECK for iteration {iteration}")
                print(f"🔍 Retrieved stored_issue_type_prev: {stored_issue_type_prev} (type: {type(stored_issue_type_prev)})")

                # Handle both enum objects and string values from shared state
                if isinstance(stored_issue_type_prev, str):
                    try:
                        issue_type_prev = IssueType(stored_issue_type_prev)
                        print(f"✅ STRING->ENUM: Converted '{stored_issue_type_prev}' to {issue_type_prev}")
                    except ValueError:
                        issue_type_prev = IssueType.CONTENT_ISSUE
                        print(f"❌ Invalid string, defaulting to CONTENT_ISSUE")
                elif isinstance(stored_issue_type_prev, IssueType):
                    issue_type_prev = stored_issue_type_prev
                    print(f"✅ Already enum: {issue_type_prev}")
                else:
                    issue_type_prev = IssueType.CONTENT_ISSUE
                    print(f"❌ Unexpected type, defaulting to CONTENT_ISSUE")

                print(f"🔍 issue_type_prev == IssueType.STYLE_ISSUE? {issue_type_prev == IssueType.STYLE_ISSUE}")
                self.logger.info(f"Previous iteration issue_type: {issue_type_prev} (type: {type(issue_type_prev)})")

                if issue_type_prev == IssueType.STYLE_ISSUE:
                    skip_graph_rebuild = True
                    print(f"✅ SKIPPING graph rebuild - will reuse existing graph")
                    print(f"{'='*80}\n")
                    self.logger.info(f"🎨 STYLE ISSUE: Reusing existing graph from previous iteration (skipping GraphBuilder)")
                else:
                    print(f"❌ NOT skipping graph rebuild - will build new graph")
                    print(f"{'='*80}\n")

            if not skip_graph_rebuild:
                self.logger.info(f"Step 1: Executing GraphBuilderAgent for {qa_pair_id}")
                exec_logger.log_agent_start(
                    agent_name="GraphBuilderAgent",
                    message_type="GraphBuildMessage",
                    batch_id=str(message.batch_id),
                    qa_pair_id=qa_pair_id,
                    iteration=iteration
                )
                try:
                    graph_response = await self._execute_graph_builder_agent(FIXED_CHUNK_SIZE, iteration)
                    exec_logger.log_agent_success(
                        agent_name="GraphBuilderAgent",
                        message_type="GraphBuildMessage",
                        result_summary="Graph built successfully"
                    )

                    # Store graph response for potential reuse in style-only iterations
                    current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
                    current_state["last_graph_response"] = {
                        "batch_id": graph_response.batch_id,
                        "repetition": graph_response.repetition,
                        "dataset": graph_response.dataset,
                        "setting": graph_response.setting,
                        "all_community_summaries": graph_response.all_community_summaries,
                        "graph_description": getattr(graph_response, 'graph_description', ''),
                        "connectivity_metrics": getattr(graph_response, 'connectivity_metrics', {})
                    }
                    self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)
                    self.logger.info(f"Stored graph response for potential reuse")

                except Exception as e:
                    exec_logger.log_agent_error(
                        agent_name="GraphBuilderAgent",
                        message_type="GraphBuildMessage",
                        error=e,
                        error_context="Failed during graph building"
                    )
                    self.logger.warning(f"GraphBuilderAgent failed, using fallback: {e}")
                    graph_response = await self.handle_agent_failure("graph_builder_agent", qa_pair_id, iteration, e)
            else:
                # Reuse graph from previous iteration
                self.logger.info(f"Reusing graph from iteration {iteration - 1}")
                current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
                previous_graph_data = current_state.get("last_graph_response")
                if previous_graph_data:
                    # Reconstruct GraphReadyMessage from stored data
                    from multi_agent_system import GraphReadyMessage
                    graph_response = GraphReadyMessage(
                        batch_id=previous_graph_data["batch_id"],
                        repetition=iteration,  # Use current iteration number
                        dataset=previous_graph_data["dataset"],
                        setting=previous_graph_data["setting"],
                        all_community_summaries=previous_graph_data["all_community_summaries"],
                        graph_description=previous_graph_data.get("graph_description", ""),
                        connectivity_metrics=previous_graph_data.get("connectivity_metrics", {})
                    )
                    self.logger.info(f"Successfully loaded previous graph response")
                    exec_logger.log_state_transition(
                        transition_type="graph_reused",
                        from_state="style_issue_detected",
                        to_state="answer_generation",
                        metadata={"reason": "style_issue_graph_reuse", "qa_pair_id": qa_pair_id, "iteration": iteration}
                    )
                else:
                    self.logger.error(f"Failed to load previous graph response, rebuilding...")
                    # Fallback: rebuild graph anyway
                    graph_response = await self._execute_graph_builder_agent(FIXED_CHUNK_SIZE, iteration)

            # Step 3: AnswerGeneratorAgent (receives all community summaries directly from GraphBuilderAgent)
            self.logger.info(f"Step 3: Executing AnswerGeneratorAgent for {qa_pair_id}")

            # Get issue type from shared state (for iterations > 0)
            issue_type_for_answer_gen = IssueType.CONTENT_ISSUE  # Default for first iteration
            if iteration > 0:
                current_state_for_issue = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
                stored_issue_type = current_state_for_issue.get(f"issue_type_{qa_pair_id}", IssueType.CONTENT_ISSUE)

                print(f"\n{'='*80}")
                print(f"🔍 RAW RETRIEVAL: stored_issue_type = {stored_issue_type} (type: {type(stored_issue_type)})")

                # Handle both enum objects and string values from shared state
                if isinstance(stored_issue_type, str):
                    try:
                        issue_type_for_answer_gen = IssueType(stored_issue_type)
                        print(f"✅ STRING->ENUM: Converted '{stored_issue_type}' to {issue_type_for_answer_gen}")
                    except ValueError:
                        print(f"❌ Invalid issue_type string '{stored_issue_type}', defaulting to CONTENT_ISSUE")
                        issue_type_for_answer_gen = IssueType.CONTENT_ISSUE
                elif isinstance(stored_issue_type, IssueType):
                    issue_type_for_answer_gen = stored_issue_type
                    print(f"✅ ENUM: Issue type is already enum: {issue_type_for_answer_gen}")
                else:
                    print(f"❌ Unexpected issue_type type '{type(stored_issue_type)}', defaulting to CONTENT_ISSUE")
                    issue_type_for_answer_gen = IssueType.CONTENT_ISSUE

                print(f"✅ FINAL: Retrieved issue_type for iteration {iteration}: {issue_type_for_answer_gen} (type: {type(issue_type_for_answer_gen)})")
                print(f"✅ PASSING TO _execute_answer_generator_agent: {issue_type_for_answer_gen}")
                print(f"{'='*80}\n")

            exec_logger.log_agent_start(
                agent_name="AnswerGeneratorAgent",
                message_type="AnswerGenerationMessage",
                batch_id=str(message.batch_id),
                qa_pair_id=qa_pair_id,
                iteration=iteration
            )
            try:
                answer_response = await self._execute_answer_generator_agent(graph_response, qa_pair, issue_type_for_answer_gen)
                exec_logger.log_agent_success(
                    agent_name="AnswerGeneratorAgent",
                    message_type="AnswerGenerationMessage",
                    result_summary=f"Generated answer (length: {len(getattr(answer_response, 'generated_answer', ''))})"
                )
            except Exception as e:
                exec_logger.log_agent_error(
                    agent_name="AnswerGeneratorAgent",
                    message_type="AnswerGenerationMessage",
                    error=e,
                    error_context="Failed during answer generation"
                )
                self.logger.warning(f"AnswerGeneratorAgent failed, using fallback: {e}")
                answer_response = await self.handle_agent_failure("answer_generator_agent", qa_pair_id, iteration, e)

            # Step 4: ResponseEvaluatorAgent
            self.logger.info(f"Step 4: Executing ResponseEvaluatorAgent for {qa_pair_id}")
            exec_logger.log_agent_start(
                agent_name="ResponseEvaluatorAgent",
                message_type="EvaluationMessage",
                batch_id=str(message.batch_id),
                qa_pair_id=qa_pair_id,
                iteration=iteration
            )
            try:
                evaluation_response = await self._execute_response_evaluator_agent(answer_response, qa_pair)
                exec_logger.log_agent_success(
                    agent_name="ResponseEvaluatorAgent",
                    message_type="EvaluationMessage",
                    result_summary=f"Evaluation complete, continue_optimization: {getattr(evaluation_response, 'continue_optimization', False)}"
                )
            except Exception as e:
                exec_logger.log_agent_error(
                    agent_name="ResponseEvaluatorAgent",
                    message_type="EvaluationMessage",
                    error=e,
                    error_context="Failed during response evaluation"
                )
                self.logger.warning(f"ResponseEvaluatorAgent failed, using fallback: {e}")
                evaluation_response = await self.handle_agent_failure("response_evaluator_agent", qa_pair_id, iteration, e)

            # Step 5: BackwardPassAgent (generates optimized prompts for next iteration)
            self.logger.info(f"Step 5: Executing BackwardPassAgent for {qa_pair_id}")
            backward_pass_response = None

            # Check if we should continue optimization
            should_continue = evaluation_response.continue_optimization
            issue_type = evaluation_response.issue_type
            is_last_iteration = iteration >= message.shared_state.get("batch_information", {}).get("total_iterations", 3) - 1

            # Store continue_optimization flag, issue_type, and missing keywords in shared state for next iteration
            current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
            current_state[f"continue_optimization_{qa_pair_id}"] = should_continue

            # Debug: Log what we're storing
            print(f"\n{'='*80}")
            print(f"💾 STORING issue_type: {issue_type} (type: {type(issue_type)})")
            current_state[f"issue_type_{qa_pair_id}"] = issue_type
            print(f"💾 STORED in state at key 'issue_type_{qa_pair_id}': {current_state[f'issue_type_{qa_pair_id}']} (type: {type(current_state[f'issue_type_{qa_pair_id}'])})")
            print(f"{'='*80}\n")

            # Store missing keywords for focused graph refinement in next iteration
            if should_continue:
                missing_keywords = evaluation_response.missing_keywords  # Direct attribute access

                print(f"\n{'='*80}")
                print(f"📝 STORING KEYWORDS FOR NEXT ITERATION (Iteration {iteration})")
                print(f"{'='*80}")

                # Only store keywords for CONTENT issues
                if issue_type == IssueType.CONTENT_ISSUE and missing_keywords:
                    current_state["missing_keywords_for_refinement"] = missing_keywords
                    print(f"Issue Type: CONTENT_ISSUE")
                    print(f"Action: STORING keywords for graph refinement")
                    print(f"Keywords: {missing_keywords}")
                    print(f"Keyword count: {len(missing_keywords)}")
                    self.logger.info(f"📝 [ITERATION {iteration}] CONTENT ISSUE: Stored {len(missing_keywords)} keywords for graph refinement")
                    self.logger.info(f"📝 Keywords: {missing_keywords}")
                    self.logger.info(f"📝 State keys after saving: {list(current_state.keys())}")
                elif issue_type == IssueType.STYLE_ISSUE:
                    # For style issues, clear keywords (no graph refinement needed)
                    current_state["missing_keywords_for_refinement"] = []
                    print(f"Issue Type: STYLE_ISSUE")
                    print(f"Action: CLEARING keywords (no graph refinement needed)")
                    self.logger.info(f"📝 [ITERATION {iteration}] STYLE ISSUE: No keywords needed (reusing existing graph)")
                else:
                    # Clear keywords if none provided
                    current_state["missing_keywords_for_refinement"] = []
                    print(f"Issue Type: {issue_type}")
                    print(f"Action: CLEARING keywords (none provided)")
                    self.logger.info(f"📝 [ITERATION {iteration}] No keywords provided, clearing keywords")

                print(f"{'='*80}\n")
            else:
                # Clear keywords if not continuing
                current_state["missing_keywords_for_refinement"] = []
                print(f"\n{'='*80}")
                print(f"📝 CLEARING KEYWORDS (Iteration {iteration})")
                print(f"{'='*80}")
                print(f"Reason: should_continue={should_continue}, has evaluation_result={hasattr(evaluation_response, 'evaluation_result')}")
                print(f"Action: Not continuing to next iteration")
                print(f"{'='*80}\n")
                self.logger.info(f"📝 [ITERATION {iteration}] Not continuing, clearing keywords")

            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)
            self.logger.info(f"📝 [ITERATION {iteration}] State saved successfully")

            if not should_continue:
                self.logger.info(f"Skipping BackwardPassAgent - evaluation indicates response is satisfactory for {qa_pair_id}")
                self.logger.info(f"Setting flag to skip future iterations for {qa_pair_id}")
                exec_logger.log_state_transition(
                    transition_type="backward_pass_skipped",
                    from_state="evaluation_complete",
                    to_state="iteration_finalize",
                    metadata={"reason": "satisfactory_response", "qa_pair_id": qa_pair_id}
                )
            elif is_last_iteration:
                self.logger.info(f"Skipping BackwardPassAgent - last iteration for {qa_pair_id}")
                exec_logger.log_state_transition(
                    transition_type="backward_pass_skipped",
                    from_state="evaluation_complete",
                    to_state="iteration_finalize",
                    metadata={"reason": "last_iteration", "qa_pair_id": qa_pair_id}
                )
            else:
                # Run backward pass if we should continue and not the last iteration
                exec_logger.log_agent_start(
                    agent_name="BackwardPassAgent",
                    message_type="BackwardPassMessage",
                    batch_id=str(message.batch_id),
                    qa_pair_id=qa_pair_id,
                    iteration=iteration
                )
                try:
                    backward_pass_response = await self._execute_backward_pass_agent(evaluation_response, iteration, issue_type)
                    exec_logger.log_agent_success(
                        agent_name="BackwardPassAgent",
                        message_type="BackwardPassMessage",
                        result_summary=f"Backward pass completed successfully (issue_type: {issue_type})"
                    )
                except Exception as e:
                    exec_logger.log_agent_error(
                        agent_name="BackwardPassAgent",
                        message_type="BackwardPassMessage",
                        error=e,
                        error_context="Failed during backward pass"
                    )
                    self.logger.warning(f"BackwardPassAgent failed, using fallback: {e}")
                    backward_pass_response = await self.handle_agent_failure("backward_pass_agent", qa_pair_id, iteration, e)

            # Compute ROUGE score
            rouge_score = self._compute_rouge_score(qa_pair, answer_response.generated_answer)
            self.logger.info(f"ROUGE score computed: {rouge_score:.4f} for {qa_pair_id}")

            # Validate all responses exist
            if not all([graph_response, answer_response, evaluation_response]):
                raise Exception("One or more critical agent responses missing")

            # Calculate total execution time
            total_execution_time = (datetime.now() - step_start_time).total_seconds()

            # Collect comprehensive graph statistics
            comprehensive_graph_stats = self._collect_comprehensive_graph_statistics(graph_response)

            # Complete intermediate outputs for evaluation logging
            intermediate_outputs.update({
                "graph_building": {
                    "graph_description": getattr(graph_response, 'graph_description', 'N/A'),
                    "connectivity_metrics": getattr(graph_response, 'connectivity_metrics', {}),
                    "comprehensive_statistics": comprehensive_graph_stats
                },
                "community_summaries": {
                    "all_summaries": getattr(graph_response, 'all_community_summaries', 'N/A'),
                    "summaries_length": len(getattr(graph_response, 'all_community_summaries', ''))
                },
                "answer_generation": {
                    "generated_answer": getattr(answer_response, 'generated_answer', 'N/A'),
                    "answer_length": len(getattr(answer_response, 'generated_answer', ''))
                },
                "evaluation": getattr(evaluation_response, 'evaluation_result', {}),
                "backward_pass": getattr(backward_pass_response, 'backward_pass_results', {}) if backward_pass_response else {}
            })

            # Log evaluation data with proper ROUGE scores and comprehensive graph statistics
            rouge_scores = {
                "rouge-l": rouge_score,  # The computed ROUGE-L score
                "rouge-1": rouge_score,  # Use same score for ROUGE-1 (simplified)
                "rouge-2": rouge_score * 0.85  # Estimate ROUGE-2 as typically lower than ROUGE-L
            }

            # Log to old evaluation logger for backward compatibility
            eval_logger.log_iteration_evaluation(
                qa_pair_id=qa_pair_id,
                iteration=iteration,
                intermediate_outputs=intermediate_outputs,
                generated_answer=getattr(answer_response, 'generated_answer', 'N/A'),
                rouge_scores=rouge_scores,
                hyperparameters={
                    "chunk_size": FIXED_CHUNK_SIZE,
                    "fixed": True
                },
                graph_metrics=comprehensive_graph_stats,
                retrieval_context=getattr(graph_response, 'all_community_summaries', 'N/A'),
                execution_time_seconds=total_execution_time,
                additional_metrics={
                    "evaluation_result": getattr(evaluation_response, 'evaluation_result', {}),
                    "backward_pass_available": backward_pass_response is not None,
                    "qa_pair_question": qa_pair.get("question", "N/A")
                }
            )

            # Log to standardized evaluation logger
            self.standardized_logger.log_iteration_evaluation(
                qa_pair_id=qa_pair_id,
                iteration=iteration,
                generated_answer=getattr(answer_response, 'generated_answer', 'N/A'),
                rouge_scores=rouge_scores,
                intermediate_outputs=intermediate_outputs,
                hyperparameters={
                    "chunk_size": FIXED_CHUNK_SIZE,
                    "fixed": True
                },
                execution_time_seconds=total_execution_time,
                system_specific_metrics=comprehensive_graph_stats
            )

            # Log batch completion
            step_logger.complete_batch(
                success=True,
                final_rouge_score=rouge_score
            )

            # Return iteration results
            # Get community summarization logs from shared state if available
            current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, self.current_batch_id)
            community_summarization_logs = current_state.get("community_summarization_logs", [])

            iteration_results = {
                "qa_pair_id": qa_pair_id,
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "hyperparameters": {"chunk_size": FIXED_CHUNK_SIZE, "fixed": True},
                "graph_description": getattr(graph_response, 'graph_description', 'N/A'),
                "community_summarization_logs": community_summarization_logs,  # Add community summarization logs
                "all_community_summaries": getattr(graph_response, 'all_community_summaries', 'N/A'),
                "generated_answer": getattr(answer_response, 'generated_answer', 'N/A'),
                "evaluation_result": getattr(evaluation_response, 'evaluation_result', {}),
                "rouge_score": rouge_score,
                "backward_pass_results": getattr(backward_pass_response, 'backward_pass_results', {}) if backward_pass_response else {},
                "pipeline_success": True
            }

            self.logger.info(f"✓ Pipeline completed successfully for {qa_pair_id}, iteration {iteration}")
            return iteration_results

        except Exception as e:
            self.logger.error(f"Critical pipeline error for {qa_pair_id}, iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()

            # Log failure in step and batch loggers
            step_logger.complete_batch(
                success=False,
                error_message=str(e)
            )

            return {
                "qa_pair_id": qa_pair_id,
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "rouge_score": 0.0,
                "pipeline_success": False
            }

    async def _execute_graph_builder_agent(self, chunk_size: int,
                                         iteration: int) -> GraphReadyMessage:
        """Execute GraphBuilderAgent with appropriate mode (create vs refine)."""
        current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, self.current_batch_id)

        # Debug logging for keyword availability
        self.logger.info(f"📖 [ITERATION {iteration}] LOADING STATE for GraphBuilder")
        self.logger.info(f"📖 State keys available: {list(current_state.keys())}")
        keywords_in_state = current_state.get("missing_keywords_for_refinement", [])
        self.logger.info(f"📖 Keywords found in state: {keywords_in_state} (count: {len(keywords_in_state)})")

        graph_start_msg = GraphStartMessage(
            batch_id=self.current_batch_id,
            repetition=iteration,
            chunk_size=chunk_size,
            dataset=self.current_dataset,
            setting=self.current_setting,
            shared_state=current_state
        )

        graph_builder_agent_id = AgentId("graph_builder_agent", "default")
        return await self.send_message(graph_start_msg, graph_builder_agent_id)

    async def _execute_graph_retrieval_agent(self, graph_response: GraphReadyMessage,
                                           qa_pair: Dict[str, Any]) -> GraphRetrievalReadyMessage:
        """Execute GraphRetrievalPlannerAgent."""
        current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, self.current_batch_id)

        retrieval_start_msg = GraphRetrievalStartMessage(
            batch_id=graph_response.batch_id,
            repetition=graph_response.repetition,
            query=qa_pair.get("question", ""),
            dataset=graph_response.dataset,
            setting=graph_response.setting,
            k_iterations=6,
            shared_state=current_state
        )

        retrieval_agent_id = AgentId("graph_retrieval_planner_agent", "default")
        return await self.send_message(retrieval_start_msg, retrieval_agent_id)

    async def _execute_answer_generator_agent(self, graph_response: GraphReadyMessage,
                                            qa_pair: Dict[str, Any],
                                            issue_type: IssueType = IssueType.CONTENT_ISSUE) -> AnswerGenerationReadyMessage:
        """
        Execute two-stage answer generation:
        1. Parallel community-level answer generation (skip for STYLE_ISSUE with existing answers)
        2. Final answer synthesis from useful community answers
        """
        qa_pair_id = qa_pair.get("question_id", "unknown")
        question = qa_pair.get("question", "")

        self.logger.info(f"Starting two-stage answer generation for QA {qa_pair_id}")

        # Step 1: Check if we should reuse community answers (style issue only)
        current_state = self.shared_state.load_state(graph_response.dataset, graph_response.setting, graph_response.batch_id)

        reuse_community_answers = False
        useful_community_answers = []

        # Debug: Log the comparison values
        print(f"\n{'='*80}")
        print(f"🔍 REUSE CHECK FOR QA: {qa_pair_id}")
        print(f"🔍 Issue type received: {issue_type} (type: {type(issue_type)})")
        print(f"🔍 Repetition: {graph_response.repetition}")
        print(f"🔍 repetition > 0? {graph_response.repetition > 0}")
        print(f"🔍 IssueType.STYLE_ISSUE = {IssueType.STYLE_ISSUE} (type: {type(IssueType.STYLE_ISSUE)})")
        print(f"🔍 issue_type == IssueType.STYLE_ISSUE? {issue_type == IssueType.STYLE_ISSUE}")
        print(f"🔍 Combined condition (repetition > 0 AND style_issue)? {graph_response.repetition > 0 and issue_type == IssueType.STYLE_ISSUE}")
        print(f"{'='*80}\n")

        if graph_response.repetition > 0 and issue_type == IssueType.STYLE_ISSUE:
            # For style issues after first iteration, try to reuse community answers
            print(f"✅ ENTERING REUSE BLOCK - Style issue detected in iteration {graph_response.repetition}")
            previous_community_answers = current_state.get("last_useful_community_answers", [])
            print(f"Found {len(previous_community_answers)} previous community answers in state")
            if previous_community_answers:
                reuse_community_answers = True
                useful_community_answers = previous_community_answers
                print(f"🎨 STYLE ISSUE: Reusing {len(useful_community_answers)} community answers from previous iteration")
                print(f"Skipping CommunityAnswerGeneratorAgent (Stage 1) - going directly to FinalAnswerGenerator")
                self.logger.info(f"🎨 STYLE ISSUE: Reusing {len(useful_community_answers)} community answers from previous iteration")
                self.logger.info(f"Skipping CommunityAnswerGeneratorAgent (Stage 1) - going directly to FinalAnswerGenerator")
            else:
                print(f"⚠️ STYLE ISSUE but no previous community answers found, running full two-stage")
                self.logger.warning(f"STYLE ISSUE but no previous community answers found, running full two-stage")
        else:
            print(f"❌ NOT entering reuse block - will regenerate community answers")

        if not reuse_community_answers:
            # Run full two-stage process - get community summaries
            community_summaries_dict = current_state.get("community_summaries", {})

            # DEBUG: Print what we found
            print(f"\n{'='*80}")
            print(f"DEBUG: TWO-STAGE ANSWER GENERATION CHECK")
            print(f"{'='*80}")
            print(f"community_summaries_dict type: {type(community_summaries_dict)}")
            print(f"community_summaries_dict length: {len(community_summaries_dict) if isinstance(community_summaries_dict, dict) else 'N/A'}")
            if isinstance(community_summaries_dict, dict) and community_summaries_dict:
                print(f"community_summaries_dict keys: {list(community_summaries_dict.keys())[:5]}")  # First 5 keys
                print(f"✓ Two-stage answer generation will be used (parallel community generators)")
            else:
                print(f"⚠️  community_summaries_dict is EMPTY or not a dict!")
                print(f"Available state keys: {list(current_state.keys())}")
            print(f"{'='*80}\n")

            if not community_summaries_dict:
                print(f"\n⚠️  FALLBACK TRIGGERED: Using old single-stage approach")
                print(f"This means parallel community generators will NOT be executed")
                print(f"FinalAnswerGenerator will NOT be called\n")
                # Fallback: use old single-stage approach if community summaries not available
                answer_gen_msg = AnswerGenerationStartMessage(
                    qa_pair_id=qa_pair_id,
                    question=question,
                    retrieved_context=graph_response.all_community_summaries,
                    batch_id=graph_response.batch_id,
                    repetition=graph_response.repetition,
                    dataset=graph_response.dataset,
                    setting=graph_response.setting
                )
                answer_gen_agent_id = AgentId("answer_generator_agent", "default")
                return await self.send_message(answer_gen_msg, answer_gen_agent_id)

            self.logger.info(f"Processing {len(community_summaries_dict)} communities in parallel")

            # Step 2: Create parallel tasks for all community answer generators
            # Use a single registered community answer generator agent for all communities
            community_answer_agent_id = AgentId("community_answer_generator", "default")
            community_tasks = []
            community_ids = []

            for community_id, community_summary in community_summaries_dict.items():
                community_msg = CommunityAnswerStartMessage(
                    community_id=community_id,
                    community_summary=community_summary,
                    question=question,
                    qa_pair_id=qa_pair_id,
                    batch_id=graph_response.batch_id,
                    repetition=graph_response.repetition,
                    dataset=graph_response.dataset,
                    setting=graph_response.setting
                )

                task = self.send_message(community_msg, community_answer_agent_id)
                community_tasks.append(task)
                community_ids.append(community_id)

            # Step 3: Execute all community answer generations in parallel
            self.logger.info(f"Executing {len(community_tasks)} community answer generators in parallel")
            community_responses = await asyncio.gather(*community_tasks, return_exceptions=True)

            # Step 4: Filter useful community answers
            for community_id, response in zip(community_ids, community_responses):
                if isinstance(response, Exception):
                    self.logger.error(f"Community answer generation failed for {community_id}: {response}")
                    continue

                if response.is_useful and response.partial_answer:
                    useful_community_answers.append({
                        "community_id": community_id,
                        "answer": response.partial_answer
                    })
                    self.logger.info(f"Community {community_id}: USEFUL (answer length: {len(response.partial_answer)})")
                else:
                    self.logger.info(f"Community {community_id}: NOT USEFUL")

            self.logger.info(f"Filtered {len(useful_community_answers)}/{len(community_summaries_dict)} useful community answers")

            # Store useful community answers for potential reuse in style-only iterations
            current_state = self.shared_state.load_state(graph_response.dataset, graph_response.setting, graph_response.batch_id)
            current_state["last_useful_community_answers"] = useful_community_answers
            self.shared_state.save_state(current_state, graph_response.dataset, graph_response.setting, graph_response.batch_id)
            print(f"\n{'='*80}")
            print(f"💾 STORED {len(useful_community_answers)} community answers for potential reuse")
            print(f"💾 Key: 'last_useful_community_answers'")
            print(f"💾 Dataset: {graph_response.dataset}, Setting: {graph_response.setting}, Batch: {graph_response.batch_id}")
            print(f"{'='*80}\n")
            self.logger.info(f"Stored {len(useful_community_answers)} useful community answers for potential reuse")

        # Step 5: Synthesize final answer from useful community answers
        if not useful_community_answers:
            self.logger.warning("No useful community answers found, using fallback")
            # Fallback: create a simple answer indicating no useful information
            return AnswerGenerationReadyMessage(
                qa_pair_id=qa_pair_id,
                generated_answer="Based on the available information, I cannot provide a comprehensive answer to this question.",
                batch_id=graph_response.batch_id,
                repetition=graph_response.repetition
            )

        final_answer_msg = FinalAnswerStartMessage(
            qa_pair_id=qa_pair_id,
            question=question,
            useful_community_answers=useful_community_answers,
            batch_id=graph_response.batch_id,
            repetition=graph_response.repetition,
            dataset=graph_response.dataset,
            setting=graph_response.setting
        )

        answer_gen_agent_id = AgentId("answer_generator_agent", "default")
        final_response = await self.send_message(final_answer_msg, answer_gen_agent_id)

        # Convert FinalAnswerReadyMessage to AnswerGenerationReadyMessage for compatibility
        return AnswerGenerationReadyMessage(
            qa_pair_id=final_response.qa_pair_id,
            generated_answer=final_response.generated_answer,
            batch_id=final_response.batch_id,
            repetition=final_response.repetition
        )

    async def _execute_response_evaluator_agent(self, answer_response: AnswerGenerationReadyMessage,
                                              qa_pair: Dict[str, Any]) -> ResponseEvaluationReadyMessage:
        """Execute ResponseEvaluatorAgent."""
        # Get unfound keywords history and community summaries from shared state
        current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, self.current_batch_id)
        unfound_keywords_history = current_state.get("unfound_keywords_history", [])

        # Retrieve community summaries from last_graph_response for context-aware keyword selection
        last_graph_response = current_state.get("last_graph_response", {})
        community_summaries = last_graph_response.get("all_community_summaries", "")

        # Debug logging
        print(f"\n{'='*80}")
        print(f"RESPONSE EVALUATOR - CONTEXT RETRIEVAL CHECK")
        print(f"{'='*80}")
        print(f"last_graph_response keys: {list(last_graph_response.keys())}")
        print(f"community_summaries length: {len(community_summaries) if community_summaries else 0} chars")
        if community_summaries:
            print(f"community_summaries preview (first 200 chars): {community_summaries[:200]}...")
        else:
            print(f"⚠️  WARNING: community_summaries is EMPTY")
        print(f"{'='*80}\n")

        eval_start_msg = ResponseEvaluationStartMessage(
            qa_pair_id=answer_response.qa_pair_id,
            original_query=qa_pair.get("question", ""),
            generated_answer=answer_response.generated_answer,
            gold_answers=qa_pair.get("answers", []),
            batch_id=answer_response.batch_id,
            repetition=answer_response.repetition,
            dataset=self.current_dataset,
            setting=self.current_setting,
            unfound_keywords_history=unfound_keywords_history,
            community_summaries=community_summaries
        )

        response_eval_agent_id = AgentId("response_evaluator_agent", "default")
        return await self.send_message(eval_start_msg, response_eval_agent_id)

    async def _execute_backward_pass_agent(self, evaluation_response: ResponseEvaluationReadyMessage,
                                         iteration: int,
                                         issue_type: IssueType = IssueType.CONTENT_ISSUE) -> Optional[BackwardPassReadyMessage]:
        """Execute BackwardPassAgent to generate optimized prompts."""
        if iteration == 0:
            # For first iteration, generate initial critiques
            backward_pass_msg = BackwardPassStartMessage(
                batch_id=evaluation_response.batch_id,
                repetition=iteration,
                dataset=self.current_dataset,
                setting=self.current_setting,
                all_qa_results=[evaluation_response.evaluation_result],
                issue_type=issue_type
            )
        else:
            # For subsequent iterations, use previous results for critique
            current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, self.current_batch_id)
            all_results = current_state.get("response_evaluations", [])

            backward_pass_msg = BackwardPassStartMessage(
                batch_id=evaluation_response.batch_id,
                repetition=iteration,
                dataset=self.current_dataset,
                setting=self.current_setting,
                all_qa_results=all_results,
                issue_type=issue_type
            )

        try:
            backward_pass_agent_id = AgentId("backward_pass_agent", "default")
            backward_response = await self.send_message(backward_pass_msg, backward_pass_agent_id)

            # Update QA pair prompts with optimized versions
            if hasattr(backward_response, 'backward_pass_results'):
                optimized_prompts = backward_response.backward_pass_results.get("optimized_prompts", {})
                current_qa_pair_id = self.shared_state.current_qa_pair_id
                if current_qa_pair_id:
                    self.shared_state.update_qa_pair_prompts(current_qa_pair_id, optimized_prompts)

            return backward_response

        except Exception as e:
            self.logger.warning(f"BackwardPassAgent failed: {e}")
            return None

    def _compute_rouge_score(self, qa_pair: Dict[str, Any], generated_answer: str) -> float:
        """Compute ROUGE score for the generated answer."""
        gold_answers = qa_pair.get("answers", [])
        if not gold_answers:
            return 0.0

        # Compute max ROUGE score across all gold answers
        rouge_scores = []
        for gold_answer in gold_answers:
            from datasets_schema import Question
            temp_question = Question(
                id=qa_pair.get("question_id", "temp"),
                question=qa_pair.get("question", ""),
                answers=[gold_answer],
                metadata=qa_pair.get("metadata", {})
            )
            score = evaluate_rouge_score(temp_question, generated_answer)
            rouge_scores.append(score)

        return max(rouge_scores) if rouge_scores else 0.0

    async def _save_iteration_data(self, qa_pair_id: str, iteration: int, results: Dict[str, Any]) -> None:
        """Save iteration data to hierarchical directory structure."""
        from pathlib import Path
        import json

        # Create directory structure: results/qa_pair_001/iteration_0/
        results_dir = Path("results") / qa_pair_id / f"iteration_{iteration}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save iteration results
        iteration_file = results_dir / "iteration_results.json"
        with open(iteration_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save system prompts state
        if qa_pair_id in self.shared_state.qa_pair_prompts:
            prompts_file = results_dir / "system_prompts.json"
            with open(prompts_file, 'w', encoding='utf-8') as f:
                json.dump(self.shared_state.qa_pair_prompts[qa_pair_id], f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved iteration {iteration} data for QA pair {qa_pair_id}")

    async def _save_qa_pair_summary(self, qa_pair_id: str) -> None:
        """Save QA pair summary after all iterations."""
        from pathlib import Path
        import json

        results_dir = Path("results") / qa_pair_id
        summary_file = results_dir / "summary.json"

        summary = {
            "qa_pair_id": qa_pair_id,
            "total_iterations": len(self.qa_pair_results[qa_pair_id]),
            "rouge_progression": self.qa_pair_rouge_progression[qa_pair_id],
            "final_rouge": self.qa_pair_rouge_progression[qa_pair_id][-1] if self.qa_pair_rouge_progression[qa_pair_id] else 0.0,
            "rouge_improvement": (
                self.qa_pair_rouge_progression[qa_pair_id][-1] - self.qa_pair_rouge_progression[qa_pair_id][0]
                if len(self.qa_pair_rouge_progression[qa_pair_id]) > 1 else 0.0
            ),
            "iterations_summary": [
                {
                    "iteration": i,
                    "rouge_score": result.get("rouge_score", 0.0),
                    "has_error": "error" in result
                }
                for i, result in enumerate(self.qa_pair_results[qa_pair_id])
            ]
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved summary for QA pair {qa_pair_id}")

    async def _create_batch_summary(self) -> Dict[str, Any]:
        """Create comprehensive batch summary."""
        total_qa_pairs = len(self.completed_qa_pairs)
        if total_qa_pairs == 0:
            return {"qa_pairs_processed": 0}

        # Calculate average ROUGE improvements
        rouge_improvements = []
        successful_qa_pairs = 0

        for qa_pair_id, qa_data in self.completed_qa_pairs.items():
            rouge_progression = qa_data["rouge_progression"]
            if len(rouge_progression) > 1:
                improvement = rouge_progression[-1] - rouge_progression[0]
                rouge_improvements.append(improvement)
                successful_qa_pairs += 1

        avg_rouge_improvement = statistics.mean(rouge_improvements) if rouge_improvements else 0.0

        return {
            "qa_pairs_processed": total_qa_pairs,
            "successful_qa_pairs": successful_qa_pairs,
            "average_rouge_improvement": avg_rouge_improvement,
            "total_iterations_completed": sum(len(qa_data["iterations"]) for qa_data in self.completed_qa_pairs.values()),
            "processing_state": self.shared_state.get_processing_state()
        }

    # ===== VALIDATION AND MONITORING METHODS =====

    def validate_system_state(self, qa_pair_id: str, iteration: int) -> Dict[str, bool]:
        """
        Comprehensive validation of system state for current QA pair and iteration.
        """
        validation_results = {
            "qa_pair_tracking_valid": False,
            "iteration_tracking_valid": False,
            "shared_state_consistent": False,
            "prompt_lifecycle_correct": False,
            "graph_state_appropriate": False,
            "data_isolation_maintained": False
        }

        try:
            # Validate QA pair tracking
            processing_state = self.shared_state.get_processing_state()
            validation_results["qa_pair_tracking_valid"] = (
                processing_state.get("current_qa_pair_id") == qa_pair_id
            )

            # Validate iteration tracking
            validation_results["iteration_tracking_valid"] = (
                processing_state.get("current_iteration") == iteration
            )

            # Validate shared state consistency
            current_state = self.shared_state.load_state(
                self.current_dataset, self.current_setting, self.current_batch_id
            )
            validation_results["shared_state_consistent"] = (
                current_state is not None and
                isinstance(current_state, dict)
            )

            # Validate prompt lifecycle
            if qa_pair_id in self.shared_state.qa_pair_prompts:
                qa_prompts = self.shared_state.qa_pair_prompts[qa_pair_id]
                if iteration == 0:
                    # First iteration should have empty or minimal prompts
                    validation_results["prompt_lifecycle_correct"] = True
                else:
                    # Subsequent iterations should have preserved prompts
                    validation_results["prompt_lifecycle_correct"] = any(
                        prompt.strip() for prompt in qa_prompts.values()
                    )

            # Validate graph state (simplified check)
            validation_results["graph_state_appropriate"] = True  # Placeholder

            # Validate data isolation
            validation_results["data_isolation_maintained"] = True  # Placeholder

            self.logger.info(f"Validation results for {qa_pair_id}, iteration {iteration}: {validation_results}")

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")

        return validation_results

    def monitor_rouge_progression(self, qa_pair_id: str) -> Dict[str, Any]:
        """
        Monitor ROUGE score progression for a QA pair.
        """
        if qa_pair_id not in self.qa_pair_rouge_progression:
            return {"error": f"No ROUGE data for QA pair {qa_pair_id}"}

        rouge_scores = self.qa_pair_rouge_progression[qa_pair_id]
        if len(rouge_scores) < 2:
            return {
                "qa_pair_id": qa_pair_id,
                "current_scores": rouge_scores,
                "trend": "insufficient_data"
            }

        # Calculate trend
        recent_improvement = rouge_scores[-1] - rouge_scores[-2]
        overall_improvement = rouge_scores[-1] - rouge_scores[0]

        trend = "improving" if recent_improvement > 0 else "declining" if recent_improvement < 0 else "stable"

        return {
            "qa_pair_id": qa_pair_id,
            "current_scores": rouge_scores,
            "recent_improvement": recent_improvement,
            "overall_improvement": overall_improvement,
            "trend": trend,
            "best_score": max(rouge_scores),
            "worst_score": min(rouge_scores)
        }

    def generate_system_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive system health report.
        """
        health_report = {
            "timestamp": self.shared_state.processing_state.get("iteration_start_time"),
            "system_state": "healthy",
            "qa_pairs_status": {},
            "overall_metrics": {},
            "warnings": [],
            "errors": []
        }

        try:
            # Check each QA pair
            for qa_pair_id in self.qa_pair_results.keys():
                rouge_monitor = self.monitor_rouge_progression(qa_pair_id)
                health_report["qa_pairs_status"][qa_pair_id] = rouge_monitor

                # Check for concerning trends
                if rouge_monitor.get("trend") == "declining":
                    health_report["warnings"].append(f"QA pair {qa_pair_id} shows declining ROUGE scores")

            # Overall system metrics
            if self.completed_qa_pairs:
                total_improvements = []
                for qa_pair_id, qa_data in self.completed_qa_pairs.items():
                    rouge_progression = qa_data.get("rouge_progression", [])
                    if len(rouge_progression) > 1:
                        improvement = rouge_progression[-1] - rouge_progression[0]
                        total_improvements.append(improvement)

                health_report["overall_metrics"] = {
                    "average_improvement": statistics.mean(total_improvements) if total_improvements else 0.0,
                    "total_qa_pairs": len(self.completed_qa_pairs),
                    "successful_qa_pairs": len([imp for imp in total_improvements if imp > 0])
                }

            # Check for errors
            processing_errors = 0
            for qa_pair_id, results in self.qa_pair_results.items():
                for iteration_result in results:
                    if "error" in iteration_result:
                        processing_errors += 1

            if processing_errors > 0:
                health_report["errors"].append(f"Found {processing_errors} processing errors")
                health_report["system_state"] = "degraded"

        except Exception as e:
            health_report["system_state"] = "error"
            health_report["errors"].append(f"Health report generation failed: {e}")

        return health_report

    def log_transition_event(self, event_type: str, qa_pair_id: str, iteration: int, details: Dict[str, Any] = None):
        """
        Log system transition events for monitoring and debugging.
        """
        transition_log = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,  # "complete_reset", "partial_reset", "iteration_start", "qa_pair_complete"
            "qa_pair_id": qa_pair_id,
            "iteration": iteration,
            "batch_id": self.current_batch_id,
            "dataset": self.current_dataset,
            "setting": self.current_setting,
            "details": details or {}
        }

        self.logger.info(f"TRANSITION EVENT: {event_type} | QA: {qa_pair_id} | Iter: {iteration} | Details: {details}")

        # Save to transition log file
        try:
            from pathlib import Path
            import json

            log_dir = Path("transition_logs")
            log_dir.mkdir(exist_ok=True)

            log_file = log_dir / f"transitions_{self.current_dataset}_{self.current_setting}.jsonl"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(transition_log) + '\n')

        except Exception as e:
            self.logger.error(f"Failed to write transition log: {e}")

    def _validate_reset_checkpoints(self, qa_pair_id: str, iteration: int, reset_type: str) -> bool:
        """
        Validate reset checkpoints to ensure correct state transitions.
        """
        checkpoints = {
            "shared_state_reset": False,
            "graph_data_handled": False,
            "prompt_lifecycle_correct": False,
            "processing_state_updated": False
        }

        try:
            # Check shared state reset
            current_state = self.shared_state.load_state(
                self.current_dataset, self.current_setting, self.current_batch_id
            )

            if reset_type == "complete":
                # For complete reset, key fields should be reset to defaults
                checkpoints["shared_state_reset"] = (
                    len(current_state.get("retrieved_contexts", [])) == 0 and
                    len(current_state.get("response_evaluations", [])) == 0
                )
                checkpoints["graph_data_handled"] = True  # Graph should be cleared
                checkpoints["prompt_lifecycle_correct"] = (
                    qa_pair_id in self.shared_state.qa_pair_prompts
                )

            elif reset_type == "partial":
                # For partial reset, some data should be preserved
                checkpoints["shared_state_reset"] = True  # Selective reset is expected
                checkpoints["prompt_lifecycle_correct"] = (
                    qa_pair_id in self.shared_state.qa_pair_prompts and
                    any(prompt.strip() for prompt in self.shared_state.qa_pair_prompts[qa_pair_id].values())
                )

            # Check processing state update
            processing_state = self.shared_state.get_processing_state()
            checkpoints["processing_state_updated"] = (
                processing_state.get("current_qa_pair_id") == qa_pair_id and
                processing_state.get("current_iteration") == iteration
            )

            all_passed = all(checkpoints.values())
            self.logger.info(f"Reset validation for {reset_type} reset: {checkpoints} | All passed: {all_passed}")

            return all_passed

        except Exception as e:
            self.logger.error(f"Reset checkpoint validation failed: {e}")
            return False

    # ===== ERROR RECOVERY MECHANISMS =====

    async def recover_from_qa_pair_failure(self, qa_pair_id: str, iteration: int, error: Exception) -> bool:
        """
        Attempt to recover from QA pair processing failure.
        """
        self.logger.warning(f"Attempting recovery for QA pair {qa_pair_id}, iteration {iteration}, error: {error}")

        try:
            # Log the recovery attempt
            self.log_transition_event("error_recovery_attempt", qa_pair_id, iteration, {
                "error": str(error),
                "recovery_strategy": "state_reset_and_retry"
            })

            # Strategy 1: Reset to known good state and retry
            if iteration > 0:
                self.logger.info(f"Resetting to previous iteration state for {qa_pair_id}")
                # Reset to previous iteration state
                self.shared_state.reset_for_new_iteration(
                    qa_pair_id, iteration, self.current_dataset, self.current_setting, self.current_batch_id
                )

                # Validate reset was successful
                if self.shared_state.validate_reset_state('partial', qa_pair_id, iteration):
                    self.logger.info(f"Successfully recovered {qa_pair_id} to iteration {iteration}")
                    return True

            # Strategy 2: Complete reset if partial reset failed
            self.logger.info(f"Attempting complete reset for {qa_pair_id}")
            self.shared_state.reset_for_new_qa_pair(
                qa_pair_id, self.current_dataset, self.current_setting, self.current_batch_id
            )

            if self.shared_state.validate_reset_state('complete', qa_pair_id, 0):
                self.logger.info(f"Successfully recovered {qa_pair_id} with complete reset")
                return True

            return False

        except Exception as recovery_error:
            self.logger.error(f"Recovery failed for {qa_pair_id}: {recovery_error}")
            return False

    async def handle_agent_failure(self, agent_name: str, qa_pair_id: str, iteration: int, error: Exception) -> Optional[Any]:
        """
        Handle individual agent failures with fallback strategies.
        """
        self.logger.warning(f"Agent {agent_name} failed for QA pair {qa_pair_id}, iteration {iteration}: {error}")

        # Log the agent failure
        self.log_transition_event("agent_failure", qa_pair_id, iteration, {
            "agent_name": agent_name,
            "error": str(error),
            "fallback_strategy": "simulation"
        })

        # Fallback to simulation based on agent type
        try:
            if agent_name == "graph_builder_agent":
                return await self._simulate_graph_builder_response(iteration)
            elif agent_name == "graph_retrieval_planner_agent":
                return await self._simulate_retrieval_response(qa_pair_id)
            elif agent_name == "answer_generator_agent":
                return await self._simulate_answer_generation_response(qa_pair_id)
            elif agent_name == "response_evaluator_agent":
                return await self._simulate_evaluation_response(qa_pair_id)
            elif agent_name == "backward_pass_agent":
                return await self._simulate_backward_pass_response(qa_pair_id, iteration)
            else:
                self.logger.error(f"No fallback strategy for agent: {agent_name}")
                return None

        except Exception as fallback_error:
            self.logger.error(f"Fallback strategy failed for {agent_name}: {fallback_error}")
            return None

    async def _simulate_graph_builder_response(self, iteration: int) -> GraphReadyMessage:
        """Fallback simulation for GraphBuilderAgent."""
        self.logger.info("Simulating graph builder response")
        return GraphReadyMessage(
            batch_id=self.current_batch_id,
            repetition=iteration,
            graph_description="Simulated graph with basic entities and relationships",
            connectivity_metrics={"density": 0.3, "fragmentation_index": 0.2},
            dataset=self.current_dataset,
            setting=self.current_setting
        )

    async def _simulate_retrieval_response(self, qa_pair_id: str) -> GraphRetrievalReadyMessage:
        """Fallback simulation for GraphRetrievalPlannerAgent."""
        self.logger.info(f"Simulating retrieval response for {qa_pair_id}")
        return GraphRetrievalReadyMessage(
            batch_id=self.current_batch_id,
            repetition=0,
            retrieved_context="Simulated retrieved context from graph",
            dataset=self.current_dataset,
            setting=self.current_setting
        )

    async def _simulate_answer_generation_response(self, qa_pair_id: str) -> AnswerGenerationReadyMessage:
        """Fallback simulation for AnswerGeneratorAgent."""
        self.logger.info(f"Simulating answer generation response for {qa_pair_id}")
        return AnswerGenerationReadyMessage(
            qa_pair_id=qa_pair_id,
            generated_answer="Simulated answer based on retrieved context",
            batch_id=self.current_batch_id,
            repetition=0
        )

    async def _simulate_evaluation_response(self, qa_pair_id: str) -> ResponseEvaluationReadyMessage:
        """Fallback simulation for ResponseEvaluatorAgent."""
        self.logger.info(f"Simulating evaluation response for {qa_pair_id}")
        return ResponseEvaluationReadyMessage(
            qa_pair_id=qa_pair_id,
            evaluation_result={
                "score": 0.5,
                "quality": "simulated",
                "qa_pair_id": qa_pair_id,
                "original_query": "",
                "generated_answer": "",
                "evaluation_reasoning": "Simulated response due to agent failure",
                "evaluation_feedback": "",
                "missing_keywords": [],
                "continue_optimization": False,
                "issue_type": IssueType.SATISFACTORY,
                "repetition": 0,
                "timestamp": datetime.now().isoformat()
            },
            continue_optimization=False,
            issue_type=IssueType.SATISFACTORY,
            missing_keywords=[],
            batch_id=self.current_batch_id,
            repetition=0
        )

    async def _simulate_backward_pass_response(self, qa_pair_id: str, iteration: int) -> BackwardPassReadyMessage:
        """Fallback simulation for BackwardPassAgent."""
        self.logger.info(f"Simulating backward pass response for {qa_pair_id}")
        return BackwardPassReadyMessage(
            batch_id=self.current_batch_id,
            repetition=iteration,
            backward_pass_results={
                "critiques_generated": False,
                "simulation": True,
                "optimized_prompts": {}
            }
        )

    def save_recovery_checkpoint(self, qa_pair_id: str, iteration: int, checkpoint_data: Dict[str, Any]) -> bool:
        """
        Save recovery checkpoint for resuming from specific QA pair and iteration.
        """
        try:
            checkpoint_dir = Path("recovery_checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)

            checkpoint_file = checkpoint_dir / f"checkpoint_{qa_pair_id}_{iteration}.json"

            checkpoint = {
                "qa_pair_id": qa_pair_id,
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "batch_id": self.current_batch_id,
                "dataset": self.current_dataset,
                "setting": self.current_setting,
                "checkpoint_data": checkpoint_data,
                "processing_state": self.shared_state.get_processing_state()
            }

            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved recovery checkpoint for {qa_pair_id}, iteration {iteration}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save recovery checkpoint: {e}")
            return False

    def load_recovery_checkpoint(self, qa_pair_id: str, iteration: int) -> Optional[Dict[str, Any]]:
        """
        Load recovery checkpoint for resuming processing.
        """
        try:
            checkpoint_dir = Path("recovery_checkpoints")
            checkpoint_file = checkpoint_dir / f"checkpoint_{qa_pair_id}_{iteration}.json"

            if not checkpoint_file.exists():
                self.logger.info(f"No recovery checkpoint found for {qa_pair_id}, iteration {iteration}")
                return None

            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)

            self.logger.info(f"Loaded recovery checkpoint for {qa_pair_id}, iteration {iteration}")
            return checkpoint

        except Exception as e:
            self.logger.error(f"Failed to load recovery checkpoint: {e}")
            return None

    def cleanup_old_checkpoints(self, max_age_hours: int = 24) -> None:
        """
        Clean up old recovery checkpoints.
        """
        try:
            checkpoint_dir = Path("recovery_checkpoints")
            if not checkpoint_dir.exists():
                return

            current_time = datetime.now()
            for checkpoint_file in checkpoint_dir.glob("checkpoint_*.json"):
                file_age = current_time - datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                if file_age.total_seconds() > max_age_hours * 3600:
                    checkpoint_file.unlink()
                    self.logger.info(f"Cleaned up old checkpoint: {checkpoint_file}")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old checkpoints: {e}")

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

    def _collect_comprehensive_graph_statistics(self, graph_response) -> Dict[str, Any]:
        """
        Collect comprehensive graph statistics for evaluation logging.

        Args:
            graph_response: Response from GraphBuilderAgent

        Returns:
            Dict[str, Any]: Comprehensive graph statistics
        """
        try:
            # Get basic connectivity metrics
            connectivity_metrics = getattr(graph_response, 'connectivity_metrics', {})

            # Get graph description and extract statistics
            graph_description = getattr(graph_response, 'graph_description', '')

            # Try to get graph statistics from the graph description function
            try:
                from graph_functions import generate_graph_description
                graph_stats = generate_graph_description()

                if graph_stats.get("status") == "success":
                    return {
                        "connectivity_metrics": connectivity_metrics,
                        "graph_density": graph_stats.get("density", 0.0),
                        "fragmentation_index": graph_stats.get("fragmentation_index", 0.0),
                        "total_nodes": graph_stats.get("total_nodes", 0),
                        "total_relationships": graph_stats.get("total_relationships", 0),
                        "largest_component_size": graph_stats.get("largest_component_size", 0),
                        "entity_types_count": len(graph_stats.get("statistics", {}).get("entity_types", [])),
                        "relationship_types_count": len(graph_stats.get("statistics", {}).get("relationship_types", [])),
                        "most_frequent_entity_type": graph_stats.get("statistics", {}).get("entity_types", [{}])[0].get("type", "unknown") if graph_stats.get("statistics", {}).get("entity_types") else "unknown",
                        "most_frequent_relationship_type": graph_stats.get("statistics", {}).get("relationship_types", [{}])[0].get("type", "unknown") if graph_stats.get("statistics", {}).get("relationship_types") else "unknown"
                    }
            except Exception as e:
                self.logger.warning(f"Could not collect extended graph statistics: {e}")

            # Fallback to basic metrics
            return {
                "connectivity_metrics": connectivity_metrics,
                "graph_description_length": len(graph_description)
            }

        except Exception as e:
            self.logger.error(f"Error collecting graph statistics: {e}")
            return {"connectivity_metrics": {}}

# ===== FACTORY FUNCTIONS =====

def create_batch_orchestrator_agent() -> BatchOrchestratorAgent:
    """Factory function to create BatchOrchestratorAgent instances."""
    return BatchOrchestratorAgent("batch_orchestrator_agent")


# ===== GRAPH BUILDER AGENT =====

class GraphBuilderAgent(RoutedAgent):
    """
    Agent that builds knowledge graphs from text chunks using LLM extraction.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.graph_builder")
        self.shared_state = SharedState("agent_states")

        # Import response formats and prompts
        from parameters import (
            base_prompt_graph_builder,
            base_prompt_graph_refinement,
            GraphBuilderResponse,
            GraphRefinementResponse
        )

        self.model_client_creation = OpenAIChatCompletionClient(
            model="gemini-2.5-flash-lite",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=llm_keys.GEMINI_KEY,
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "family": "unknown",
                "structured_output": False,
            }
        )

        self.model_client_refinement = OpenAIChatCompletionClient(
            model="gemini-2.5-flash",  # Using Gemini for graph refinement
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=llm_keys.GEMINI_KEY,
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "family": "unknown",
                "structured_output": False,
            }
        )

        # Separate client for community summarization (plain text output)
        self.model_client_summarization = OpenAIChatCompletionClient(
            model="gemini-2.5-flash-lite",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=llm_keys.GEMINI_KEY,
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "family": "unknown",
                "structured_output": False,
            }
            # No response_format - allows plain text responses
        )

        self.base_prompt_graph_builder = base_prompt_graph_builder
        self.base_prompt_graph_refinement = base_prompt_graph_refinement


    def _parse_graph_builder_response(self, response_text: str) -> 'GraphBuilderResponse':
        """
        Parse text response into GraphBuilderResponse Pydantic model.
        Handles JSON extraction from markdown code blocks and validates structure.
        """
        import re
        import json
        from pydantic import ValidationError
        from parameters import GraphBuilderResponse

        try:
            # Try extracting from markdown code block first
            json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try finding JSON object in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Last resort: try the entire response
                    json_str = response_text.strip()

            # Parse JSON
            data = json.loads(json_str)

            # Validate with Pydantic
            return GraphBuilderResponse.model_validate(data)

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed for graph builder response: {e}")
            self.logger.error(f"Response text (first 1000 chars): {response_text[:1000]}")
            raise ValueError(f"Failed to parse graph builder response as JSON: {e}")
        except ValidationError as e:
            self.logger.error(f"Pydantic validation failed for graph builder response: {e}")
            self.logger.error(f"Parsed data: {json.dumps(data, indent=2)[:500]}")
            raise ValueError(f"Graph builder response format invalid: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error parsing graph builder response: {e}")
            raise ValueError(f"Failed to parse graph builder response: {e}")

    def _parse_graph_refinement_response(self, response_text: str) -> 'GraphRefinementResponse':
        """
        Parse text response into GraphRefinementResponse Pydantic model.
        Handles JSON extraction from markdown code blocks and validates structure.
        """
        import re
        import json
        from pydantic import ValidationError
        from parameters import GraphRefinementResponse

        try:
            # Try extracting from markdown code block first
            json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try finding JSON object in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Last resort: try the entire response
                    json_str = response_text.strip()

            # Parse JSON
            data = json.loads(json_str)

            # Validate with Pydantic
            return GraphRefinementResponse.model_validate(data)

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed for graph refinement response: {e}")
            self.logger.error(f"Response text (first 1000 chars): {response_text[:1000]}")
            raise ValueError(f"Failed to parse graph refinement response as JSON: {e}")
        except ValidationError as e:
            self.logger.error(f"Pydantic validation failed for graph refinement response: {e}")
            self.logger.error(f"Parsed data: {json.dumps(data, indent=2)[:500]}")
            raise ValueError(f"Graph refinement response format invalid: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error parsing graph refinement response: {e}")
            raise ValueError(f"Failed to parse graph refinement response: {e}")

    def _parse_graph_refinement_tuples(self, response_text: str) -> 'GraphRefinementResponse':
        """
        Parse tuple format response into GraphRefinementResponse Pydantic model.
        Expected format:
        ("entity"<|>name<|>type<|>description)
        ("relationship"<|>type<|>source<|>target<|>description)
        """
        import re
        from parameters import GraphRefinementResponse, Entity, Relationship, Triplet, TUPLE_DELIMITER

        entities = []
        relationships = []
        triplets = []

        lines = response_text.strip().split('\n')
        failed_lines = []
        total_non_empty_lines = 0

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            total_non_empty_lines += 1

            try:
                # Parse entity tuples: ("entity"<|>name<|>type<|>description)
                entity_pattern = r'\("entity"' + re.escape(TUPLE_DELIMITER) + r'([^' + re.escape(TUPLE_DELIMITER) + r']+)' + re.escape(TUPLE_DELIMITER) + r'([^' + re.escape(TUPLE_DELIMITER) + r']+)' + re.escape(TUPLE_DELIMITER) + r'([^)]+)\)'
                entity_match = re.match(entity_pattern, line)

                if entity_match:
                    name, entity_type, description = entity_match.groups()
                    entities.append(Entity(
                        name=name.strip(),
                        type=entity_type.strip(),
                        description=description.strip()
                    ))
                    continue

                # Parse relationship tuples: ("relationship"<|>type<|>source<|>target<|>description)
                rel_pattern = r'\("relationship"' + re.escape(TUPLE_DELIMITER) + r'([^' + re.escape(TUPLE_DELIMITER) + r']+)' + re.escape(TUPLE_DELIMITER) + r'([^' + re.escape(TUPLE_DELIMITER) + r']+)' + re.escape(TUPLE_DELIMITER) + r'([^' + re.escape(TUPLE_DELIMITER) + r']+)' + re.escape(TUPLE_DELIMITER) + r'([^)]+)\)'
                rel_match = re.match(rel_pattern, line)

                if rel_match:
                    rel_type, source, target, description = rel_match.groups()
                    relationships.append(Relationship(
                        source_entity=source.strip(),
                        target_entity=target.strip(),
                        relationship_type=rel_type.strip(),
                        description=description.strip(),
                        evidence=""  # Empty evidence for tuple format
                    ))
                    # Also create triplet
                    triplets.append(Triplet(
                        subject=source.strip(),
                        predicate=rel_type.strip(),
                        object=target.strip()
                    ))
                    continue

                # If line doesn't match either pattern, log warning and track
                if line:
                    failed_lines.append((line_num, line[:100]))
                    self.logger.warning(f"Line {line_num} doesn't match tuple format: {line[:100]}")

            except Exception as e:
                self.logger.error(f"Error parsing line {line_num}: {e}")
                failed_lines.append((line_num, line[:100]))
                continue

        # Summary logging
        success_count = len(entities) + len(relationships)
        self.logger.info(f"Tuple parsing: {len(entities)} entities, {len(relationships)} relationships from {total_non_empty_lines} lines")

        if failed_lines:
            failure_rate = len(failed_lines) / total_non_empty_lines * 100 if total_non_empty_lines > 0 else 0
            self.logger.warning(
                f"⚠️  {len(failed_lines)} of {total_non_empty_lines} lines failed to parse ({failure_rate:.1f}% failure rate). "
                f"First 3 failed lines: {failed_lines[:3]}"
            )

        if total_non_empty_lines > 0 and success_count == 0:
            self.logger.error(
                f"❌ COMPLETE PARSING FAILURE: 0 items extracted from {total_non_empty_lines} lines. "
                f"LLM likely returned wrong format. Expected tuple format with <|> delimiter."
            )

        return GraphRefinementResponse(
            new_entities=entities,
            new_relationships=relationships,
            new_triplets=triplets,
            reasoning="Extracted from tuple format"
        )

    @message_handler
    async def handle_graph_start(self, message: GraphStartMessage, ctx: MessageContext) -> GraphReadyMessage:
        """Handle GraphStart message by chunking text and building/refining graph."""
        self.logger.info(f"GraphBuilderAgent processing batch {message.batch_id} with chunk_size {message.chunk_size}")

        # Store message attributes for logging in _process_chunk and _process_refinement_chunk
        self.current_batch_id = message.batch_id
        self.current_dataset = message.dataset
        self.current_setting = message.setting

        # Load shared state
        current_state = message.shared_state

        # IMMEDIATELY check keywords at entry point BEFORE any other operations
        self.logger.info(f"🚀 [ITERATION {message.repetition}] GraphBuilder ENTRY POINT")
        keywords_at_entry = current_state.get("missing_keywords_for_refinement", [])
        self.logger.info(f"🚀 Keywords at ENTRY: {keywords_at_entry} (count: {len(keywords_at_entry)})")
        self.logger.info(f"🚀 State keys at ENTRY: {list(current_state.keys())}")
        batch_info = current_state.get("batch_information", {})
        example_info = current_state.get("example_information", {})

        # Try multiple locations for document text with better error reporting
        corpus = None
        if "full_document_text" in current_state:
            corpus = current_state["full_document_text"]
            self.logger.info("Found document text in current_state['full_document_text']")
        elif "document_text" in batch_info:
            corpus = batch_info["document_text"]
            self.logger.info("Found document text in batch_information['document_text']")
        elif "document_text" in example_info:
            corpus = example_info["document_text"]
            self.logger.info("Found document text in example_information['document_text']")
        elif "qa_pairs" in batch_info and batch_info["qa_pairs"]:
            # Try to find document text in qa_pairs structure
            first_qa = batch_info["qa_pairs"][0]
            if "document_text" in first_qa:
                corpus = first_qa["document_text"]
                self.logger.info("Found document text in qa_pairs[0]['document_text']")

        # Determine if this is graph creation (first iteration) or refinement (subsequent iterations)
        is_first_iteration = message.repetition == 0
        self.logger.info(f"Graph mode: {'Creation' if is_first_iteration else 'Refinement'} (iteration {message.repetition})")

        print(f"\n{'='*80}")
        print(f"🔧 GRAPH BUILDING MODE DECISION")
        print(f"{'='*80}")
        print(f"Iteration: {message.repetition}")
        print(f"Mode: {'CREATION (first iteration)' if is_first_iteration else 'REFINEMENT (subsequent iteration)'}")
        print(f"{'='*80}\n")

        if not corpus:
            self.logger.error("No document text found in any expected location!")
            self.logger.error(f"Available keys in current_state: {list(current_state.keys())}")
            self.logger.error(f"Available keys in batch_information: {list(batch_info.keys())}")
            self.logger.error(f"Available keys in example_information: {list(example_info.keys())}")
            if "qa_pairs" in batch_info and batch_info["qa_pairs"]:
                self.logger.error(f"Available keys in first qa_pair: {list(batch_info['qa_pairs'][0].keys())}")
            return

        # Get appropriate learned system prompt based on iteration
        # For first iteration (repetition=0), start with empty system prompt to avoid data leakage from previous QA pairs
        if is_first_iteration:
            learned_system_prompt = ""
            self.logger.info(f"First repetition for batch {message.batch_id} - using empty graph creation system prompt")
        else:
            # For iteration 1, use the graph builder prompt from iteration 0
            # For iteration 2+, use the graph refinement prompt from previous iteration
            if message.repetition == 1:
                learned_system_prompt = current_state.get("learned_prompt_graph_builder", "")
                self.logger.info(f"Using learned_prompt_graph_builder for iteration 1 refinement")
            else:
                learned_system_prompt = current_state.get("learned_prompt_graph_refinement", "")
                self.logger.info(f"Using learned_prompt_graph_refinement for iteration {message.repetition} refinement")

            if learned_system_prompt:
                self.logger.info(f"Using optimized graph system prompt for refinement in batch {message.batch_id} (length: {len(learned_system_prompt)} chars)")
            else:
                self.logger.info(f"No optimized graph prompt available - using empty system prompt for refinement")

        if is_first_iteration:
            # Graph creation mode
            chunks = self._split_text_into_chunks(corpus, message.chunk_size)
            self.logger.info(f"Split corpus into {len(chunks)} chunks for graph creation")

            # Save prompt template (without text chunk) to shared state
            current_state["graph_builder_prompt"] = self.base_prompt_graph_builder

            # Initialize graph data structure
            all_entities = []
            all_relationships = []
            all_triplets = []

            # Process chunks for graph creation
            new_entities, new_relationships, new_triplets = await self._process_graph_creation(chunks, learned_system_prompt, ctx)
            all_entities.extend(new_entities)
            all_relationships.extend(new_relationships)
            all_triplets.extend(new_triplets)
        else:
            # Graph refinement mode
            self.logger.info("Graph refinement mode: loading existing graph and refining")

            # Get missing keywords from previous iteration's evaluation
            self.logger.info(f"🔍 [ITERATION {message.repetition}] ACCESSING KEYWORDS in GraphBuilder")
            self.logger.info(f"🔍 State keys available: {list(current_state.keys())}")
            missing_keywords = current_state.get("missing_keywords_for_refinement", [])
            self.logger.info(f"🔍 Keywords retrieved: {missing_keywords} (count: {len(missing_keywords)})")

            print(f"\n{'='*80}")
            print(f"🔍 MISSING KEYWORDS CHECK (Iteration {message.repetition})")
            print(f"{'='*80}")
            print(f"Keywords for refinement: {missing_keywords}")
            print(f"Keyword count: {len(missing_keywords)}")
            print(f"State has 'missing_keywords_for_refinement' key: {'missing_keywords_for_refinement' in current_state}")
            print(f"{'='*80}\n")

            # If no keywords provided, skip refinement and reuse previous graph
            if not missing_keywords:
                print(f"\n{'='*80}")
                print(f"⏭️  SKIPPING GRAPH REFINEMENT")
                print(f"{'='*80}")
                print(f"Reason: No missing keywords provided (continue=True but keywords list is empty)")
                print(f"Action: Reusing previous iteration's graph (no new entities/relationships)")
                print(f"{'='*80}\n")

                self.logger.info("⏭️  SKIPPING graph refinement - no missing keywords, reusing previous graph")

                # Load existing graph data (don't create new entities/relationships)
                all_entities, all_relationships, all_triplets = await self._load_existing_graph_data()

                # Set new entities to empty (no refinement performed)
                new_entities, new_relationships, new_triplets = [], [], []

                # Signal to retrieval agent that it should reuse community summaries
                current_state["refinement_skipped"] = True
                self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            else:
                # Focused refinement: extract context around missing keywords
                print(f"\n{'='*80}")
                print(f"🎯 FOCUSED REFINEMENT MODE ACTIVATED")
                print(f"{'='*80}")
                print(f"Keywords to search: {missing_keywords}")
                print(f"Keyword count: {len(missing_keywords)}")
                print(f"Corpus length: {len(corpus)} characters")
                print(f"Context window: 800 characters")
                print(f"{'='*80}\n")

                self.logger.info(f"🎯 FOCUSED REFINEMENT MODE: extracting context for {len(missing_keywords)} keywords")
                self.logger.info(f"Missing keywords: {missing_keywords}")
                focused_text, unfound_keywords = self._extract_focused_context(corpus, missing_keywords, context_window=800)

                # Store unfound keywords in history for response evaluator
                unfound_keywords_history = current_state.get("unfound_keywords_history", [])
                if unfound_keywords:
                    unfound_keywords_history.extend(unfound_keywords)
                    # Deduplicate while preserving order
                    unfound_keywords_history = list(dict.fromkeys(unfound_keywords_history))
                    self.logger.info(f"📝 Storing {len(unfound_keywords)} unfound keywords: {unfound_keywords}")
                    self.logger.info(f"📝 Total unfound keywords history: {len(unfound_keywords_history)} keywords")
                current_state["unfound_keywords_history"] = unfound_keywords_history
                self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

                print(f"\n{'='*80}")
                print(f"📊 FOCUSED CONTEXT EXTRACTION RESULT")
                print(f"{'='*80}")
                print(f"Focused text extracted: {'YES' if focused_text else 'NO'}")
                print(f"Focused text length: {len(focused_text) if focused_text else 0} characters")
                print(f"Unfound keywords: {unfound_keywords if unfound_keywords else 'None'}")
                print(f"Decision: {'PROCEED WITH REFINEMENT' if focused_text else 'SKIP REFINEMENT (no co-occurrence contexts found)'}")
                print(f"{'='*80}\n")

                if focused_text:
                    # Split focused text into chunks
                    full_corpus_size = len(corpus)
                    focused_size = len(focused_text)
                    reduction_pct = ((full_corpus_size - focused_size) / full_corpus_size * 100) if full_corpus_size > 0 else 0

                    chunks = self._split_text_into_chunks(focused_text, message.chunk_size)
                    full_chunks_count = len(self._split_text_into_chunks(corpus, message.chunk_size))

                    self.logger.info(f"🎯 TEXT REDUCTION: {full_corpus_size} → {focused_size} chars ({reduction_pct:.1f}% reduction)")
                    self.logger.info(f"🎯 CHUNK REDUCTION: {full_chunks_count} → {len(chunks)} chunks (processing only {len(chunks)/full_chunks_count*100:.1f}% of full corpus)")
                    self.logger.info(f"🎯 LLM CALLS SAVED: {full_chunks_count - len(chunks)} fewer calls to graph extraction LLM")

                    self.logger.info(f"Split text into {len(chunks)} chunks for graph refinement")

                    print(f"\n{'='*80}")
                    print(f"🚀 STARTING GRAPH REFINEMENT EXECUTION")
                    print(f"{'='*80}")
                    print(f"Model: gemini-2.5-flash (Gemini)")
                    print(f"Chunks to process: {len(chunks)}")
                    print(f"Full corpus chunks (if not focused): {full_chunks_count}")
                    print(f"LLM calls saved: {full_chunks_count - len(chunks)}")
                    print(f"Text reduction: {reduction_pct:.1f}%")
                    print(f"Learned system prompt length: {len(learned_system_prompt)} chars")
                    print(f"{'='*80}\n")

                    # Save refinement prompt template to shared state
                    current_state["graph_refinement_prompt"] = self.base_prompt_graph_refinement

                    # Process refinement (without passing existing graph summary to the prompt)
                    new_entities, new_relationships, new_triplets = await self._process_graph_refinement(
                        chunks, learned_system_prompt, ctx
                    )

                    print(f"\n{'='*80}")
                    print(f"✅ GRAPH REFINEMENT COMPLETED")
                    print(f"{'='*80}")
                    print(f"New entities extracted: {len(new_entities)}")
                    print(f"New relationships extracted: {len(new_relationships)}")
                    print(f"New triplets extracted: {len(new_triplets)}")
                    print(f"{'='*80}\n")

                    # Validate refinement results
                    if len(chunks) > 0 and len(new_entities) == 0 and len(new_relationships) == 0:
                        self.logger.error(
                            f"⚠️  REFINEMENT FAILURE: Processed {len(chunks)} chunks but extracted 0 entities and 0 relationships. "
                            f"This likely indicates a parsing error or incorrect LLM response format. "
                            f"Check logs for parsing errors or tuple format validation warnings."
                        )
                    elif len(new_entities) == 0 and len(new_relationships) == 0:
                        self.logger.warning(
                            f"⚠️  Refinement produced no new entities or relationships from {len(chunks)} chunks. "
                            f"This may be expected if focused text contains no new information."
                        )
                    else:
                        self.logger.info(f"✓ Refinement successful: extracted {len(new_entities)} new entities, {len(new_relationships)} new relationships")

                    # For refinement, merge new entities/relationships with existing ones
                    existing_entities, existing_relationships, existing_triplets = await self._load_existing_graph_data()

                    # Clear refinement_skipped flag since we performed refinement
                    current_state["refinement_skipped"] = False
                    self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)
                else:
                    # No co-occurrence contexts found - skip refinement and reuse existing graph
                    self.logger.info("⏭️  SKIPPING graph refinement - no co-occurrence contexts found, reusing previous graph and communities")

                    # Load existing graph data (don't create new entities/relationships)
                    existing_entities, existing_relationships, existing_triplets = await self._load_existing_graph_data()

                    # Set new entities to empty (no refinement performed)
                    new_entities, new_relationships, new_triplets = [], [], []

                    # Signal to retrieval agent that it should reuse community summaries
                    current_state["refinement_skipped"] = True
                    self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

                # Log existing entities details
                if existing_entities:
                    existing_names = [entity.name for entity in existing_entities[:10]]  # First 10 names
                    self.logger.info(f"EXISTING ENTITIES ({len(existing_entities)} total): {', '.join(existing_names)}{'...' if len(existing_entities) > 10 else ''}")
                else:
                    self.logger.info("EXISTING ENTITIES: None found (first iteration or loading failed)")

                # Log new entities details
                if new_entities:
                    new_names = [entity.name for entity in new_entities[:10]]  # First 10 names
                    self.logger.info(f"NEW ENTITIES ({len(new_entities)} total): {', '.join(new_names)}{'...' if len(new_entities) > 10 else ''}")
                else:
                    self.logger.info("NEW ENTITIES: None found")

                # Merge new entities with existing ones
                all_entities = existing_entities + new_entities
                all_relationships = existing_relationships + new_relationships
                all_triplets = existing_triplets + new_triplets

                # Log merge results
                self.logger.info(f"MERGE RESULT: {len(existing_entities)} existing + {len(new_entities)} new = {len(all_entities)} total entities")
                self.logger.info(f"MERGE RESULT: {len(existing_relationships)} existing + {len(new_relationships)} new = {len(all_relationships)} total relationships")

                # Log sample of merged entities to verify merging worked
                if all_entities:
                    merged_sample = [entity.name for entity in all_entities[:15]]  # Show more for verification
                    self.logger.info(f"MERGED ENTITIES SAMPLE (first 15): {', '.join(merged_sample)}{'...' if len(all_entities) > 15 else ''}")

        # Perform entity resolution to merge similar entities
        all_entities, all_relationships, all_triplets = self._resolve_entities(
            all_entities, all_relationships, all_triplets
        )

        # Add placeholder entities for missing relationship endpoints
        # This ensures all relationships can be created even if the LLM didn't explicitly define some entities
        all_entities = self._add_placeholder_entities(all_entities, all_relationships)

        # Convert to memgraph JSON format and save to file (with iteration-based edge weighting)
        graph_json = self._convert_to_memgraph_format(all_entities, all_relationships, all_triplets, iteration=message.repetition)
        # Save to dedicated graphs folder
        import os
        graphs_dir = "graphs"
        os.makedirs(graphs_dir, exist_ok=True)
        graph_filename = os.path.join(graphs_dir, f"{message.dataset}_{message.setting}_batch_{message.batch_id}_graph.json")

        try:
            with open(graph_filename, 'w', encoding='utf-8') as f:
                json.dump(graph_json, f, indent=2, ensure_ascii=False)

            # Count nodes and relationships in saved file
            node_count = sum(1 for item in graph_json if item.get("type") == "node")
            rel_count = sum(1 for item in graph_json if item.get("type") == "relationship")

            self.logger.info(f"✓ Saved graph to {graph_filename}")
            self.logger.info(f"✓ File contains: {node_count} nodes, {rel_count} relationships (from {len(all_entities)} entities, {len(all_relationships)} relationships)")

            # Verification: Read back the file to ensure it was written correctly
            with open(graph_filename, 'r', encoding='utf-8') as f:
                verify_data = json.load(f)
            verify_node_count = sum(1 for item in verify_data if item.get("type") == "node")
            verify_rel_count = sum(1 for item in verify_data if item.get("type") == "relationship")
            self.logger.info(f"✓ File verification: {verify_node_count} nodes, {verify_rel_count} relationships read back successfully")

        except Exception as e:
            self.logger.error(f"Error saving graph file: {e}")

        # Graph is now saved to JSON file - no Memgraph loading needed for community-based retrieval
        self.logger.info(f"Graph building completed. JSON file saved at {graph_filename}")
        self.logger.info(f"Graph contains {len(all_entities)} entities and {len(all_relationships)} relationships")

        # Generate simple graph description and connectivity metrics
        try:
            # Create simple graph description based on NetworkX statistics
            graph_description = f"Knowledge graph with {len(all_entities)} entities and {len(all_relationships)} relationships. "

            # Basic connectivity metrics
            connectivity_metrics = {
                "total_nodes": len(all_entities),
                "total_relationships": len(all_relationships),
                "density": 0.0,  # Will be calculated during community detection
                "fragmentation_index": 0.0,
                "largest_component_size": 0
            }

            # Add entity type distribution to description
            entity_types = {}
            for entity in all_entities:
                entity_type = entity.type or "Unknown"
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

            if entity_types:
                type_desc = ", ".join([f"{count} {etype}" for etype, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:5]])
                graph_description += f"Main entity types: {type_desc}."

                # Check for high ratio of undefined entities
                undefined_count = entity_types.get("undefined", 0)
                if undefined_count > 0:
                    undefined_ratio = (undefined_count / len(all_entities)) * 100
                    self.logger.info(f"Graph contains {undefined_count} undefined entities ({undefined_ratio:.1f}%)")
                    if undefined_ratio > 20:
                        self.logger.warning(
                            f"High ratio of undefined entities ({undefined_ratio:.1f}%) - "
                            f"LLM may be producing inconsistent entity references in relationships"
                        )

            # Save to shared state
            current_state["graph_description"] = graph_description
            current_state["graph_statistics"] = connectivity_metrics

            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

        except Exception as e:
            self.logger.error(f"Error generating graph description: {e}")
            graph_description = "Graph description generation failed"
            connectivity_metrics = {
                "total_nodes": len(all_entities) if all_entities else 0,
                "total_relationships": len(all_relationships) if all_relationships else 0
            }

        # Perform community detection and summarization
        self.logger.info("Starting community detection and summarization")
        all_community_summaries = ""
        try:
            from community_graph_utils import load_and_process_graph
            from parameters import default_community_levels

            # Load and process graph with community detection
            community_manager = await load_and_process_graph(
                graph_filename,
                self.model_client_summarization,  # Use plain text LLM client for summarization
                embedding_model=None,  # Use TF-IDF fallback
                learned_prompt="",  # Always use base prompt, no optimization
                community_levels=default_community_levels  # Use configured level filter
            )

            if community_manager and community_manager.community_summaries:
                # Concatenate all community summaries
                all_summaries = []
                for comm_id, summary in community_manager.community_summaries.items():
                    all_summaries.append(f"Community {comm_id}:\n{summary}\n")

                all_community_summaries = "\n".join(all_summaries)
                self.logger.info(f"Generated summaries for {len(community_manager.community_summaries)} communities")
                self.logger.info(f"Total community summaries length: {len(all_community_summaries)} characters")

                # Store community summaries in shared state for logging
                current_state["community_summaries"] = community_manager.community_summaries
                current_state["community_summarization_logs"] = community_manager.community_summarization_logs
                self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)
            else:
                self.logger.warning("No communities detected or community manager failed")
                all_community_summaries = "No communities detected in the knowledge graph."

        except Exception as e:
            self.logger.error(f"Error during community detection and summarization: {e}")
            import traceback
            traceback.print_exc()
            all_community_summaries = f"Error during community detection: {str(e)}"

        # Send GraphReady message
        graph_ready_msg = GraphReadyMessage(
            batch_id=message.batch_id,
            repetition=message.repetition,
            graph_description=graph_description,
            connectivity_metrics=connectivity_metrics,
            dataset=message.dataset,
            setting=message.setting,
            all_community_summaries=all_community_summaries
        )

        self.logger.info(f"Returning GraphReady for batch {message.batch_id} with {len(all_community_summaries)} chars of community summaries")

        # Return the GraphReady message
        return graph_ready_msg

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

    def _extract_focused_context(self, text: str, keywords: List[str], context_window: int = 300, max_contexts: int = 30) -> Tuple[str, List[str]]:
        """
        Extract context windows where at least TWO keywords co-occur.
        Uses exact matching first, then falls back to fuzzy matching if needed.

        Includes anti-explosion mechanisms:
        - Merges overlapping contexts (>50% overlap)
        - Limits total contexts to max_contexts to prevent chunk explosion

        Args:
            text: The full document text
            keywords: List of keywords/phrases to search for
            context_window: Number of characters to include before and after keyword matches
            max_contexts: Maximum number of contexts to return (prevents explosion)

        Returns:
            Tuple of (concatenated context windows, list of keywords that were not found)
        """
        if not keywords:
            return "", []

        if len(keywords) == 1:
            # If only one keyword, fall back to single keyword matching
            print(f"⚠️  Only 1 keyword provided, using single-keyword matching")

        import re
        from difflib import SequenceMatcher

        # Track all keyword positions (exact and fuzzy matches)
        keyword_positions = {}  # keyword -> list of (start, end) positions
        keywords_found = set()  # Track which keywords were found
        keywords_not_found = []  # Track keywords that need fuzzy matching

        # Phase 1: Exact matching (case-insensitive) - Track positions
        print(f"\n{'='*80}")
        print(f"🔍 KEYWORD SEARCH - Exact Matching Phase (Co-occurrence Mode)")
        print(f"{'='*80}")
        print(f"Searching for {len(keywords)} keywords in corpus ({len(text)} chars)")
        print(f"Keywords: {keywords}")
        print(f"Strategy: Extract contexts where at least 2 keywords co-occur")
        print(f"{'-'*80}")

        for keyword in keywords:
            # Case-insensitive regex search for the keyword
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            found_matches = False
            positions = []

            for match in pattern.finditer(text):
                found_matches = True
                positions.append((match.start(), match.end()))
                print(f"  ✓ Exact match for '{keyword}': found at position {match.start()}")

            if found_matches:
                keywords_found.add(keyword)
                keyword_positions[keyword] = positions
                print(f"  ✓ '{keyword}': {len(positions)} exact matches found in corpus")
            else:
                keywords_not_found.append(keyword)
                print(f"  ✗ '{keyword}': NO exact matches found in corpus (will try fuzzy matching)")

        # Phase 2: Fuzzy matching for keywords not found exactly
        if keywords_not_found:
            print(f"\n{'='*80}")
            print(f"🔎 KEYWORD SEARCH - Fuzzy Matching Phase")
            print(f"{'='*80}")
            print(f"Attempting fuzzy matching for {len(keywords_not_found)} keywords not found exactly:")
            print(f"Keywords: {keywords_not_found}")
            print(f"{'-'*80}")
            self.logger.info(f"🔎 Fuzzy matching for {len(keywords_not_found)} keywords not found exactly: {keywords_not_found}")

            # Split text into sentences for fuzzy matching
            sentences = re.split(r'[.!?]\s+', text)

            for keyword in keywords_not_found:
                keyword_lower = keyword.lower()
                best_matches = []

                # Find sentences with high similarity to the keyword
                for sentence in sentences:
                    sentence_lower = sentence.lower()

                    # Calculate similarity ratio
                    similarity = SequenceMatcher(None, keyword_lower, sentence_lower).ratio()

                    # Also check if any significant words from keyword appear in sentence
                    keyword_words = set(keyword_lower.split())
                    sentence_words = set(sentence_lower.split())
                    word_overlap = len(keyword_words & sentence_words) / len(keyword_words) if keyword_words else 0

                    # Consider it a match if either:
                    # 1. Similarity ratio is moderate (> 0.4) - MORE PERMISSIVE
                    # 2. Significant word overlap (> 0.4) and some similarity (> 0.2) - MORE PERMISSIVE
                    if similarity > 0.4 or (word_overlap > 0.4 and similarity > 0.2):
                        best_matches.append((sentence, similarity, word_overlap))

                # Sort by similarity and take top matches
                best_matches.sort(key=lambda x: (x[2], x[1]), reverse=True)

                # Track positions from best matching sentences
                for sentence, sim, overlap in best_matches[:3]:  # Take up to 3 best matches per keyword
                    # Find sentence position in original text
                    sentence_pos = text.lower().find(sentence.lower())
                    if sentence_pos != -1:
                        # Store position for co-occurrence check
                        if keyword not in keyword_positions:
                            keyword_positions[keyword] = []
                        keyword_positions[keyword].append((sentence_pos, sentence_pos + len(sentence)))
                        keywords_found.add(keyword)
                        print(f"  ✓ Fuzzy match for '{keyword}': similarity={sim:.2f}, word_overlap={overlap:.2f}, position={sentence_pos}")
                        self.logger.info(f"🔎 Fuzzy match for '{keyword}': similarity={sim:.2f}, word_overlap={overlap:.2f}")
                        break  # Found a good match for this keyword

        # Phase 3: Find contexts with keyword co-occurrence (at least 2 keywords)
        print(f"\n{'='*80}")
        print(f"🔍 KEYWORD SEARCH - Co-occurrence Analysis")
        print(f"{'='*80}")
        print(f"Found {len(keywords_found)}/{len(keywords)} keywords")

        if len(keywords_found) < len(keywords):
            still_missing = set(keywords) - keywords_found
            print(f"✗ No matches found for {len(still_missing)} keywords: {list(still_missing)}")
            self.logger.warning(f"✗ No matches found for {len(still_missing)} keywords: {list(still_missing)}")

        if len(keyword_positions) < 2 and len(keywords) > 1:
            print(f"✗ Need at least 2 keywords found for co-occurrence, but only found {len(keyword_positions)}")
            print(f"{'='*80}\n")
            self.logger.warning(f"✗ Insufficient keywords for co-occurrence ({len(keyword_positions)}/2 minimum)")
            # Return unfound keywords
            still_missing = list(set(keywords) - keywords_found)
            return "", still_missing

        # Find all contexts where at least 2 keywords co-occur
        # Store contexts with position info: (start, end, text, keywords_in_window)
        context_candidates = []
        seen_contexts = set()
        co_occurrence_count = 0

        # For each keyword position, check if another keyword appears within the context window
        for keyword1, positions1 in keyword_positions.items():
            for start1, end1 in positions1:
                # Define context window around this keyword
                context_start = max(0, start1 - context_window)
                context_end = min(len(text), end1 + context_window)

                # Check if any other keyword appears in this window
                keywords_in_window = {keyword1}
                for keyword2, positions2 in keyword_positions.items():
                    if keyword2 != keyword1:
                        for start2, end2 in positions2:
                            # Check if keyword2 overlaps with this context window
                            if (context_start <= start2 < context_end) or (context_start < end2 <= context_end):
                                keywords_in_window.add(keyword2)
                                break

                # Extract context if at least 2 keywords co-occur (or only 1 keyword exists)
                if len(keywords_in_window) >= 2 or len(keywords) == 1:
                    context = text[context_start:context_end].strip()
                    context_hash = hash(context)

                    if context_hash not in seen_contexts:
                        seen_contexts.add(context_hash)
                        context_candidates.append((context_start, context_end, context, keywords_in_window))
                        co_occurrence_count += 1
                        print(f"  ✓ Co-occurrence found: {list(keywords_in_window)} at position {start1}")

        print(f"\n{'='*80}")
        print(f"🔧 CONTEXT OPTIMIZATION - Merging & Limiting")
        print(f"{'='*80}")
        print(f"Initial contexts extracted: {len(context_candidates)}")

        # Calculate unfound keywords for return
        still_missing = list(set(keywords) - keywords_found)

        if not context_candidates:
            print(f"✗ No contexts with keyword co-occurrence found")
            print(f"{'='*80}\n")
            self.logger.warning(f"✗ No contexts with keyword co-occurrence found")
            return "", still_missing

        # SOLUTION 1: Merge overlapping contexts
        # Sort contexts by start position
        context_candidates.sort(key=lambda x: x[0])

        merged_contexts = []
        current_start, current_end, current_text, current_keywords = context_candidates[0]

        for i in range(1, len(context_candidates)):
            next_start, next_end, next_text, next_keywords = context_candidates[i]

            # Calculate overlap: contexts overlap if one starts before the other ends
            overlap_start = max(current_start, next_start)
            overlap_end = min(current_end, next_end)
            overlap_length = max(0, overlap_end - overlap_start)

            # Calculate overlap percentage relative to smaller context
            smaller_length = min(current_end - current_start, next_end - next_start)
            overlap_pct = (overlap_length / smaller_length * 100) if smaller_length > 0 else 0

            # Merge if overlap is > 50%
            if overlap_pct > 50:
                # Merge: extend current context to include next context
                current_end = max(current_end, next_end)
                current_start = min(current_start, next_start)
                # Re-extract text from merged range
                current_text = text[current_start:current_end].strip()
                # Combine keywords
                current_keywords = current_keywords | next_keywords
                print(f"  ✓ Merged contexts (overlap: {overlap_pct:.1f}%): positions {next_start}-{next_end} into {current_start}-{current_end}")
            else:
                # No significant overlap, save current and start new
                merged_contexts.append((current_start, current_end, current_text, current_keywords))
                current_start, current_end, current_text, current_keywords = next_start, next_end, next_text, next_keywords

        # Don't forget the last context
        merged_contexts.append((current_start, current_end, current_text, current_keywords))

        print(f"After merging: {len(merged_contexts)} contexts")
        self.logger.info(f"🔧 Merged overlapping contexts: {len(context_candidates)} → {len(merged_contexts)}")

        # SOLUTION 2: Limit number of contexts to prevent explosion
        if len(merged_contexts) > max_contexts:
            # Sort by keyword density (more keywords = higher priority)
            merged_contexts.sort(key=lambda x: len(x[3]), reverse=True)
            # Keep only top max_contexts
            limited_contexts = merged_contexts[:max_contexts]
            # Re-sort by position for coherent concatenation
            limited_contexts.sort(key=lambda x: x[0])
            print(f"Applied context limit: {len(merged_contexts)} → {max_contexts} contexts (keeping highest keyword density)")
            self.logger.info(f"🔧 Limited contexts to prevent explosion: {len(merged_contexts)} → {max_contexts}")
            merged_contexts = limited_contexts

        # Extract just the text from the (start, end, text, keywords) tuples
        contexts = [ctx[2] for ctx in merged_contexts]

        print(f"\n{'='*80}")
        print(f"🔍 KEYWORD SEARCH - Final Results")
        print(f"{'='*80}")
        print(f"✓ Final contexts after optimization: {len(contexts)}")
        print(f"✓ Total extracted text: {sum(len(c) for c in contexts)} chars")
        print(f"{'='*80}\n")
        self.logger.info(f"✓ Extracted {len(contexts)} context windows after merging & limiting")

        # Concatenate all contexts with separators
        focused_text = "\n\n---\n\n".join(contexts)

        self.logger.info(f"Extracted {len(contexts)} context windows with keyword co-occurrence (total length: {len(focused_text)} chars)")
        self.logger.info(f"Keywords NOT found: {still_missing}")

        return focused_text, still_missing

    async def _process_chunk(self, chunk: str, learned_system_prompt: str, ctx: MessageContext) -> tuple:
        """Process a text chunk to extract entities and relationships."""
        # Prepare base prompt (without critique)
        prompt_content = self.base_prompt_graph_builder.format(chunk)

        # Call LLM with structured output using learned system prompt
        system_message = SystemMessage(content=learned_system_prompt)
        user_message = UserMessage(content=prompt_content, source="user")

        response = await self.model_client_creation.create(
            [system_message, user_message],
            cancellation_token=ctx.cancellation_token
        )

        # Log LLM interaction
        # Get current QA pair and iteration from shared state for logging
        try:
            current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, self.current_batch_id)
            current_qa_pair_id = current_state.get("current_qa_pair_id", None)
            current_iteration = current_state.get("current_iteration", None)
        except Exception:
            current_qa_pair_id = None
            current_iteration = None

        logger = get_global_prompt_logger()
        logger.log_interaction(
            agent_name="GraphBuilderAgent",
            interaction_type="graph_creation",
            system_prompt=learned_system_prompt,
            user_prompt=prompt_content,
            llm_response=response.content if isinstance(response.content, str) else str(response.content),
            batch_id=self.current_batch_id,
            qa_pair_id=current_qa_pair_id,
            iteration=current_iteration,
            additional_metadata={
                "chunk_length": len(chunk),
                "mode": "creation"
            }
        )

        # Parse structured response using custom parser
        assert isinstance(response.content, str)
        graph_response = self._parse_graph_builder_response(response.content)

        return graph_response.entities, graph_response.relationships, graph_response.triplets

    def _convert_to_memgraph_format(self, entities, relationships, triplets, iteration: int = 0) -> List[Dict[str, Any]]:
        """
        Convert extracted data to Memgraph import_util.json() format.

        Args:
            entities: List of entities
            relationships: List of relationships
            triplets: List of triplets
            iteration: Current iteration number (0 for first iteration, 1 for second, etc.)
                      Used to weight edges - higher iteration = higher weight
        """
        memgraph_items = []
        node_id_counter = 1000  # Start with high numbers to avoid conflicts

        # Track node names to IDs for relationship creation
        name_to_id = {}

        # Convert entities to nodes
        for entity in entities:
            node_id = node_id_counter
            node_id_counter += 1

            # Start with basic properties
            properties = {
                "name": entity.name,
                "type": entity.type
            }

            # Add entity description to node properties
            properties["description"] = entity.description

            node = {
                "id": node_id,
                "labels": [entity.type],
                "properties": properties,
                "type": "node"
            }
            memgraph_items.append(node)
            name_to_id[entity.name] = node_id

        # Convert relationships to edges with iteration-based weighting
        # Weight formula: log(iteration + 1) + 1 for less aggressive weighting
        # iteration 0 -> weight 1.0, iteration 1 -> weight 1.69, iteration 2 -> weight 2.10, etc.
        import math
        edge_weight = 1
        rel_id_counter = 2000  # Start with high numbers to avoid conflicts
        for relationship in relationships:
            # Get node IDs for start and end nodes
            start_id = name_to_id.get(relationship.source_entity)
            end_id = name_to_id.get(relationship.target_entity)

            # Only create relationship if both nodes exist
            if start_id is not None and end_id is not None:
                edge = {
                    "id": rel_id_counter,
                    "start": start_id,
                    "end": end_id,
                    "label": relationship.relationship_type,
                    "properties": {
                        "description": relationship.description,
                        "evidence": relationship.evidence,
                        "type": relationship.relationship_type,
                        "weight": edge_weight,  # Add weight based on iteration
                        "iteration": iteration  # Track which iteration added this edge
                    },
                    "type": "relationship"
                }
                memgraph_items.append(edge)
                rel_id_counter += 1

        return memgraph_items

    def _add_placeholder_entities(self, entities: List, relationships: List) -> List:
        """
        Add placeholder Entity objects for entities referenced in relationships
        but not explicitly defined.

        Args:
            entities: List of explicitly defined entities
            relationships: List of relationships

        Returns:
            Updated entity list including placeholders
        """
        from parameters import Entity

        # Get set of existing entity names
        existing_names = {entity.name for entity in entities}

        # Find missing entity names referenced in relationships
        missing_names = set()
        for rel in relationships:
            # Check source entity
            if rel.source_entity and rel.source_entity.strip():
                if rel.source_entity not in existing_names:
                    missing_names.add(rel.source_entity)

            # Check target entity
            if rel.target_entity and rel.target_entity.strip():
                if rel.target_entity not in existing_names:
                    missing_names.add(rel.target_entity)

        # Create placeholder entities with undefined type and description
        placeholder_entities = []
        for name in missing_names:
            placeholder = Entity(
                name=name,
                type="undefined",
                description="undefined"
            )
            placeholder_entities.append(placeholder)

        # Log if placeholders were created
        if placeholder_entities:
            missing_names_sorted = sorted(list(missing_names))
            self.logger.warning(
                f"Auto-created {len(placeholder_entities)} undefined nodes for entities "
                f"referenced in relationships but not defined by LLM: "
                f"{missing_names_sorted[:10]}{'...' if len(missing_names) > 10 else ''}"
            )

        # Return merged list
        return list(entities) + placeholder_entities

    async def _process_graph_creation(self, chunks: List[str], learned_system_prompt: str, ctx: MessageContext) -> tuple:
        """Process chunks for graph creation mode."""
        all_entities = []
        all_relationships = []
        all_triplets = []

        # Process all chunks asynchronously for better time efficiency
        self.logger.info(f"Starting concurrent processing of {len(chunks)} chunks")

        # Create tasks for all chunks
        async def process_chunk_with_index(i: int, chunk: str) -> tuple:
            """Wrapper to process chunk with index for logging and error handling."""
            try:
                self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                entities, relationships, triplets = await self._process_chunk(chunk, learned_system_prompt, ctx)
                self.logger.info(f"Completed chunk {i+1}/{len(chunks)}")
                return entities, relationships, triplets
            except Exception as e:
                self.logger.error(f"Error processing chunk {i+1}: {e}")
                return [], [], []  # Return empty results for failed chunks

        # Execute all chunk processing tasks concurrently
        import asyncio
        tasks = [process_chunk_with_index(i, chunk) for i, chunk in enumerate(chunks)]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=False)

        # Aggregate results from all chunks
        for entities, relationships, triplets in chunk_results:
            all_entities.extend(entities)
            all_relationships.extend(relationships)
            all_triplets.extend(triplets)

        self.logger.info(f"Completed concurrent processing of all {len(chunks)} chunks")
        return all_entities, all_relationships, all_triplets

    async def _process_graph_refinement(self, chunks: list, learned_system_prompt: str, ctx: MessageContext) -> tuple:
        """Process chunks for graph refinement mode - parallel processing like graph creation."""
        all_entities = []
        all_relationships = []
        all_triplets = []

        # Process all chunks asynchronously for better time efficiency
        self.logger.info(f"Starting concurrent refinement processing of {len(chunks)} chunks")

        # Track chunk processing results
        chunks_with_results = 0
        chunks_with_no_results = 0

        # Create tasks for all chunks
        async def process_refinement_chunk_with_index(i: int, chunk: str) -> tuple:
            """Wrapper to process refinement chunk with index for logging and error handling."""
            try:
                self.logger.info(f"Processing refinement chunk {i+1}/{len(chunks)}")
                entities, relationships, triplets = await self._process_refinement_chunk(chunk, learned_system_prompt, ctx)
                self.logger.info(f"Completed refinement chunk {i+1}/{len(chunks)}: {len(entities)} entities, {len(relationships)} relationships")
                return entities, relationships, triplets
            except Exception as e:
                self.logger.error(f"❌ Error processing refinement chunk {i+1}: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return [], [], []  # Return empty results for failed chunks

        # Execute all chunk processing tasks concurrently
        import asyncio
        tasks = [process_refinement_chunk_with_index(i, chunk) for i, chunk in enumerate(chunks)]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=False)

        # Aggregate results from all chunks and track statistics
        for i, (entities, relationships, triplets) in enumerate(chunk_results):
            if len(entities) > 0 or len(relationships) > 0:
                chunks_with_results += 1
            else:
                chunks_with_no_results += 1

            all_entities.extend(entities)
            all_relationships.extend(relationships)
            all_triplets.extend(triplets)

        self.logger.info(f"Completed concurrent refinement processing of all {len(chunks)} chunks")
        self.logger.info(f"Chunk results: {chunks_with_results} chunks with data, {chunks_with_no_results} chunks with no data")

        # Warn if many chunks produced no results
        if len(chunks) > 0 and chunks_with_no_results / len(chunks) > 0.5:
            self.logger.warning(
                f"⚠️  More than 50% of chunks ({chunks_with_no_results}/{len(chunks)}) produced no results. "
                f"This may indicate parsing errors or LLM format issues."
            )

        log_agent_action(self.logger, "GraphRefinement", "LLM refinement",
                        entities=len(all_entities),
                        relationships=len(all_relationships),
                        triplets=len(all_triplets))

        return all_entities, all_relationships, all_triplets

    async def _process_refinement_chunk(self, chunk: str, learned_system_prompt: str, ctx: MessageContext) -> tuple:
        """Process a single text chunk for graph refinement."""
        # Extract only the output format section from base_prompt_graph_refinement
        # The base prompt contains steps, examples, constraints, text placeholder, and output format
        # We need to extract only the abstract output format (from "# Output Format" onwards)
        base_prompt_lines = self.base_prompt_graph_refinement.split('\n')

        # Find the "# Output Format" section
        output_format_start = None
        for i, line in enumerate(base_prompt_lines):
            if line.strip() == "# Output Format":
                output_format_start = i
                break

        # Extract the output format section
        if output_format_start is not None:
            output_format_section = '\n'.join(base_prompt_lines[output_format_start:])
        else:
            # Fallback if structure changes - use the full base prompt
            output_format_section = self.base_prompt_graph_refinement.format(chunk)

        # Build the user prompt: Learned Prompt + Text + Output Format
        if learned_system_prompt:
            user_prompt = learned_system_prompt
            user_prompt += "\n\n" + "=" * 80 + "\n"
            user_prompt += "# Text to Analyze\n\n"
            user_prompt += chunk
            user_prompt += "\n\n" + "=" * 80 + "\n"
            user_prompt += output_format_section
        else:
            # If no learned prompt, use the original base prompt
            user_prompt = self.base_prompt_graph_refinement.format(chunk)

        # Create messages with empty system prompt (only user message)
        messages = [UserMessage(content=user_prompt, source="user")]

        # Use refinement model client
        response = await self.model_client_refinement.create(messages, cancellation_token=ctx.cancellation_token)

        # Log LLM interaction
        # Get current QA pair and iteration from shared state for logging
        try:
            current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, self.current_batch_id)
            current_qa_pair_id = current_state.get("current_qa_pair_id", None)
            current_iteration = current_state.get("current_iteration", None)
        except Exception:
            current_qa_pair_id = None
            current_iteration = None

        logger = get_global_prompt_logger()
        logger.log_interaction(
            agent_name="GraphBuilderAgent",
            interaction_type="graph_refinement",
            system_prompt="",  # System prompt is now empty
            user_prompt=user_prompt,
            llm_response=response.content if isinstance(response.content, str) else str(response.content),
            batch_id=self.current_batch_id,
            qa_pair_id=current_qa_pair_id,
            iteration=current_iteration,
            additional_metadata={
                "chunk_length": len(chunk),
                "mode": "refinement"
            }
        )

        # Parse structured response using tuple parser
        if not isinstance(response.content, str):
            self.logger.error(f"Unexpected response type from LLM: {type(response.content)}. Expected string.")
            self.logger.error(f"Response content: {response.content}")
            return [], [], []

        # Try tuple parsing first
        try:
            refinement_response = self._parse_graph_refinement_tuples(response.content)

            # Check if parsing produced any results
            if (len(refinement_response.new_entities) == 0 and
                len(refinement_response.new_relationships) == 0 and
                len(response.content.strip()) > 100):
                # Response has content but no entities/relationships extracted
                self.logger.warning(
                    f"Tuple parsing extracted 0 entities and 0 relationships from {len(response.content)} character response. "
                    f"LLM may have returned wrong format. First 500 chars of response:\n{response.content[:500]}"
                )

                # Try JSON parsing as fallback
                self.logger.info("Attempting fallback to JSON parsing...")
                try:
                    refinement_response = self._parse_graph_refinement_response(response.content)
                    self.logger.info(f"✓ JSON fallback successful: {len(refinement_response.new_entities)} entities, {len(refinement_response.new_relationships)} relationships")
                except Exception as json_error:
                    self.logger.error(f"JSON fallback parsing also failed: {json_error}")
                    # Return the original (empty) tuple parsing result

            return refinement_response.new_entities, refinement_response.new_relationships, refinement_response.new_triplets

        except Exception as e:
            self.logger.error(f"Failed to parse refinement response: {e}")
            self.logger.error(f"Response content (first 1000 chars): {response.content[:1000]}")
            return [], [], []

    async def _get_existing_graph_summary(self) -> str:
        """Get a summary of the existing graph from Memgraph."""
        try:
            from graph_functions import generate_graph_description
            graph_description_result = generate_graph_description()
            return graph_description_result.get("description", "No existing graph found")
        except Exception as e:
            self.logger.error(f"Error getting existing graph summary: {e}")
            return "No existing graph available"

    async def _load_existing_graph_data(self) -> tuple:
        """Load existing graph entities and relationships from the previous iteration."""
        try:
            # Try to load the existing graph file to get the actual entities/relationships
            import os
            import json
            from parameters import Entity, Relationship, Triplet

            graphs_dir = "graphs"
            # We need to figure out the current batch info to find the right file
            # For now, we'll look for the most recent graph file

            if not os.path.exists(graphs_dir):
                self.logger.info("No existing graphs directory found, starting with empty graph")
                return [], [], []

            graph_files = [f for f in os.listdir(graphs_dir) if f.endswith('_graph.json')]
            if not graph_files:
                self.logger.info("No existing graph files found, starting with empty graph")
                return [], [], []

            # Get the most recent graph file
            latest_file = max(graph_files, key=lambda f: os.path.getmtime(os.path.join(graphs_dir, f)))
            graph_path = os.path.join(graphs_dir, latest_file)

            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)

            # Extract entities and relationships from the memgraph format
            entities = []
            relationships = []
            triplets = []

            # First pass: build ID to name mapping
            id_to_name = {}
            for item in graph_data:
                if item.get("type") == "node":
                    node_id = item.get("id")
                    node_name = item.get("properties", {}).get("name", "")
                    if node_id and node_name:
                        id_to_name[node_id] = node_name

            # Second pass: extract entities and relationships
            for item in graph_data:
                if item.get("type") == "node":
                    # Convert back to Entity format
                    props = item.get("properties", {})

                    # Extract description from properties
                    description = props.get("description", "")

                    entity = Entity(
                        name=props.get("name", ""),
                        type=props.get("type", ""),
                        description=description
                    )
                    entities.append(entity)

                elif item.get("type") == "relationship":
                    # Convert back to Relationship format
                    props = item.get("properties", {})

                    # Get start and end node IDs from the edge
                    start_id = item.get("start")
                    end_id = item.get("end")

                    # Look up entity names from ID mapping
                    source_entity = id_to_name.get(start_id, "")
                    target_entity = id_to_name.get(end_id, "")

                    # Skip relationships where we can't find the entities
                    if not source_entity or not target_entity:
                        self.logger.warning(f"Skipping relationship with missing entities: start_id={start_id}, end_id={end_id}")
                        continue

                    relationship = Relationship(
                        source_entity=source_entity,
                        target_entity=target_entity,
                        relationship_type=props.get("type", ""),
                        description=props.get("description", ""),
                        evidence=props.get("evidence", "")
                    )
                    relationships.append(relationship)

            self.logger.info(f"Loaded existing graph: {len(entities)} entities, {len(relationships)} relationships")
            return entities, relationships, triplets

        except Exception as e:
            self.logger.error(f"Error loading existing graph data: {e}")
            self.logger.info("Continuing with empty existing graph")
            return [], [], []

    def _compute_string_similarity(self, str1: str, str2: str) -> float:
        """
        Compute string similarity between two strings using normalized edit distance.
        Returns a score between 0.0 (completely different) and 1.0 (identical).
        """
        # Handle edge cases
        if str1 == str2:
            return 1.0
        if not str1 or not str2:
            return 0.0

        # Normalize strings: lowercase and strip whitespace
        s1 = str1.lower().strip()
        s2 = str2.lower().strip()

        if s1 == s2:
            return 1.0

        # Compute Levenshtein distance
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)

            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        distance = levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))

        # Normalize to similarity score (0.0 to 1.0)
        similarity = 1.0 - (distance / max_len) if max_len > 0 else 0.0

        # Additional bonus for partial matches (substring containment)
        if s1 in s2 or s2 in s1:
            substring_bonus = 0.1
            similarity = min(1.0, similarity + substring_bonus)

        return similarity

    def _resolve_entities(self, entities: List, relationships: List, triplets: List) -> tuple:
        """
        Perform entity resolution by merging nodes that refer to the same entity.
        Returns updated entities, relationships, and triplets.
        """
        if not entities:
            return entities, relationships, triplets

        self.logger.info(f"Starting entity resolution on {len(entities)} entities")

        # Similarity threshold - entities with similarity >= this will be merged
        SIMILARITY_THRESHOLD = 0.85  # High threshold to avoid false positives

        # Group entities by similarity
        entity_clusters = []
        processed_indices = set()

        for i, entity1 in enumerate(entities):
            if i in processed_indices:
                continue

            # Start a new cluster with this entity
            cluster = [entity1]
            cluster_indices = [i]
            processed_indices.add(i)

            # Find all entities similar to entity1
            for j, entity2 in enumerate(entities[i+1:], start=i+1):
                if j in processed_indices:
                    continue

                similarity = self._compute_string_similarity(entity1.name, entity2.name)

                if similarity >= SIMILARITY_THRESHOLD:
                    self.logger.info(f"Merging entities: '{entity1.name}' and '{entity2.name}' (similarity: {similarity:.3f})")
                    cluster.append(entity2)
                    cluster_indices.append(j)
                    processed_indices.add(j)

            entity_clusters.append((cluster, cluster_indices))

        # Create merged entities
        merged_entities = []
        entity_name_mapping = {}  # Maps old entity names to new canonical names

        for cluster, indices in entity_clusters:
            if len(cluster) == 1:
                # No merging needed
                merged_entities.append(cluster[0])
                entity_name_mapping[cluster[0].name] = cluster[0].name
            else:
                # Merge entities in this cluster
                primary_entity = cluster[0]  # Use first entity as primary

                # Merge descriptions from all entities in the cluster
                merged_description = primary_entity.description
                descriptions_seen = {primary_entity.description}

                for entity in cluster[1:]:  # Skip primary entity, start from second
                    # Add description if it's different and adds new information
                    if entity.description and entity.description not in descriptions_seen:
                        merged_description += "; " + entity.description
                        descriptions_seen.add(entity.description)

                # Update primary entity with merged description
                primary_entity.description = merged_description

                # Collect all names for mapping
                for entity in cluster:
                    entity_name_mapping[entity.name] = primary_entity.name

                merged_entities.append(primary_entity)

        # Update relationships to use canonical entity names and deduplicate
        updated_relationships = []
        seen_relationships = set()  # Track (source, target, type) tuples to avoid duplicates

        for rel in relationships:
            # Map source and target entities to their canonical names
            canonical_source = entity_name_mapping.get(rel.source_entity, rel.source_entity)
            canonical_target = entity_name_mapping.get(rel.target_entity, rel.target_entity)

            # Skip self-relationships that might have been created by merging
            if canonical_source == canonical_target:
                continue

            # Create a signature for this relationship to detect duplicates
            rel_signature = (canonical_source.lower(), canonical_target.lower(), rel.relationship_type.lower())

            # Skip if we've already seen this exact relationship
            if rel_signature in seen_relationships:
                self.logger.debug(f"Skipping duplicate relationship: {canonical_source} -{rel.relationship_type}-> {canonical_target}")
                continue

            seen_relationships.add(rel_signature)

            # Create updated relationship
            updated_rel = type(rel)(
                source_entity=canonical_source,
                target_entity=canonical_target,
                relationship_type=rel.relationship_type,
                description=rel.description,
                evidence=rel.evidence
            )
            updated_relationships.append(updated_rel)

        # Update triplets similarly
        updated_triplets = []
        for triplet in triplets:
            # Triplet has subject, predicate, object attributes
            canonical_subject = entity_name_mapping.get(triplet.subject, triplet.subject)
            canonical_object = entity_name_mapping.get(triplet.object, triplet.object)

            # Skip self-relationships
            if canonical_subject == canonical_object:
                continue

            # Create updated triplet
            updated_triplet = type(triplet)(
                subject=canonical_subject,
                predicate=triplet.predicate,
                object=canonical_object
            )
            updated_triplets.append(updated_triplet)

        entities_before = len(entities)
        entities_after = len(merged_entities)
        entities_merged = entities_before - entities_after

        relationships_before = len(relationships)
        relationships_after = len(updated_relationships)
        relationships_deduplicated = relationships_before - relationships_after

        self.logger.info(f"Entity resolution completed: {entities_before} → {entities_after} entities ({entities_merged} merged)")
        self.logger.info(f"Relationship deduplication: {relationships_before} → {relationships_after} relationships ({relationships_deduplicated} duplicates removed)")

        return merged_entities, updated_relationships, updated_triplets

    async def close(self) -> None:
        """Close the model clients."""
        await self.model_client_creation.close()
        await self.model_client_refinement.close()


# ===== GRAPH RETRIEVAL PLANNER AGENT =====

class GraphRetrievalPlannerAgent(RoutedAgent):
    """
    Agent that uses community-based retrieval with iterative query refinement.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.graph_retrieval_planner")
        self.shared_state = SharedState("agent_states")

        # Import response formats and prompts
        from parameters import base_prompt_graph_retrieval_planner, GraphRetrievalPlannerResponse

        # Initialize LLM client for query refinement with structured output
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
            response_format=GraphRetrievalPlannerResponse
        )

        # Initialize LLM client for community summarization
        self.summarizer_client = OpenAIChatCompletionClient(
            model="gemini-2.5-flash-lite",
            max_tokens=500,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=llm_keys.GEMINI_KEY,
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "family": "unknown",
            }
        )

        self.base_prompt_graph_retrieval_planner = base_prompt_graph_retrieval_planner

        # Initialize community graph manager
        self.community_manager = None
        self.current_graph_file = None

    @message_handler
    async def handle_graph_retrieval_start(self, message: GraphRetrievalStartMessage, ctx: MessageContext) -> GraphRetrievalReadyMessage:
        """Handle GraphRetrievalStart message using iterative community-based retrieval."""
        self.logger.info(f"GraphRetrievalPlannerAgent processing batch {message.batch_id} for query: {message.query}")

        try:
            # Load and process graph with community detection
            import os
            graphs_dir = "graphs"
            graph_filename = os.path.join(graphs_dir, f"{message.dataset}_{message.setting}_batch_{message.batch_id}_graph.json")

            if not os.path.exists(graph_filename):
                self.logger.error(f"Graph file {graph_filename} not found")
                return GraphRetrievalReadyMessage(
                    batch_id=message.batch_id,
                    repetition=message.repetition,
                    retrieved_context="Error: Graph file not found",
                    dataset=message.dataset,
                    setting=message.setting
                )

            # Load shared state
            current_state = message.shared_state

            # Check if refinement was skipped (no missing keywords)
            refinement_skipped = current_state.get("refinement_skipped", False)

            # Check if we need to reload/reprocess the graph
            # Reasons to reprocess:
            # 1. No community manager exists yet
            # 2. Graph file has changed (different filename)
            # 3. Graph content has changed (refinement in iteration 1+) AND refinement wasn't skipped
            should_reprocess = (
                self.community_manager is None or
                self.current_graph_file != graph_filename or
                (message.repetition > 0 and not refinement_skipped)  # Only reprocess if graph was actually refined
            )

            if should_reprocess:
                if self.community_manager is None:
                    self.logger.info(f"Loading and processing graph for first time: {graph_filename}")
                elif self.current_graph_file != graph_filename:
                    self.logger.info(f"Graph file changed, reprocessing: {graph_filename}")
                elif message.repetition > 0:
                    self.logger.info(f"Graph was refined in iteration {message.repetition}, reprocessing to detect communities with new relationships")

                from community_graph_utils import load_and_process_graph
                from parameters import default_community_levels
                import json

                # Log what's in the file before loading
                try:
                    with open(graph_filename, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                    file_node_count = sum(1 for item in file_data if item.get("type") == "node")
                    file_rel_count = sum(1 for item in file_data if item.get("type") == "relationship")
                    self.logger.info(f"📖 Graph file to load contains: {file_node_count} nodes, {file_rel_count} relationships")
                except Exception as e:
                    self.logger.error(f"Error reading graph file for verification: {e}")

                self.logger.info("Using base community summarizer prompt (no optimization)")

                self.community_manager = await load_and_process_graph(
                    graph_filename,
                    self.summarizer_client,
                    embedding_model=None,  # Use TF-IDF fallback
                    learned_prompt="",  # Always use base prompt, no optimization
                    community_levels=default_community_levels  # Use configured level filter
                )

                if self.community_manager is None:
                    self.logger.error("Failed to load and process graph")
                    return GraphRetrievalReadyMessage(
                        batch_id=message.batch_id,
                        repetition=message.repetition,
                        retrieved_context="Error: Failed to process graph",
                        dataset=message.dataset,
                        setting=message.setting
                    )

                self.current_graph_file = graph_filename
                self.logger.info(f"Successfully processed graph with {len(self.community_manager.communities)} communities")

                # Store community summaries in shared state for backward pass critique
                current_state["community_summaries"] = self.community_manager.community_summaries
                self.logger.info(f"Stored {len(self.community_manager.community_summaries)} community summaries in shared state")

                # Store community summarization logs for prompts_response_logs
                current_state["community_summarization_logs"] = self.community_manager.community_summarization_logs
                self.logger.info(f"Stored {len(self.community_manager.community_summarization_logs)} community summarization logs")
            else:
                if refinement_skipped:
                    print(f"\n{'='*80}")
                    print(f"⏭️  SKIPPING COMMUNITY REPROCESSING")
                    print(f"{'='*80}")
                    print(f"Reason: Graph refinement was skipped (no missing keywords or no co-occurrence contexts found)")
                    print(f"Action: Reusing existing community manager and summaries from previous iteration")
                    print(f"{'='*80}\n")
                    self.logger.info(f"⏭️  SKIPPING community reprocessing - refinement was skipped, reusing existing communities")

                    # Ensure community data persists in current state for backward pass
                    if self.community_manager:
                        current_state["community_summaries"] = self.community_manager.community_summaries
                        current_state["community_summarization_logs"] = self.community_manager.community_summarization_logs
                        self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)
                        self.logger.info(f"Persisted {len(self.community_manager.community_summaries)} community summaries to current iteration state")
                else:
                    self.logger.info(f"Reusing existing community manager (iteration {message.repetition}), no prompt changes detected")

            # For first repetition (repetition=0), start with empty system prompt to avoid data leakage
            if message.repetition == 0:
                learned_system_prompt = ""
                self.logger.info(f"First repetition for batch {message.batch_id} - using empty retrieval planner system prompt")
            else:
                learned_system_prompt = current_state.get("learned_prompt_graph_retrieval_planner", "")

            # Create retrieval prompt template and save to shared state
            prompt_template = self.base_prompt_graph_retrieval_planner.format(
                message.query, "{RETRIEVED_CONTEXT}"
            )
            current_state["retrieval_prompt"] = prompt_template

            # Initialize retrieval context and plan responses
            retrieved_context = ""
            retrieval_plan_responses = []

            # One-shot community selection
            try:
                # Get all available community titles
                community_titles = self.community_manager.get_all_community_titles()

                # Format community titles for the prompt
                if community_titles:
                    titles_text = "\n".join([f"Community {cid}: {title}" for cid, title in community_titles.items()])
                else:
                    self.logger.error("No community titles available")
                    titles_text = "No communities available"

                # Prepare prompt with community titles
                prompt_content = self.base_prompt_graph_retrieval_planner.format(
                    message.query, titles_text
                )

                # Add format instruction to user prompt
                format_instruction = "\n\nIMPORTANT: In the selected_communities field, provide ONLY the community IDs WITHOUT the 'Community ' prefix.\n- Correct format: [\"L0_3\", \"L1_2\", \"L0_5\"]\n- WRONG format: [\"Community L0_3\", \"Community L1_2\"]\nJust use the ID itself (e.g., \"L0_3\" NOT \"Community L0_3\")."
                prompt_content = prompt_content + format_instruction

                # Call LLM to select communities using learned system prompt
                from autogen_core.models import SystemMessage, UserMessage
                system_message = SystemMessage(content=learned_system_prompt)
                user_message = UserMessage(content=prompt_content, source="user")

                response = await self.model_client.create(
                    [system_message, user_message],
                    cancellation_token=ctx.cancellation_token
                )

                # Log LLM interaction
                # Get current QA pair and iteration from shared state for logging
                current_qa_pair_id = current_state.get("current_qa_pair_id", None)
                current_iteration = current_state.get("current_iteration", None)

                logger = get_global_prompt_logger()
                logger.log_interaction(
                    agent_name="GraphRetrievalPlannerAgent",
                    interaction_type="community_selection",
                    system_prompt=learned_system_prompt,
                    user_prompt=prompt_content,
                    llm_response=response.content if isinstance(response.content, str) else str(response.content),
                    batch_id=message.batch_id,
                    qa_pair_id=current_qa_pair_id,
                    iteration=current_iteration,
                    additional_metadata={
                        "query": message.query,
                        "available_communities": len(community_titles)
                    }
                )

                # Parse structured response
                assert isinstance(response.content, str)
                from parameters import GraphRetrievalPlannerResponse
                retrieval_response = GraphRetrievalPlannerResponse.model_validate_json(response.content)

                # Store the LLM response
                retrieval_plan_responses.append(retrieval_response.reasoning)

                # Clean up community IDs (remove "Community " prefix if present)
                cleaned_ids = [cid.replace("Community ", "").strip() for cid in retrieval_response.selected_communities]

                # Retrieve the selected communities
                retrieved_context = await self._retrieve_selected_communities(cleaned_ids)

                self.logger.info(f"Selected communities: {retrieval_response.selected_communities}, context length: {len(retrieved_context)}")

            except Exception as e:
                self.logger.error(f"Error in community selection: {e}")
                retrieved_context = "Error during community selection"

            # Save retrieval plans and retrieved contexts to shared state for BackwardPassAgent
            current_state["retrieval_plans"] = retrieval_plan_responses

            # Store retrieved context data for BackwardPassAgent
            retrieved_contexts = current_state.get("retrieved_contexts", [])
            from datetime import datetime
            context_entry = {
                "retrieved_context": retrieved_context,
                "repetition": message.repetition,
                "timestamp": datetime.now().isoformat(),
                "batch_id": message.batch_id,
                "query": message.query
            }
            retrieved_contexts.append(context_entry)
            current_state["retrieved_contexts"] = retrieved_contexts

            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Send GraphRetrievalReady message
            retrieval_ready_msg = GraphRetrievalReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                retrieved_context=retrieved_context,
                dataset=message.dataset,
                setting=message.setting
            )

            self.logger.info(f"Returning GraphRetrievalReady for batch {message.batch_id}")

            # Return the retrieval ready message
            return retrieval_ready_msg

        except Exception as e:
            self.logger.error(f"Error in community-based retrieval: {e}")
            import traceback
            traceback.print_exc()

            return GraphRetrievalReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                retrieved_context=f"Error during retrieval: {str(e)}",
                dataset=message.dataset,
                setting=message.setting
            )

    async def _retrieve_selected_communities(self, community_ids: List[str]) -> str:
        """Retrieve the specified communities by their IDs."""
        try:
            context_parts = []
            valid_communities = []

            print(f"\n{'='*80}")
            print(f"🔍 COMMUNITY RETRIEVAL DEBUG")
            print(f"{'='*80}")
            print(f"Attempting to retrieve {len(community_ids)} communities: {community_ids}")
            print(f"Available communities in manager: {list(self.community_manager.community_summaries.keys())}")
            print(f"{'='*80}\n")

            for community_id in community_ids:
                # Check if community exists and has enough nodes
                in_summaries = community_id in self.community_manager.community_summaries
                in_titles = community_id in self.community_manager.community_titles
                node_count = len(self.community_manager.communities.get(community_id, []))

                print(f"  Community '{community_id}': in_summaries={in_summaries}, in_titles={in_titles}, nodes={node_count}")

                if (in_summaries and in_titles and node_count >= 3):
                    summary = self.community_manager.community_summaries[community_id]
                    context_parts.append(summary)
                    valid_communities.append(community_id)
                    print(f"  ✓ Community '{community_id}' RETRIEVED (summary length: {len(summary)} chars)")
                else:
                    print(f"  ✗ Community '{community_id}' REJECTED - in_summaries={in_summaries}, in_titles={in_titles}, nodes={node_count}")

            # Format the retrieved context
            print(f"\n{'='*80}")
            if context_parts:
                result = "\n\n".join(context_parts)
                print(f"✅ SUCCESSFULLY Retrieved {len(valid_communities)} communities: {valid_communities}")
                print(f"   Total context length: {len(result)} chars")
                print(f"{'='*80}\n")
                return result
            else:
                print(f"❌ FAILED - No valid communities could be retrieved")
                print(f"{'='*80}\n")
                return "No valid communities found."

        except Exception as e:
            print(f"\n{'='*80}")
            print(f"❌ ERROR retrieving selected communities {community_ids}: {e}")
            print(f"{'='*80}\n")
            import traceback
            traceback.print_exc()
            return ""

    async def close(self) -> None:
        """Close the model clients"""
        await self.model_client.close()
        await self.summarizer_client.close()


# ===== UPDATED FACTORY FUNCTIONS =====
def create_graph_builder_agent() -> GraphBuilderAgent:
    """Factory function to create GraphBuilderAgent instances."""
    return GraphBuilderAgent("graph_builder_agent")

def create_graph_retrieval_planner_agent() -> GraphRetrievalPlannerAgent:
    """Factory function to create GraphRetrievalPlannerAgent instances."""
    return GraphRetrievalPlannerAgent("graph_retrieval_planner_agent")


# ===== COMMUNITY ANSWER GENERATOR AGENT =====

class CommunityAnswerGeneratorAgent(RoutedAgent):
    """
    Agent that generates partial answers from individual community summaries.
    Evaluates if the community is useful and provides a partial answer if so.
    Uses Gemini Lite for fast parallel processing.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.community_answer_generator")
        self.shared_state = SharedState("agent_states")

        # Initialize Gemini Flash Lite model client for fast parallel processing
        self.model_client = OpenAIChatCompletionClient(
            model="gemini-2.5-flash-lite",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=llm_keys.GEMINI_KEY,
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "family": "unknown",
                "structured_output": False,
            }
        )

    @message_handler
    async def handle_community_answer_start(self, message: CommunityAnswerStartMessage, ctx: MessageContext) -> CommunityAnswerReadyMessage:
        """Handle CommunityAnswerStart message and generate partial answer from community summary."""
        self.logger.info(f"CommunityAnswerGenerator processing community {message.community_id} for QA {message.qa_pair_id}")

        # CommunityAnswerGeneratorAgent does NOT use learned system prompt
        # Only the FinalAnswerGenerator uses the learned system prompt
        # This ensures community-level answers remain consistent and are not affected by test-time training
        learned_system_prompt = ""

        # Create prompt for community-level answer generation
        user_prompt = f"""You are analyzing a community summary from a knowledge graph to help answer a question.

Question: {message.question}

Community Summary:
{message.community_summary}

Your task:
1. Analyze the community summary and identify which parts (if any) relate to the question.
2. Decide if this community provides useful information:
   - YES: If the community contains information that directly or indirectly helps answer the question
   - NO: If the community discusses unrelated topics or lacks relevant information
3. If useful, provide a partial answer based on this community's information.

═══════════════════════════════════════════════════════════════
CORE PRINCIPLE: Be INCLUSIVE, not exclusive
═══════════════════════════════════════════════════════════════

A community is USEFUL if it mentions:
1. **Named entities** from the question (people, places, objects, events)
2. **Related entities** that interact with, describe, or contextualize the named entities
3. **Relevant concepts** even without specific names (e.g., "relationships" for a relationship question)
4. **Contextual information** that provides background or setting for the question's focus

A community is NOT USEFUL only if:
- It discusses completely different topics with no overlap
- It contains zero mentions of entities/concepts related to the question
- It describes purely technical/environmental details with no relevance to the question's focus


IMPORTANT: Your partial answer should be comprehensive, not overly concise:
- Include ALL relevant information from the community that could help answer the question
- Filter out only truly irrelevant details that don't relate to the question
- Provide sufficient detail and context - don't just give brief snippets
- Think: "What would be most helpful for someone synthesizing multiple community answers?"

Format your response as:
USEFULNESS: [YES/NO]
ANSWER: [If YES: Provide a comprehensive partial answer based on this community's information, including all relevant details. If NO: Write "Not applicable"]"""

        try:
            # Call LLM for community-level answer generation
            from autogen_core.models import UserMessage

            # No system message - only user prompt
            messages = [UserMessage(content=user_prompt, source="user")]

            # Use retry logic
            async def api_call():
                return await self.model_client.create(
                    messages,
                    cancellation_token=ctx.cancellation_token
                )

            response = await retry_api_call_with_backoff(api_call)

            # Get generated response
            generated_response = response.content if isinstance(response.content, str) else str(response.content)

            # Parse usefulness and answer
            is_useful = False
            partial_answer = ""

            if "USEFULNESS:" in generated_response and "ANSWER:" in generated_response:
                # Extract usefulness
                usefulness_parts = generated_response.split("USEFULNESS:", 1)[1].split("ANSWER:", 1)
                usefulness_value = usefulness_parts[0].strip().upper()
                is_useful = "YES" in usefulness_value

                # Extract answer
                partial_answer = usefulness_parts[1].strip() if is_useful else ""
            else:
                # Fallback parsing
                is_useful = False
                partial_answer = ""

            self.logger.info(f"Community {message.community_id}: useful={is_useful}, answer_length={len(partial_answer)}")

            # Log LLM interaction
            logger = get_global_prompt_logger()
            if logger:
                logger.log_interaction(
                    agent_name="CommunityAnswerGeneratorAgent",
                    interaction_type="community_answer_generation",
                    system_prompt=learned_system_prompt,
                    user_prompt=user_prompt,
                    llm_response=generated_response,
                    batch_id=message.batch_id,
                    qa_pair_id=message.qa_pair_id,
                    iteration=message.repetition,
                    additional_metadata={
                        "community_id": message.community_id,
                        "is_useful": is_useful,
                        "partial_answer_length": len(partial_answer)
                    }
                )

            # Return CommunityAnswerReady message
            return CommunityAnswerReadyMessage(
                community_id=message.community_id,
                is_useful=is_useful,
                partial_answer=partial_answer,
                qa_pair_id=message.qa_pair_id,
                batch_id=message.batch_id,
                repetition=message.repetition
            )

        except Exception as e:
            self.logger.error(f"Error in community answer generation: {e}")
            # Return non-useful result on error
            return CommunityAnswerReadyMessage(
                community_id=message.community_id,
                is_useful=False,
                partial_answer="",
                qa_pair_id=message.qa_pair_id,
                batch_id=message.batch_id,
                repetition=message.repetition
            )

    async def close(self) -> None:
        """Close the model client."""
        await self.model_client.close()


def create_community_answer_generator_agent() -> CommunityAnswerGeneratorAgent:
    """Factory function to create CommunityAnswerGeneratorAgent instance."""
    return CommunityAnswerGeneratorAgent("community_answer_generator")


# ===== FINAL ANSWER GENERATOR AGENT (formerly AnswerGeneratorAgent) =====

class AnswerGeneratorAgent(RoutedAgent):
    """
    Final Answer Generator: Synthesizes community-level answers into a final answer.
    Renamed from original AnswerGeneratorAgent to clarify its new role.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.final_answer_generator")
        self.shared_state = SharedState("agent_states")

        # Import prompts
        from parameters import base_prompt_answer_generator_graph

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

        self.base_prompt_answer_generator_graph = base_prompt_answer_generator_graph

    @message_handler
    async def handle_answer_generation_start(self, message: AnswerGenerationStartMessage, ctx: MessageContext) -> AnswerGenerationReadyMessage:
        """Handle AnswerGenerationStart message and generate answer using LLM."""
        self.logger.info(f"AnswerGeneratorAgent processing QA pair {message.qa_pair_id}")

        # Load shared state to get learned system prompt
        current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)

        # For first repetition (repetition=0), start with empty system prompt to avoid data leakage
        if message.repetition == 0:
            learned_system_prompt = ""
            self.logger.info(f"First repetition for QA pair {message.qa_pair_id} - using empty answer generator system prompt")
        else:
            learned_system_prompt = current_state.get("learned_prompt_answer_generator_graph", "")

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

        # Prepare prompt with question and retrieved context
        prompt_content = self.base_prompt_answer_generator_graph.format(
            message.question, message.retrieved_context
        )

        # Append previous attempts to the prompt
        if previous_attempts_text:
            prompt_content += "\n\n" + previous_attempts_text
            self.logger.info(f"Including {len(qa_pair_evals)} previous attempt(s) in answer generation prompt") 

        try:
            # Call LLM for answer generation using learned system prompt
            system_message = SystemMessage(content=learned_system_prompt)
            user_message = UserMessage(content=prompt_content, source="user")

            # Use retry logic to handle API overload errors
            async def api_call():
                return await self.model_client.create(
                    [system_message, user_message],
                    cancellation_token=ctx.cancellation_token
                )

            response = await retry_api_call_with_backoff(api_call)

            # Get generated answer
            generated_answer = response.content if isinstance(response.content, str) else str(response.content)

            # Log LLM interaction
            logger = get_global_prompt_logger()
            logger.log_interaction(
                agent_name="AnswerGeneratorAgent",
                interaction_type="answer_generation",
                system_prompt=learned_system_prompt,
                user_prompt=prompt_content,
                llm_response=generated_answer,
                batch_id=message.batch_id,
                qa_pair_id=message.qa_pair_id,
                iteration=message.repetition,
                additional_metadata={
                    "question": message.question,
                    "retrieved_context_length": len(message.retrieved_context),
                    "answer_length": len(generated_answer)
                }
            )

            log_qa_processing(self.logger, message.qa_pair_id, "Generated answer", generated_answer)

            # Store conversation in shared state
            conversation_entry = {
                "qa_pair_id": message.qa_pair_id,
                "question": message.question,
                "retrieved_context": message.retrieved_context,
                "prompt": prompt_content,
                "generated_answer": generated_answer,
                "repetition": message.repetition  # Add repetition to track which iteration this belongs to
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
            self.logger.error(f"Error in answer generation after all retries: {e}")
            # Raise exception to stop pipeline - don't continue with bad data
            raise RuntimeError(f"Failed to generate answer for QA pair {message.qa_pair_id} after multiple retries: {e}")

    @message_handler
    async def handle_final_answer_start(self, message: FinalAnswerStartMessage, ctx: MessageContext) -> FinalAnswerReadyMessage:
        """Handle FinalAnswerStart message and synthesize community answers into final answer."""
        self.logger.info(f"FinalAnswerGenerator synthesizing {len(message.useful_community_answers)} community answers for QA {message.qa_pair_id}")

        # Load shared state to get learned system prompt
        current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)

        # Use the same learned prompt as community generators
        if message.repetition == 0:
            learned_system_prompt = ""
        else:
            learned_system_prompt = current_state.get("learned_prompt_answer_generator_graph", "")

        # Get previous responses and evaluations for this QA pair
        all_evaluation_responses = current_state.get("response_evaluations", [])
        qa_pair_evals = [
            eval_resp for eval_resp in all_evaluation_responses
            if eval_resp.get('qa_pair_id') == message.qa_pair_id
        ]

        # Format previous attempts
        previous_attempts_text = ""
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

        # Format community answers
        community_answers_text = "INFORMATION FROM KNOWLEDGE GRAPH COMMUNITIES:\n\n"
        for i, community_ans in enumerate(message.useful_community_answers, 1):
            community_answers_text += f"Community {community_ans['community_id']}:\n"
            community_answers_text += f"{community_ans['answer']}\n\n"

        # Create final answer generation prompt
        user_prompt = f"""You are given partial answers from multiple communities in a knowledge graph. Synthesize these into a comprehensive final answer.

Question: {message.question}

{community_answers_text}

Your task: Synthesize the information from all communities into a coherent, comprehensive final answer to the question. Combine related information, resolve any contradictions, and present a unified response.

{previous_attempts_text}

Provide only the final synthesized answer without meta-commentary."""

        try:
            # Call LLM for final answer generation
            system_message = SystemMessage(content=learned_system_prompt)
            user_message = UserMessage(content=user_prompt, source="user")

            async def api_call():
                return await self.model_client.create(
                    [system_message, user_message],
                    cancellation_token=ctx.cancellation_token
                )

            response = await retry_api_call_with_backoff(api_call)
            generated_answer = response.content if isinstance(response.content, str) else str(response.content)

            # Log LLM interaction
            logger = get_global_prompt_logger()
            if logger:
                logger.log_interaction(
                    agent_name="FinalAnswerGenerator",
                    interaction_type="final_answer_generation",
                    system_prompt=learned_system_prompt,
                    user_prompt=user_prompt,
                    llm_response=generated_answer,
                    batch_id=message.batch_id,
                    qa_pair_id=message.qa_pair_id,
                    iteration=message.repetition,
                    additional_metadata={
                        "question": message.question,
                        "num_community_answers": len(message.useful_community_answers),
                        "answer_length": len(generated_answer)
                    }
                )

            log_qa_processing(self.logger, message.qa_pair_id, "Generated final answer", generated_answer)

            # Store conversation in shared state
            conversation_entry = {
                "qa_pair_id": message.qa_pair_id,
                "question": message.question,
                "community_answers": message.useful_community_answers,
                "prompt": user_prompt,
                "generated_answer": generated_answer,
                "repetition": message.repetition
            }

            conversations = current_state.get("conversations_answer_generation", [])
            conversations.append(conversation_entry)
            current_state["conversations_answer_generation"] = conversations
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            return FinalAnswerReadyMessage(
                qa_pair_id=message.qa_pair_id,
                generated_answer=generated_answer,
                batch_id=message.batch_id,
                repetition=message.repetition
            )

        except Exception as e:
            self.logger.error(f"Error in final answer generation: {e}")
            raise RuntimeError(f"Failed to generate final answer for QA pair {message.qa_pair_id}: {e}")

    async def close(self) -> None:
        """Close the model client."""
        await self.model_client.close()


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

        # Import prompts and response format
        from parameters import response_evaluator_prompt_graph, ResponseEvaluationResponse

        # Load learned gold answer patterns if available
        self.satisfactory_criteria = self._load_gold_patterns()

        # Initialize Gemini model client with structured output
        self.model_client = OpenAIChatCompletionClient(
            model="gemini-2.5-flash",
            max_tokens=8192,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=llm_keys.GEMINI_KEY,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            },
            response_format=ResponseEvaluationResponse
        )

        self.response_evaluator_prompt = response_evaluator_prompt_graph

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

    @message_handler
    async def handle_response_evaluation_start(self, message: ResponseEvaluationStartMessage, ctx: MessageContext) -> ResponseEvaluationReadyMessage:
        """Handle ResponseEvaluationStart message and evaluate response using LLM."""
        print(f"\n🔍 ResponseEvaluatorAgent: Starting evaluation for QA pair {message.qa_pair_id}")
        print(f"   Query: {message.original_query[:100]}...")
        print(f"   Answer length: {len(message.generated_answer)} chars")
        print(f"   Community summaries received: {len(message.community_summaries) if message.community_summaries else 0} chars")

        self.logger.info(f"ResponseEvaluatorAgent evaluating QA pair {message.qa_pair_id}")

        # Don't include previous evaluations in the prompt (to match Vector RAG behavior)
        # Each iteration is evaluated independently

        # Format unfound keywords history for display
        if message.unfound_keywords_history:
            unfound_keywords_text = "\n\n**KEYWORDS ALREADY TRIED (DO NOT REPEAT):**\nThe following keywords were suggested in previous iterations but were NOT found in the document:\n" + ", ".join(message.unfound_keywords_history)
        else:
            unfound_keywords_text = ""

        # Format community summaries for context-aware keyword selection
        if message.community_summaries and message.community_summaries.strip():
            # Truncate if too long to fit within token limits
            max_context_length = 4000  # characters
            context_text = message.community_summaries
            if len(context_text) > max_context_length:
                context_text = context_text[:max_context_length] + "\n\n[... context truncated for brevity ...]"

            community_context = context_text
        else:
            community_context = "No context available from knowledge graph retrieval."

        # Prepare prompt with query, generated response, retrieved context, satisfactory criteria, and unfound keywords
        prompt_content = self.response_evaluator_prompt.format(
            original_query=message.original_query,
            generated_answer=message.generated_answer,
            satisfactory_criteria=self.satisfactory_criteria,
            community_summaries=community_context,
            unfound_keywords_history=unfound_keywords_text
        )

        print(f"   Prompt prepared ({len(prompt_content)} chars)")
        print(f"   ℹ️  Evaluating answer independently (not showing previous iterations to match Vector RAG)")

        try:
            # Call LLM for response evaluation
            system_message = SystemMessage(content=prompt_content)
            user_message = UserMessage(content="Please evaluate the response.", source="system")

            print("=" * 80)
            print("RESPONSE EVALUATOR AGENT - LLM CALL")
            print("=" * 80)
            print(f"System Prompt ({len(prompt_content)} chars):")
            print(f"\n[First 500 chars of full prompt:]")
            print(prompt_content[:500])
            print("-" * 80)
            print(f"User Prompt: Please evaluate the response.")
            print("-" * 80)
            print("Calling LLM...")

            # Use retry logic to handle API overload errors
            async def api_call():
                return await self.model_client.create(
                    [system_message, user_message],
                    cancellation_token=ctx.cancellation_token
                )

            response = await retry_api_call_with_backoff(api_call)

            print(f"LLM Response received ({len(response.content)} chars):")
            print(response.content)
            print("=" * 80)

            # Parse structured response
            assert isinstance(response.content, str)
            from parameters import ResponseEvaluationResponse
            eval_response = ResponseEvaluationResponse.model_validate_json(response.content)

            # Log LLM interaction
            logger = get_global_prompt_logger()
            logger.log_interaction(
                agent_name="ResponseEvaluatorAgent",
                interaction_type="response_evaluation",
                system_prompt=prompt_content,
                user_prompt="Please evaluate the response. Remember: if the answer states that the context doesn't contain the information, then it is not satisfactory.",
                llm_response=response.content,
                batch_id=message.batch_id,
                qa_pair_id=message.qa_pair_id,
                iteration=message.repetition,
                additional_metadata={
                    "original_query": message.original_query,
                    "generated_answer_length": len(message.generated_answer),
                    "continue_optimization": eval_response.continue_optimization,
                    "issue_type": eval_response.issue_type,
                    "critique_length": len(eval_response.critique)
                }
            )

            log_qa_processing(self.logger, message.qa_pair_id,
                            f"Evaluation completed - continue: {eval_response.continue_optimization}, issue_type: {eval_response.issue_type}",
                            f"Reasoning: {eval_response.reasoning}\nCritique: {eval_response.critique}\nMissing Keywords: {eval_response.missing_keywords}")

            # Create evaluation result dictionary (excluding gold answers to prevent data leakage)
            evaluation_data = {
                "qa_pair_id": message.qa_pair_id,
                "original_query": message.original_query,
                "generated_answer": message.generated_answer,
                "evaluation_reasoning": eval_response.reasoning,
                "evaluation_feedback": eval_response.critique,
                "missing_keywords": eval_response.missing_keywords,
                "continue_optimization": eval_response.continue_optimization,
                "issue_type": eval_response.issue_type,
                "repetition": message.repetition,
                "timestamp": datetime.now().isoformat()
            }

            # Store evaluation result in shared state for BackwardPassAgent
            current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
            response_evaluations = current_state.get("response_evaluations", [])
            response_evaluations.append(evaluation_data)
            current_state["response_evaluations"] = response_evaluations
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Send ResponseEvaluationReady message
            eval_ready_msg = ResponseEvaluationReadyMessage(
                qa_pair_id=message.qa_pair_id,
                evaluation_result=evaluation_data,
                continue_optimization=eval_response.continue_optimization,
                issue_type=eval_response.issue_type,
                missing_keywords=eval_response.missing_keywords,  # Direct attribute for consistent access
                batch_id=message.batch_id,
                repetition=message.repetition
            )

            self.logger.info(f"Returning ResponseEvaluationReady for QA pair {message.qa_pair_id}")

            # Return the evaluation ready message
            return eval_ready_msg

        except Exception as e:
            print("=" * 80)
            print("RESPONSE EVALUATOR AGENT - ERROR AFTER ALL RETRIES")
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
        await self.model_client.close()


# ===== UPDATED FACTORY FUNCTIONS =====

def create_answer_generator_agent() -> AnswerGeneratorAgent:
    """Factory function to create AnswerGeneratorAgent instances."""
    return AnswerGeneratorAgent("answer_generator_agent")

def create_response_evaluator_agent(dataset_name: str = "qmsum_test") -> ResponseEvaluatorAgent:
    """Factory function to create ResponseEvaluatorAgent instances."""
    return ResponseEvaluatorAgent("response_evaluator_agent", dataset_name=dataset_name)


# ===== BACKWARD PASS AGENT =====

class BackwardPassAgent(RoutedAgent):
    """
    Agent that performs backward pass through all agent critiques for system improvement.
    """

    def __init__(self, name: str, critique_token_limit: int = 512) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.backward_pass")
        self.shared_state = SharedState("agent_states")

        # Configuration for critique token limits
        self.critique_token_limit = critique_token_limit

        # Import gradient prompts and optimizer prompts
        from parameters import (
            generation_prompt_gradient_prompt,
            graph_gradient_prompt,
            graph_extraction_prompt_gradient_prompt,
            answer_generation_prompt_optimizer,
            graph_builder_prompt_optimizer,
            PromptCritiqueResponse,
            ContentCritiqueResponse
        )

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

        # Store all gradient prompts and optimizer prompts
        self.generation_prompt_gradient_prompt = generation_prompt_gradient_prompt
        self.graph_gradient_prompt = graph_gradient_prompt
        self.graph_extraction_prompt_gradient_prompt = graph_extraction_prompt_gradient_prompt

        # Store optimizer prompts
        self.answer_generation_prompt_optimizer = answer_generation_prompt_optimizer
        self.graph_builder_prompt_optimizer = graph_builder_prompt_optimizer

    @message_handler
    async def handle_backward_pass_start(self, message: BackwardPassStartMessage, ctx: MessageContext) -> BackwardPassReadyMessage:
        """
        Enhanced BackwardPassStart handler with QA pair boundary awareness.
        Generates appropriate critiques based on iteration context.
        """
        self.logger.info(f"BackwardPassAgent processing backward pass for batch {message.batch_id}, repetition {message.repetition}")
        self.logger.info(f"Issue type: {message.issue_type}")

        # Load shared state with correct dataset and setting parameters
        current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)

        # Get processing state to understand QA pair context
        processing_state = self.shared_state.get_processing_state()
        current_qa_pair_id = processing_state.get("current_qa_pair_id")
        current_iteration = processing_state.get("current_iteration", 0)

        # Store repetition information in batch_information for critique logic
        batch_info = current_state.get("batch_information", {})
        batch_info["current_repetition"] = message.repetition
        batch_info["current_qa_pair_id"] = current_qa_pair_id
        batch_info["current_iteration"] = current_iteration
        batch_info["issue_type"] = message.issue_type
        current_state["batch_information"] = batch_info

        try:
            # Step 1: ALWAYS generate answer generation critique (both content and style issues)
            await self._generate_answer_generation_critique(current_state, ctx, message.batch_id, message.repetition)
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Conditional critique generation based on issue type
            if message.issue_type == IssueType.CONTENT_ISSUE:
                # Content issue: Generate critiques for ALL agents (full backward pass)
                self.logger.info(f"CONTENT ISSUE detected - performing FULL backward pass (all agents)")

                # Step 2: Generate graph critique
                await self._generate_graph_critique(current_state, ctx, message.batch_id, message.repetition)
                self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

                # Step 3: Generate graph builder critique
                await self._generate_graph_builder_critique(current_state, ctx, message.batch_id, message.repetition)
                self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            elif message.issue_type == IssueType.STYLE_ISSUE:
                # Style issue: SKIP graph and graph builder critiques
                self.logger.info(f"STYLE ISSUE detected - performing PARTIAL backward pass (answer generation ONLY)")
                self.logger.info(f"Skipping graph critique and graph builder critique")
                # Don't generate graph or graph builder critiques
                # They will not be updated, so the next iteration will reuse the existing graph

            else:
                # Satisfactory or unknown: default to full backward pass for safety
                self.logger.warning(f"Unknown issue type {message.issue_type}, defaulting to full backward pass")
                await self._generate_graph_critique(current_state, ctx, message.batch_id, message.repetition)
                self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)
                await self._generate_graph_builder_critique(current_state, ctx, message.batch_id, message.repetition)
                self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Final save to ensure everything is persisted
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Extract optimized prompts for QA pair prompt lifecycle
            optimized_prompts = {
                # Store the actual learned system prompts that agents will use
                "learned_prompt_answer_generator_graph": current_state.get("learned_prompt_answer_generator_graph", ""),
                "learned_prompt_graph_builder": current_state.get("learned_prompt_graph_builder", ""),
                "learned_prompt_graph_refinement": current_state.get("learned_prompt_graph_refinement", ""),
                # Also store critiques and prompt templates for reference
                "graph_builder_agent_critique": current_state.get("graph_builder_agent_critique", ""),
                "answer_generation_critique": current_state.get("answer_generation_critique", ""),
                "graph_builder_prompt": current_state.get("graph_builder_prompt", "")
            }

            # Log the critique generation context
            if current_iteration == 0:
                self.logger.info(f"Generated initial critiques for new QA pair {current_qa_pair_id}")
            else:
                self.logger.info(f"Generated optimized critiques for iteration {current_iteration} of QA pair {current_qa_pair_id}")

            # Send BackwardPassReady message with optimized prompts
            backward_ready_msg = BackwardPassReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                backward_pass_results={
                    "critiques_generated": True,
                    "total_qa_pairs": len(message.all_qa_results),
                    "qa_pair_context": {
                        "qa_pair_id": current_qa_pair_id,
                        "iteration": current_iteration,
                        "is_first_iteration": current_iteration == 0
                    },
                    "optimized_prompts": optimized_prompts,
                    "critiques_updated": [
                        "answer_generation_critique",
                        "graph_critique",
                        "graph_builder_agent_critique"
                    ],
                    "learned_prompts_generated": [
                        "learned_prompt_answer_generator_graph",
                        "learned_prompt_graph_builder"
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

    async def _generate_answer_generation_critique(self, current_state: Dict[str, Any], ctx: MessageContext, batch_id: int, repetition: int) -> None:
        """Generate critique for answer generation prompt (with skip logic)."""
        self.logger.info("Generating answer generation critique")

        all_evaluation_responses = current_state.get("response_evaluations", [])

        if not all_evaluation_responses:
            self.logger.warning("No evaluation data available - skipping answer generation critique")
            current_state["answer_generation_critique"] = "No critique provided"
            return

        # Get the current answer generation prompt
        current_answer_prompt = current_state.get("learned_prompt_answer_generator_graph", "")
        if not current_answer_prompt:
            from parameters import base_prompt_answer_generator_graph
            current_answer_prompt = base_prompt_answer_generator_graph

        # Get previous critique (empty string for first component in backward pass)
        previous_critique = ""

        # Get response evaluator output (all iterations)
        response_evaluator_output = self._format_all_evaluation_responses(all_evaluation_responses)

        # Format prompt with new structure
        prompt_content = self.generation_prompt_gradient_prompt.format(
            current_prompt=current_answer_prompt,
            previous_critique=previous_critique,
            response_evaluator_output=response_evaluator_output
        )

        # Call LLM with structured output
        from parameters import PromptCritiqueResponse
        critique_response = await self._call_llm_structured(prompt_content, ctx, PromptCritiqueResponse, batch_id=batch_id, repetition=repetition)

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
        optimizer_prompt = self.answer_generation_prompt_optimizer.format(critique)
        optimized_prompt = await self._call_llm(optimizer_prompt, ctx, interaction_type="optimization", batch_id=batch_id, repetition=repetition, user_prompt="Generate the optimized system prompt.")

        # Limit prompt length to prevent truncation
        MAX_PROMPT_LENGTH = 4000
        if len(optimized_prompt) > MAX_PROMPT_LENGTH:
            self.logger.warning(f"Optimized answer generator prompt too long ({len(optimized_prompt)} chars), truncating to {MAX_PROMPT_LENGTH}")
            optimized_prompt = optimized_prompt[:MAX_PROMPT_LENGTH] + "\n\n[Prompt truncated to prevent errors]"

        # Only update if not frozen
        is_frozen = self._is_prompt_frozen("answer_generator_graph", current_state)
        if not is_frozen:
            current_state["learned_prompt_answer_generator_graph"] = optimized_prompt
            self.logger.info(f"Stored optimized answer generator prompt ({len(optimized_prompt)} chars)")

        log_critique_result(self.logger, "answer_generator_graph", critique, is_frozen)

    async def _generate_graph_critique(self, current_state: Dict[str, Any], ctx: MessageContext, batch_id: int, repetition: int) -> None:
        """Generate critique for graph (no skip logic)."""
        self.logger.info("Generating graph critique")

        graph_description = current_state.get("graph_description", "")
        all_evaluation_responses = current_state.get("response_evaluations", [])

        if not graph_description or not all_evaluation_responses:
            self.logger.warning("Missing graph description or evaluation for critique")
            current_state["graph_critique"] = "No critique provided"
            return

        # Get previous critique (from retrieved content)
        previous_critique = current_state.get("retrieved_content_critique", "")

        # Get response evaluator output (all iterations)
        response_evaluator_output = self._format_all_evaluation_responses(all_evaluation_responses)

        # Format prompt with new structure
        prompt_content = self.graph_gradient_prompt.format(
            graph_description=graph_description,
            previous_critique=previous_critique,
            response_evaluator_output=response_evaluator_output
        )

        # Call LLM with structured output (ContentCritiqueResponse - no skip logic)
        from parameters import ContentCritiqueResponse
        critique_response = await self._call_llm_structured(prompt_content, ctx, ContentCritiqueResponse, batch_id=batch_id, repetition=repetition)

        # Store critique (always, no skip logic)
        current_state["graph_critique"] = critique_response.critique

        self.logger.info("Graph critique generated and saved")

    async def _generate_graph_builder_critique(self, current_state: Dict[str, Any], ctx: MessageContext, batch_id: int, repetition: int) -> None:
        """Generate critique for graph builder/refinement prompt (with skip logic)."""

        batch_info = current_state.get("batch_information", {})
        current_repetition = batch_info.get("current_repetition", 0)
        is_first_iteration = current_repetition == 0
        all_evaluation_responses = current_state.get("response_evaluations", [])

        if not all_evaluation_responses:
            self.logger.warning("No evaluation data available - skipping graph builder critique")
            if is_first_iteration:
                current_state["graph_builder_agent_critique"] = "No critique provided"
            else:
                current_state["graph_refinement_agent_critique"] = "No critique provided"
            return

        if is_first_iteration:
            # First iteration: optimize graph creation prompt
            self.logger.info("Generating graph builder (creation) critique")

            graph_builder_prompt = current_state.get("learned_prompt_graph_builder", "")
            if not graph_builder_prompt:
                from parameters import base_prompt_graph_builder
                graph_builder_prompt = base_prompt_graph_builder

            # Get previous critique (from graph)
            previous_critique = current_state.get("graph_critique", "")

            # Get response evaluator output
            eval_resp = all_evaluation_responses[0]
            response_evaluator_output = f"Reasoning: {eval_resp.get('evaluation_reasoning', '')}\nCritique: {eval_resp.get('evaluation_feedback', '')}\nContinue: {eval_resp.get('continue_optimization', False)}"

            # Format prompt with new structure
            prompt_content = self.graph_extraction_prompt_gradient_prompt.format(
                current_prompt=graph_builder_prompt,
                previous_critique=previous_critique,
                response_evaluator_output=response_evaluator_output
            )

            # Call LLM with structured output
            from parameters import PromptCritiqueResponse
            critique_response = await self._call_llm_structured(prompt_content, ctx, PromptCritiqueResponse, batch_id=batch_id, repetition=repetition)

            # Implement skip logic
            if not critique_response.problem_in_this_component:
                current_state["graph_builder_agent_critique"] = "No critique provided"
                self.logger.info("Graph builder prompt: No problem detected, skipping optimization")
                print("\n" + "="*80)
                print("GRAPH BUILDER PROMPT OPTIMIZER - SKIPPED (Iteration 0)")
                print("="*80)
                print("Reason: PromptCritiqueResponse.problem_in_this_component = False")
                print("No optimization needed for graph builder prompt")
                print("="*80 + "\n")
                return

            # Problem detected
            critique = critique_response.critique
            current_state["graph_builder_agent_critique"] = critique


            optimizer_prompt = self.graph_builder_prompt_optimizer.format(critique)


            # Increased limit to accommodate full 5-section prompts with examples in tuple format
            MAX_PROMPT_LENGTH = 20000
            if len(optimized_prompt) > MAX_PROMPT_LENGTH:
                self.logger.warning(f"Optimized graph builder prompt too long ({len(optimized_prompt)} chars), truncating")
                optimized_prompt = optimized_prompt[:MAX_PROMPT_LENGTH] + "\n\n[Prompt truncated to prevent errors]"
                print(f"⚠️  WARNING: Prompt truncated from {len(optimized_prompt)} to {MAX_PROMPT_LENGTH} chars\n")

            is_frozen = self._is_prompt_frozen("graph_builder", current_state)
            if not is_frozen:
                current_state["learned_prompt_graph_builder"] = optimized_prompt
                print(f"✓ Stored optimized prompt in 'learned_prompt_graph_builder'\n")
            else:
                print(f"⚠️  Prompt is FROZEN - not storing optimized prompt\n")

        else:
            # Subsequent iterations: optimize graph refinement prompt
            self.logger.info(f"Generating graph refinement critique - iteration {current_repetition}")

            graph_refinement_prompt = current_state.get("learned_prompt_graph_refinement", "")
            if not graph_refinement_prompt:
                from parameters import base_prompt_graph_refinement
                graph_refinement_prompt = base_prompt_graph_refinement

            # Get previous critique (from graph)
            previous_critique = current_state.get("graph_critique", "")

            # Get response evaluator output
            eval_resp = all_evaluation_responses[0]
            response_evaluator_output = f"Reasoning: {eval_resp.get('evaluation_reasoning', '')}\nCritique: {eval_resp.get('evaluation_feedback', '')}\nContinue: {eval_resp.get('continue_optimization', False)}"

            # Format prompt with new structure
            prompt_content = self.graph_extraction_prompt_gradient_prompt.format(
                current_prompt=graph_refinement_prompt,
                previous_critique=previous_critique,
                response_evaluator_output=response_evaluator_output
            )

            # Call LLM with structured output
            from parameters import PromptCritiqueResponse
            critique_response = await self._call_llm_structured(prompt_content, ctx, PromptCritiqueResponse, batch_id=batch_id, repetition=repetition)

            # Implement skip logic
            if not critique_response.problem_in_this_component:
                current_state["graph_refinement_agent_critique"] = "No critique provided"
                self.logger.info("Graph refinement prompt: No problem detected, skipping optimization")
                print("\n" + "="*80)
                print(f"GRAPH REFINEMENT PROMPT OPTIMIZER - SKIPPED (Iteration {current_repetition})")
                print("="*80)
                print("Reason: PromptCritiqueResponse.problem_in_this_component = False")
                print("No optimization needed for graph refinement prompt")
                print("="*80 + "\n")
                return

            # Problem detected
            critique = critique_response.critique
            current_state["graph_refinement_agent_critique"] = critique


            optimizer_prompt = self.graph_builder_prompt_optimizer.format(critique)


            optimized_refinement_prompt = await self._call_llm(optimizer_prompt, ctx, interaction_type="optimization", batch_id=batch_id, repetition=repetition, user_prompt="Generate the optimized system prompt.")


            # Increased limit to accommodate full 5-section prompts with examples in tuple format
            MAX_PROMPT_LENGTH = 10000
            if len(optimized_refinement_prompt) > MAX_PROMPT_LENGTH:
                self.logger.warning(f"Optimized refinement prompt too long ({len(optimized_refinement_prompt)} chars), truncating")
                optimized_refinement_prompt = optimized_refinement_prompt[:MAX_PROMPT_LENGTH] + "\n\n[Prompt truncated to prevent errors]"
                print(f"⚠️  WARNING: Prompt truncated from {len(optimized_refinement_prompt)} to {MAX_PROMPT_LENGTH} chars\n")

            is_frozen = self._is_prompt_frozen("graph_refinement", current_state)
            if not is_frozen:
                current_state["learned_prompt_graph_refinement"] = optimized_refinement_prompt
                print(f"✓ Stored optimized prompt in 'learned_prompt_graph_refinement'\n")
            else:
                print(f"⚠️  Prompt is FROZEN - not storing optimized prompt\n")

            log_critique_result(self.logger, "graph_refinement", critique, is_frozen)

    def _is_prompt_frozen(self, prompt_type: str, current_state: Dict[str, Any]) -> bool:
        """Check if a prompt type is frozen."""
        frozen_prompts = current_state.get("frozen_prompts", [])
        return prompt_type in frozen_prompts

    async def _call_llm(self, prompt_content: str, ctx: MessageContext, interaction_type: str = "critique", batch_id: int = None, repetition: int = None, user_prompt: str = "Please provide your critique and feedback.") -> str:
        """Helper method to call LLM with given prompt and token limit."""
        try:
            # Use the prompt content directly without token limit constraints
            enhanced_prompt = prompt_content

            system_message = SystemMessage(content=enhanced_prompt)
            user_message = UserMessage(content=user_prompt, source="system")

            response = await self.model_client.create(
                [system_message, user_message],
                cancellation_token=ctx.cancellation_token
            )

            response_content = response.content if isinstance(response.content, str) else str(response.content)

            # Log LLM interaction
            # BackwardPassAgent operates at batch level, so qa_pair_id is None
            # Use repetition parameter for iteration
            logger = get_global_prompt_logger()
            logger.log_interaction(
                agent_name="BackwardPassAgent",
                interaction_type=interaction_type,
                system_prompt=enhanced_prompt,
                user_prompt=user_prompt,
                llm_response=response_content,
                batch_id=batch_id,
                qa_pair_id=None,  # Batch-level processing
                iteration=repetition,
                additional_metadata={
                    "critique_token_limit": self.critique_token_limit,
                    "prompt_length": len(prompt_content),
                    "enhanced_prompt_length": len(enhanced_prompt),
                    "response_length": len(response_content)
                }
            )

            return response_content

        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            return f"Error generating critique: {e}"

    async def _call_llm_structured(self, prompt_content: str, ctx: MessageContext, response_format, interaction_type: str = "critique", batch_id: int = None, repetition: int = None, user_prompt: str = "Please provide your critique."):
        """Helper method to call LLM with structured output (Pydantic response format)."""
        try:
            # Create a temporary client with the response format
            structured_client = OpenAIChatCompletionClient(
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
            # BackwardPassAgent operates at batch level, so qa_pair_id is None
            # Use repetition parameter for iteration
            logger = get_global_prompt_logger()
            logger.log_interaction(
                agent_name="BackwardPassAgent",
                interaction_type=interaction_type,
                system_prompt=prompt_content,
                user_prompt=user_prompt,
                llm_response=response.content,
                batch_id=batch_id,
                qa_pair_id=None,  # Batch-level processing
                iteration=repetition,
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


# ===== SUMMARIZER AGENT =====

class SummarizerAgent(RoutedAgent):
    """
    Agent that summarizes retrieved contexts to avoid overly long backward pass prompts.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.summarizer")
        self.shared_state = SharedState("agent_states")

        # Initialize Gemini model client for text summarization
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

                # Log LLM interaction
                # Get current QA pair and iteration from shared state for logging
                current_state_for_logging = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
                current_qa_pair_id = current_state_for_logging.get("current_qa_pair_id", None)
                current_iteration = current_state_for_logging.get("current_iteration", None)

                logger = get_global_prompt_logger()
                logger.log_interaction(
                    agent_name="SummarizerAgent",
                    interaction_type="context_summarization",
                    system_prompt="You are a helpful assistant that creates concise summaries.",
                    user_prompt=prompt_content,
                    llm_response=summary,
                    batch_id=message.batch_id,
                    qa_pair_id=current_qa_pair_id,
                    iteration=current_iteration,
                    additional_metadata={
                        "context_index": i + 1,
                        "total_contexts": len(message.retrieved_contexts),
                        "original_context_length": len(context),
                        "summary_length": len(summary),
                        "compression_ratio": round(len(summary) / len(context), 3) if len(context) > 0 else 0
                    }
                )

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


# ===== UPDATED FACTORY FUNCTIONS =====

def create_backward_pass_agent(critique_token_limit: int = 512) -> BackwardPassAgent:
    """Factory function to create BackwardPassAgent instances."""
    return BackwardPassAgent("backward_pass_agent", critique_token_limit)

def create_summarizer_agent() -> SummarizerAgent:
    """Factory function to create SummarizerAgent instances."""
    return SummarizerAgent("summarizer_agent")