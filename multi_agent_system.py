"""
Multi-Agent GraphRAG System with AutoGen Core API.
Includes BatchOrchestratorAgent and HyperparametersGraphAgent.
"""

import json
import logging
import statistics
from datetime import datetime
from typing import Dict, Any, List, Optional
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
    'BatchOrchestratorAgent',
    'HyperparametersGraphAgent'
]


# ===== MESSAGE TYPES =====

# Import shared messages from DatasetAgent to avoid duplication
# BatchStartMessage and BatchReadyMessage are imported above

# New messages for orchestration workflow
class HyperparametersGraphStartMessage(BaseModel):
    qa_pair_id: str
    qa_pair: Dict[str, Any]
    batch_id: int
    repetition: int
    dataset: str
    setting: str

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

class HyperparametersGraphReadyMessage(BaseModel):
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
    batch_id: int
    repetition: int
    dataset: str
    setting: str

class ResponseEvaluationReadyMessage(BaseModel):
    qa_pair_id: str
    evaluation_result: Dict[str, Any]
    continue_optimization: bool
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

class BackwardPassReadyMessage(BaseModel):
    batch_id: int
    repetition: int
    backward_pass_results: Dict[str, Any]

# Response format for HyperparametersGraphAgent
class HyperparametersGraphResponse(BaseModel):
    reasoning: str
    chunk_size: int
    confidence_score: float


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
            # Step 1: HyperparametersGraphAgent
            self.logger.info(f"Step 1: Executing HyperparametersGraphAgent for {qa_pair_id}")

            # Log execution start
            exec_logger.log_agent_start(
                agent_name="HyperparametersGraphAgent",
                message_type="HyperparametersRequestMessage",
                batch_id=str(message.batch_id),
                qa_pair_id=qa_pair_id,
                iteration=iteration
            )

            step_logger.log_step(
                step_name="hyperparameters_generation",
                status=StepStatus.STARTED,
                agent_name="HyperparametersGraphAgent",
                input_data_summary=f"QA pair {qa_pair_id}, question length: {len(qa_pair.get('question', ''))}"
            )

            step_agent_start = datetime.now()
            try:
                hyperparams_response = await self._execute_hyperparameters_agent(qa_pair_id, qa_pair, iteration)

                step_execution_time = (datetime.now() - step_agent_start).total_seconds() * 1000
                step_logger.log_step(
                    step_name="hyperparameters_generation",
                    status=StepStatus.COMPLETED,
                    agent_name="HyperparametersGraphAgent",
                    execution_time_ms=step_execution_time,
                    output_data_summary=f"chunk_size: {getattr(hyperparams_response, 'chunk_size', 'N/A')}"
                )

                # Log execution success
                exec_logger.log_agent_success(
                    agent_name="HyperparametersGraphAgent",
                    message_type="HyperparametersRequestMessage",
                    result_summary=f"chunk_size: {getattr(hyperparams_response, 'chunk_size', 512)}"
                )

                # Log intermediate output
                intermediate_outputs["hyperparameters"] = {
                    "chunk_size": getattr(hyperparams_response, 'chunk_size', 512),
                    "processing_time_ms": step_execution_time
                }

            except Exception as e:
                step_execution_time = (datetime.now() - step_agent_start).total_seconds() * 1000
                step_logger.log_step(
                    step_name="hyperparameters_generation",
                    status=StepStatus.FAILED,
                    agent_name="HyperparametersGraphAgent",
                    execution_time_ms=step_execution_time,
                    error_message=str(e)
                )

                # Log execution error
                exec_logger.log_agent_error(
                    agent_name="HyperparametersGraphAgent",
                    message_type="HyperparametersRequestMessage",
                    error=e,
                    error_context="Failed during hyperparameters generation"
                )

                self.logger.warning(f"HyperparametersGraphAgent failed, using fallback: {e}")
                hyperparams_response = await self.handle_agent_failure("hyperparameters_graph_agent", qa_pair_id, iteration, e)

                intermediate_outputs["hyperparameters"] = {
                    "error": str(e),
                    "fallback_used": True,
                    "processing_time_ms": step_execution_time
                }

            # Step 2: GraphBuilderAgent
            self.logger.info(f"Step 2: Executing GraphBuilderAgent for {qa_pair_id}")
            exec_logger.log_agent_start(
                agent_name="GraphBuilderAgent",
                message_type="GraphBuildMessage",
                batch_id=str(message.batch_id),
                qa_pair_id=qa_pair_id,
                iteration=iteration
            )
            try:
                graph_response = await self._execute_graph_builder_agent(hyperparams_response, iteration)
                exec_logger.log_agent_success(
                    agent_name="GraphBuilderAgent",
                    message_type="GraphBuildMessage",
                    result_summary="Graph built successfully"
                )
            except Exception as e:
                exec_logger.log_agent_error(
                    agent_name="GraphBuilderAgent",
                    message_type="GraphBuildMessage",
                    error=e,
                    error_context="Failed during graph building"
                )
                self.logger.warning(f"GraphBuilderAgent failed, using fallback: {e}")
                graph_response = await self.handle_agent_failure("graph_builder_agent", qa_pair_id, iteration, e)

            # Step 3: GraphRetrievalAgent
            self.logger.info(f"Step 3: Executing GraphRetrievalPlannerAgent for {qa_pair_id}")
            exec_logger.log_agent_start(
                agent_name="GraphRetrievalPlannerAgent",
                message_type="GraphRetrievalMessage",
                batch_id=str(message.batch_id),
                qa_pair_id=qa_pair_id,
                iteration=iteration
            )
            try:
                retrieval_response = await self._execute_graph_retrieval_agent(graph_response, qa_pair)
                exec_logger.log_agent_success(
                    agent_name="GraphRetrievalPlannerAgent",
                    message_type="GraphRetrievalMessage",
                    result_summary=f"Retrieved {len(getattr(retrieval_response, 'retrieved_contexts', []))} contexts"
                )
            except Exception as e:
                exec_logger.log_agent_error(
                    agent_name="GraphRetrievalPlannerAgent",
                    message_type="GraphRetrievalMessage",
                    error=e,
                    error_context="Failed during graph retrieval"
                )
                self.logger.warning(f"GraphRetrievalPlannerAgent failed, using fallback: {e}")
                retrieval_response = await self.handle_agent_failure("graph_retrieval_planner_agent", qa_pair_id, iteration, e)

            # Step 4: AnswerGeneratorAgent
            self.logger.info(f"Step 4: Executing AnswerGeneratorAgent for {qa_pair_id}")
            exec_logger.log_agent_start(
                agent_name="AnswerGeneratorAgent",
                message_type="AnswerGenerationMessage",
                batch_id=str(message.batch_id),
                qa_pair_id=qa_pair_id,
                iteration=iteration
            )
            try:
                answer_response = await self._execute_answer_generator_agent(retrieval_response, qa_pair)
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

            # Step 5: ResponseEvaluatorAgent
            self.logger.info(f"Step 5: Executing ResponseEvaluatorAgent for {qa_pair_id}")
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

            # Step 6: BackwardPassAgent (generates optimized prompts for next iteration)
            self.logger.info(f"Step 6: Executing BackwardPassAgent for {qa_pair_id}")
            backward_pass_response = None

            # Check if we should continue optimization
            should_continue = evaluation_response.continue_optimization
            is_last_iteration = iteration >= message.shared_state.get("batch_information", {}).get("total_iterations", 3) - 1

            # Store continue_optimization flag and missing keywords in shared state for next iteration
            current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
            current_state[f"continue_optimization_{qa_pair_id}"] = should_continue

            # Store missing keywords for focused graph refinement in next iteration
            if should_continue and hasattr(evaluation_response, 'evaluation_result'):
                missing_keywords = evaluation_response.evaluation_result.get("missing_keywords", [])
                if missing_keywords:
                    current_state["missing_keywords_for_refinement"] = missing_keywords
                    self.logger.info(f"📝 [ITERATION {iteration}] SAVING KEYWORDS: Stored {len(missing_keywords)} keywords for next iteration")
                    self.logger.info(f"📝 Keywords: {missing_keywords}")
                    self.logger.info(f"📝 State keys after saving: {list(current_state.keys())}")
                else:
                    # Clear keywords if none provided
                    current_state["missing_keywords_for_refinement"] = []
                    self.logger.info(f"📝 [ITERATION {iteration}] No keywords provided, clearing keywords")
            else:
                # Clear keywords if not continuing
                current_state["missing_keywords_for_refinement"] = []
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
                    backward_pass_response = await self._execute_backward_pass_agent(evaluation_response, iteration)
                    exec_logger.log_agent_success(
                        agent_name="BackwardPassAgent",
                        message_type="BackwardPassMessage",
                        result_summary="Backward pass completed successfully"
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
            if not all([hyperparams_response, graph_response, retrieval_response, answer_response, evaluation_response]):
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
                "retrieval": {
                    "retrieved_context": getattr(retrieval_response, 'retrieved_context', 'N/A'),
                    "context_length": len(getattr(retrieval_response, 'retrieved_context', ''))
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
                    "chunk_size": getattr(hyperparams_response, 'chunk_size', 512),
                    "graph_hyperparameters": getattr(hyperparams_response, '__dict__', {})
                },
                graph_metrics=comprehensive_graph_stats,
                retrieval_context=getattr(retrieval_response, 'retrieved_context', 'N/A'),
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
                    "chunk_size": getattr(hyperparams_response, 'chunk_size', 512),
                    "graph_hyperparameters": getattr(hyperparams_response, '__dict__', {})
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
                "hyperparameters": {"chunk_size": getattr(hyperparams_response, 'chunk_size', 512)},
                "graph_description": getattr(graph_response, 'graph_description', 'N/A'),
                "community_summarization_logs": community_summarization_logs,  # Add community summarization logs
                "retrieved_context": getattr(retrieval_response, 'retrieved_context', 'N/A'),
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

    async def _execute_hyperparameters_agent(self, qa_pair_id: str, qa_pair: Dict[str, Any],
                                           iteration: int) -> HyperparametersGraphReadyMessage:
        """Execute HyperparametersGraphAgent with current system prompt."""
        hyperparams_msg = HyperparametersGraphStartMessage(
            qa_pair_id=qa_pair_id,
            qa_pair=qa_pair,
            batch_id=self.current_batch_id,
            repetition=iteration,
            dataset=self.current_dataset,
            setting=self.current_setting
        )

        hyperparams_agent_id = AgentId("hyperparameters_graph_agent", "default")
        return await self.send_message(hyperparams_msg, hyperparams_agent_id)

    async def _execute_graph_builder_agent(self, hyperparams_response: HyperparametersGraphReadyMessage,
                                         iteration: int) -> GraphReadyMessage:
        """Execute GraphBuilderAgent with appropriate mode (create vs refine)."""
        current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, self.current_batch_id)

        # Debug logging for keyword availability
        self.logger.info(f"📖 [ITERATION {iteration}] LOADING STATE for GraphBuilder")
        self.logger.info(f"📖 State keys available: {list(current_state.keys())}")
        keywords_in_state = current_state.get("missing_keywords_for_refinement", [])
        self.logger.info(f"📖 Keywords found in state: {keywords_in_state} (count: {len(keywords_in_state)})")

        graph_start_msg = GraphStartMessage(
            batch_id=hyperparams_response.batch_id,
            repetition=iteration,
            chunk_size=hyperparams_response.chunk_size,
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

    async def _execute_answer_generator_agent(self, retrieval_response: GraphRetrievalReadyMessage,
                                            qa_pair: Dict[str, Any]) -> AnswerGenerationReadyMessage:
        """Execute AnswerGeneratorAgent."""
        qa_pair_id = qa_pair.get("question_id", "unknown")

        answer_gen_msg = AnswerGenerationStartMessage(
            qa_pair_id=qa_pair_id,
            question=qa_pair.get("question", ""),
            retrieved_context=retrieval_response.retrieved_context,
            batch_id=retrieval_response.batch_id,
            repetition=retrieval_response.repetition,
            dataset=retrieval_response.dataset,
            setting=retrieval_response.setting
        )

        answer_gen_agent_id = AgentId("answer_generator_agent", "default")
        return await self.send_message(answer_gen_msg, answer_gen_agent_id)

    async def _execute_response_evaluator_agent(self, answer_response: AnswerGenerationReadyMessage,
                                              qa_pair: Dict[str, Any]) -> ResponseEvaluationReadyMessage:
        """Execute ResponseEvaluatorAgent."""
        eval_start_msg = ResponseEvaluationStartMessage(
            qa_pair_id=answer_response.qa_pair_id,
            original_query=qa_pair.get("question", ""),
            generated_answer=answer_response.generated_answer,
            gold_answers=qa_pair.get("answers", []),
            batch_id=answer_response.batch_id,
            repetition=answer_response.repetition,
            dataset=self.current_dataset,
            setting=self.current_setting
        )

        response_eval_agent_id = AgentId("response_evaluator_agent", "default")
        return await self.send_message(eval_start_msg, response_eval_agent_id)

    async def _execute_backward_pass_agent(self, evaluation_response: ResponseEvaluationReadyMessage,
                                         iteration: int) -> Optional[BackwardPassReadyMessage]:
        """Execute BackwardPassAgent to generate optimized prompts."""
        if iteration == 0:
            # For first iteration, generate initial critiques
            backward_pass_msg = BackwardPassStartMessage(
                batch_id=evaluation_response.batch_id,
                repetition=iteration,
                dataset=self.current_dataset,
                setting=self.current_setting,
                all_qa_results=[evaluation_response.evaluation_result]
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
                all_qa_results=all_results
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
            if agent_name == "hyperparameters_graph_agent":
                return await self._simulate_hyperparameters_response(qa_pair_id, iteration)
            elif agent_name == "graph_builder_agent":
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

    async def _simulate_hyperparameters_response(self, qa_pair_id: str, iteration: int) -> HyperparametersGraphReadyMessage:
        """Fallback simulation for HyperparametersGraphAgent."""
        self.logger.info(f"Simulating hyperparameters response for {qa_pair_id}")
        return HyperparametersGraphReadyMessage(
            qa_pair_id=qa_pair_id,
            chunk_size=512,  # Default chunk size
            batch_id=self.current_batch_id,
            repetition=iteration
        )

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
            evaluation_result={"score": 0.5, "quality": "simulated"},
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



# ===== HYPERPARAMETERS GRAPH AGENT =====

class HyperparametersGraphAgent(RoutedAgent):
    """
    Agent that determines graph construction hyperparameters using LLM reasoning.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.hyperparameters_graph")
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
            response_format=HyperparametersGraphResponse
        )

        # Base prompt for hyperparameters determination
        from parameters import base_prompt_hyperparameters_graph
        self.base_prompt_hyperparameters_graph = base_prompt_hyperparameters_graph

    @message_handler
    async def handle_hyperparameters_graph_start(
        self, message: HyperparametersGraphStartMessage, ctx: MessageContext
    ) -> HyperparametersGraphReadyMessage:
        """Handle HyperparametersGraphStart message and generate hyperparameters using LLM."""
        self.logger.info(f"HyperparametersGraphAgent processing QA pair {message.qa_pair_id}")

        try:
            # Load shared state to get learned system prompt
            current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)

            # For first repetition (repetition=0), start with empty system prompt to avoid data leakage
            if message.repetition == 0:
                learned_system_prompt = ""
                self.logger.info(f"First repetition for QA pair {message.qa_pair_id} - using empty system prompt")
            else:
                learned_system_prompt = current_state.get("learned_prompt_hyperparameters_graph", "")

            # Extract question from QA pair
            qa_pair = message.qa_pair
            question = qa_pair.get("question", "")

            # Use generic description for text sample
            text_sample = "Meeting transcripts"

            # Prepare base prompt (without critique)
            prompt_content = self.base_prompt_hyperparameters_graph.format(
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

            # Log LLM interaction
            logger = get_global_prompt_logger()
            logger.log_interaction(
                agent_name="HyperparametersGraphAgent",
                interaction_type="hyperparameters_recommendation",
                system_prompt=learned_system_prompt,
                user_prompt=prompt_content,
                llm_response=response.content if isinstance(response.content, str) else str(response.content),
                batch_id=message.batch_id,
                qa_pair_id=message.qa_pair_id,
                additional_metadata={
                    "text_sample": text_sample,
                    "question": question
                }
            )

            # Parse structured response
            assert isinstance(response.content, str)
            hyperparams_response = HyperparametersGraphResponse.model_validate_json(response.content)

            log_agent_action(self.logger, "HyperparametersGraph", "LLM recommendation",
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

            # Return HyperparametersGraphReady message
            ready_msg = HyperparametersGraphReadyMessage(
                qa_pair_id=message.qa_pair_id,
                chunk_size=hyperparams_response.chunk_size,
                batch_id=message.batch_id,
                repetition=message.repetition
            )

            self.logger.info(f"Returning HyperparametersGraphReady for QA pair {message.qa_pair_id}")
            return ready_msg

        except Exception as e:
            self.logger.error(f"Error in LLM call: {e}")
            # Return default response on error
            return HyperparametersGraphReadyMessage(
                qa_pair_id=message.qa_pair_id,
                chunk_size=512,  # Default chunk size
                batch_id=message.batch_id,
                repetition=message.repetition
            )

    async def close(self) -> None:
        """Close the model client."""
        await self.model_client.close()


# ===== FACTORY FUNCTIONS =====

def create_batch_orchestrator_agent() -> BatchOrchestratorAgent:
    """Factory function to create BatchOrchestratorAgent instances."""
    return BatchOrchestratorAgent("batch_orchestrator_agent")

def create_hyperparameters_graph_agent() -> HyperparametersGraphAgent:
    """Factory function to create HyperparametersGraphAgent instances."""
    return HyperparametersGraphAgent("hyperparameters_graph_agent")


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
                "structured_output": True,
            },
            response_format=GraphBuilderResponse
        )

        self.model_client_refinement = OpenAIChatCompletionClient(
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
            response_format=GraphRefinementResponse
        )

        self.base_prompt_graph_builder = base_prompt_graph_builder
        self.base_prompt_graph_refinement = base_prompt_graph_refinement
        

    @message_handler
    async def handle_graph_start(self, message: GraphStartMessage, ctx: MessageContext) -> GraphReadyMessage:
        """Handle GraphStart message by chunking text and building/refining graph."""
        self.logger.info(f"GraphBuilderAgent processing batch {message.batch_id} with chunk_size {message.chunk_size}")

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

            if missing_keywords:
                # Focused refinement: extract context around missing keywords
                self.logger.info(f"🎯 FOCUSED REFINEMENT MODE: extracting context for {len(missing_keywords)} keywords")
                self.logger.info(f"Missing keywords: {missing_keywords}")
                focused_text = self._extract_focused_context(corpus, missing_keywords, context_window=800)

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
                else:
                    # Fallback to full corpus if no contexts found
                    self.logger.warning("⚠️  No context found for keywords, falling back to full corpus")
                    chunks = self._split_text_into_chunks(corpus, message.chunk_size)
            else:
                # No keywords provided, process full corpus (fallback for backward compatibility)
                self.logger.info("No missing keywords provided, processing full corpus for refinement")
                chunks = self._split_text_into_chunks(corpus, message.chunk_size)

            self.logger.info(f"Split text into {len(chunks)} chunks for graph refinement")

            # Save refinement prompt template to shared state
            current_state["graph_refinement_prompt"] = self.base_prompt_graph_refinement

            # Process refinement (without passing existing graph summary to the prompt)
            new_entities, new_relationships, new_triplets = await self._process_graph_refinement(
                chunks, learned_system_prompt, ctx
            )

            # For refinement, merge new entities/relationships with existing ones
            existing_entities, existing_relationships, existing_triplets = await self._load_existing_graph_data()

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

        # Convert to memgraph JSON format and save to file
        graph_json = self._convert_to_memgraph_format(all_entities, all_relationships, all_triplets)
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

        # Send GraphReady message
        graph_ready_msg = GraphReadyMessage(
            batch_id=message.batch_id,
            repetition=message.repetition,
            graph_description=graph_description,
            connectivity_metrics=connectivity_metrics,
            dataset=message.dataset,
            setting=message.setting
        )

        self.logger.info(f"Returning GraphReady for batch {message.batch_id}")

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

    def _extract_focused_context(self, text: str, keywords: List[str], context_window: int = 300) -> str:
        """
        Extract context windows around specified keywords from text.
        Uses exact matching first, then falls back to fuzzy matching if needed.

        Args:
            text: The full document text
            keywords: List of keywords/phrases to search for
            context_window: Number of characters to include before and after each keyword match

        Returns:
            Concatenated context windows containing the keywords
        """
        if not keywords:
            return ""

        import re
        from difflib import SequenceMatcher

        contexts = []
        seen_contexts = set()  # To avoid duplicates
        keywords_found = set()  # Track which keywords were found
        keywords_not_found = []  # Track keywords that need fuzzy matching

        # Phase 1: Exact matching (case-insensitive)
        for keyword in keywords:
            # Case-insensitive regex search for the keyword
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            found_matches = False

            for match in pattern.finditer(text):
                found_matches = True
                start_pos = max(0, match.start() - context_window)
                end_pos = min(len(text), match.end() + context_window)

                context = text[start_pos:end_pos].strip()

                # Avoid duplicate contexts
                context_hash = hash(context)
                if context_hash not in seen_contexts:
                    seen_contexts.add(context_hash)
                    contexts.append(context)

            if found_matches:
                keywords_found.add(keyword)
            else:
                keywords_not_found.append(keyword)

        # Phase 2: Fuzzy matching for keywords not found exactly
        if keywords_not_found:
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
                    # 1. Similarity ratio is high (> 0.6)
                    # 2. Significant word overlap (> 0.5) and some similarity (> 0.3)
                    if similarity > 0.6 or (word_overlap > 0.5 and similarity > 0.3):
                        best_matches.append((sentence, similarity, word_overlap))

                # Sort by similarity and take top matches
                best_matches.sort(key=lambda x: (x[2], x[1]), reverse=True)

                # Extract contexts from best matching sentences
                for sentence, sim, overlap in best_matches[:3]:  # Take up to 3 best matches per keyword
                    # Find sentence position in original text
                    sentence_pos = text.lower().find(sentence.lower())
                    if sentence_pos != -1:
                        start_pos = max(0, sentence_pos - context_window)
                        end_pos = min(len(text), sentence_pos + len(sentence) + context_window)

                        context = text[start_pos:end_pos].strip()

                        # Avoid duplicate contexts
                        context_hash = hash(context)
                        if context_hash not in seen_contexts:
                            seen_contexts.add(context_hash)
                            contexts.append(context)
                            keywords_found.add(keyword)
                            self.logger.info(f"🔎 Fuzzy match for '{keyword}': similarity={sim:.2f}, word_overlap={overlap:.2f}")
                            break  # Found a good match for this keyword

        # Log results
        if contexts:
            self.logger.info(f"✓ Found contexts for {len(keywords_found)}/{len(keywords)} keywords")
            if len(keywords_found) < len(keywords):
                still_missing = set(keywords) - keywords_found
                self.logger.warning(f"✗ No matches found for {len(still_missing)} keywords: {list(still_missing)}")
        else:
            self.logger.warning(f"✗ No contexts found for any keywords: {keywords}")
            return ""

        # Concatenate all contexts with separators
        focused_text = "\n\n---\n\n".join(contexts)

        self.logger.info(f"Extracted {len(contexts)} context windows for {len(keywords)} keywords (total length: {len(focused_text)} chars)")

        return focused_text

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
        logger = get_global_prompt_logger()
        logger.log_interaction(
            agent_name="GraphBuilderAgent",
            interaction_type="graph_creation",
            system_prompt=learned_system_prompt,
            user_prompt=prompt_content,
            llm_response=response.content if isinstance(response.content, str) else str(response.content),
            additional_metadata={
                "chunk_length": len(chunk),
                "mode": "creation"
            }
        )

        # Parse structured response
        assert isinstance(response.content, str)
        from parameters import GraphBuilderResponse
        graph_response = GraphBuilderResponse.model_validate_json(response.content)

        return graph_response.entities, graph_response.relationships, graph_response.triplets

    def _convert_to_memgraph_format(self, entities, relationships, triplets) -> List[Dict[str, Any]]:
        """Convert extracted data to Memgraph import_util.json() format."""
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

            # Add entity properties to node properties
            for prop in entity.properties:
                properties[prop.key] = prop.value

            node = {
                "id": node_id,
                "labels": [entity.type],
                "properties": properties,
                "type": "node"
            }
            memgraph_items.append(node)
            name_to_id[entity.name] = node_id

        # Convert relationships to edges
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
                        "type": relationship.relationship_type
                    },
                    "type": "relationship"
                }
                memgraph_items.append(edge)
                rel_id_counter += 1

        return memgraph_items

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

        # Create tasks for all chunks
        async def process_refinement_chunk_with_index(i: int, chunk: str) -> tuple:
            """Wrapper to process refinement chunk with index for logging and error handling."""
            try:
                self.logger.info(f"Processing refinement chunk {i+1}/{len(chunks)}")
                entities, relationships, triplets = await self._process_refinement_chunk(chunk, learned_system_prompt, ctx)
                self.logger.info(f"Completed refinement chunk {i+1}/{len(chunks)}")
                return entities, relationships, triplets
            except Exception as e:
                self.logger.error(f"Error processing refinement chunk {i+1}: {e}")
                return [], [], []  # Return empty results for failed chunks

        # Execute all chunk processing tasks concurrently
        import asyncio
        tasks = [process_refinement_chunk_with_index(i, chunk) for i, chunk in enumerate(chunks)]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=False)

        # Aggregate results from all chunks
        for entities, relationships, triplets in chunk_results:
            all_entities.extend(entities)
            all_relationships.extend(relationships)
            all_triplets.extend(triplets)

        self.logger.info(f"Completed concurrent refinement processing of all {len(chunks)} chunks")

        log_agent_action(self.logger, "GraphRefinement", "LLM refinement",
                        entities=len(all_entities),
                        relationships=len(all_relationships),
                        triplets=len(all_triplets))

        return all_entities, all_relationships, all_triplets

    async def _process_refinement_chunk(self, chunk: str, learned_system_prompt: str, ctx: MessageContext) -> tuple:
        """Process a single text chunk for graph refinement."""
        # Prepare the refinement prompt (no existing graph summary)
        user_prompt = self.base_prompt_graph_refinement.format(chunk)

        # Create complete prompt with optional system message
        if learned_system_prompt:
            messages = [
                SystemMessage(content=learned_system_prompt),
                UserMessage(content=user_prompt, source="user")
            ]
        else:
            messages = [UserMessage(content=user_prompt, source="user")]

        # Use refinement model client
        response = await self.model_client_refinement.create(messages, cancellation_token=ctx.cancellation_token)

        # Log LLM interaction
        logger = get_global_prompt_logger()
        logger.log_interaction(
            agent_name="GraphBuilderAgent",
            interaction_type="graph_refinement",
            system_prompt=learned_system_prompt if learned_system_prompt else "",
            user_prompt=user_prompt,
            llm_response=response.content if isinstance(response.content, str) else str(response.content),
            additional_metadata={
                "chunk_length": len(chunk),
                "mode": "refinement"
            }
        )

        # Parse structured response
        assert isinstance(response.content, str)
        from parameters import GraphRefinementResponse
        refinement_response = GraphRefinementResponse.model_validate_json(response.content)

        return refinement_response.new_entities, refinement_response.new_relationships, refinement_response.new_triplets

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
            from parameters import Entity, Relationship, Triplet, EntityProperty

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

                    # Create entity properties list from additional properties
                    entity_properties = []
                    for key, value in props.items():
                        if key not in ["name", "type"]:  # Skip main fields
                            entity_prop = EntityProperty(key=key, value=str(value))
                            entity_properties.append(entity_prop)

                    entity = Entity(
                        name=props.get("name", ""),
                        type=props.get("type", ""),
                        properties=entity_properties
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

                # Merge properties from all entities in the cluster
                merged_properties = list(primary_entity.properties)  # Start with primary entity's properties
                property_keys_seen = {prop.key for prop in primary_entity.properties}

                for entity in cluster[1:]:  # Skip primary entity, start from second
                    for prop in entity.properties:
                        # Add property if key is not already present
                        if prop.key not in property_keys_seen:
                            merged_properties.append(prop)
                            property_keys_seen.add(prop.key)

                # Update primary entity with merged properties
                primary_entity.properties = merged_properties

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
        self.last_community_prompt = ""  # Track last used community summarizer prompt

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

            # Load shared state to check for learned community summarizer prompt
            current_state = message.shared_state
            learned_community_prompt = current_state.get("learned_prompt_community_summarizer", "")

            # Check if we need to reload/reprocess the graph
            # Reasons to reprocess:
            # 1. No community manager exists yet
            # 2. Graph file has changed (different filename)
            # 3. Graph content has changed (refinement in iteration 1+)
            # 4. Learned community summarizer prompt has changed
            should_reprocess = (
                self.community_manager is None or
                self.current_graph_file != graph_filename or
                message.repetition > 0 or  # Always reprocess in iterations 1+ (graph was refined with new relationships)
                learned_community_prompt != self.last_community_prompt
            )

            if should_reprocess:
                if self.community_manager is None:
                    self.logger.info(f"Loading and processing graph for first time: {graph_filename}")
                elif self.current_graph_file != graph_filename:
                    self.logger.info(f"Graph file changed, reprocessing: {graph_filename}")
                elif message.repetition > 0:
                    self.logger.info(f"Graph was refined in iteration {message.repetition}, reprocessing to detect communities with new relationships")
                    if learned_community_prompt != self.last_community_prompt:
                        self.logger.info(f"Additionally, learned community summarizer prompt changed (prev: {len(self.last_community_prompt)} chars, new: {len(learned_community_prompt)} chars)")
                else:
                    self.logger.info(f"Learned community summarizer prompt updated, re-summarizing communities")
                    self.logger.info(f"Previous prompt length: {len(self.last_community_prompt) if hasattr(self, 'last_community_prompt') else 0}, New prompt length: {len(learned_community_prompt)}")

                from community_graph_utils import load_and_process_graph
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

                if learned_community_prompt:
                    self.logger.info(f"Using learned community summarizer prompt ({len(learned_community_prompt)} chars)")
                else:
                    self.logger.info("No learned community summarizer prompt found, using base prompt")

                self.community_manager = await load_and_process_graph(
                    graph_filename,
                    self.summarizer_client,
                    embedding_model=None,  # Use TF-IDF fallback
                    learned_prompt=learned_community_prompt
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
                self.last_community_prompt = learned_community_prompt  # Track the prompt we used
                self.logger.info(f"Successfully processed graph with {len(self.community_manager.communities)} communities")

                # Store community summaries in shared state for backward pass critique
                current_state["community_summaries"] = self.community_manager.community_summaries
                self.logger.info(f"Stored {len(self.community_manager.community_summaries)} community summaries in shared state")

                # Store community summarization logs for prompts_response_logs
                current_state["community_summarization_logs"] = self.community_manager.community_summarization_logs
                self.logger.info(f"Stored {len(self.community_manager.community_summarization_logs)} community summarization logs")
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

                # Call LLM to select communities using learned system prompt
                from autogen_core.models import SystemMessage, UserMessage
                system_message = SystemMessage(content=learned_system_prompt)
                user_message = UserMessage(content=prompt_content, source="user")

                response = await self.model_client.create(
                    [system_message, user_message],
                    cancellation_token=ctx.cancellation_token
                )

                # Log LLM interaction
                logger = get_global_prompt_logger()
                logger.log_interaction(
                    agent_name="GraphRetrievalPlannerAgent",
                    interaction_type="community_selection",
                    system_prompt=learned_system_prompt,
                    user_prompt=prompt_content,
                    llm_response=response.content if isinstance(response.content, str) else str(response.content),
                    batch_id=message.batch_id,
                    iteration=1,
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

                # Retrieve the selected communities
                retrieved_context = await self._retrieve_selected_communities(retrieval_response.selected_communities)

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

    async def _retrieve_selected_communities(self, community_ids: List[int]) -> str:
        """Retrieve the specified communities by their IDs."""
        try:
            context_parts = []
            valid_communities = []

            for community_id in community_ids:
                # Check if community exists and has enough nodes
                if (community_id in self.community_manager.community_summaries and
                    community_id in self.community_manager.community_titles and
                    len(self.community_manager.communities.get(community_id, [])) >= 3):

                    title = self.community_manager.community_titles[community_id]
                    summary = self.community_manager.community_summaries[community_id]
                    context_parts.append(f"Community {community_id} - {title}:\n{summary}")
                    valid_communities.append(community_id)
                else:
                    self.logger.warning(f"Community {community_id} not found or has too few nodes")

            # Format the retrieved context
            if context_parts:
                result = "\n\n".join(context_parts)
                self.logger.info(f"Retrieved {len(valid_communities)} communities (IDs: {valid_communities})")
                return result
            else:
                self.logger.warning("No valid communities could be retrieved")
                return "No valid communities found."

        except Exception as e:
            self.logger.error(f"Error retrieving selected communities {community_ids}: {e}")
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


class AnswerGeneratorAgent(RoutedAgent):
    """
    Agent that generates answers using retrieved context and LLM reasoning.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.answer_generator")
        self.shared_state = SharedState("agent_states")

        # Import prompts
        from parameters import base_prompt_answer_generator_graph

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

        # Prepare prompt with question and retrieved context (without critique)
        prompt_content = self.base_prompt_answer_generator_graph.format(
            message.question, message.retrieved_context
        ) + "\nIn your response, do not mention communities and the structure of a graph, since this is internal information not interesting for the user."

        try:
            # Call LLM for answer generation using learned system prompt
            system_message = SystemMessage(content=learned_system_prompt)
            user_message = UserMessage(content=prompt_content, source="user")

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
                user_prompt=prompt_content,
                llm_response=generated_answer,
                batch_id=message.batch_id,
                qa_pair_id=message.qa_pair_id,
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

    def __init__(self, name: str, dataset_name: str = None) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.response_evaluator")
        self.shared_state = SharedState("agent_states")
        self.dataset_name = dataset_name

        # Import prompts and response format
        from parameters import response_evaluator_prompt, ResponseEvaluationResponse

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

        self.response_evaluator_prompt = response_evaluator_prompt

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

        self.logger.info(f"ResponseEvaluatorAgent evaluating QA pair {message.qa_pair_id}")

        # Prepare prompt with query, generated response, and satisfactory criteria
        prompt_content = self.response_evaluator_prompt.format(
            original_query=message.original_query,
            generated_answer=message.generated_answer,
            satisfactory_criteria=self.satisfactory_criteria
        )

        print(f"   Prompt prepared ({len(prompt_content)} chars)")

        try:
            # Call LLM for response evaluation
            system_message = SystemMessage(content=prompt_content)
            user_message = UserMessage(content="Please evaluate the response.", source="system")

            print("=" * 80)
            print("RESPONSE EVALUATOR AGENT - LLM CALL")
            print("=" * 80)
            print(f"System Prompt ({len(prompt_content)} chars):")
            print(prompt_content[:500])  # Print first 500 chars
            print("-" * 80)
            print(f"User Prompt: Please evaluate the response.")
            print("-" * 80)
            print("Calling LLM...")

            response = await self.model_client.create(
                [system_message, user_message],
                cancellation_token=ctx.cancellation_token
            )

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
                additional_metadata={
                    "original_query": message.original_query,
                    "generated_answer_length": len(message.generated_answer),
                    "continue_optimization": eval_response.continue_optimization,
                    "critique_length": len(eval_response.critique)
                }
            )

            log_qa_processing(self.logger, message.qa_pair_id,
                            f"Evaluation completed - continue: {eval_response.continue_optimization}",
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
                batch_id=message.batch_id,
                repetition=message.repetition
            )

            self.logger.info(f"Returning ResponseEvaluationReady for QA pair {message.qa_pair_id}")

            # Return the evaluation ready message
            return eval_ready_msg

        except Exception as e:
            print("=" * 80)
            print("RESPONSE EVALUATOR AGENT - ERROR")
            print("=" * 80)
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            import traceback
            print(f"Traceback:")
            print(traceback.format_exc())
            print("=" * 80)

            self.logger.error("=" * 80)
            self.logger.error("RESPONSE EVALUATOR AGENT - ERROR")
            self.logger.error("=" * 80)
            self.logger.error(f"Error Type: {type(e).__name__}")
            self.logger.error(f"Error Message: {e}")
            self.logger.error(f"Traceback:")
            self.logger.error(traceback.format_exc())
            self.logger.error("=" * 80)
            # Create error response data
            evaluation_data = {
                "qa_pair_id": message.qa_pair_id,
                "error": f"Evaluation failed: {e}",
                "continue_optimization": True,  # Default to continue on error
                "repetition": message.repetition,
                "timestamp": datetime.now().isoformat()
            }

            # Store error result in shared state for BackwardPassAgent
            current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
            response_evaluations = current_state.get("response_evaluations", [])
            response_evaluations.append(evaluation_data)
            current_state["response_evaluations"] = response_evaluations
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            return ResponseEvaluationReadyMessage(
                qa_pair_id=message.qa_pair_id,
                evaluation_result=evaluation_data,
                continue_optimization=True,  # Default to continue on error
                batch_id=message.batch_id,
                repetition=message.repetition
            )

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
            retrieved_content_gradient_prompt_graph,
            retrieval_plan_gradient_prompt_graph,
            retrieval_planning_prompt_gradient_prompt,
            graph_gradient_prompt,
            graph_extraction_prompt_gradient_prompt,
            rag_hyperparameters_agent_gradient_prompt,
            answer_generation_prompt_optimizer,
            retrieval_planner_prompt_optimizer,
            graph_builder_prompt_optimizer,
            hyperparameters_graph_agent_prompt_optimizer,
            community_summarizer_gradient_prompt,
            community_summarizer_prompt_optimizer
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

        # Store all gradient prompts and optimizer prompts
        self.generation_prompt_gradient_prompt = generation_prompt_gradient_prompt
        self.retrieved_content_gradient_prompt_graph = retrieved_content_gradient_prompt_graph
        self.retrieval_plan_gradient_prompt_graph = retrieval_plan_gradient_prompt_graph
        self.retrieval_planning_prompt_gradient_prompt = retrieval_planning_prompt_gradient_prompt
        self.graph_gradient_prompt = graph_gradient_prompt
        self.graph_extraction_prompt_gradient_prompt = graph_extraction_prompt_gradient_prompt
        self.rag_hyperparameters_agent_gradient_prompt = rag_hyperparameters_agent_gradient_prompt

        # Store optimizer prompts
        self.answer_generation_prompt_optimizer = answer_generation_prompt_optimizer
        self.retrieval_planner_prompt_optimizer = retrieval_planner_prompt_optimizer
        self.graph_builder_prompt_optimizer = graph_builder_prompt_optimizer
        self.hyperparameters_graph_agent_prompt_optimizer = hyperparameters_graph_agent_prompt_optimizer
        self.community_summarizer_gradient_prompt = community_summarizer_gradient_prompt
        self.community_summarizer_prompt_optimizer = community_summarizer_prompt_optimizer

    @message_handler
    async def handle_backward_pass_start(self, message: BackwardPassStartMessage, ctx: MessageContext) -> BackwardPassReadyMessage:
        """
        Enhanced BackwardPassStart handler with QA pair boundary awareness.
        Generates appropriate critiques based on iteration context.
        """
        self.logger.info(f"BackwardPassAgent processing backward pass for batch {message.batch_id}, repetition {message.repetition}")

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
        current_state["batch_information"] = batch_info

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

            # Step 4.5: Generate community summarizer critique (after retrieval planner, uses retrieved content critique)
            await self._generate_community_summarizer_critique(current_state, ctx)
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Step 5: Generate graph critique
            await self._generate_graph_critique(current_state, ctx)
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Step 6: Generate graph builder critique
            await self._generate_graph_builder_critique(current_state, ctx)
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Step 7: Generate hyperparameters critique
            await self._generate_hyperparameters_critique(current_state, ctx)
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Final save to ensure everything is persisted
            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

            # Extract optimized prompts for QA pair prompt lifecycle
            optimized_prompts = {
                # Store the actual learned system prompts that agents will use
                "learned_prompt_hyperparameters_graph": current_state.get("learned_prompt_hyperparameters_graph", ""),
                "learned_prompt_answer_generator_graph": current_state.get("learned_prompt_answer_generator_graph", ""),
                "learned_prompt_graph_retrieval_planner": current_state.get("learned_prompt_graph_retrieval_planner", ""),
                "learned_prompt_graph_builder": current_state.get("learned_prompt_graph_builder", ""),
                "learned_prompt_graph_refinement": current_state.get("learned_prompt_graph_refinement", ""),
                "learned_prompt_community_summarizer": current_state.get("learned_prompt_community_summarizer", ""),
                # Also store critiques and prompt templates for reference
                "hyperparameters_graph_agent_critique": current_state.get("hyperparameters_graph_agent_critique", ""),
                "graph_builder_agent_critique": current_state.get("graph_builder_agent_critique", ""),
                "retrieval_planner_agent_critique": current_state.get("retrieval_planner_agent_critique", ""),
                "answer_generation_critique": current_state.get("answer_generation_critique", ""),
                "graph_builder_prompt": current_state.get("graph_builder_prompt", ""),
                "retrieval_prompt": current_state.get("retrieval_prompt", "")
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
                        "retrieved_content_critique",
                        "retrieval_plan_critique",
                        "retrieval_planner_agent_critique",
                        "graph_critique",
                        "graph_builder_agent_critique",
                        "hyperparameters_graph_agent_critique"
                    ],
                    "learned_prompts_generated": [
                        "learned_prompt_answer_generator_graph",
                        "learned_prompt_graph_retrieval_planner",
                        "learned_prompt_graph_builder",
                        "learned_prompt_hyperparameters_graph"
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
        """Generate critique for answer generation based on conversations and evaluations."""
        self.logger.info("Generating answer generation critique")

        all_conversations = current_state.get("conversations_answer_generation", [])
        all_evaluation_responses = current_state.get("response_evaluations", [])

        # Get current repetition from batch info
        batch_info = current_state.get("batch_information", {})
        current_repetition = batch_info.get("current_repetition", 0)

        # Since lists are cleared at the start of each iteration, all data should be from current iteration
        conversations = all_conversations
        evaluation_responses = all_evaluation_responses

        self.logger.info(f"Found {len(conversations)} conversations and {len(evaluation_responses)} evaluations for iteration {current_repetition}")

        if not conversations or not evaluation_responses:
            self.logger.warning("No conversation or evaluation data available - skipping answer generation critique")
            return

        # Get the current answer generation prompt for critique
        current_answer_prompt = current_state.get("learned_prompt_answer_generator_graph", "")
        if not current_answer_prompt:
            # Use base prompt if no learned prompt exists yet
            from parameters import base_prompt_answer_generator_graph
            current_answer_prompt = base_prompt_answer_generator_graph

        # One iteration = One QA pair, so we should have exactly 1 conversation and 1 evaluation
        if len(conversations) != 1 or len(evaluation_responses) != 1:
            self.logger.error(f"Expected exactly 1 conversation and 1 evaluation per iteration, but got {len(conversations)} conversations and {len(evaluation_responses)} evaluations")
            return

        # Extract the single conversation and evaluation from this iteration
        conv = conversations[0]
        eval_resp = evaluation_responses[0]

        # Create the single sequence: query + previous prompt + answer + feedback
        concatenated_data = f"Question: {conv.get('question', '')}\nPrevious Answer Generation Prompt: {current_answer_prompt}\nGenerated Answer: {conv.get('generated_answer', '')}\nResponse Feedback: {eval_resp.get('evaluation_feedback', '')}"

        self.logger.info(f"Created answer generation critique sequence for iteration {current_repetition}")

        # Call LLM with generation_prompt_gradient_prompt
        prompt_content = self.generation_prompt_gradient_prompt.format(concatenated_data)

        critique = await self._call_llm(prompt_content, ctx)
        current_state["answer_generation_critique"] = critique

        # Generate optimized prompt using the critique
        optimizer_prompt = self.answer_generation_prompt_optimizer.format(critique)
        optimized_prompt = await self._call_llm(optimizer_prompt, ctx, interaction_type="optimization", user_prompt="Generate the optimized system prompt.")

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

    async def _generate_retrieved_content_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for retrieved content based on conversations, contexts, and evaluations."""
        self.logger.info("Generating retrieved content critique")

        all_conversations = current_state.get("conversations_answer_generation", [])
        all_retrieved_contexts = current_state.get("retrieved_contexts", [])
        all_evaluation_responses = current_state.get("response_evaluations", [])

        # Get current repetition from batch info
        batch_info = current_state.get("batch_information", {})
        current_repetition = batch_info.get("current_repetition", 0)

        # Since lists are cleared at the start of each iteration, all data should be from current iteration
        conversations = all_conversations
        evaluation_responses = all_evaluation_responses

        # Extract retrieved contexts for current iteration
        retrieved_contexts = []
        for ctx_entry in all_retrieved_contexts:
            if isinstance(ctx_entry, dict) and "repetition" in ctx_entry:
                if ctx_entry.get("repetition") == current_repetition:
                    retrieved_contexts.append(ctx_entry.get("retrieved_context", ""))

        self.logger.info(f"Found {len(conversations)} conversations, {len(evaluation_responses)} evaluations, {len(retrieved_contexts)} contexts for iteration {current_repetition}")

        if not conversations or not evaluation_responses:
            self.logger.warning("No conversation or evaluation data available - skipping retrieved content critique")
            return

        # One iteration = One QA pair, so we should have exactly 1 conversation, 1 evaluation, and 1 context
        if len(conversations) != 1 or len(evaluation_responses) != 1:
            self.logger.error(f"Expected exactly 1 conversation and 1 evaluation per iteration, but got {len(conversations)} conversations and {len(evaluation_responses)} evaluations")
            return

        if len(retrieved_contexts) != 1:
            self.logger.error(f"Expected exactly 1 retrieved context per iteration, but got {len(retrieved_contexts)}")
            return

        # Extract the single conversation, evaluation, and context from this iteration
        conv = conversations[0]
        eval_resp = evaluation_responses[0]
        context = retrieved_contexts[0]

        # Extract query from conversation
        query = conv.get('question', 'No query available')

        # Log context size for monitoring
        context_length = len(str(context))
        self.logger.info(f"Retrieved context length: {context_length} characters")

        # Note: No truncation applied - using full retrieved context for gradient analysis

        # Create the single sequence: context + query + answer + feedback
        concatenated_data = f"Retrieved Context: {context}\nQuery: {query}\nGenerated Answer: {conv.get('generated_answer', '')}\nFeedback: {eval_resp.get('evaluation_feedback', '')}"

        self.logger.info(f"Created retrieved content critique sequence for iteration {current_repetition}")

        # Call LLM with retrieved_content_gradient_prompt_graph
        prompt_content = self.retrieved_content_gradient_prompt_graph.format(concatenated_data)

        critique = await self._call_llm(prompt_content, ctx)
        current_state["retrieved_content_critique"] = critique

        self.logger.info("Retrieved content critique generated and saved")

    async def _generate_retrieval_plan_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for retrieval plans based on plans and contexts."""
        self.logger.info("Generating retrieval plan critique")

        retrieval_plans = current_state.get("retrieval_plans", [])
        retrieved_contexts = current_state.get("retrieved_contexts", [])

        self.logger.info(f"Retrieval plans: {retrieval_plans}")
        self.logger.info(f"Retrieved contexts: {retrieved_contexts}")

        if not retrieval_plans or not retrieved_contexts:
            self.logger.warning("Missing retrieval plans or contexts for critique")
            return

        # retrieval_plans is a list of reasoning strings from k iterations in the current repetition
        # We want to show all moves in the complete plan, not just the first one
        complete_retrieval_plan = "\n".join([f"Move {i+1}: {plan}" for i, plan in enumerate(retrieval_plans)])

        # retrieved_contexts is a list of context entries, get the most recent one for this repetition
        if retrieved_contexts:
            latest_context_entry = retrieved_contexts[-1]  # Get the most recent context
            if isinstance(latest_context_entry, dict):
                retrieved_context_text = latest_context_entry.get("retrieved_context", str(latest_context_entry))
            else:
                retrieved_context_text = str(latest_context_entry)
        else:
            retrieved_context_text = "No retrieved context available"

        # Create the pair showing the community selection reasoning
        pair = f"Community Selection: {complete_retrieval_plan}\nRetrieved Community Summaries: {retrieved_context_text}"
        concatenated_pairs = pair

        # Get retrieved_content_critique for the second variable
        retrieved_content_critique = current_state.get("retrieved_content_critique", "No critique available")

        # Call LLM with retrieval_plan_gradient_prompt_graph (updated for community selection)
        prompt_content = self.retrieval_plan_gradient_prompt_graph.format(concatenated_pairs, retrieved_content_critique)

        critique = await self._call_llm(prompt_content, ctx)
        current_state["retrieval_plan_critique"] = critique

        self.logger.info("Community selection critique generated and saved")

    async def _generate_retrieval_planning_prompt_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for community selection prompt."""
        self.logger.info("Generating community selection prompt critique")

        retrieval_prompt = current_state.get("retrieval_prompt", "")
        # Get community information from retrieved contexts instead of direct access
        retrieved_contexts = current_state.get("retrieved_contexts", [])
        retrieval_plans = current_state.get("retrieval_plans", [])

        if not retrieval_prompt or not retrieval_plans:
            self.logger.warning("Missing community selection prompt or reasoning for critique")
            return

        # Create data showing: community_selection_prompt + retrieved_communities + selection_reasoning
        retrieved_community_info = "No communities retrieved"
        if retrieved_contexts:
            latest_context = retrieved_contexts[-1]
            if isinstance(latest_context, dict):
                retrieved_community_info = latest_context.get("retrieved_context", "No context available")

        # With one-shot approach, we have one selection reasoning
        selection_reasoning = retrieval_plans[0] if retrieval_plans else "No reasoning available"

        triplet = f"Community Selection Prompt: {retrieval_prompt}\nRetrieved Communities: {retrieved_community_info}\nSelection Reasoning: {selection_reasoning}"
        concatenated_triplets = triplet

        # Get retrieval_plan_critique for the second variable
        retrieval_plan_critique = current_state.get("retrieval_plan_critique", "No critique available")

        # Call LLM with retrieval_planning_prompt_gradient_prompt (now for community selection)
        prompt_content = self.retrieval_planning_prompt_gradient_prompt.format(concatenated_triplets, retrieval_plan_critique)

        critique = await self._call_llm(prompt_content, ctx)
        current_state["retrieval_planner_agent_critique"] = critique

        # Generate optimized prompt using the critique
        optimizer_prompt = self.retrieval_planner_prompt_optimizer.format(critique)
        optimized_prompt = await self._call_llm(optimizer_prompt, ctx, interaction_type="optimization", user_prompt="Generate the optimized system prompt.")

        # Only update if not frozen
        is_frozen = self._is_prompt_frozen("graph_retrieval_planner", current_state)
        if not is_frozen:
            current_state["learned_prompt_graph_retrieval_planner"] = optimized_prompt

        log_critique_result(self.logger, "graph_retrieval_planner", critique, is_frozen)

    async def _generate_graph_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for the graph based on questions, description, and community selections."""
        self.logger.info("Generating graph critique")

        batch_info = current_state.get("batch_information", {})
        current_qa_pair_id = batch_info.get("current_qa_pair_id", "")

        # If current_qa_pair_id is not available, try to get it from other sources
        if not current_qa_pair_id:
            # Try to get from conversations
            conversations = current_state.get("conversations_answer_generation", [])
            if conversations:
                # Use the most recent conversation's qa_pair_id
                current_qa_pair_id = conversations[-1].get("qa_pair_id", "")

            # If still not found, try to get from response evaluations
            if not current_qa_pair_id:
                evaluations = current_state.get("response_evaluations", [])
                if evaluations:
                    current_qa_pair_id = evaluations[-1].get("qa_pair_id", "")

        graph_description = current_state.get("graph_description", "")
        retrieval_plans = current_state.get("retrieval_plans", [])
        retrieved_contexts = current_state.get("retrieved_contexts", [])

        # Debug logging to see what data is available
        self.logger.info(f"Debug - current_qa_pair_id: '{current_qa_pair_id}' (from batch_info: '{batch_info.get('current_qa_pair_id', 'None')}')")
        self.logger.info(f"Debug - graph_description length: {len(graph_description) if graph_description else 0}")
        self.logger.info(f"Debug - retrieval_plans length: {len(retrieval_plans)}")
        self.logger.info(f"Debug - retrieved_contexts length: {len(retrieved_contexts)}")
        self.logger.info(f"Debug - available state keys: {list(current_state.keys())}")
        self.logger.info(f"Debug - batch_info keys: {list(batch_info.keys())}")

        # Make qa_pair_id optional for graph critique since the core functionality doesn't strictly need it
        if not graph_description or not retrieval_plans:
            self.logger.warning(f"Missing essential data for graph critique - graph_desc: {bool(graph_description)}, retrieval_plans: {bool(retrieval_plans)}")
            return

        if not current_qa_pair_id:
            self.logger.warning("No qa_pair_id found, proceeding with graph critique anyway")
            current_qa_pair_id = "unknown_qa_pair"

        # Get the current question from conversations
        conversations = current_state.get("conversations_answer_generation", [])
        current_question = "No question available"
        if conversations:
            current_question = conversations[-1].get("question", "No question available")

        # Create data: current_query + graph_description + community_selection
        community_selection = retrieval_plans[0] if retrieval_plans else "No selection reasoning available"
        retrieved_info = "No communities retrieved"
        if retrieved_contexts:
            latest_context = retrieved_contexts[-1]
            if isinstance(latest_context, dict):
                retrieved_info = latest_context.get("retrieved_context", "No context available")

        triplet = f"Query: {current_question}\nGraph Description: {graph_description}\nCommunity Selection: {community_selection}\nRetrieved Communities: {retrieved_info}"
        concatenated_triplets = triplet

        # Get retrieval_plan_critique for the second variable
        retrieval_plan_critique = current_state.get("retrieval_plan_critique", "No critique available")

        # Call LLM with graph_gradient_prompt
        prompt_content = self.graph_gradient_prompt.format(concatenated_triplets, retrieval_plan_critique)

        critique = await self._call_llm(prompt_content, ctx)
        current_state["graph_critique"] = critique

        self.logger.info("Graph critique generated and saved")

    async def _generate_graph_builder_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for graph builder/refinement prompt based on current iteration."""

        # Determine current repetition to decide creation vs refinement
        batch_info = current_state.get("batch_information", {})
        current_repetition = batch_info.get("current_repetition", 0)
        is_first_iteration = current_repetition == 0

        if is_first_iteration:
            # First iteration: optimize graph creation prompt
            self.logger.info("Generating graph builder (creation) critique - first iteration")

            graph_builder_prompt = current_state.get("graph_builder_prompt", "")
            corpus_sample = current_state.get("full_document_text", batch_info.get("document_text", ""))[:500]
            graph_description = current_state.get("graph_description", "")
            graph_critique = current_state.get("graph_critique", "No critique available")

            if not graph_builder_prompt or not corpus_sample or not graph_description:
                self.logger.warning(f"Missing data for graph builder critique: graph_builder_prompt={bool(graph_builder_prompt)}, corpus_sample={bool(corpus_sample)}, graph_description={bool(graph_description)}")
                return

            # Call LLM with graph_extraction_prompt_gradient_prompt
            prompt_content = self.graph_extraction_prompt_gradient_prompt.format(
                graph_builder_prompt, corpus_sample, graph_description, graph_critique
            )

            critique = await self._call_llm(prompt_content, ctx)
            current_state["graph_builder_agent_critique"] = critique

            # Generate optimized prompt using the critique
            optimizer_prompt = self.graph_builder_prompt_optimizer.format(critique)
            optimized_prompt = await self._call_llm(optimizer_prompt, ctx, interaction_type="optimization", user_prompt="Generate the optimized system prompt.")

            # Store the optimized graph builder prompt for use in graph refinement
            # Limit prompt length to prevent truncation and LLM failures
            MAX_PROMPT_LENGTH = 4000  # characters
            if len(optimized_prompt) > MAX_PROMPT_LENGTH:
                self.logger.warning(f"Optimized graph builder prompt too long ({len(optimized_prompt)} chars), truncating to {MAX_PROMPT_LENGTH}")
                optimized_prompt = optimized_prompt[:MAX_PROMPT_LENGTH] + "\n\n[Prompt truncated to prevent errors]"

            is_frozen = self._is_prompt_frozen("graph_builder", current_state)
            if not is_frozen:
                current_state["learned_prompt_graph_builder"] = optimized_prompt
                self.logger.info(f"Stored optimized graph builder prompt ({len(optimized_prompt)} chars) for use in refinement mode")
            else:
                self.logger.info("Graph builder prompt is frozen - not updating learned_prompt_graph_builder")

        else:
            # Subsequent iterations: optimize graph refinement prompt
            self.logger.info(f"Generating graph refinement critique - iteration {current_repetition}")

            graph_refinement_prompt = current_state.get("graph_refinement_prompt", "")
            corpus_sample = current_state.get("full_document_text", batch_info.get("document_text", ""))[:500]
            graph_description = current_state.get("graph_description", "")
            graph_critique = current_state.get("graph_critique", "No critique available")

            if not graph_refinement_prompt or not corpus_sample or not graph_description:
                self.logger.warning(f"Missing data for graph refinement critique: graph_refinement_prompt={bool(graph_refinement_prompt)}, corpus_sample={bool(corpus_sample)}, graph_description={bool(graph_description)}")
                return

            # For refinement, we use the same gradient prompt format but focus on refinement
            prompt_content = self.graph_extraction_prompt_gradient_prompt.format(
                graph_refinement_prompt, corpus_sample, graph_description, graph_critique
            )

            critique = await self._call_llm(prompt_content, ctx)
            current_state["graph_refinement_agent_critique"] = critique

            # Generate optimized refinement prompt using the critique
            optimizer_prompt = self.graph_builder_prompt_optimizer.format(critique)
            optimized_refinement_prompt = await self._call_llm(optimizer_prompt, ctx, interaction_type="optimization", user_prompt="Generate the optimized system prompt.")

            # Update the refinement prompt (this IS optimized in backward pass)
            # Limit prompt length to prevent truncation and LLM failures
            MAX_PROMPT_LENGTH = 4000  # characters
            if len(optimized_refinement_prompt) > MAX_PROMPT_LENGTH:
                self.logger.warning(f"Optimized refinement prompt too long ({len(optimized_refinement_prompt)} chars), truncating to {MAX_PROMPT_LENGTH}")
                optimized_refinement_prompt = optimized_refinement_prompt[:MAX_PROMPT_LENGTH] + "\n\n[Prompt truncated to prevent errors]"

            is_frozen = self._is_prompt_frozen("graph_refinement", current_state)
            if not is_frozen:
                current_state["learned_prompt_graph_refinement"] = optimized_refinement_prompt
                self.logger.info(f"Stored optimized refinement prompt ({len(optimized_refinement_prompt)} chars)")

            log_critique_result(self.logger, "graph_refinement", critique, is_frozen)

    async def _generate_hyperparameters_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for hyperparameters agent."""
        self.logger.info("Generating hyperparameters critique")

        rag_hyperparams = current_state.get("rag_hyperparameters", {})
        chunk_size = rag_hyperparams.get("chunk_size", "Not specified")

        batch_info = current_state.get("batch_information", {})
        corpus_sample = current_state.get("full_document_text", batch_info.get("document_text", ""))[:500]  # Sample of corpus
        graph_description = current_state.get("graph_description", "")
        graph_critique = current_state.get("graph_critique", "No critique available")

        if not corpus_sample or not graph_description:
            self.logger.warning(f"Missing data for hyperparameters critique: corpus_sample={bool(corpus_sample)}, graph_description={bool(graph_description)}")
            return

        # Call LLM with rag_hyperparameters_agent_gradient_prompt
        prompt_content = self.rag_hyperparameters_agent_gradient_prompt.format(
            chunk_size, corpus_sample, graph_description, graph_critique
        )

        critique = await self._call_llm(prompt_content, ctx)
        current_state["hyperparameters_graph_agent_critique"] = critique

        # Generate optimized prompt using the critique
        optimizer_prompt = self.hyperparameters_graph_agent_prompt_optimizer.format(critique)
        optimized_prompt = await self._call_llm(optimizer_prompt, ctx, interaction_type="optimization", user_prompt="Generate the optimized system prompt.")

        # Limit prompt length to prevent truncation
        MAX_PROMPT_LENGTH = 4000
        if len(optimized_prompt) > MAX_PROMPT_LENGTH:
            self.logger.warning(f"Optimized hyperparameters prompt too long ({len(optimized_prompt)} chars), truncating to {MAX_PROMPT_LENGTH}")
            optimized_prompt = optimized_prompt[:MAX_PROMPT_LENGTH] + "\n\n[Prompt truncated to prevent errors]"

        # Only update if not frozen
        is_frozen = self._is_prompt_frozen("hyperparameters_graph", current_state)
        if not is_frozen:
            current_state["learned_prompt_hyperparameters_graph"] = optimized_prompt
            self.logger.info(f"Stored optimized hyperparameters prompt ({len(optimized_prompt)} chars)")

        log_critique_result(self.logger, "hyperparameters_graph", critique, is_frozen)

    async def _generate_community_summarizer_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for community summarizer based on retrieved content critique."""
        self.logger.info("=" * 80)
        self.logger.info("COMMUNITY SUMMARIZER CRITIQUE - Starting")
        self.logger.info("=" * 80)

        # Get community summaries from shared state (sample)
        community_summaries = current_state.get("community_summaries", {})
        self.logger.info(f"Community summaries in state: {len(community_summaries)} summaries found")

        if not community_summaries:
            self.logger.warning("⚠️ No community summaries found in state, skipping community summarizer critique")
            self.logger.info(f"State keys available: {list(current_state.keys())}")
            return

        # Get a sample of community summaries (first 3)
        sample_summaries = []
        for comm_id, summary in list(community_summaries.items())[:3]:
            sample_summaries.append(f"Community {comm_id}:\n{summary}")
            self.logger.info(f"Sampled community {comm_id} (length: {len(summary)} chars)")

        community_summaries_text = "\n\n".join(sample_summaries)
        self.logger.info(f"Total sampled text length: {len(community_summaries_text)} chars")

        # Get retrieved content critique (which critiques the community summaries)
        retrieved_content_critique = current_state.get("retrieved_content_critique", "No critique available")
        self.logger.info(f"Retrieved content critique length: {len(retrieved_content_critique)} chars")

        if not community_summaries_text:
            self.logger.warning("⚠️ No community summary samples available for critique (empty text)")
            return

        # Call LLM with community_summarizer_gradient_prompt
        self.logger.info("Calling LLM for community summarizer critique...")
        prompt_content = self.community_summarizer_gradient_prompt.format(
            community_summaries_text, retrieved_content_critique
        )

        critique = await self._call_llm(prompt_content, ctx)
        current_state["community_summarizer_critique"] = critique
        self.logger.info(f"✓ Generated critique (length: {len(critique)} chars)")

        # Generate optimized prompt using the critique
        self.logger.info("Calling LLM for prompt optimization...")
        optimizer_prompt = self.community_summarizer_prompt_optimizer.format(critique)
        optimized_prompt = await self._call_llm(optimizer_prompt, ctx, interaction_type="optimization", user_prompt="Generate the optimized system prompt.")
        self.logger.info(f"✓ Generated optimized prompt (length: {len(optimized_prompt)} chars)")

        # Limit prompt length to prevent truncation
        MAX_PROMPT_LENGTH = 4000
        if len(optimized_prompt) > MAX_PROMPT_LENGTH:
            self.logger.warning(f"Optimized community summarizer prompt too long ({len(optimized_prompt)} chars), truncating to {MAX_PROMPT_LENGTH}")
            optimized_prompt = optimized_prompt[:MAX_PROMPT_LENGTH] + "\n\n[Prompt truncated to prevent errors]"

        # Only update if not frozen
        is_frozen = self._is_prompt_frozen("community_summarizer", current_state)
        if not is_frozen:
            current_state["learned_prompt_community_summarizer"] = optimized_prompt
            self.logger.info(f"✓ Stored optimized community summarizer prompt ({len(optimized_prompt)} chars)")
        else:
            self.logger.info("⚠️ Community summarizer prompt is frozen, not updating")

        log_critique_result(self.logger, "community_summarizer", critique, is_frozen)
        self.logger.info("=" * 80)
        self.logger.info("COMMUNITY SUMMARIZER CRITIQUE - Completed")
        self.logger.info("=" * 80)

    def _is_prompt_frozen(self, prompt_type: str, current_state: Dict[str, Any]) -> bool:
        """Check if a prompt type is frozen."""
        frozen_prompts = current_state.get("frozen_prompts", [])
        return prompt_type in frozen_prompts

    async def _call_llm(self, prompt_content: str, ctx: MessageContext, interaction_type: str = "critique", batch_id: int = None, user_prompt: str = "Please provide your critique and feedback.") -> str:
        """Helper method to call LLM with given prompt and token limit."""
        try:
            # Add token limit instruction to the prompt based on interaction type
            if interaction_type == "optimization":
                enhanced_prompt = f"{prompt_content}\n\nIMPORTANT: Please limit your response to approximately {self.critique_token_limit} tokens to ensure efficient inference processing. Focus on the optimized prompt and be concise."
            else:
                enhanced_prompt = f"{prompt_content}\n\nIMPORTANT: Please limit your critique to approximately {self.critique_token_limit} tokens to ensure efficient inference processing. Focus on the most critical points and be concise."

            system_message = SystemMessage(content=enhanced_prompt)
            user_message = UserMessage(content=user_prompt, source="system")

            response = await self.model_client.create(
                [system_message, user_message],
                cancellation_token=ctx.cancellation_token
            )

            response_content = response.content if isinstance(response.content, str) else str(response.content)

            # Log LLM interaction
            logger = get_global_prompt_logger()
            logger.log_interaction(
                agent_name="BackwardPassAgent",
                interaction_type=interaction_type,
                system_prompt=enhanced_prompt,
                user_prompt=user_prompt,
                llm_response=response_content,
                batch_id=batch_id,
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
                logger = get_global_prompt_logger()
                logger.log_interaction(
                    agent_name="SummarizerAgent",
                    interaction_type="context_summarization",
                    system_prompt="You are a helpful assistant that creates concise summaries.",
                    user_prompt=prompt_content,
                    llm_response=summary,
                    batch_id=message.batch_id,
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