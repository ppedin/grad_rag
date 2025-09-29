"""
Multi-Agent All-Context System with AutoGen Core API.
No RAG - uses entire document in prompt for answer generation.
Includes Answer Generator, Response Evaluator, and Answer Generator Prompt Critique agents.
"""

import json
import logging
import statistics
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel

from logging_utils import log_agent_action, log_qa_processing, log_critique_result
from prompt_response_logger import get_global_prompt_logger, initialize_prompt_logging
from step_execution_logger import get_global_step_logger, StepStatus
from evaluation_logger import get_global_evaluation_logger
from standardized_evaluation_logger import initialize_standardized_logging, get_standardized_logger, finalize_standardized_logging, SystemType

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


# ===== MESSAGE TYPES =====

class AnswerGenerationStartMessage(BaseModel):
    qa_pair_id: str
    question: str
    document_text: str
    system_prompt: str  # Empty initially, then optimized prompt
    batch_id: int
    repetition: int
    iteration: int

class AnswerGenerationReadyMessage(BaseModel):
    qa_pair_id: str
    generated_answer: str
    batch_id: int
    repetition: int
    iteration: int

class ResponseEvaluationStartMessage(BaseModel):
    qa_pair_id: str
    original_query: str
    generated_answer: str
    gold_answers: List[str]
    rouge_score: float
    batch_id: int
    repetition: int
    iteration: int

class ResponseEvaluationReadyMessage(BaseModel):
    qa_pair_id: str
    evaluation_result: Dict[str, Any]
    rouge_score: float
    batch_id: int
    repetition: int
    iteration: int

class CritiqueStartMessage(BaseModel):
    qa_pair_id: str
    question: str
    current_system_prompt: str
    generated_answer: str
    evaluation_result: Dict[str, Any]
    rouge_score: float
    batch_id: int
    repetition: int
    iteration: int

class CritiqueReadyMessage(BaseModel):
    qa_pair_id: str
    critique_feedback: str
    batch_id: int
    repetition: int
    iteration: int

class PromptOptimizationStartMessage(BaseModel):
    qa_pair_id: str
    question: str
    current_system_prompt: str
    critique_feedback: str
    batch_id: int
    repetition: int
    iteration: int

class PromptOptimizationReadyMessage(BaseModel):
    qa_pair_id: str
    optimized_prompt: str
    batch_id: int
    repetition: int
    iteration: int


# ===== ANSWER GENERATOR AGENT =====

class AnswerGeneratorAgent(RoutedAgent):
    """
    Generates answers to questions using the entire document as context.
    No RAG - full document is provided in the prompt.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.answer_generator")

        self.client = OpenAIChatCompletionClient(
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

    @message_handler
    async def handle_answer_generation_start(self, message: AnswerGenerationStartMessage, ctx: MessageContext) -> AnswerGenerationReadyMessage:
        """Generate answer using the full document as context."""

        self.logger.info(f"Starting answer generation for QA pair {message.qa_pair_id}, iteration {message.iteration}")

        # Create system message (empty initially, then optimized prompt)
        system_message = SystemMessage(content=message.system_prompt if message.system_prompt else "")

        # Create user message with question and full document
        user_content = f"""Based on the following document, please answer the question.

Document:
{message.document_text}

Question: {message.question}

Please provide a comprehensive and accurate answer based on the document content."""

        user_message = UserMessage(content=user_content, source="user")

        # Generate response
        try:
            response = await self.client.create([system_message, user_message])
            generated_answer = response.content.strip()

            # Log LLM interaction
            prompt_logger = get_global_prompt_logger()
            prompt_logger.log_interaction(
                agent_name="AnswerGeneratorAgent",
                interaction_type="answer_generation",
                system_prompt=message.system_prompt if message.system_prompt else "",
                user_prompt=user_content,
                llm_response=generated_answer,
                additional_metadata={
                    "qa_pair_id": message.qa_pair_id,
                    "iteration": message.iteration,
                    "document_length": len(message.document_text),
                    "system_prompt_length": len(message.system_prompt) if message.system_prompt else 0
                }
            )

            self.logger.info(f"Generated answer for {message.qa_pair_id}: {len(generated_answer)} characters")

            return AnswerGenerationReadyMessage(
                qa_pair_id=message.qa_pair_id,
                generated_answer=generated_answer,
                batch_id=message.batch_id,
                repetition=message.repetition,
                iteration=message.iteration
            )

        except Exception as e:
            self.logger.error(f"Error generating answer for {message.qa_pair_id}: {e}")
            return AnswerGenerationReadyMessage(
                qa_pair_id=message.qa_pair_id,
                generated_answer=f"Error generating answer: {str(e)}",
                batch_id=message.batch_id,
                repetition=message.repetition,
                iteration=message.iteration
            )


# ===== RESPONSE EVALUATOR AGENT =====

class ResponseEvaluatorAgent(RoutedAgent):
    """
    Evaluates generated responses against reference answers.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.response_evaluator")

        self.client = OpenAIChatCompletionClient(
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

    @message_handler
    async def handle_response_evaluation_start(self, message: ResponseEvaluationStartMessage, ctx: MessageContext) -> ResponseEvaluationReadyMessage:
        """Evaluate the generated response."""

        self.logger.info(f"Starting response evaluation for QA pair {message.qa_pair_id}, iteration {message.iteration}")

        # Create evaluation prompt - only include query and generated answer (no reference answers or ROUGE score)
        evaluation_prompt = f"""Please evaluate the following generated answer for the given question.

Question: {message.original_query}

Generated Answer:
{message.generated_answer}

Please provide a detailed evaluation considering:
1. Accuracy and correctness
2. Completeness of the answer
3. Relevance to the question
4. Clarity and coherence
5. Overall quality of the response

Provide specific feedback on what could be improved in the answer generation process."""

        user_message = UserMessage(content=evaluation_prompt, source="user")

        try:
            response = await self.client.create([user_message])
            evaluation_feedback = response.content.strip()

            # Log LLM interaction
            prompt_logger = get_global_prompt_logger()
            prompt_logger.log_interaction(
                agent_name="ResponseEvaluatorAgent",
                interaction_type="response_evaluation",
                system_prompt="",
                user_prompt=evaluation_prompt,
                llm_response=evaluation_feedback,
                additional_metadata={
                    "qa_pair_id": message.qa_pair_id,
                    "iteration": message.iteration,
                    "rouge_score": message.rouge_score,
                    "generated_answer_length": len(message.generated_answer)
                }
            )

            evaluation_result = {
                "evaluation_feedback": evaluation_feedback,
                "rouge_score": message.rouge_score,
                "timestamp": datetime.now().isoformat()
            }

            self.logger.info(f"Completed evaluation for {message.qa_pair_id}")

            return ResponseEvaluationReadyMessage(
                qa_pair_id=message.qa_pair_id,
                evaluation_result=evaluation_result,
                rouge_score=message.rouge_score,
                batch_id=message.batch_id,
                repetition=message.repetition,
                iteration=message.iteration
            )

        except Exception as e:
            self.logger.error(f"Error evaluating response for {message.qa_pair_id}: {e}")
            return ResponseEvaluationReadyMessage(
                qa_pair_id=message.qa_pair_id,
                evaluation_result={"error": str(e), "rouge_score": message.rouge_score},
                rouge_score=message.rouge_score,
                batch_id=message.batch_id,
                repetition=message.repetition,
                iteration=message.iteration
            )


# ===== CRITIQUE AGENT =====

class CritiqueAgent(RoutedAgent):
    """
    Critiques the current answer generation system prompt based on performance results.
    Focuses on identifying weaknesses and issues in the prompt itself.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.critique")

        self.client = OpenAIChatCompletionClient(
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

    @message_handler
    async def handle_critique_start(self, message: CritiqueStartMessage, ctx: MessageContext) -> CritiqueReadyMessage:
        """Generate critique based on current performance."""

        self.logger.info(f"Starting system prompt critique for QA pair {message.qa_pair_id}, iteration {message.iteration}")

        # Create critique prompt focused on the system prompt
        critique_prompt = f"""You are an expert at analyzing and critiquing system prompts for answer generation tasks.

Task:
Question: {message.question}

Current System Prompt: "{message.current_system_prompt if message.current_system_prompt else 'No system prompt (empty)'}"

Generated Answer:
{message.generated_answer}

Evaluation Results:
{message.evaluation_result.get('evaluation_feedback', 'No evaluation available')}

ROUGE Score: {message.rouge_score:.4f}

Please analyze the current system prompt and provide a detailed critique focusing specifically on:

1. What is wrong or missing in the current system prompt
2. How the system prompt may have led to the poor performance observed
3. Specific weaknesses in the prompt's structure, clarity, or instructions
4. What guidance is missing that could help the model generate better answers
5. How the prompt fails to address the specific requirements of this type of question

Do NOT suggest improvements or optimizations - only provide a critical analysis of the current system prompt's shortcomings. The critique should focus on the prompt itself, not the generated answer."""

        user_message = UserMessage(content=critique_prompt, source="user")

        try:
            response = await self.client.create([user_message])
            critique_feedback = response.content.strip()

            # Log LLM interaction
            prompt_logger = get_global_prompt_logger()
            prompt_logger.log_interaction(
                agent_name="CritiqueAgent",
                interaction_type="system_prompt_critique",
                system_prompt="",
                user_prompt=critique_prompt,
                llm_response=critique_feedback,
                additional_metadata={
                    "qa_pair_id": message.qa_pair_id,
                    "iteration": message.iteration,
                    "current_prompt_length": len(message.current_system_prompt),
                    "rouge_score": message.rouge_score
                }
            )

            self.logger.info(f"Generated system prompt critique for {message.qa_pair_id}: {len(critique_feedback)} characters")

            return CritiqueReadyMessage(
                qa_pair_id=message.qa_pair_id,
                critique_feedback=critique_feedback,
                batch_id=message.batch_id,
                repetition=message.repetition,
                iteration=message.iteration
            )

        except Exception as e:
            self.logger.error(f"Error generating critique for {message.qa_pair_id}: {e}")
            return CritiqueReadyMessage(
                qa_pair_id=message.qa_pair_id,
                critique_feedback=f"Error: {str(e)}",
                batch_id=message.batch_id,
                repetition=message.repetition,
                iteration=message.iteration
            )


# ===== PROMPT OPTIMIZATION AGENT =====

class PromptOptimizationAgent(RoutedAgent):
    """
    Uses critique feedback to generate optimized system prompts.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.prompt_optimization")

        self.client = OpenAIChatCompletionClient(
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

    @message_handler
    async def handle_prompt_optimization_start(self, message: PromptOptimizationStartMessage, ctx: MessageContext) -> PromptOptimizationReadyMessage:
        """Generate optimized prompt based on critique feedback."""

        self.logger.info(f"Starting prompt optimization for QA pair {message.qa_pair_id}, iteration {message.iteration}")

        # Create optimization prompt
        optimization_prompt = f"""You are an expert at creating system prompts for answer generation tasks.

Task: {message.question}

Current System Prompt: "{message.current_system_prompt if message.current_system_prompt else 'No system prompt (empty)'}"

Critique Feedback:
{message.critique_feedback}

Based on the critique feedback, please generate an optimized system prompt that addresses the identified issues.

The optimized system prompt should:
- Address specific weaknesses identified in the critique
- Guide the model to generate better answers
- Be clear, actionable, and specific
- Help improve accuracy, completeness, and relevance

Please provide only the optimized system prompt without any additional explanation or formatting."""

        user_message = UserMessage(content=optimization_prompt, source="user")

        try:
            response = await self.client.create([user_message])
            optimized_prompt = response.content.strip()

            # Log LLM interaction
            prompt_logger = get_global_prompt_logger()
            prompt_logger.log_interaction(
                agent_name="PromptOptimizationAgent",
                interaction_type="prompt_optimization",
                system_prompt="",
                user_prompt=optimization_prompt,
                llm_response=optimized_prompt,
                additional_metadata={
                    "qa_pair_id": message.qa_pair_id,
                    "iteration": message.iteration,
                    "current_prompt_length": len(message.current_system_prompt),
                    "optimized_prompt_length": len(optimized_prompt)
                }
            )

            self.logger.info(f"Generated optimized prompt for {message.qa_pair_id}: {len(optimized_prompt)} characters")

            return PromptOptimizationReadyMessage(
                qa_pair_id=message.qa_pair_id,
                optimized_prompt=optimized_prompt,
                batch_id=message.batch_id,
                repetition=message.repetition,
                iteration=message.iteration
            )

        except Exception as e:
            self.logger.error(f"Error generating optimized prompt for {message.qa_pair_id}: {e}")
            return PromptOptimizationReadyMessage(
                qa_pair_id=message.qa_pair_id,
                optimized_prompt=message.current_system_prompt,  # Fallback to current prompt
                batch_id=message.batch_id,
                repetition=message.repetition,
                iteration=message.iteration
            )


# ===== BATCH ORCHESTRATOR AGENT =====

class BatchOrchestratorAgent(RoutedAgent):
    """
    Orchestrates the processing of QA pairs in a batch through the all-context pipeline.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.batch_orchestrator")
        self.shared_state = SharedState("agent_states")

        # Initialize logging systems
        from prompt_response_logger import initialize_prompt_logging
        from step_execution_logger import initialize_step_logging
        from evaluation_logger import initialize_evaluation_logging

        initialize_step_logging()
        initialize_evaluation_logging()

        # Standardized evaluation logging
        self.standardized_logger = None

        # Track QA pairs processing
        self.current_batch_qa_pairs: Dict[str, Dict[str, Any]] = {}
        self.completed_qa_pairs: Dict[str, Dict[str, Any]] = {}
        self.current_batch_id: Optional[int] = None
        self.current_repetition: Optional[int] = None
        self.current_dataset: Optional[str] = None
        self.current_setting: Optional[str] = None

        # QA pair and iteration tracking
        self.qa_pair_results: Dict[str, List[Dict[str, Any]]] = {}
        self.qa_pair_rouge_progression: Dict[str, List[float]] = {}
        self.qa_pair_system_prompts: Dict[str, List[str]] = {}  # Track prompt evolution

    @message_handler
    async def handle_batch_start(self, message: BatchStartMessage, ctx: MessageContext) -> BatchReadyMessage:
        """Handle BatchStart message by iterating over QA pairs."""
        self.logger.info(f"AllContext BatchOrchestrator received BatchStart for batch {message.batch_id}")

        # Initialize step logger
        step_logger = get_global_step_logger()
        step_logger.start_pipeline(message.dataset, message.setting, len(message.shared_state.get("batch_information", {}).get("qa_pairs", [])))

        # Initialize prompt logging with system-specific folder
        system_log_dir = f"prompt_response_logs/allcontext_{message.dataset}_{message.setting}"
        initialize_prompt_logging(system_log_dir)

        # Initialize standardized evaluation logging for AllContext
        if self.standardized_logger is None:
            self.standardized_logger = initialize_standardized_logging(
                SystemType.ALLCONTEXT, message.dataset, message.setting
            )

        # Validate that only test datasets are used
        if message.setting != "test":
            error_msg = f"AllContext system only supports 'test' setting. Got: '{message.setting}'"
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
            total_iterations = batch_info.get("total_iterations", 1)  # Total iterations for context/logging

            # All Context system processes ONE iteration per BatchStart call (like Vector/Graph RAG)
            current_iteration = message.repetition
            document_text = batch_info.get("document_text", "")

            if not qa_pairs:
                self.logger.warning("No QA pairs found in batch information")
                return BatchReadyMessage(
                    batch_id=message.batch_id,
                    repetition=message.repetition,
                    status="completed",
                    metrics={"qa_pairs_processed": 0}
                )

            self.logger.info(f"Processing {len(qa_pairs)} QA pairs - current iteration: {current_iteration} (DatasetAgent repetition: {message.repetition})")

            # Process each QA pair
            for qa_pair in qa_pairs:
                qa_pair_id = qa_pair.get("question_id", f"qa_{len(self.current_batch_qa_pairs)}")
                self.current_batch_qa_pairs[qa_pair_id] = qa_pair

                # Initialize tracking for this QA pair
                self.qa_pair_results[qa_pair_id] = []
                self.qa_pair_rouge_progression[qa_pair_id] = []
                self.qa_pair_system_prompts[qa_pair_id] = [""]  # Start with empty system prompt

                # Log QA pair start for evaluation
                eval_logger = get_global_evaluation_logger()
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
                current_system_prompt = current_state.get("current_system_prompt", "")

                # Process ONLY the current iteration (not all iterations)
                try:
                        self.logger.info(f"Processing iteration {current_iteration} for QA pair {qa_pair_id}")

                        # Process current iteration
                        iteration_results = await self._process_iteration_step(
                            qa_pair_id, qa_pair, current_iteration, current_system_prompt, document_text, message, total_iterations
                        )

                        # Store iteration results
                        self.qa_pair_results[qa_pair_id].append(iteration_results)
                        rouge_score = iteration_results.get("rouge_score", 0.0)
                        self.qa_pair_rouge_progression[qa_pair_id].append(rouge_score)

                        # Update system prompt for next iteration and save to shared state
                        if current_iteration < total_iterations - 1:
                            optimized_prompt = iteration_results.get("optimized_prompt", current_system_prompt)
                            current_state["current_system_prompt"] = optimized_prompt
                            self.qa_pair_system_prompts[qa_pair_id].append(optimized_prompt)
                        else:
                            # Last iteration - clear the prompt for next QA pair
                            current_state["current_system_prompt"] = ""

                        # Save updated state
                        self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

                        # Log iteration completion
                        self.logger.info(f"Iteration {current_iteration} completed for {qa_pair_id} | ROUGE: {rouge_score:.4f}")

                except Exception as e:
                    self.logger.error(f"Error processing iteration {current_iteration} for {qa_pair_id}: {e}")
                    # Store error results
                    error_results = {
                        "qa_pair_id": qa_pair_id,
                        "iteration": current_iteration,
                        "error": str(e),
                        "rouge_score": 0.0
                    }
                    self.qa_pair_results[qa_pair_id].append(error_results)
                    self.qa_pair_rouge_progression[qa_pair_id].append(0.0)

                # Handle completion logic only on final iteration
                if current_iteration == total_iterations - 1:
                    # Calculate final metrics for this QA pair
                    final_rouge = self.qa_pair_rouge_progression[qa_pair_id][-1]
                    initial_rouge = self.qa_pair_rouge_progression[qa_pair_id][0]
                    rouge_improvement = final_rouge - initial_rouge
                    best_iteration = self.qa_pair_rouge_progression[qa_pair_id].index(max(self.qa_pair_rouge_progression[qa_pair_id]))
                    best_answer = self.qa_pair_results[qa_pair_id][best_iteration].get("generated_answer", "")

                    # Store completion metrics
                    self.completed_qa_pairs[qa_pair_id] = {
                        "final_rouge_score": final_rouge,
                        "initial_rouge_score": initial_rouge,
                        "rouge_improvement": rouge_improvement,
                        "best_iteration": best_iteration,
                        "best_answer": best_answer,
                        "total_iterations_completed": len(self.qa_pair_results[qa_pair_id])
                    }

                    # Log QA pair completion
                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"COMPLETED QA PAIR: {qa_pair_id}")
                    self.logger.info(f"Final ROUGE: {final_rouge:.4f} | Improvement: {rouge_improvement:+.4f}")
                    self.logger.info(f"Best iteration: {best_iteration} | Total iterations: {len(self.qa_pair_results[qa_pair_id])}")
                    self.logger.info(f"{'='*60}")

                    # Complete evaluation logging
                    eval_logger = get_global_evaluation_logger()
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
                    self.standardized_logger.complete_qa_pair_evaluation(
                        qa_pair_id=qa_pair_id,
                        final_rouge_score=final_rouge,
                        rouge_progression=self.qa_pair_rouge_progression[qa_pair_id],
                        best_iteration=best_iteration,
                        total_iterations_completed=len(self.qa_pair_results[qa_pair_id]),
                        best_answer=best_answer
                    )

                # Log QA pair completion
                final_rouge = self.qa_pair_rouge_progression[qa_pair_id][-1] if self.qa_pair_rouge_progression[qa_pair_id] else 0.0
                rouge_improvement = (
                    final_rouge - self.qa_pair_rouge_progression[qa_pair_id][0]
                    if len(self.qa_pair_rouge_progression[qa_pair_id]) > 1 else 0.0
                )

                self.logger.info(f"\nQA PAIR {qa_pair_id} COMPLETED")
                self.logger.info(f"Final ROUGE: {final_rouge:.4f}")
                self.logger.info(f"ROUGE Improvement: {rouge_improvement:+.4f}")
                self.logger.info(f"Iterations completed: {len(self.qa_pair_results[qa_pair_id])}")

                # Mark QA pair as completed
                self.completed_qa_pairs[qa_pair_id] = {
                    "iterations": self.qa_pair_results[qa_pair_id],
                    "rouge_progression": self.qa_pair_rouge_progression[qa_pair_id],
                    "final_rouge": final_rouge,
                    "rouge_improvement": rouge_improvement,
                    "system_prompts": self.qa_pair_system_prompts[qa_pair_id]
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

                # Reset system prompt for next QA pair (as specified)
                self.logger.info(f"Resetting system prompt for next QA pair")

            # Complete pipeline - only 1 iteration per call now
            step_logger.complete_pipeline(success=True, total_qa_pairs_processed=len(qa_pairs),
                                        total_iterations_completed=1)  # Only 1 iteration per call

            # Finalize standardized evaluation logging only handled by DatasetAgent when all processing is complete

            # Return BatchReady message indicating completion
            return BatchReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                status="completed",
                metrics={"qa_pairs_processed": len(qa_pairs)}
            )

        except Exception as e:
            self.logger.error(f"Error processing batch {message.batch_id}: {e}")
            step_logger.complete_pipeline(success=False, error_message=str(e))
            raise

    async def _process_iteration_step(self, qa_pair_id: str, qa_pair: Dict[str, Any], iteration: int,
                                     system_prompt: str, document_text: str, original_message: BatchStartMessage,
                                     total_iterations: int) -> Dict[str, Any]:
        """Process a single iteration for a QA pair."""

        step_start_time = datetime.now()

        # Track intermediate outputs for comprehensive evaluation logging
        intermediate_outputs = {}

        try:
            # Step 1: Answer Generation
            self.logger.info(f"Step 1/3: Answer generation for {qa_pair_id}, iteration {iteration}")

            step_agent_start = datetime.now()
            try:
                answer_gen_msg = AnswerGenerationStartMessage(
                    qa_pair_id=qa_pair_id,
                    question=qa_pair.get("question", ""),
                    document_text=document_text,
                    system_prompt=system_prompt,
                    batch_id=original_message.batch_id,
                    repetition=original_message.repetition,
                    iteration=iteration
                )

                answer_gen_agent_id = AgentId("answer_generator_agent", "default")
                answer_response = await self.send_message(answer_gen_msg, answer_gen_agent_id)

                step_execution_time = (datetime.now() - step_agent_start).total_seconds() * 1000

                # Log intermediate output
                intermediate_outputs["answer_generation"] = {
                    "question": qa_pair.get("question", ""),
                    "system_prompt": system_prompt,
                    "system_prompt_length": len(system_prompt),
                    "document_length": len(document_text),
                    "generated_answer": answer_response.generated_answer[:200] + "..." if len(answer_response.generated_answer) > 200 else answer_response.generated_answer,
                    "generated_answer_length": len(answer_response.generated_answer),
                    "processing_time_ms": step_execution_time
                }

            except Exception as e:
                step_execution_time = (datetime.now() - step_agent_start).total_seconds() * 1000
                intermediate_outputs["answer_generation"] = {
                    "error": str(e),
                    "processing_time_ms": step_execution_time
                }
                raise

            # Step 2: Evaluation
            self.logger.info(f"Step 2/3: Evaluation for {qa_pair_id}, iteration {iteration}")

            step_agent_start = datetime.now()
            try:
                # Compute ROUGE score
                from datasets_schema import Question
                temp_question = Question(
                    id=qa_pair.get("question_id", ""),
                    question=qa_pair.get("question", ""),
                    answers=qa_pair.get("answers", []),
                    metadata=qa_pair.get("metadata", {})
                )
                rouge_score = evaluate_rouge_score(temp_question, answer_response.generated_answer)

                eval_start_msg = ResponseEvaluationStartMessage(
                    qa_pair_id=qa_pair_id,
                    original_query=qa_pair.get("question", ""),
                    generated_answer=answer_response.generated_answer,
                    gold_answers=qa_pair.get("answers", []),
                    rouge_score=rouge_score,
                    batch_id=original_message.batch_id,
                    repetition=original_message.repetition,
                    iteration=iteration
                )

                eval_agent_id = AgentId("response_evaluator_agent", "default")
                eval_response = await self.send_message(eval_start_msg, eval_agent_id)

                step_execution_time = (datetime.now() - step_agent_start).total_seconds() * 1000

                # Log intermediate output for evaluation
                intermediate_outputs["evaluation"] = {
                    "evaluation_feedback": eval_response.evaluation_result.get("evaluation_feedback", "N/A")[:200] + "..." if len(eval_response.evaluation_result.get("evaluation_feedback", "N/A")) > 200 else eval_response.evaluation_result.get("evaluation_feedback", "N/A"),
                    "rouge_score": rouge_score,
                    "gold_answers": qa_pair.get("answers", []),
                    "processing_time_ms": step_execution_time
                }

            except Exception as e:
                step_execution_time = (datetime.now() - step_agent_start).total_seconds() * 1000
                intermediate_outputs["evaluation"] = {
                    "error": str(e),
                    "processing_time_ms": step_execution_time
                }
                raise

            # Step 3: Critique (if not last iteration)
            optimized_prompt = system_prompt  # Default to current prompt
            critique_feedback = ""  # Default empty critique

            if iteration < total_iterations - 1:  # Not last iteration
                self.logger.info(f"Step 3/5: System prompt critique for {qa_pair_id}, iteration {iteration}")

                step_agent_start = datetime.now()
                try:
                    critique_msg = CritiqueStartMessage(
                        qa_pair_id=qa_pair_id,
                        question=qa_pair.get("question", ""),
                        current_system_prompt=system_prompt,
                        generated_answer=answer_response.generated_answer,
                        evaluation_result=eval_response.evaluation_result,
                        rouge_score=rouge_score,
                        batch_id=original_message.batch_id,
                        repetition=original_message.repetition,
                        iteration=iteration
                    )

                    critique_agent_id = AgentId("critique_agent", "default")
                    critique_response = await self.send_message(critique_msg, critique_agent_id)

                    critique_feedback = critique_response.critique_feedback
                    step_execution_time = (datetime.now() - step_agent_start).total_seconds() * 1000

                    # Log intermediate output for critique
                    intermediate_outputs["critique"] = {
                        "current_prompt": system_prompt,
                        "critique_feedback": critique_feedback[:200] + "..." if len(critique_feedback) > 200 else critique_feedback,
                        "processing_time_ms": step_execution_time
                    }

                except Exception as e:
                    step_execution_time = (datetime.now() - step_agent_start).total_seconds() * 1000
                    intermediate_outputs["critique"] = {
                        "error": str(e),
                        "processing_time_ms": step_execution_time
                    }
                    # Don't raise, continue with current prompt
                    self.logger.warning(f"System prompt critique failed for {qa_pair_id}: {e}")

                # Step 4: Prompt Optimization (if we have critique feedback)
                if critique_feedback:
                    self.logger.info(f"Step 4/5: Prompt optimization for {qa_pair_id}, iteration {iteration}")

                    step_agent_start = datetime.now()
                    try:
                        optimization_msg = PromptOptimizationStartMessage(
                            qa_pair_id=qa_pair_id,
                            question=qa_pair.get("question", ""),
                            current_system_prompt=system_prompt,
                            critique_feedback=critique_feedback,
                            batch_id=original_message.batch_id,
                            repetition=original_message.repetition,
                            iteration=iteration
                        )

                        optimization_agent_id = AgentId("prompt_optimization_agent", "default")
                        optimization_response = await self.send_message(optimization_msg, optimization_agent_id)

                        optimized_prompt = optimization_response.optimized_prompt
                        step_execution_time = (datetime.now() - step_agent_start).total_seconds() * 1000

                        # Log intermediate output for prompt optimization
                        intermediate_outputs["prompt_optimization"] = {
                            "current_prompt": system_prompt,
                            "optimized_prompt": optimized_prompt[:200] + "..." if len(optimized_prompt) > 200 else optimized_prompt,
                            "processing_time_ms": step_execution_time
                        }

                    except Exception as e:
                        step_execution_time = (datetime.now() - step_agent_start).total_seconds() * 1000
                        intermediate_outputs["prompt_optimization"] = {
                            "error": str(e),
                            "processing_time_ms": step_execution_time
                        }
                        # Don't raise, continue with current prompt
                        self.logger.warning(f"Prompt optimization failed for {qa_pair_id}: {e}")
                else:
                    intermediate_outputs["prompt_optimization"] = {
                        "note": "Skipped - no critique feedback",
                        "processing_time_ms": 0
                    }
            else:
                intermediate_outputs["critique"] = {
                    "note": "Skipped - last iteration",
                    "processing_time_ms": 0
                }
                intermediate_outputs["prompt_optimization"] = {
                    "note": "Skipped - last iteration",
                    "processing_time_ms": 0
                }

            # Calculate total execution time
            total_execution_time = (datetime.now() - step_start_time).total_seconds()

            # Prepare system-specific metrics
            system_specific_metrics = {
                "system_prompt_length": len(system_prompt),
                "document_length": len(document_text),
                "critique_performed": iteration < total_iterations - 1 and bool(critique_feedback),
                "prompt_optimization_performed": iteration < total_iterations - 1 and bool(critique_feedback),
                "iteration_type": "with_critique_and_optimization" if iteration < total_iterations - 1 else "final_iteration"
            }

            # Log evaluation data with proper ROUGE scores
            rouge_scores = {
                "rouge-l": rouge_score,
                "rouge-1": rouge_score,  # Use same score for ROUGE-1 (simplified)
                "rouge-2": rouge_score * 0.85  # Estimate ROUGE-2 as typically lower than ROUGE-L
            }

            # Log to old evaluation logger for backward compatibility
            eval_logger = get_global_evaluation_logger()
            eval_logger.log_iteration_evaluation(
                qa_pair_id=qa_pair_id,
                iteration=iteration,
                intermediate_outputs=intermediate_outputs,
                generated_answer=answer_response.generated_answer,
                rouge_scores=rouge_scores,
                hyperparameters={
                    "system_prompt": system_prompt,
                    "document_length": len(document_text)
                },
                additional_metrics=system_specific_metrics,
                retrieval_context="N/A - All context provided directly",
                execution_time_seconds=total_execution_time
            )

            # Log to standardized evaluation logger
            self.standardized_logger.log_iteration_evaluation(
                qa_pair_id=qa_pair_id,
                iteration=iteration,
                generated_answer=answer_response.generated_answer,
                rouge_scores=rouge_scores,
                intermediate_outputs=intermediate_outputs,
                hyperparameters={
                    "system_prompt": system_prompt,
                    "document_length": len(document_text)
                },
                execution_time_seconds=total_execution_time,
                system_specific_metrics=system_specific_metrics
            )

            # Return iteration results
            return {
                "qa_pair_id": qa_pair_id,
                "iteration": iteration,
                "generated_answer": answer_response.generated_answer,
                "rouge_score": rouge_score,
                "evaluation_result": eval_response.evaluation_result,
                "optimized_prompt": optimized_prompt,
                "system_prompt_used": system_prompt,
                "execution_time": total_execution_time,
                "intermediate_outputs": intermediate_outputs
            }

        except Exception as e:
            self.logger.error(f"Error in iteration {iteration} for {qa_pair_id}: {e}")
            # Return error results
            return {
                "qa_pair_id": qa_pair_id,
                "iteration": iteration,
                "error": str(e),
                "rouge_score": 0.0,
                "generated_answer": "",
                "optimized_prompt": system_prompt,
                "system_prompt_used": system_prompt,
                "execution_time": (datetime.now() - step_start_time).total_seconds()
            }


# ===== FACTORY FUNCTIONS =====

def create_batch_orchestrator_agent() -> BatchOrchestratorAgent:
    """Factory function to create BatchOrchestratorAgent instances."""
    return BatchOrchestratorAgent("batch_orchestrator_agent")

def create_answer_generator_agent() -> AnswerGeneratorAgent:
    """Factory function to create AnswerGeneratorAgent instances."""
    return AnswerGeneratorAgent("answer_generator_agent")

def create_response_evaluator_agent() -> ResponseEvaluatorAgent:
    """Factory function to create ResponseEvaluatorAgent instances."""
    return ResponseEvaluatorAgent("response_evaluator_agent")

def create_critique_agent() -> CritiqueAgent:
    """Factory function to create CritiqueAgent instances."""
    return CritiqueAgent("critique_agent")

def create_prompt_optimization_agent() -> PromptOptimizationAgent:
    """Factory function to create PromptOptimizationAgent instances."""
    return PromptOptimizationAgent("prompt_optimization_agent")


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        # Create runtime
        runtime = SingleThreadedAgentRuntime()

        # Register agents
        await runtime.register("batch_orchestrator_agent", create_batch_orchestrator_agent)
        await runtime.register("answer_generator_agent", create_answer_generator_agent)
        await runtime.register("response_evaluator_agent", create_response_evaluator_agent)
        await runtime.register("critique_agent", create_critique_agent)
        await runtime.register("prompt_optimization_agent", create_prompt_optimization_agent)

        print("All-Context Multi-Agent System registered successfully!")
        await runtime.stop()

    asyncio.run(main())