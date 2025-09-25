"""
Multi-Agent GraphRAG System with AutoGen Core API.
Includes BatchOrchestratorAgent and HyperparametersGraphAgent.
"""

import json
import logging
import statistics
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel

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
    k_iterations: int = 3
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
    rouge_score: float
    batch_id: int
    repetition: int

class ResponseEvaluationReadyMessage(BaseModel):
    qa_pair_id: str
    evaluation_result: Dict[str, Any]
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

# Response format for HyperparametersGraphAgent
class HyperparametersGraphResponse(BaseModel):
    reasoning: str
    chunk_size: int
    confidence_score: float


# ===== BATCH ORCHESTRATOR AGENT =====

class BatchOrchestratorAgent(RoutedAgent):
    """
    Orchestrates the processing of QA pairs in a batch through the multi-agent pipeline.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.batch_orchestrator")
        self.shared_state = SharedState("agent_states")

        # Track QA pairs processing
        self.current_batch_qa_pairs: Dict[str, Dict[str, Any]] = {}
        self.completed_qa_pairs: Dict[str, Dict[str, Any]] = {}
        self.current_batch_id: Optional[int] = None
        self.current_repetition: Optional[int] = None
        self.current_dataset: Optional[str] = None
        self.current_setting: Optional[str] = None

    @message_handler
    async def handle_batch_start(self, message: BatchStartMessage, ctx: MessageContext) -> BatchReadyMessage:
        """Handle BatchStart message by iterating over QA pairs."""
        self.logger.info(f"BatchOrchestrator received BatchStart for batch {message.batch_id}")

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

            # Start processing each QA pair
            for qa_pair in qa_pairs:
                qa_pair_id = qa_pair.get("question_id", f"qa_{len(self.current_batch_qa_pairs)}")
                self.current_batch_qa_pairs[qa_pair_id] = qa_pair

                # Send HyperparametersGraphStart message
                hyperparams_msg = HyperparametersGraphStartMessage(
                    qa_pair_id=qa_pair_id,
                    qa_pair=qa_pair,
                    batch_id=message.batch_id,
                    repetition=message.repetition,
                    dataset=self.current_dataset,
                    setting=self.current_setting
                )

                self.logger.info(f"Sending HyperparametersGraphStart for QA pair {qa_pair_id}")

                # Create agent ID for HyperparametersGraphAgent
                hyperparams_agent_id = AgentId("hyperparameters_graph_agent", "default")

                # Get hyperparameters response
                hyperparams_response = await self.send_message(hyperparams_msg, hyperparams_agent_id)

                # Process the hyperparameters response
                await self._process_hyperparameters_response(hyperparams_response, message, ctx)

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

    async def _process_hyperparameters_response(self, hyperparams_response: HyperparametersGraphReadyMessage,
                                              original_message: BatchStartMessage, ctx: MessageContext) -> None:
        """Process the hyperparameters response and continue the pipeline."""
        self.logger.info(f"Processing hyperparameters response for QA pair {hyperparams_response.qa_pair_id}")

        # Load current shared state to pass to GraphBuilderAgent
        # Use the same dataset/setting from the current batch
        current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, hyperparams_response.batch_id)

        # Send GraphStart message with chunk_size
        graph_start_msg = GraphStartMessage(
            batch_id=hyperparams_response.batch_id,
            repetition=hyperparams_response.repetition,
            chunk_size=hyperparams_response.chunk_size,
            dataset=self.current_dataset,
            setting=self.current_setting,
            shared_state=current_state
        )

        self.logger.info(f"Sending GraphStart for batch {hyperparams_response.batch_id} with chunk_size {hyperparams_response.chunk_size}")

        # Send to GraphBuilderAgent and get response
        try:
            graph_builder_agent_id = AgentId("graph_builder_agent", "default")
            graph_response = await self.send_message(graph_start_msg, graph_builder_agent_id)
            self.logger.info(f"Received GraphReady response")

            # Continue processing with graph response
            await self._process_graph_response(graph_response, ctx)

        except Exception as e:
            self.logger.warning(f"GraphBuilderAgent not available, falling back to simulation: {e}")
            # Fallback to simulation if GraphBuilderAgent is not available
            await self._simulate_graph_ready(graph_start_msg, ctx)

    async def _process_graph_response(self, graph_response: GraphReadyMessage, ctx: MessageContext) -> None:
        """Process graph response and continue with retrieval."""
        self.logger.info(f"Processing graph response for batch {graph_response.batch_id}")

        # Load current shared state to get all QA pairs
        current_state = self.shared_state.load_state(graph_response.dataset, graph_response.setting, graph_response.batch_id)
        batch_info = current_state.get("batch_information", {})
        qa_pairs = batch_info.get("qa_pairs", [])

        # Process each QA pair for retrieval
        for qa_pair in qa_pairs:
            question = qa_pair.get("question", "")

            # Send GraphRetrievalStart message
            retrieval_start_msg = GraphRetrievalStartMessage(
                batch_id=graph_response.batch_id,
                repetition=graph_response.repetition,
                query=question,
                dataset=graph_response.dataset,
                setting=graph_response.setting,
                k_iterations=3,
                shared_state=current_state
            )

            self.logger.info(f"Sending GraphRetrievalStart for batch {graph_response.batch_id}, question: {question[:50]}...")

            # Send to GraphRetrievalPlannerAgent
            try:
                retrieval_agent_id = AgentId("graph_retrieval_planner_agent", "default")
                retrieval_response = await self.send_message(retrieval_start_msg, retrieval_agent_id)
                self.logger.info(f"Received GraphRetrievalReady response")

                # Continue with answer generation
                await self._process_retrieval_response(retrieval_response, ctx)

            except Exception as e:
                self.logger.warning(f"GraphRetrievalPlannerAgent not available, falling back to simulation: {e}")
                # Fallback to simulation if agent is not available
                await self._simulate_retrieval_ready(retrieval_start_msg, ctx)

    async def _process_retrieval_response(self, retrieval_response: GraphRetrievalReadyMessage, ctx: MessageContext) -> None:
        """Process retrieval response and continue with answer generation."""
        self.logger.info(f"Processing retrieval response for batch {retrieval_response.batch_id}")

        # Load current shared state to get all QA pairs
        current_state = self.shared_state.load_state(retrieval_response.dataset, retrieval_response.setting, retrieval_response.batch_id)
        batch_info = current_state.get("batch_information", {})
        qa_pairs = batch_info.get("qa_pairs", [])

        # Store retrieved context for backward pass critiques
        retrieved_contexts = current_state.get("retrieved_contexts", [])
        retrieved_contexts.append(retrieval_response.retrieved_context)
        current_state["retrieved_contexts"] = retrieved_contexts
        self.shared_state.save_state(current_state, retrieval_response.dataset, retrieval_response.setting, retrieval_response.batch_id)

        # Process each QA pair for answer generation
        for qa_pair in qa_pairs:
            qa_pair_id = qa_pair.get("question_id", f"qa_{len(qa_pairs)}")
            question = qa_pair.get("question", "")

            # Send AnswerGenerationStart message
            answer_gen_msg = AnswerGenerationStartMessage(
                qa_pair_id=qa_pair_id,
                question=question,
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
                self.logger.info(f"Received AnswerGenerationReady response")

                # Continue with evaluation
                await self._process_answer_response(answer_response, ctx)

            except Exception as e:
                self.logger.warning(f"AnswerGeneratorAgent not available, falling back to simulation: {e}")
                # Fallback to simulation if agent is not available
                await self._simulate_answer_generation_ready(answer_gen_msg, ctx)

    async def _process_answer_response(self, answer_response: AnswerGenerationReadyMessage, ctx: MessageContext) -> None:
        """Process answer generation response and continue with evaluation."""
        self.logger.info(f"Processing answer response for QA pair {answer_response.qa_pair_id}")

        qa_pair = self.current_batch_qa_pairs.get(answer_response.qa_pair_id)
        if not qa_pair:
            self.logger.error(f"QA pair {answer_response.qa_pair_id} not found")
            return

        # Compute ROUGE score
        gold_answers = qa_pair.get("answers", [])
        rouge_score = 0.0

        if gold_answers:
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
                score = evaluate_rouge_score(temp_question, answer_response.generated_answer)
                rouge_scores.append(score)
            rouge_score = max(rouge_scores) if rouge_scores else 0.0

        self.logger.info(f"Computed ROUGE score {rouge_score:.4f} for QA pair {answer_response.qa_pair_id}")

        # Save ROUGE score to shared state
        current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, answer_response.batch_id)
        rouge_scores_list = current_state.get("rouge_scores", [])
        rouge_scores_list.append(rouge_score)
        current_state["rouge_scores"] = rouge_scores_list
        self.shared_state.save_state(current_state, self.current_dataset, self.current_setting, answer_response.batch_id)

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
            self.logger.info(f"All QA pairs completed for batch {eval_response.batch_id}")
            await self._start_backward_pass(eval_response.batch_id, eval_response.repetition, ctx)

    @message_handler
    async def handle_hyperparameters_graph_ready(
        self, message: HyperparametersGraphReadyMessage, ctx: MessageContext
    ) -> None:
        """Handle HyperparametersGraphReady message and send GraphStart."""
        self.logger.info(f"Received HyperparametersGraphReady for QA pair {message.qa_pair_id}")

        qa_pair = self.current_batch_qa_pairs.get(message.qa_pair_id)
        if not qa_pair:
            self.logger.error(f"QA pair {message.qa_pair_id} not found")
            return

        # Load current shared state to pass to GraphBuilderAgent
        current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, message.batch_id)

        # Store RAG hyperparameters for backward pass critiques
        rag_hyperparams = {
            "chunk_size": message.chunk_size,
            "chunk_size_confidence": 0.85  # Mock confidence score
        }
        current_state["rag_hyperparameters"] = rag_hyperparams
        self.shared_state.save_state(current_state, self.current_dataset, self.current_setting, message.batch_id)

        # Send GraphStart message with chunk_size
        graph_start_msg = GraphStartMessage(
            batch_id=message.batch_id,
            repetition=message.repetition,
            chunk_size=message.chunk_size,
            dataset="dataset",
            setting="train",
            shared_state=current_state
        )

        self.logger.info(f"Sending GraphStart for batch {message.batch_id} with chunk_size {message.chunk_size}")

        # Send to GraphBuilderAgent
        try:
            graph_builder_agent_id = AgentId("graph_builder_agent", "default")
            await self.send_message(graph_start_msg, graph_builder_agent_id)
            self.logger.info(f"GraphStart message sent to GraphBuilderAgent")
        except Exception as e:
            self.logger.warning(f"GraphBuilderAgent not available, falling back to simulation: {e}")
            # Fallback to simulation if GraphBuilderAgent is not available
            await self._simulate_graph_ready(graph_start_msg, ctx)

    @message_handler
    async def handle_graph_ready(self, message: GraphReadyMessage, ctx: MessageContext) -> None:
        """Handle GraphReady message and send GraphRetrievalStart."""
        self.logger.info(f"Received GraphReady for batch {message.batch_id}")

        # Load current shared state to get all QA pairs
        current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
        batch_info = current_state.get("batch_information", {})
        qa_pairs = batch_info.get("qa_pairs", [])

        # Store graph description for backward pass critiques
        current_state["graph_description"] = message.graph_description
        self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

        # Process each QA pair for retrieval
        for qa_pair in qa_pairs:
            question = qa_pair.get("question", "")

            # Send GraphRetrievalStart message
            retrieval_start_msg = GraphRetrievalStartMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                query=question,
                dataset=message.dataset,
                setting=message.setting,
                k_iterations=3,
                shared_state=current_state
            )

            self.logger.info(f"Sending GraphRetrievalStart for batch {message.batch_id}, question: {question[:50]}...")

            # Send to GraphRetrievalPlannerAgent
            try:
                retrieval_agent_id = AgentId("graph_retrieval_planner_agent", "default")
                await self.send_message(retrieval_start_msg, retrieval_agent_id)
                self.logger.info(f"GraphRetrievalStart message sent to GraphRetrievalPlannerAgent")
            except Exception as e:
                self.logger.warning(f"GraphRetrievalPlannerAgent not available, falling back to simulation: {e}")
                # Fallback to simulation if agent is not available
                await self._simulate_retrieval_ready(retrieval_start_msg, ctx)

    @message_handler
    async def handle_graph_retrieval_ready(
        self, message: GraphRetrievalReadyMessage, ctx: MessageContext
    ) -> None:
        """Handle GraphRetrievalReady message and send AnswerGenerationStart."""
        self.logger.info(f"Received GraphRetrievalReady for batch {message.batch_id}")

        # Load current shared state to get all QA pairs
        current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
        batch_info = current_state.get("batch_information", {})
        qa_pairs = batch_info.get("qa_pairs", [])

        # Process each QA pair for answer generation
        for qa_pair in qa_pairs:
            qa_pair_id = qa_pair.get("question_id", f"qa_{len(qa_pairs)}")
            question = qa_pair.get("question", "")

            # Send AnswerGenerationStart message
            answer_gen_msg = AnswerGenerationStartMessage(
                qa_pair_id=qa_pair_id,
                question=question,
                retrieved_context=message.retrieved_context,
                batch_id=message.batch_id,
                repetition=message.repetition,
                dataset=message.dataset,
                setting=message.setting
            )

            self.logger.info(f"Sending AnswerGenerationStart for QA pair {qa_pair_id}")

            # Send to AnswerGeneratorAgent
            try:
                answer_gen_agent_id = AgentId("answer_generator_agent", "default")
                await self.send_message(answer_gen_msg, answer_gen_agent_id)
                self.logger.info(f"AnswerGenerationStart message sent to AnswerGeneratorAgent")
            except Exception as e:
                self.logger.warning(f"AnswerGeneratorAgent not available, falling back to simulation: {e}")
                # Fallback to simulation if agent is not available
                await self._simulate_answer_generation_ready(answer_gen_msg, ctx)

    @message_handler
    async def handle_answer_generation_ready(
        self, message: AnswerGenerationReadyMessage, ctx: MessageContext
    ) -> None:
        """Handle AnswerGenerationReady message, compute ROUGE score, and send ResponseEvaluationStart."""
        self.logger.info(f"Received AnswerGenerationReady for QA pair {message.qa_pair_id}")

        qa_pair = self.current_batch_qa_pairs.get(message.qa_pair_id)
        if not qa_pair:
            self.logger.error(f"QA pair {message.qa_pair_id} not found")
            return

        # Compute ROUGE score
        gold_answers = qa_pair.get("answers", [])
        rouge_score = 0.0

        if gold_answers:
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
                score = evaluate_rouge_score(temp_question, message.generated_answer)
                rouge_scores.append(score)
            rouge_score = max(rouge_scores) if rouge_scores else 0.0

        self.logger.info(f"Computed ROUGE score {rouge_score:.4f} for QA pair {message.qa_pair_id}")

        # Save ROUGE score to shared state
        current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, message.batch_id)
        rouge_scores_list = current_state.get("rouge_scores", [])
        rouge_scores_list.append(rouge_score)
        current_state["rouge_scores"] = rouge_scores_list
        self.shared_state.save_state(current_state, self.current_dataset, self.current_setting, message.batch_id)

        # Send ResponseEvaluationStart message
        eval_start_msg = ResponseEvaluationStartMessage(
            qa_pair_id=message.qa_pair_id,
            original_query=qa_pair.get("question", ""),
            generated_answer=message.generated_answer,
            gold_answers=gold_answers,
            rouge_score=rouge_score,
            batch_id=message.batch_id,
            repetition=message.repetition
        )

        self.logger.info(f"Sending ResponseEvaluationStart for QA pair {message.qa_pair_id}")

        # Send to ResponseEvaluatorAgent
        try:
            response_eval_agent_id = AgentId("response_evaluator_agent", "default")
            await self.send_message(eval_start_msg, response_eval_agent_id)
            self.logger.info(f"ResponseEvaluationStart message sent to ResponseEvaluatorAgent")
        except Exception as e:
            self.logger.warning(f"ResponseEvaluatorAgent not available, falling back to simulation: {e}")
            # Fallback to simulation if agent is not available
            await self._simulate_response_evaluation_ready(eval_start_msg, ctx)

    @message_handler
    async def handle_response_evaluation_ready(
        self, message: ResponseEvaluationReadyMessage, ctx: MessageContext
    ) -> None:
        """Handle ResponseEvaluationReady message and track completion."""
        self.logger.info(f"Received ResponseEvaluationReady for QA pair {message.qa_pair_id}")

        # Add to shared state
        current_state = self.shared_state.load_state(self.current_dataset, self.current_setting, message.batch_id)
        response_evaluations = current_state.get("response_evaluations", [])
        response_evaluations.append(message.evaluation_result)
        current_state["response_evaluations"] = response_evaluations
        self.shared_state.save_state(current_state, self.current_dataset, self.current_setting, message.batch_id)

        # Mark QA pair as completed
        self.completed_qa_pairs[message.qa_pair_id] = message.evaluation_result

        # Check if all QA pairs are completed
        if len(self.completed_qa_pairs) == len(self.current_batch_qa_pairs):
            self.logger.info(f"All QA pairs completed for batch {message.batch_id}")
            await self._start_backward_pass(message.batch_id, message.repetition, ctx)

    @message_handler
    async def handle_backward_pass_ready(
        self, message: BackwardPassReadyMessage, ctx: MessageContext
    ) -> None:
        """Handle BackwardPassReady message and send BatchReady to DatasetAgent."""
        self.logger.info(f"Received BackwardPassReady for batch {message.batch_id}")

        # Send BatchReady message to DatasetAgent
        batch_ready_msg = BatchReadyMessage(
            batch_id=message.batch_id,
            repetition=message.repetition,
            status="completed",
            metrics=message.backward_pass_results
        )

        self.logger.info(f"Sending BatchReady to DatasetAgent for batch {message.batch_id}")

        # Send to DatasetAgent
        dataset_agent_id = AgentId("dataset_agent", "default")
        await self.send_message(batch_ready_msg, dataset_agent_id)

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

        # Send BatchReady message to DatasetAgent
        batch_ready_msg = BatchReadyMessage(
            batch_id=backward_response.batch_id,
            repetition=backward_response.repetition,
            status="completed",
            metrics=backward_response.backward_pass_results
        )

        self.logger.info(f"Sending BatchReady to DatasetAgent for batch {backward_response.batch_id}")

        # Note: This is the final step - we don't need to send this, the BatchOrchestratorAgent's
        # handle_batch_start method already returns the BatchReady message to the DatasetAgent

    # Simulation methods for testing
    async def _simulate_graph_ready(self, graph_start: GraphStartMessage, ctx: MessageContext) -> None:
        # Store simulation data in shared state for backward pass critiques
        current_state = self.shared_state.load_state(graph_start.dataset, graph_start.setting, graph_start.batch_id)

        # Store mock graph builder prompt (simulating what GraphBuilderAgent would store)
        current_state["graph_builder_prompt"] = "Mock graph builder prompt for entity and relationship extraction"

        self.shared_state.save_state(current_state, graph_start.dataset, graph_start.setting, graph_start.batch_id)

        graph_ready_msg = GraphReadyMessage(
            batch_id=graph_start.batch_id,
            repetition=graph_start.repetition,
            graph_description="Mock graph description with entities and relationships",
            connectivity_metrics={"density": 0.3, "fragmentation_index": 0.2},
            dataset=graph_start.dataset,
            setting=graph_start.setting
        )
        await self.handle_graph_ready(graph_ready_msg, ctx)

    async def _simulate_retrieval_ready(self, retrieval_start: GraphRetrievalStartMessage, ctx: MessageContext) -> None:
        # Store simulation data in shared state for backward pass critiques
        current_state = self.shared_state.load_state(retrieval_start.dataset, retrieval_start.setting, retrieval_start.batch_id)

        # Store mock retrieval prompt and plans (simulating what GraphRetrievalPlannerAgent would store)
        current_state["retrieval_prompt"] = "Mock retrieval prompt template for graph queries"
        current_state["retrieval_plans"] = ["Mock retrieval plan 1", "Mock retrieval plan 2"]

        # Store mock retrieved contexts for backward pass
        retrieved_contexts = current_state.get("retrieved_contexts", [])
        retrieved_contexts.append("Mock retrieved context from graph")
        current_state["retrieved_contexts"] = retrieved_contexts

        self.shared_state.save_state(current_state, retrieval_start.dataset, retrieval_start.setting, retrieval_start.batch_id)

        retrieval_ready_msg = GraphRetrievalReadyMessage(
            batch_id=retrieval_start.batch_id,
            repetition=retrieval_start.repetition,
            retrieved_context="Mock retrieved context from graph",
            dataset=retrieval_start.dataset,
            setting=retrieval_start.setting
        )
        await self.handle_graph_retrieval_ready(retrieval_ready_msg, ctx)

    async def _simulate_answer_generation_ready(self, answer_gen: AnswerGenerationStartMessage, ctx: MessageContext) -> None:
        answer_ready_msg = AnswerGenerationReadyMessage(
            qa_pair_id=answer_gen.qa_pair_id,
            generated_answer="Mock generated answer based on retrieved context",
            batch_id=answer_gen.batch_id,
            repetition=answer_gen.repetition
        )
        await self.handle_answer_generation_ready(answer_ready_msg, ctx)

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
        await self.handle_response_evaluation_ready(eval_ready_msg, ctx)

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
        await self.handle_backward_pass_ready(backward_ready_msg, ctx)


# ===== HYPERPARAMETERS GRAPH AGENT =====

class HyperparametersGraphAgent(RoutedAgent):
    """
    Agent that determines graph construction hyperparameters using LLM reasoning.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.hyperparameters_graph")
        self.shared_state = SharedState("agent_states")

        # Initialize OpenAI model client with structured output
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=llm_keys.OPENAI_KEY,
            response_format=HyperparametersGraphResponse
        )

        # Base prompt for hyperparameters determination
        self.base_prompt_hyperparameters_graph = """
        You are an expert in graph-based retrieval-augmented generation (GraphRAG) systems.
        Your task is to determine optimal hyperparameters for graph construction based on the input text and any previous critiques.

        Consider the following factors when determining the chunk_size:
        1. Text complexity and length
        2. Entity density and relationship complexity
        3. Previous performance feedback from critiques
        4. Balance between granularity and context preservation

        Previous critique (if any): {critique}

        Text to analyze: {text}

        Question: {question}

        Provide your reasoning and determine an optimal chunk_size (between 100 and 2000 characters).
        Also provide a confidence score (0.0 to 1.0) for your recommendation.
        """

    @message_handler
    async def handle_hyperparameters_graph_start(
        self, message: HyperparametersGraphStartMessage, ctx: MessageContext
    ) -> HyperparametersGraphReadyMessage:
        """Handle HyperparametersGraphStart message and generate hyperparameters using LLM."""
        self.logger.info(f"HyperparametersGraphAgent processing QA pair {message.qa_pair_id}")

        try:
            # Load shared state to get critique
            current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
            critique = current_state.get("hyperparameters_graph_agent_critique", "")

            # Extract text and question from QA pair
            qa_pair = message.qa_pair
            question = qa_pair.get("question", "")

            # Get document text from current state (assuming it's available)
            batch_info = current_state.get("batch_information", {})
            document_text = batch_info.get("document_text", "")[:1000]  # Limit for prompt

            # Prepare prompt
            prompt_content = self.base_prompt_hyperparameters_graph.format(
                critique=critique or "No previous critique available.",
                text=document_text,
                question=question
            )

            # Call LLM with structured output
            system_message = SystemMessage(content=prompt_content)
            user_message = UserMessage(content="Please analyze and provide hyperparameters.", source="system")

            response = await self.model_client.create(
                [system_message, user_message],
                cancellation_token=ctx.cancellation_token
            )

            # Parse structured response
            assert isinstance(response.content, str)
            hyperparams_response = HyperparametersGraphResponse.model_validate_json(response.content)

            self.logger.info(f"LLM recommended chunk_size: {hyperparams_response.chunk_size} "
                           f"(confidence: {hyperparams_response.confidence_score:.2f})")

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
        from parameters import base_prompt_graph_builder, GraphBuilderResponse

        # Initialize OpenAI model client with structured output
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=llm_keys.OPENAI_KEY,
            response_format=GraphBuilderResponse
        )

        self.base_prompt_graph_builder = base_prompt_graph_builder
        

    @message_handler
    async def handle_graph_start(self, message: GraphStartMessage, ctx: MessageContext) -> GraphReadyMessage:
        """Handle GraphStart message by chunking text and building graph."""
        self.logger.info(f"GraphBuilderAgent processing batch {message.batch_id} with chunk_size {message.chunk_size}")

        # Load shared state
        current_state = message.shared_state
        batch_info = current_state.get("batch_information", {})
        corpus = batch_info.get("document_text", "")

        if not corpus:
            self.logger.error("No document text found in batch information")
            return

        # Get critique from shared state
        critique = current_state.get("graph_builder_agent_critique", "")

        # Split corpus into chunks
        chunks = self._split_text_into_chunks(corpus, message.chunk_size)
        self.logger.info(f"Split corpus into {len(chunks)} chunks")

        # Save prompt template (without text chunk) to shared state
        # The base_prompt has two {} placeholders: first for text chunk, second for critique
        # Format the critique first, then replace the text placeholder
        prompt_with_critique = self.base_prompt_graph_builder.format("{TEXT_CHUNK}", critique or "No critique available.")
        current_state["graph_builder_prompt"] = prompt_with_critique

        # Initialize graph data structure
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
                entities, relationships, triplets = await self._process_chunk(chunk, critique, ctx)
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
                json.dump(graph_json, f, indent=2)
            self.logger.info(f"Saved graph to {graph_filename}")
        except Exception as e:
            self.logger.error(f"Error saving graph file: {e}")

        # Load graph into Memgraph server
        try:
            from graph_functions import load_graph_from_json_flexible, clear_graph
            import os

            # Clear existing graph to free up memory
            self.logger.info("Clearing existing graph from Memgraph to free memory")
            clear_result = clear_graph()
            if clear_result.get("status") == "success":
                self.logger.info(f"Graph cleared successfully: {clear_result.get('message')}")
            else:
                self.logger.warning(f"Failed to clear graph: {clear_result.get('message')}")

            # Use the filename relative to the mounted volume
            graph_file_basename = os.path.basename(graph_filename)
            load_result = load_graph_from_json_flexible(graph_file_basename)

            if load_result.get("status") == "success":
                self.logger.info(f"Successfully loaded graph {graph_file_basename} into Memgraph server")
            else:
                self.logger.error(f"Failed to load graph: {load_result.get('message')}")
        except Exception as e:
            self.logger.error(f"Error loading graph into Memgraph: {e}")

        # Generate graph description and connectivity metrics
        try:
            from graph_functions import generate_graph_description
            graph_description_result = generate_graph_description()

            graph_description = graph_description_result.get("description", "")
            connectivity_metrics = {
                "density": graph_description_result.get("density", 0.0),
                "fragmentation_index": graph_description_result.get("fragmentation_index", 0.0),
                "largest_component_size": graph_description_result.get("largest_component_size", 0),
                "total_nodes": graph_description_result.get("total_nodes", 0),
                "total_relationships": graph_description_result.get("total_relationships", 0)
            }

            # Save to shared state
            current_state["graph_description"] = graph_description
            current_state["graph_statistics"] = connectivity_metrics

            self.shared_state.save_state(current_state, message.dataset, message.setting, message.batch_id)

        except Exception as e:
            self.logger.error(f"Error generating graph description: {e}")
            graph_description = "Graph description generation failed"
            connectivity_metrics = {}

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

    async def _process_chunk(self, chunk: str, critique: str, ctx: MessageContext) -> tuple:
        """Process a text chunk to extract entities and relationships."""
        # Prepare prompt
        prompt_content = self.base_prompt_graph_builder.format(chunk, critique or "No critique available.")

        # Call LLM with structured output
        system_message = SystemMessage(content=prompt_content)
        user_message = UserMessage(content="Please extract entities and relationships.", source="system")

        response = await self.model_client.create(
            [system_message, user_message],
            cancellation_token=ctx.cancellation_token
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

            node = {
                "id": node_id,
                "labels": [entity.type],
                "properties": {
                    "name": entity.name,
                    "type": entity.type
                },
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

                # Collect all names for mapping
                for entity in cluster:
                    entity_name_mapping[entity.name] = primary_entity.name

                merged_entities.append(primary_entity)

        # Update relationships to use canonical entity names
        updated_relationships = []
        for rel in relationships:
            # Map source and target entities to their canonical names
            canonical_source = entity_name_mapping.get(rel.source_entity, rel.source_entity)
            canonical_target = entity_name_mapping.get(rel.target_entity, rel.target_entity)

            # Skip self-relationships that might have been created by merging
            if canonical_source == canonical_target:
                continue

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

        self.logger.info(f"Entity resolution completed: {entities_before} → {entities_after} entities ({entities_merged} merged)")

        return merged_entities, updated_relationships, updated_triplets

    async def close(self) -> None:
        """Close the model client."""
        await self.model_client.close()


# ===== GRAPH RETRIEVAL PLANNER AGENT =====

class GraphRetrievalPlannerAgent(RoutedAgent):
    """
    Agent that plans and executes graph retrieval strategies using iterative LLM calls.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.graph_retrieval_planner")
        self.shared_state = SharedState("agent_states")

        # Import response formats and prompts
        from parameters import base_prompt_graph_retrieval_planner, GraphRetrievalPlannerResponse

        # Initialize OpenAI model client with structured output
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=llm_keys.OPENAI_KEY,
            response_format=GraphRetrievalPlannerResponse
        )

        self.base_prompt_graph_retrieval_planner = base_prompt_graph_retrieval_planner

    @message_handler
    async def handle_graph_retrieval_start(self, message: GraphRetrievalStartMessage, ctx: MessageContext) -> GraphRetrievalReadyMessage:
        """Handle GraphRetrievalStart message and execute iterative retrieval."""
        self.logger.info(f"GraphRetrievalPlannerAgent processing batch {message.batch_id} for query: {message.query}")

        # Ensure the correct graph is loaded in Memgraph
        import os
        graphs_dir = "graphs"
        graph_filename = os.path.join(graphs_dir, f"{message.dataset}_{message.setting}_batch_{message.batch_id}_graph.json")

        if os.path.exists(graph_filename):
            try:
                from graph_functions import load_graph_from_json_flexible, clear_graph

                # Clear existing graph to ensure we have the correct one loaded
                self.logger.info("Clearing existing graph from Memgraph before loading correct graph")
                clear_result = clear_graph()
                if clear_result.get("status") == "success":
                    self.logger.info(f"Graph cleared successfully: {clear_result.get('message')}")
                else:
                    self.logger.warning(f"Failed to clear graph: {clear_result.get('message')}")

                # Use the filename relative to the mounted volume
                graph_file_basename = os.path.basename(graph_filename)
                load_result = load_graph_from_json_flexible(graph_file_basename)

                if load_result.get("status") == "success":
                    self.logger.info(f"Successfully loaded correct graph {graph_file_basename} into Memgraph for retrieval")
                else:
                    self.logger.error(f"Failed to load graph: {load_result.get('message')}")
            except Exception as e:
                self.logger.error(f"Error loading graph {graph_filename}: {e}")
        else:
            self.logger.warning(f"Graph file {graph_filename} not found for retrieval")

        # Load shared state
        current_state = message.shared_state
        critique = current_state.get("retrieval_planner_agent_critique", "")

        # Get graph description from shared state
        graph_description = current_state.get("graph_description", "")

        # Create retrieval prompt template and save to shared state
        prompt_template = self.base_prompt_graph_retrieval_planner.format(
            message.query, graph_description, "{RETRIEVED_CONTEXT}", critique or "No critique available."
        )
        current_state["retrieval_prompt"] = prompt_template

        # Initialize retrieval context and plan responses
        retrieved_context = ""
        retrieval_plan_responses = []

        # Execute k iterations of retrieval
        for iteration in range(message.k_iterations):
            self.logger.info(f"Retrieval iteration {iteration + 1}/{message.k_iterations}")

            try:
                # Prepare prompt with current context
                prompt_content = self.base_prompt_graph_retrieval_planner.format(
                    message.query, graph_description, retrieved_context, critique or "No critique available."
                )

                # Call LLM to get next retrieval step
                system_message = SystemMessage(content=prompt_content)
                user_message = UserMessage(content="Please plan the next retrieval step.", source="system")

                response = await self.model_client.create(
                    [system_message, user_message],
                    cancellation_token=ctx.cancellation_token
                )

                # Parse structured response
                assert isinstance(response.content, str)
                from parameters import GraphRetrievalPlannerResponse
                retrieval_response = GraphRetrievalPlannerResponse.model_validate_json(response.content)

                # Store the LLM response
                retrieval_plan_responses.append(retrieval_response.reasoning)

                # Execute the function call and retrieve context
                new_context = await self._execute_retrieval_function(retrieval_response.function_call)

                # Add to retrieved context
                if new_context:
                    retrieved_context += f"\n\nIteration {iteration + 1} results:\n{new_context}"

                self.logger.info(f"Completed iteration {iteration + 1}, context length: {len(retrieved_context)}")

            except Exception as e:
                self.logger.error(f"Error in retrieval iteration {iteration + 1}: {e}")
                continue

        # Save retrieval plans to shared state
        current_state["retrieval_plans"] = retrieval_plan_responses
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

    async def _execute_retrieval_function(self, function_call) -> str:
        """Execute the graph retrieval function based on LLM response."""
        try:
            function_name = function_call.function_name

            # Import graph functions
            from graph_functions import (
                search_nodes_by_keyword, search_nodes_by_types, get_neighbors,
                search_relations_by_type, identify_communities, analyze_path, find_hub_nodes
            )

            if function_name == "search_nodes_by_keyword":
                result = search_nodes_by_keyword(function_call.keyword)
            elif function_name == "search_nodes_by_types":
                result = search_nodes_by_types(function_call.node_type)
            elif function_name == "get_neighbors":
                result = get_neighbors(function_call.node_name)
            elif function_name == "search_relations_by_type":
                result = search_relations_by_type(function_call.relation_type)
            elif function_name == "identify_communities":
                result = identify_communities(function_call.node_name)
            elif function_name == "analyze_path":
                result = analyze_path(function_call.start_node_name, function_call.end_node_name)
            elif function_name == "find_hub_nodes":
                result = find_hub_nodes()
            else:
                self.logger.error(f"Unknown function: {function_name}")
                return ""

            # Convert result to string format
            if isinstance(result, dict):
                graph_string = result.get("graph_string", "")
                if graph_string:
                    return graph_string
                return json.dumps(result, indent=2)
            elif isinstance(result, list):
                return json.dumps(result, indent=2)
            else:
                return str(result)

        except Exception as e:
            self.logger.error(f"Error executing retrieval function {function_call.function_name}: {e}")
            return ""

    async def close(self) -> None:
        """Close the model client."""
        await self.model_client.close()


# ===== UPDATED FACTORY FUNCTIONS =====

def create_graph_builder_agent() -> GraphBuilderAgent:
    """Factory function to create GraphBuilderAgent instances."""
    return GraphBuilderAgent("graph_builder_agent")

def create_graph_retrieval_planner_agent() -> GraphRetrievalPlannerAgent:
    """Factory function to create GraphRetrievalPlannerAgent instances."""
    return GraphRetrievalPlannerAgent("graph_retrieval_planner_agent")


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
        from parameters import base_prompt_answer_generator_graph

        # Initialize OpenAI model client for simple text response
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=llm_keys.OPENAI_KEY
        )

        self.base_prompt_answer_generator_graph = base_prompt_answer_generator_graph

    @message_handler
    async def handle_answer_generation_start(self, message: AnswerGenerationStartMessage, ctx: MessageContext) -> AnswerGenerationReadyMessage:
        """Handle AnswerGenerationStart message and generate answer using LLM."""
        self.logger.info(f"AnswerGeneratorAgent processing QA pair {message.qa_pair_id}")

        # Load shared state to get critique
        current_state = self.shared_state.load_state(message.dataset, message.setting, message.batch_id)
        critique = current_state.get("answer_generation_critique", "")

        # Prepare prompt with question, retrieved context, and critique
        prompt_content = self.base_prompt_answer_generator_graph.format(
            message.question, message.retrieved_context, critique or "No critique available."
        )

        try:
            # Call LLM for answer generation
            system_message = SystemMessage(content=prompt_content)
            user_message = UserMessage(content="Please generate an answer based on the retrieved context.", source="system")

            response = await self.model_client.create(
                [system_message, user_message],
                cancellation_token=ctx.cancellation_token
            )

            # Get generated answer
            generated_answer = response.content if isinstance(response.content, str) else str(response.content)

            self.logger.info(f"Generated answer for QA pair {message.qa_pair_id}: {generated_answer[:100]}...")

            # Store conversation in shared state
            conversation_entry = {
                "qa_pair_id": message.qa_pair_id,
                "question": message.question,
                "retrieved_context": message.retrieved_context,
                "prompt": prompt_content,
                "generated_answer": generated_answer,
                "critique": critique
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

        # Initialize OpenAI model client for simple text response
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=llm_keys.OPENAI_KEY
        )

        self.response_evaluator_prompt = response_evaluator_prompt

    @message_handler
    async def handle_response_evaluation_start(self, message: ResponseEvaluationStartMessage, ctx: MessageContext) -> ResponseEvaluationReadyMessage:
        """Handle ResponseEvaluationStart message and evaluate response using LLM."""
        self.logger.info(f"ResponseEvaluatorAgent evaluating QA pair {message.qa_pair_id}")

        # Prepare gold responses (join multiple answers if available)
        gold_response = " | ".join(message.gold_answers) if message.gold_answers else "No gold response available"

        # Prepare prompt with query, generated response, gold response, and ROUGE score
        prompt_content = self.response_evaluator_prompt.format(
            message.original_query,
            message.generated_answer,
            gold_response,
            message.rouge_score
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

            self.logger.info(f"Evaluation completed for QA pair {message.qa_pair_id}: {evaluation_result[:100]}...")

            # Create evaluation result dictionary
            evaluation_data = {
                "qa_pair_id": message.qa_pair_id,
                "original_query": message.original_query,
                "generated_answer": message.generated_answer,
                "gold_answers": message.gold_answers,
                "rouge_score": message.rouge_score,
                "evaluation_feedback": evaluation_result,
                "timestamp": "2025-09-22T15:30:00"  # Could be made dynamic
            }

            # Send ResponseEvaluationReady message
            eval_ready_msg = ResponseEvaluationReadyMessage(
                qa_pair_id=message.qa_pair_id,
                evaluation_result=evaluation_data,
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


# ===== UPDATED FACTORY FUNCTIONS =====

def create_answer_generator_agent() -> AnswerGeneratorAgent:
    """Factory function to create AnswerGeneratorAgent instances."""
    return AnswerGeneratorAgent("answer_generator_agent")

def create_response_evaluator_agent() -> ResponseEvaluatorAgent:
    """Factory function to create ResponseEvaluatorAgent instances."""
    return ResponseEvaluatorAgent("response_evaluator_agent")


# ===== BACKWARD PASS AGENT =====

class BackwardPassAgent(RoutedAgent):
    """
    Agent that performs backward pass through all agent critiques for system improvement.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.backward_pass")
        self.shared_state = SharedState("agent_states")

        # Import gradient prompts
        from parameters import (
            generation_prompt_gradient_prompt,
            retrieved_content_gradient_prompt_graph,
            retrieval_plan_gradient_prompt_graph,
            retrieval_planning_prompt_gradient_prompt,
            graph_gradient_prompt,
            graph_extraction_prompt_gradient_prompt,
            rag_hyperparameters_agent_gradient_prompt
        )

        # Initialize OpenAI model client for simple text response
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=llm_keys.OPENAI_KEY
        )

        # Store all gradient prompts
        self.generation_prompt_gradient_prompt = generation_prompt_gradient_prompt
        self.retrieved_content_gradient_prompt_graph = retrieved_content_gradient_prompt_graph
        self.retrieval_plan_gradient_prompt_graph = retrieval_plan_gradient_prompt_graph
        self.retrieval_planning_prompt_gradient_prompt = retrieval_planning_prompt_gradient_prompt
        self.graph_gradient_prompt = graph_gradient_prompt
        self.graph_extraction_prompt_gradient_prompt = graph_extraction_prompt_gradient_prompt
        self.rag_hyperparameters_agent_gradient_prompt = rag_hyperparameters_agent_gradient_prompt

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

            # Send BackwardPassReady message
            backward_ready_msg = BackwardPassReadyMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                backward_pass_results={
                    "critiques_generated": True,
                    "total_qa_pairs": len(message.all_qa_results),
                    "critiques_updated": [
                        "answer_generation_critique",
                        "retrieved_content_critique",
                        "retrieval_plan_critique",
                        "retrieval_planner_agent_critique",
                        "graph_critique",
                        "graph_builder_agent_critique",
                        "hyperparameters_graph_agent_critique"
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

        conversations = current_state.get("conversations_answer_generation", [])
        evaluation_responses = current_state.get("response_evaluations", [])

        if not conversations or not evaluation_responses:
            self.logger.warning("No conversations or evaluations found for answer generation critique")
            return

        # Create triplets: conversation (prompt-answer) + evaluation
        triplets = []
        for i, conv in enumerate(conversations):
            if i < len(evaluation_responses):
                eval_resp = evaluation_responses[i]
                triplet = f"Conversation: {conv.get('prompt', '')} -> {conv.get('generated_answer', '')}\nEvaluation: {eval_resp.get('evaluation_feedback', '')}"
                triplets.append(triplet)

        concatenated_feedback = "\n\n".join(triplets)

        # Call LLM with generation_prompt_gradient_prompt
        prompt_content = self.generation_prompt_gradient_prompt.format(concatenated_feedback)

        critique = await self._call_llm(prompt_content, ctx)
        current_state["answer_generation_critique"] = critique

        self.logger.info("Answer generation critique generated and saved")

    async def _generate_retrieved_content_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for retrieved content based on conversations, contexts, and evaluations."""
        self.logger.info("Generating retrieved content critique")

        conversations = current_state.get("conversations_answer_generation", [])
        retrieved_contexts = current_state.get("retrieved_contexts", [])
        evaluation_responses = current_state.get("response_evaluations", [])

        if not conversations or not evaluation_responses:
            self.logger.warning("Missing data for retrieved content critique")
            return

        # Create quadruplets: retrieved context + conversation (prompt-answer) + evaluation response
        quadruplets = []
        for i, conv in enumerate(conversations):
            if i < len(evaluation_responses):
                context = retrieved_contexts[i] if i < len(retrieved_contexts) else "No context available"
                eval_resp = evaluation_responses[i]
                quadruplet = f"Retrieved Context: {context}\nConversation: {conv.get('prompt', '')} -> {conv.get('generated_answer', '')}\nEvaluation: {eval_resp.get('evaluation_feedback', '')}"
                quadruplets.append(quadruplet)

        concatenated_data = "\n\n".join(quadruplets)

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

        # Create pairs (retrieval_plan, retrieved_context) with same indices
        pairs = []
        for i in range(min(len(retrieval_plans), len(retrieved_contexts))):
            pair = f"Retrieval Plan: {retrieval_plans[i]}\nRetrieved Context: {retrieved_contexts[i]}"
            pairs.append(pair)

        concatenated_pairs = "\n\n".join(pairs)

        # Get retrieved_content_critique for the second variable
        retrieved_content_critique = current_state.get("retrieved_content_critique", "No critique available")

        # Call LLM with retrieval_plan_gradient_prompt_graph
        prompt_content = self.retrieval_plan_gradient_prompt_graph.format(concatenated_pairs, retrieved_content_critique)

        critique = await self._call_llm(prompt_content, ctx)
        current_state["retrieval_plan_critique"] = critique

        self.logger.info("Retrieval plan critique generated and saved")

    async def _generate_retrieval_planning_prompt_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for retrieval planning prompt."""
        self.logger.info("Generating retrieval planning prompt critique")

        retrieval_prompt = current_state.get("retrieval_prompt", "")
        graph_description = current_state.get("graph_description", "")
        retrieval_plans = current_state.get("retrieval_plans", [])

        if not retrieval_prompt or not retrieval_plans:
            self.logger.warning("Missing retrieval prompt or plans for critique")
            return

        # Create triplets: retrieval_prompt + graph_description + retrieval_plan
        triplets = []
        for plan in retrieval_plans:
            triplet = f"Retrieval Prompt: {retrieval_prompt}\nGraph Description: {graph_description}\nRetrieval Plan: {plan}"
            triplets.append(triplet)

        concatenated_triplets = "\n\n".join(triplets)

        # Get retrieval_plan_critique for the second variable
        retrieval_plan_critique = current_state.get("retrieval_plan_critique", "No critique available")

        # Call LLM with retrieval_planning_prompt_gradient_prompt
        prompt_content = self.retrieval_planning_prompt_gradient_prompt.format(concatenated_triplets, retrieval_plan_critique)

        critique = await self._call_llm(prompt_content, ctx)
        current_state["retrieval_planner_agent_critique"] = critique

        self.logger.info("Retrieval planning prompt critique generated and saved")

    async def _generate_graph_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for the graph based on questions, description, and retrieval plans."""
        self.logger.info("Generating graph critique")

        batch_info = current_state.get("batch_information", {})
        qa_pairs = batch_info.get("qa_pairs", [])
        graph_description = current_state.get("graph_description", "")
        retrieval_plans = current_state.get("retrieval_plans", [])

        if not qa_pairs or not graph_description or not retrieval_plans:
            self.logger.warning("Missing data for graph critique")
            return

        # Create triplets: query + graph description + retrieval plan
        triplets = []
        for i, qa_pair in enumerate(qa_pairs):
            question = qa_pair.get("question", "")
            plan = retrieval_plans[i] if i < len(retrieval_plans) else "No plan available"
            triplet = f"Query: {question}\nGraph Description: {graph_description}\nRetrieval Plan: {plan}"
            triplets.append(triplet)

        concatenated_triplets = "\n\n".join(triplets)

        # Get retrieval_plan_critique for the second variable
        retrieval_plan_critique = current_state.get("retrieval_plan_critique", "No critique available")

        # Call LLM with graph_gradient_prompt
        prompt_content = self.graph_gradient_prompt.format(concatenated_triplets, retrieval_plan_critique)

        critique = await self._call_llm(prompt_content, ctx)
        current_state["graph_critique"] = critique

        self.logger.info("Graph critique generated and saved")

    async def _generate_graph_builder_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for graph builder prompt."""
        self.logger.info("Generating graph builder critique")

        graph_builder_prompt = current_state.get("graph_builder_prompt", "")
        batch_info = current_state.get("batch_information", {})
        corpus_sample = batch_info.get("document_text", "")[:500]  # Sample of corpus
        graph_description = current_state.get("graph_description", "")
        graph_critique = current_state.get("graph_critique", "No critique available")

        if not graph_builder_prompt or not corpus_sample or not graph_description:
            self.logger.warning("Missing data for graph builder critique")
            return

        # Call LLM with graph_extraction_prompt_gradient_prompt
        prompt_content = self.graph_extraction_prompt_gradient_prompt.format(
            graph_builder_prompt, corpus_sample, graph_description, graph_critique
        )

        critique = await self._call_llm(prompt_content, ctx)
        current_state["graph_builder_agent_critique"] = critique

        self.logger.info(f"Graph builder agent critique: {critique}")

        self.logger.info("Graph builder critique generated and saved")

    async def _generate_hyperparameters_critique(self, current_state: Dict[str, Any], ctx: MessageContext) -> None:
        """Generate critique for hyperparameters agent."""
        self.logger.info("Generating hyperparameters critique")

        rag_hyperparams = current_state.get("rag_hyperparameters", {})
        chunk_size = rag_hyperparams.get("chunk_size", "Not specified")

        batch_info = current_state.get("batch_information", {})
        corpus_sample = batch_info.get("document_text", "")[:500]  # Sample of corpus
        graph_description = current_state.get("graph_description", "")
        graph_critique = current_state.get("graph_critique", "No critique available")

        if not corpus_sample or not graph_description:
            self.logger.warning("Missing data for hyperparameters critique")
            return

        # Call LLM with rag_hyperparameters_agent_gradient_prompt
        prompt_content = self.rag_hyperparameters_agent_gradient_prompt.format(
            chunk_size, corpus_sample, graph_description, graph_critique
        )

        critique = await self._call_llm(prompt_content, ctx)
        current_state["hyperparameters_graph_agent_critique"] = critique
        self.logger.info(f"hyperparameters graph agent critique: {critique}")

        self.logger.info("Hyperparameters critique generated and saved")

    async def _call_llm(self, prompt_content: str, ctx: MessageContext) -> str:
        """Helper method to call LLM with given prompt."""
        try:
            system_message = SystemMessage(content=prompt_content)
            user_message = UserMessage(content="Please provide your critique and feedback.", source="system")

            response = await self.model_client.create(
                [system_message, user_message],
                cancellation_token=ctx.cancellation_token
            )

            return response.content if isinstance(response.content, str) else str(response.content)

        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            return f"Error generating critique: {e}"

    async def close(self) -> None:
        """Close the model client."""
        await self.model_client.close()


# ===== UPDATED FACTORY FUNCTIONS =====

def create_backward_pass_agent() -> BackwardPassAgent:
    """Factory function to create BackwardPassAgent instances."""
    return BackwardPassAgent("backward_pass_agent")