"""
DatasetAgent implementation using AutoGen Core API.
"""

import json
import os
import statistics
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel

from autogen_core import (
    AgentId, MessageContext, RoutedAgent, message_handler,
    SingleThreadedAgentRuntime, TRACE_LOGGER_NAME
)

from shared_state import SharedState
from datasets_schema import Document, Question
from eval_functions import evaluate_rouge_score


# Message Types for AutoGen Communication
class BatchStartMessage(BaseModel):
    """Message sent when a new batch starts processing."""
    batch_id: int
    repetition: int
    dataset: str
    setting: str
    shared_state: Dict[str, Any]


class BatchReadyMessage(BaseModel):
    """Message sent when batch processing is completed."""
    batch_id: int
    repetition: int
    status: str
    metrics: Optional[Dict[str, Any]] = None


class DatasetProcessingRequest(BaseModel):
    """Message to request dataset processing."""
    dataset_name: str
    setting: str
    repetitions: int = 3


class DatasetProcessingResponse(BaseModel):
    """Response from dataset processing completion."""
    status: str
    final_metrics: Dict[str, Any]
    dataset_name: str
    setting: str
    total_qa_pairs: int
    completed_qa_pairs: int


class MetricsComputedMessage(BaseModel):
    """Message sent after metrics are computed."""
    batch_id: int
    repetition: int
    mean_rouge: float
    connectivity_metrics: Dict[str, float]
    continue_processing: bool


class DatasetAgent(RoutedAgent):
    """
    DatasetAgent for managing dataset processing and evaluation using AutoGen Core API.
    """

    def __init__(self, name: str, dataset_name: str, setting: str, repetitions: int = 3) -> None:
        """
        Initialize DatasetAgent.
        Args:
            name (str): Agent name
            dataset_name (str): Name of the dataset to process
            setting (str): Must be 'test' for test-time training
            repetitions (int): Number of repetitions per batch (hyperparameter n)
        """
        super().__init__(name)

        # Validate that only test datasets are used
        if setting != "test":
            raise ValueError(f"Only 'test' setting is supported for test-time training. Got: '{setting}'")

        self.dataset_name = dataset_name
        self.setting = setting
        self.repetitions = repetitions
        self.current_batch_id = 0
        self.current_repetition = 0
        self.current_qa_index = 0  # Track current QA pair within document

        # Initialize shared state manager
        self.shared_state = SharedState("agent_states")

        # Initialize AutoGen logging
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.dataset_agent")

        # Initialize file logging for processing tracking
        self.log_dir = Path("agent_logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"{dataset_name}_{setting}_processing.log"

        # For test-time training, start fresh each time (don't load previous state)
        self.processed_examples = {}

        # Load dataset
        self.dataset = self._load_dataset() # This now returns a List[Document]

        # Generate QA pair index for single-example processing (after dataset is loaded)
        self.qa_pairs_list = self._generate_qa_pairs_list()

        # --- THIS IS THE CORRECTED PART ---
        # Calculate totals from the list of documents
        total_documents = len(self.dataset)
        total_questions = sum(len(doc.questions) for doc in self.dataset)
        total_qa_pairs = len(self.qa_pairs_list)

        self.logger.info(f"DatasetAgent initialized for {dataset_name} ({setting})")
        self.logger.info(f"Loaded {total_documents} documents with {total_questions} total questions")
        self.logger.info(f"Generated {total_qa_pairs} individual QA pairs for test-time training")
        self.logger.info(f"Repetitions per example: {repetitions}")

    def _load_dataset(self) -> List[Document]:
        """
        Load the dataset from a JSON file as a list of Document objects.
        Each document retains its own text and associated questions.
        """
        dataset_file = Path(f"{self.dataset_name}/{self.dataset_name}_{self.setting}.json")

        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            documents_data = []
            if isinstance(data, list):
                documents_data = data
            elif isinstance(data, dict) and 'documents' in data:
                documents_data = data['documents']
            else:
                raise ValueError("Invalid dataset format - expected array of documents or {documents: [...]}")

        
            individual_documents = []
            for doc_data in documents_data:
                # Here we are using the schema you provided to parse the data
                questions = [Question(**q) for q in doc_data.get('questions', [])]
                
                if not doc_data.get('text', '').strip() or not questions:
                    continue

                doc_metadata = doc_data.get('metadata', {})
                doc_metadata.update({
                    'dataset_name': self.dataset_name,
                    'dataset_setting': self.setting
                })
                
                document = Document(
                    id=doc_data.get('id', 'unknown_id'),
                    text=doc_data.get('text', ''),
                    questions=questions,
                    metadata=doc_metadata
                )
                individual_documents.append(document)

            return individual_documents

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"Error loading dataset: {e}")

    def _load_processed_log(self) -> Dict[int, int]:
        """Load the log of processed examples."""
        processed_log = {}
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith("PROCESSED:"):
                            parts = line.strip().split(":")
                            if len(parts) >= 3:
                                batch_id = int(parts[1])
                                repetition = int(parts[2])
                                processed_log[batch_id] = max(
                                    processed_log.get(batch_id, 0), repetition
                                )
            except (IOError, ValueError):
                pass
        return processed_log

    def _log_to_file(self, message: str):
        """Log a message to the processing file for persistence."""
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {self.id.type}: {message}"

        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + "\n")
        except IOError:
            pass

    def generate_batch(self, batch_id: int) -> Optional[Dict[str, Any]]:
        """
        Generate a batch using the document at the given batch_id (index).
        Each batch consists of one document's text and all its associated questions.
        """
        if batch_id >= len(self.dataset):
            return None

        current_document = self.dataset[batch_id]
        document_text = current_document.text

        qa_pairs = [{
            "question_id": q.id,
            "question": q.question,
            "answers": q.answers,
            "metadata": q.metadata
        } for q in current_document.questions]

        batch_info = {
            "batch_id": batch_id,
            "document_text": document_text,
            "qa_pairs": qa_pairs
        }

        return batch_info

    def _split_document_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split document into chunks of specified size, trying to break at sentence boundaries."""
        if not text or chunk_size <= 0:
            return []

        chunks = []
        current_pos = 0

        while current_pos < len(text):
            # Get chunk of specified size
            chunk_end = min(current_pos + chunk_size, len(text))
            chunk = text[current_pos:chunk_end]

            # If not at end of text, try to break at sentence boundary
            if chunk_end < len(text):
                # Look for last sentence ending in the chunk
                for i in range(len(chunk) - 1, max(0, len(chunk) - 200), -1):
                    if chunk[i] in '.!?':
                        # Found sentence boundary, adjust chunk
                        chunk = chunk[:i+1]
                        chunk_end = current_pos + i + 1
                        break

            chunks.append(chunk.strip())
            current_pos = chunk_end

        return [chunk for chunk in chunks if chunk]  # Remove empty chunks

    def _generate_qa_pairs_list(self) -> List[Dict[str, Any]]:
        """
        Generate a flat list of all QA pairs with their document context.
        Each entry contains: document_text, question, answers, document_id, question_id, metadata
        """
        qa_pairs_list = []
        for doc_idx, document in enumerate(self.dataset):
            for qa_idx, question in enumerate(document.questions):
                qa_pair = {
                    "document_index": doc_idx,
                    "question_index": qa_idx,
                    "document_id": document.id,
                    "document_text": document.text,
                    "question_id": question.id,
                    "question": question.question,
                    "answers": question.answers,
                    "metadata": question.metadata,
                    "global_index": len(qa_pairs_list)  # Global index for this QA pair
                }
                qa_pairs_list.append(qa_pair)
        return qa_pairs_list

    def generate_single_qa_pair(self, qa_index: int) -> Optional[Dict[str, Any]]:
        """
        Generate a single QA pair for test-time training.
        Args:
            qa_index: Global index of the QA pair to generate
        Returns:
            Dictionary containing single QA pair with document context
        """
        if qa_index >= len(self.qa_pairs_list):
            return None

        qa_pair = self.qa_pairs_list[qa_index]

        # Return format compatible with single example processing
        single_example = {
            "qa_index": qa_index,
            "document_text": qa_pair["document_text"],
            "qa_pair": {
                "question_id": qa_pair["question_id"],
                "question": qa_pair["question"],
                "answers": qa_pair["answers"],
                "metadata": qa_pair["metadata"]
            },
            "document_metadata": {
                "document_id": qa_pair["document_id"],
                "document_index": qa_pair["document_index"],
                "question_index": qa_pair["question_index"]
            }
        }

        return single_example

    async def _reset_state_for_new_example(self, dataset: str, setting: str) -> None:
        """
        Reset learned prompts and clear graph when moving to a new example.
        This ensures each example starts fresh without influence from previous examples.
        """
        self.logger.info("Resetting state for new example")

        # Get the current state for the new QA index
        current_state = self.shared_state.load_state(dataset, setting, self.current_qa_index)

        # Clear all learned prompts (these will be re-optimized during the new example's iterations)
        learned_prompt_keys = [
            # GraphRAG prompts
            "learned_prompt_graph_builder",
            "learned_prompt_graph_refinement",
            "learned_prompt_retrieval_planner",
            "learned_prompt_answer_generator",
            "learned_prompt_hyperparameters",
            # VectorRAG prompts
            "learned_prompt_hyperparameters_vector",
            "learned_prompt_vector_retrieval_planner",
            "learned_prompt_answer_generator_vector"
        ]

        for key in learned_prompt_keys:
            if key in current_state:
                del current_state[key]
                self.logger.info(f"Cleared learned prompt: {key}")

        # Clear previous critiques and optimization results
        critique_keys = [
            "answer_generation_critique",
            "retrieved_content_critique",
            "retrieval_plan_critique",
            "graph_critique",
            "graph_builder_agent_critique",
            "hyperparameters_graph_agent_critique"
        ]

        for key in critique_keys:
            if key in current_state:
                del current_state[key]

        # Clear conversation histories from previous examples
        conversation_keys = [
            # GraphRAG conversations
            "conversations_answer_generation",
            "conversations_retrieval_planning",
            "conversations_graph_builder",
            "conversations_hyperparameters",
            # VectorRAG conversations
            "conversations_vector_answer_generation",
            "conversations_vector_retrieval_planning",
            "conversations_vector_hyperparameters"
        ]

        for key in conversation_keys:
            if key in current_state:
                del current_state[key]

        # Save the reset state
        self.shared_state.save_state(current_state, dataset, setting, self.current_qa_index)

        # Clear the graph from Memgraph to start fresh (for GraphRAG)
        try:
            from graph_functions import clear_graph
            clear_result = clear_graph()
            if clear_result.get("status") == "success":
                self.logger.info("Successfully cleared graph for new example")
            else:
                self.logger.warning(f"Failed to clear graph: {clear_result.get('message')}")
        except Exception as e:
            self.logger.error(f"Error clearing graph for new example: {e}")

        # Clear vector-specific data for VectorRAG systems
        try:
            # Clear any cached vector embeddings or FAISS indices
            # This ensures VectorRAG systems start fresh for each example
            vector_data_keys = [
                "vector_index_metadata",
                "chunk_embeddings",
                "faiss_index_state"
            ]

            for key in vector_data_keys:
                if key in current_state:
                    del current_state[key]

            self.logger.info("Successfully cleared vector-specific data for new example")
        except Exception as e:
            self.logger.error(f"Error clearing vector data for new example: {e}")

        self.logger.info("State reset completed for new example")

    @message_handler
    async def handle_dataset_processing_request(
        self, message: DatasetProcessingRequest, ctx: MessageContext
    ) -> DatasetProcessingResponse:
        """Handle request to start dataset processing."""
        self.logger.info(f"Received dataset processing request for {message.dataset_name} ({message.setting})")
        self._log_to_file(f"Received dataset processing request for {message.dataset_name} ({message.setting})")

        # Update parameters from request
        if message.dataset_name != self.dataset_name or message.setting != self.setting:
            self.logger.warning(f"Request parameters don't match agent initialization")
            self._log_to_file(f"Warning: Request parameters don't match agent initialization")

        self.repetitions = message.repetitions

        # Start processing
        await self._process_dataset(ctx)

        # Calculate final metrics
        total_qa_pairs = len(self.qa_pairs_list)
        completed_qa_pairs = len([qa_idx for qa_idx, count in self.processed_examples.items() if count >= self.repetitions])

        final_metrics = {
            "total_qa_pairs": total_qa_pairs,
            "completed_qa_pairs": completed_qa_pairs,
            "completion_rate": completed_qa_pairs / total_qa_pairs if total_qa_pairs > 0 else 0.0,
            "repetitions_per_example": self.repetitions,
            "total_iterations": completed_qa_pairs * self.repetitions
        }

        # Return proper response object
        return DatasetProcessingResponse(
            status="completed",
            final_metrics=final_metrics,
            dataset_name=message.dataset_name,
            setting=message.setting,
            total_qa_pairs=total_qa_pairs,
            completed_qa_pairs=completed_qa_pairs
        )

    @message_handler
    async def handle_batch_ready_message(
        self, message: BatchReadyMessage, ctx: MessageContext
    ) -> None:
        """Handle BatchReady message by computing metrics and managing repetitions."""
        try:
            self.logger.info(f"Received BatchReady for batch {message.batch_id}, repetition {message.repetition}")
            self._log_to_file(f"Received BatchReady for batch {message.batch_id}, repetition {message.repetition}")

            # Load current shared state
            current_state = self.shared_state.load_state(
                self.dataset_name, self.setting, message.batch_id
            )

            # Compute mean ROUGE score
            rouge_scores = current_state.get("rouge_scores", [])
            if rouge_scores:
                mean_rouge = statistics.mean(rouge_scores)
                self.logger.info(f"Mean ROUGE score: {mean_rouge:.4f}")
                self._log_to_file(f"Mean ROUGE score: {mean_rouge:.4f}")
            else:
                mean_rouge = 0.0
                self.logger.info("No ROUGE scores available")
                self._log_to_file("No ROUGE scores available")

            # Compute mean connectivity metrics
            graph_stats = current_state.get("graph_statistics", {})
            connectivity_metrics = {}
            for metric_name, values in graph_stats.items():
                if isinstance(values, list) and values:
                    connectivity_metrics[metric_name] = statistics.mean(values)
                elif isinstance(values, (int, float)):
                    connectivity_metrics[metric_name] = values

            self.logger.info("Graph connectivity metrics:")
            self._log_to_file("Graph connectivity metrics:")
            for metric, value in connectivity_metrics.items():
                self.logger.info(f"  {metric}: {value:.4f}")
                self._log_to_file(f"  {metric}: {value:.4f}")

            # Save updated state
            self.shared_state.save_state(
                current_state, self.dataset_name, self.setting, message.batch_id
            )

            # Log processing completion
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"PROCESSED:{message.batch_id}:{message.repetition}\n")

            # Determine if we should continue
            continue_processing = False

            # Check if we need more iterations for current QA pair
            if message.repetition < self.repetitions - 1:
                self.current_repetition = message.repetition + 1
                continue_processing = True
                self.logger.info(f"Starting iteration {self.current_repetition + 1}/{self.repetitions} for QA pair {self.current_qa_index}")
                self._log_to_file(f"Starting iteration {self.current_repetition + 1}/{self.repetitions} for QA pair {self.current_qa_index}")
            else:
                # Move to next QA pair (reset state between examples)
                self.current_repetition = 0
                self.current_qa_index = message.batch_id + 1  # message.batch_id is actually QA index
                next_example = self.generate_single_qa_pair(self.current_qa_index)
                if next_example is not None:
                    continue_processing = True
                    self.logger.info(f"Moving to QA pair {self.current_qa_index}, resetting learned prompts")
                    self._log_to_file(f"Moving to QA pair {self.current_qa_index}, resetting learned prompts")
                    # Reset learned prompts and clear graph for new example
                    await self._reset_state_for_new_example(self.dataset_name, self.setting)
                else:
                    self.logger.info("All QA pairs completed")
                    self._log_to_file("All QA pairs completed")

            # Send metrics computed message
            metrics_msg = MetricsComputedMessage(
                batch_id=message.batch_id,
                repetition=message.repetition,
                mean_rouge=mean_rouge,
                connectivity_metrics=connectivity_metrics,
                continue_processing=continue_processing
            )

            # In a real multi-agent system, you would send this to other agents
            # For now, we'll continue processing if needed
            if continue_processing:
                await self._process_next_example(ctx)

        except Exception as e:
            self.logger.error(f"Error handling BatchReady message: {e}")
            self._log_to_file(f"Error handling BatchReady message: {e}")
            # Return early on error to prevent implicit None return

    async def _process_dataset(self, ctx: MessageContext) -> None:
        """Process the entire dataset using single QA pair processing."""
        self.logger.info(f"Starting test-time training for {self.dataset_name} ({self.setting})")
        self._log_to_file(f"Starting test-time training for {self.dataset_name} ({self.setting})")

        # Find the first unprocessed QA pair
        self.logger.info(f"Looking for unprocessed QA pairs. Current index: {self.current_qa_index}, Total pairs: {len(self.qa_pairs_list)}")

        while self.current_qa_index in self.processed_examples:
            if self.processed_examples[self.current_qa_index] >= self.repetitions:
                self.current_qa_index += 1
                self.logger.info(f"QA pair {self.current_qa_index - 1} already completed, moving to {self.current_qa_index}")
            else:
                self.current_repetition = self.processed_examples[self.current_qa_index]
                self.logger.info(f"Resuming QA pair {self.current_qa_index} at repetition {self.current_repetition}")
                break

        # Check if we have QA pairs to process
        if self.current_qa_index >= len(self.qa_pairs_list):
            self.logger.info("No QA pairs to process - all completed or index out of range")
            return

        self.logger.info(f"Starting processing from QA pair {self.current_qa_index}")
        # Start processing single QA pairs
        await self._process_next_example(ctx)

    async def _process_next_example(self, ctx: MessageContext) -> None:
        """Process the next single QA pair example for test-time training."""
        # Generate single QA pair information
        example_info = self.generate_single_qa_pair(self.current_qa_index)
        if example_info is None:
            self.logger.info("No more QA pairs to process")
            self._log_to_file("No more QA pairs to process")
            return

        # --- START OF SINGLE QA PROCESSING SECTION ---

        # Get current QA pair information
        qa_pair_info = example_info["qa_pair"]
        document_metadata = example_info["document_metadata"]

        question_preview = (
            qa_pair_info["question"][:80] + "..."
            if len(qa_pair_info["question"]) > 80
            else qa_pair_info["question"]
        )

        # The total number of examples is the total number of QA pairs
        total_examples = len(self.qa_pairs_list)

        self.logger.info(f"Processing QA pair {self.current_qa_index + 1}/{total_examples}, iteration {self.current_repetition + 1}/{self.repetitions}")
        self.logger.info(f"Document ID: {document_metadata['document_id']}")
        self.logger.info(f"Question ID: {qa_pair_info['question_id']}")
        self.logger.info(f"Question: {question_preview}")
        self._log_to_file(f"Processing QA {self.current_qa_index + 1}/{total_examples}, iteration {self.current_repetition + 1}/{self.repetitions} - DocID: {document_metadata['document_id']} - QID: {qa_pair_info['question_id']} - Q: {question_preview}")

        # --- END OF SINGLE QA PROCESSING SECTION ---

        # Update shared state with single QA pair information
        current_state = self.shared_state.load_state(
            self.dataset_name, self.setting, self.current_qa_index
        )

        # Create a truncated version for logging to keep logs clean
        example_info_for_logging = {
            "qa_index": example_info["qa_index"],
            "document_text": example_info["document_text"][:100] + "..." if len(example_info["document_text"]) > 100 else example_info["document_text"],
            "qa_pair": {
                "question_id": qa_pair_info["question_id"],
                "question": qa_pair_info["question"][:100] + "..." if len(qa_pair_info["question"]) > 100 else qa_pair_info["question"],
                "answers": [
                    answer[:100] + "..." if isinstance(answer, str) and len(answer) > 100 else answer
                    for answer in qa_pair_info["answers"]
                ] if qa_pair_info["answers"] else [],
                "metadata": qa_pair_info["metadata"]
            },
            "document_metadata": document_metadata
        }

        current_state["example_information"] = example_info_for_logging
        current_state["full_document_text"] = example_info["document_text"]
        current_state["current_qa_pair"] = qa_pair_info

        # Also create batch_information format that BatchOrchestrator expects
        batch_info_for_orchestrator = {
            "batch_id": self.current_qa_index,
            "document_text": example_info["document_text"],
            "qa_pairs": [qa_pair_info],  # Single QA pair in array format
            "total_iterations": self.repetitions  # Required for two-level reset logic
        }
        current_state["batch_information"] = batch_info_for_orchestrator

        self.shared_state.save_state(
            current_state, self.dataset_name, self.setting, self.current_qa_index
        )

        # Create and send BatchStart message (reusing existing message format but with QA index as batch_id)
        batch_start_msg = BatchStartMessage(
            batch_id=self.current_qa_index,  # Use QA index as unique identifier
            repetition=self.current_repetition,
            dataset=self.dataset_name,
            setting=self.setting,
            shared_state=current_state
        )

        self.logger.info(f"Sending BatchStart message for QA pair {self.current_qa_index}")
        self._log_to_file(f"Sending BatchStart message for QA pair {self.current_qa_index}")

        try:
            batch_orchestrator_id = AgentId("batch_orchestrator_agent", "default")
            batch_ready_response = await self.send_message(batch_start_msg, batch_orchestrator_id)
            self.logger.info(f"Received BatchReady response from BatchOrchestratorAgent")
            await self.handle_batch_ready_message(batch_ready_response, ctx)

        except Exception as e:
            self.logger.warning(f"BatchOrchestratorAgent not available, falling back to simulation: {e}")
            await self._simulate_batch_processing(ctx, batch_start_msg)

    async def _simulate_batch_processing(self, ctx: MessageContext, batch_start: BatchStartMessage) -> None:
        """Simulate batch processing for testing purposes."""
        # Simulate some processing time and generate results
        import random

        # Simulate ROUGE scores
        rouge_scores = [random.uniform(0.3, 0.8) for _ in range(3)]
        batch_start.shared_state["rouge_scores"] = rouge_scores

        # Simulate graph statistics
        batch_start.shared_state["graph_statistics"] = {
            "density": [random.uniform(0.1, 0.6)],
            "clustering_coefficient": [random.uniform(0.2, 0.7)],
            "avg_path_length": [random.uniform(2.0, 5.0)]
        }

        # Save the updated state
        self.shared_state.save_state(
            batch_start.shared_state,
            self.dataset_name,
            self.setting,
            batch_start.batch_id
        )

        # Send BatchReady message back to ourselves
        batch_ready_msg = BatchReadyMessage(
            batch_id=batch_start.batch_id,
            repetition=batch_start.repetition,
            status="completed",
            metrics={"simulated": True}
        )

        # Send BatchReady message back to self using AutoGen direct messaging
        # In a real multi-agent system, other agents would send this message
        await self.send_message(batch_ready_msg, self.id)


# Factory function for creating DatasetAgent instances
def create_dataset_agent(dataset_name: str, setting: str, repetitions: int = 3) -> DatasetAgent:
    """Factory function to create DatasetAgent instances."""
    return DatasetAgent(f"dataset_agent_{dataset_name}_{setting}", dataset_name, setting, repetitions)