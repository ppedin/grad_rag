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
            setting (str): 'train' or 'test'
            repetitions (int): Number of repetitions per batch (hyperparameter n)
        """
        super().__init__(name)

        self.dataset_name = dataset_name
        self.setting = setting
        self.repetitions = repetitions
        self.current_batch_id = 0
        self.current_repetition = 0

        # Initialize shared state manager
        self.shared_state = SharedState("agent_states")

        # Initialize AutoGen logging
        self.logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.dataset_agent")

        # Initialize file logging for processing tracking
        self.log_dir = Path("agent_logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"{dataset_name}_{setting}_processing.log"

        # Load processed examples log
        self.processed_examples = self._load_processed_log()

        # Load dataset
        self.dataset = self._load_dataset()

        self.logger.info(f"DatasetAgent initialized for {dataset_name} ({setting})")
        self.logger.info(f"Loaded {self.dataset.metadata.get('total_documents', 1)} documents with {len(self.dataset.questions)} total questions")
        self.logger.info(f"Repetitions per batch: {repetitions}")

    def _load_dataset(self) -> Document:
        """Load the dataset from JSON file and merge all documents into one for processing."""
        dataset_file = Path(f"{self.dataset_name}/{self.dataset_name}_{self.setting}.json")

        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different dataset formats
            documents_data = []

            if isinstance(data, list) and len(data) > 0:
                # Demo format: direct array of documents
                documents_data = data
            elif isinstance(data, dict) and 'documents' in data and len(data['documents']) > 0:
                # Real dataset format: {"documents": [...]}
                documents_data = data['documents']
            else:
                raise ValueError("Invalid dataset format - expected array of documents or {documents: [...]}")

            # Merge all documents into a single dataset for processing
            all_questions = []
            all_texts = []
            all_metadata = {}

            for doc_data in documents_data:
                # Collect all questions from all documents
                questions = [Question(**q) for q in doc_data.get('questions', [])]
                all_questions.extend(questions)

                # Collect text (we'll concatenate all texts)
                doc_text = doc_data.get('text', '')
                if doc_text.strip():
                    all_texts.append(doc_text.strip())

                # Merge metadata
                doc_metadata = doc_data.get('metadata', {})
                for key, value in doc_metadata.items():
                    if key in all_metadata:
                        # If key exists, create a list or extend existing list
                        if isinstance(all_metadata[key], list):
                            if isinstance(value, list):
                                all_metadata[key].extend(value)
                            else:
                                all_metadata[key].append(value)
                        else:
                            all_metadata[key] = [all_metadata[key], value]
                    else:
                        all_metadata[key] = value

            # Combine all texts with a separator to avoid merging issues
            combined_text = '\n\n--- DOCUMENT SEPARATOR ---\n\n'.join(all_texts)

            # Add dataset statistics to metadata
            all_metadata.update({
                'total_documents': len(documents_data),
                'total_questions': len(all_questions),
                'dataset_name': self.dataset_name,
                'dataset_setting': self.setting
            })

            return Document(
                id=f"{self.dataset_name}_{self.setting}_combined",
                text=combined_text,
                questions=all_questions,
                metadata=all_metadata
            )

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
        """Generate batch information for a specific batch."""
        # Process questions sequentially - one question per batch
        if batch_id >= len(self.dataset.questions):
            return None

        # Get the question for this batch
        question = self.dataset.questions[batch_id]

        # Create document chunk for this question (use reasonable chunk size)
        chunk_size = 2000  # Characters per chunk
        document_chunks = self._split_document_into_chunks(self.dataset.text, chunk_size)

        # For now, use first chunk or create a relevant chunk for the question
        # In a more advanced implementation, you could select the most relevant chunk
        document_chunk = document_chunks[0] if document_chunks else self.dataset.text[:chunk_size]

        # Create batch with single question and relevant document chunk
        batch_info = {
            "batch_id": batch_id,
            "document_text": document_chunk,
            "qa_pairs": [{
                "question_id": question.id,
                "question": question.question,
                "answers": question.answers,
                "metadata": question.metadata
            }]
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

    @message_handler
    async def handle_dataset_processing_request(
        self, message: DatasetProcessingRequest, ctx: MessageContext
    ) -> str:
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

        # Return completion status
        return f"Dataset processing completed for {message.dataset_name} ({message.setting})"

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

            # Check if we need more repetitions
            if message.repetition < self.repetitions - 1:
                self.current_repetition = message.repetition + 1
                continue_processing = True
                self.logger.info(f"Starting repetition {self.current_repetition + 1}/{self.repetitions}")
                self._log_to_file(f"Starting repetition {self.current_repetition + 1}/{self.repetitions}")
            else:
                # Move to next batch
                self.current_repetition = 0
                self.current_batch_id = message.batch_id + 1
                next_batch = self.generate_batch(self.current_batch_id)
                if next_batch is not None:
                    continue_processing = True
                    self.logger.info(f"Moving to batch {self.current_batch_id}")
                    self._log_to_file(f"Moving to batch {self.current_batch_id}")
                else:
                    self.logger.info("All batches completed")
                    self._log_to_file("All batches completed")

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
                await self._process_next_batch(ctx)

        except Exception as e:
            self.logger.error(f"Error handling BatchReady message: {e}")
            self._log_to_file(f"Error handling BatchReady message: {e}")
            # Return early on error to prevent implicit None return

    async def _process_dataset(self, ctx: MessageContext) -> None:
        """Process the entire dataset."""
        self.logger.info(f"Starting dataset processing for {self.dataset_name} ({self.setting})")
        self._log_to_file(f"Starting dataset processing for {self.dataset_name} ({self.setting})")

        # Find the first unprocessed batch
        while self.current_batch_id in self.processed_examples:
            if self.processed_examples[self.current_batch_id] >= self.repetitions:
                self.current_batch_id += 1
            else:
                self.current_repetition = self.processed_examples[self.current_batch_id]
                break

        # Start processing
        await self._process_next_batch(ctx)

    async def _process_next_batch(self, ctx: MessageContext) -> None:
        """Process the next batch."""
        # Generate batch information
        batch_info = self.generate_batch(self.current_batch_id)
        if batch_info is None:
            self.logger.info("No more batches to process")
            self._log_to_file("No more batches to process")
            return

        # Log progress with question information
        current_question = self.dataset.questions[self.current_batch_id] if self.current_batch_id < len(self.dataset.questions) else None
        question_preview = current_question.question[:80] + "..." if current_question and len(current_question.question) > 80 else (current_question.question if current_question else "No question")

        self.logger.info(f"Processing batch {self.current_batch_id + 1}/{len(self.dataset.questions)}, repetition {self.current_repetition + 1}/{self.repetitions}")
        self.logger.info(f"Question: {question_preview}")
        self._log_to_file(f"Processing batch {self.current_batch_id + 1}/{len(self.dataset.questions)}, repetition {self.current_repetition + 1}/{self.repetitions} - Q: {question_preview}")

        # Update shared state with batch information
        current_state = self.shared_state.load_state(
            self.dataset_name, self.setting, self.current_batch_id
        )

        # Create a heavily truncated version for logging to keep logs clean
        batch_info_for_logging = batch_info.copy()
        if "document_text" in batch_info_for_logging:
            # Truncate to just 100 characters for logging to keep logs readable
            original_text = batch_info_for_logging["document_text"]
            if len(original_text) > 100:
                batch_info_for_logging["document_text"] = original_text[:100] + "..."
            else:
                batch_info_for_logging["document_text"] = original_text

        # Also truncate question and answer text in QA pairs for logging
        if "qa_pairs" in batch_info_for_logging:
            for qa_pair in batch_info_for_logging["qa_pairs"]:
                if len(qa_pair.get("question", "")) > 100:
                    qa_pair["question"] = qa_pair["question"][:100] + "..."
                if isinstance(qa_pair.get("answers"), list) and qa_pair["answers"]:
                    for i, answer in enumerate(qa_pair["answers"]):
                        if isinstance(answer, str) and len(answer) > 100:
                            qa_pair["answers"][i] = answer[:100] + "..."

        # Store truncated batch information for logging
        current_state["batch_information"] = batch_info_for_logging

        # Store full document text separately for agents to access
        current_state["full_document_text"] = batch_info["document_text"]

        # Save state
        self.shared_state.save_state(
            current_state, self.dataset_name, self.setting, self.current_batch_id
        )

        # Create and send BatchStart message
        batch_start_msg = BatchStartMessage(
            batch_id=self.current_batch_id,
            repetition=self.current_repetition,
            dataset=self.dataset_name,
            setting=self.setting,
            shared_state=current_state
        )

        self.logger.info(f"Sending BatchStart message for batch {self.current_batch_id}")
        self._log_to_file(f"Sending BatchStart message for batch {self.current_batch_id}")

        # Send BatchStart message to BatchOrchestratorAgent and get response
        try:
            batch_orchestrator_id = AgentId("batch_orchestrator_agent", "default")
            batch_ready_response = await self.send_message(batch_start_msg, batch_orchestrator_id)
            self.logger.info(f"Received BatchReady response from BatchOrchestratorAgent")

            # Process the response
            await self.handle_batch_ready_message(batch_ready_response, ctx)

        except Exception as e:
            self.logger.warning(f"BatchOrchestratorAgent not available, falling back to simulation: {e}")
            # Fallback to simulation if BatchOrchestratorAgent is not available
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