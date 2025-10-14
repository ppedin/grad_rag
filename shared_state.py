"""
Shared state management for AutoGen agents.
"""

import json
import os
import shutil
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path


class SharedState:
    """Manages shared state between AutoGen agents."""

    def __init__(self, state_dir: str = "agent_states"):
        """
        Initialize shared state manager.

        Args:
            state_dir (str): Directory to store state files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)

        # Default shared state structure
        self.default_state = {
            "batch_information": {},
            "hyperparameters_graph_agent_critique": "",
            "graph_builder_agent_critique": "",
            "retrieval_planner_agent_critique": "",
            "answer_generation_critique": "",
            "response_evaluations": [],
            "retrieved_contexts": [],
            "conversations_answer_generation": [],
            "retrieval_plans": [],
            "retrieval_prompt": "",
            "retrieval_plan_critique": "",
            "graph_description": "",
            "graph_critique": "",
            "graph_builder_prompt": "",
            "rag_hyperparameters": {},
            "rouge_scores": [],
            "graph_statistics": {},
            "vector_statistics": [],  # Vector statistics for VectorRAG
            "full_document_text": "",  # Document text that should persist through iterations
            "example_information": {}   # Example information that should persist through iterations
        }

        # QA pair level tracking
        self.current_qa_pair_id: Optional[str] = None
        self.current_iteration: int = 0
        self.qa_pair_prompts: Dict[str, Dict[str, str]] = {}
        self.processing_state = {
            "current_qa_pair_id": None,
            "current_iteration": 0,
            "total_iterations": 0,
            "qa_pair_start_time": None,
            "iteration_start_time": None
        }

    def get_state_file_path(self, dataset: str, setting: str, batch_id: int = None) -> Path:
        """
        Get the path for a specific state file.

        Args:
            dataset (str): Dataset name
            setting (str): 'train' or 'test'
            batch_id (int, optional): Batch identifier

        Returns:
            Path: Path to the state file
        """
        if batch_id is not None:
            filename = f"{dataset}_{setting}_batch_{batch_id}_state.json"
        else:
            filename = f"{dataset}_{setting}_latest_state.json"
        return self.state_dir / filename

    def load_state(self, dataset: str, setting: str, batch_id: int = None) -> Dict[str, Any]:
        """
        Load shared state from file.

        Args:
            dataset (str): Dataset name
            setting (str): 'train' or 'test'
            batch_id (int, optional): Batch identifier

        Returns:
            Dict[str, Any]: Shared state dictionary
        """
        state_file = self.get_state_file_path(dataset, setting, batch_id)

        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                # Ensure all required keys exist
                for key, default_value in self.default_state.items():
                    if key not in state:
                        state[key] = default_value
                return state
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading state file {state_file}: {e}")
                return self.default_state.copy()
        else:
            return self.default_state.copy()

    def save_state(self, state: Dict[str, Any], dataset: str, setting: str, batch_id: int = None) -> bool:
        """
        Save shared state to file.

        Args:
            state (Dict[str, Any]): State to save
            dataset (str): Dataset name
            setting (str): 'train' or 'test'
            batch_id (int, optional): Batch identifier

        Returns:
            bool: True if successful, False otherwise
        """
        state_file = self.get_state_file_path(dataset, setting, batch_id)

        try:
            # Add timestamp
            state_with_meta = state.copy()
            state_with_meta["_metadata"] = {
                "last_updated": datetime.now().isoformat(),
                "dataset": dataset,
                "setting": setting,
                "batch_id": batch_id
            }

            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_with_meta, f, indent=2, ensure_ascii=False)

            # Also save as latest
            if batch_id is not None:
                latest_file = self.get_state_file_path(dataset, setting, None)
                with open(latest_file, 'w', encoding='utf-8') as f:
                    json.dump(state_with_meta, f, indent=2, ensure_ascii=False)

            return True
        except (IOError, json.JSONEncodeError) as e:
            print(f"Error saving state file {state_file}: {e}")
            return False

    def get_all_states(self, dataset: str, setting: str) -> List[Dict[str, Any]]:
        """
        Get all saved states for a dataset and setting.

        Args:
            dataset (str): Dataset name
            setting (str): 'train' or 'test'

        Returns:
            List[Dict[str, Any]]: List of all states
        """
        pattern = f"{dataset}_{setting}_batch_*_state.json"
        states = []

        for state_file in self.state_dir.glob(pattern):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    states.append(state)
            except (json.JSONDecodeError, IOError):
                continue

        # Sort by batch_id if available
        def get_batch_id(state):
            return state.get("_metadata", {}).get("batch_id", 0) or 0

        states.sort(key=get_batch_id)
        return states

    def reset_for_new_qa_pair(self, qa_pair_id: str, dataset: str, setting: str, batch_id: int = None) -> Dict[str, Any]:
        """
        Complete reset for new QA pair - everything starts fresh.

        Args:
            qa_pair_id (str): New QA pair identifier
            dataset (str): Dataset name
            setting (str): 'train' or 'test'
            batch_id (int, optional): Batch identifier

        Returns:
            Dict[str, Any]: Fresh state for new QA pair
        """
        # Load current state to preserve document text and example information
        current_state = self.load_state(dataset, setting, batch_id)

        # Archive previous QA pair data if exists
        if self.current_qa_pair_id and self.current_qa_pair_id != qa_pair_id:
            self._archive_qa_pair_data(self.current_qa_pair_id, dataset, setting, batch_id)

        # Complete reset - start with fresh default state
        fresh_state = self.default_state.copy()

        # Preserve document text and example information that should persist
        fresh_state["full_document_text"] = current_state.get("full_document_text", "")
        fresh_state["example_information"] = current_state.get("example_information", {})

        # Preserve document_index for fault-tolerant FAISS index naming
        if "document_index" in current_state:
            fresh_state["document_index"] = current_state["document_index"]
        fresh_state["batch_information"] = current_state.get("batch_information", {})

        # CRITICAL FIX: Explicitly clear all learned prompt keys to prevent leakage between QA pairs
        learned_prompt_keys = [
            "learned_prompt_hyperparameters_vector",
            "learned_prompt_answer_generator_vector",
            "learned_prompt_vector_retrieval_planner",
            "learned_prompt_hyperparameters_graph",
            "learned_prompt_answer_generator_graph",
            "learned_prompt_graph_retrieval_planner",
            "learned_prompt_graph_builder"
        ]
        for key in learned_prompt_keys:
            fresh_state[key] = ""

        # Reset QA pair level tracking
        self.current_qa_pair_id = qa_pair_id
        self.current_iteration = 0
        self.qa_pair_prompts[qa_pair_id] = {
            # Initialize learned system prompts as empty for new QA pair (GraphRAG)
            "learned_prompt_hyperparameters_graph": "",
            "learned_prompt_answer_generator_graph": "",
            "learned_prompt_graph_retrieval_planner": "",
            "learned_prompt_graph_builder": "",
            # Initialize learned system prompts as empty for new QA pair (VectorRAG)
            "learned_prompt_hyperparameters_vector": "",
            "learned_prompt_answer_generator_vector": "",
            "learned_prompt_vector_retrieval_planner": "",
            # Initialize critiques and prompt templates
            "hyperparameters_graph_agent_critique": "",
            "graph_builder_agent_critique": "",
            "retrieval_planner_agent_critique": "",
            "answer_generation_critique": "",
            "graph_builder_prompt": "",
            "retrieval_prompt": ""
        }

        # Update processing state
        self.processing_state.update({
            "current_qa_pair_id": qa_pair_id,
            "current_iteration": 0,
            "qa_pair_start_time": datetime.now().isoformat(),
            "iteration_start_time": datetime.now().isoformat()
        })

        # Save fresh state
        self.save_state(fresh_state, dataset, setting, batch_id)

        print(f"COMPLETE RESET: Starting new QA pair {qa_pair_id} (iteration 0)")
        return fresh_state

    def reset_for_new_iteration(self, qa_pair_id: str, iteration: int, dataset: str, setting: str, batch_id: int = None) -> Dict[str, Any]:
        """
        Partial reset for new iteration of same QA pair - preserve system prompts.

        Args:
            qa_pair_id (str): QA pair identifier
            iteration (int): Iteration number
            dataset (str): Dataset name
            setting (str): 'train' or 'test'
            batch_id (int, optional): Batch identifier

        Returns:
            Dict[str, Any]: State with preserved system prompts
        """
        # Load current state to preserve system prompts
        current_state = self.load_state(dataset, setting, batch_id)

        # Archive previous iteration data
        self._archive_iteration_data(qa_pair_id, self.current_iteration, current_state, dataset, setting, batch_id)

        # Create new state preserving only system prompts
        new_state = self.default_state.copy()

        # Preserve document text and example information that should persist through iterations
        new_state["full_document_text"] = current_state.get("full_document_text", "")
        new_state["example_information"] = current_state.get("example_information", {})

        # Preserve document_index for fault-tolerant FAISS index naming
        if "document_index" in current_state:
            new_state["document_index"] = current_state["document_index"]
        new_state["batch_information"] = current_state.get("batch_information", {})

        # Preserve missing keywords from previous iteration's evaluation for focused refinement
        new_state["missing_keywords_for_refinement"] = current_state.get("missing_keywords_for_refinement", [])

        # Preserve all response evaluations from previous iterations
        new_state["response_evaluations"] = current_state.get("response_evaluations", [])

        # Preserve graph and community data for style-issue reuse
        new_state["last_graph_response"] = current_state.get("last_graph_response", {})
        new_state["last_useful_community_answers"] = current_state.get("last_useful_community_answers", [])
        new_state["community_summaries"] = current_state.get("community_summaries", {})

        # Preserve continue_optimization and issue_type flags for all QA pairs
        for key, value in current_state.items():
            if key.startswith("continue_optimization_") or key.startswith("issue_type_"):
                new_state[key] = value

        # Preserve system prompts from previous iteration if they exist
        if qa_pair_id in self.qa_pair_prompts:
            preserved_prompts = self.qa_pair_prompts[qa_pair_id]

            # Preserve the actual learned system prompts that agents use (GraphRAG)
            new_state["learned_prompt_hyperparameters_graph"] = preserved_prompts.get("learned_prompt_hyperparameters_graph", "")
            new_state["learned_prompt_answer_generator_graph"] = preserved_prompts.get("learned_prompt_answer_generator_graph", "")
            new_state["learned_prompt_graph_retrieval_planner"] = preserved_prompts.get("learned_prompt_graph_retrieval_planner", "")
            new_state["learned_prompt_graph_builder"] = preserved_prompts.get("learned_prompt_graph_builder", "")
            new_state["learned_prompt_graph_refinement"] = preserved_prompts.get("learned_prompt_graph_refinement", "")
            new_state["learned_prompt_community_summarizer"] = preserved_prompts.get("learned_prompt_community_summarizer", "")

            # Preserve the actual learned system prompts that agents use (VectorRAG)
            new_state["learned_prompt_hyperparameters_vector"] = preserved_prompts.get("learned_prompt_hyperparameters_vector", "")
            new_state["learned_prompt_answer_generator_vector"] = preserved_prompts.get("learned_prompt_answer_generator_vector", "")
            new_state["learned_prompt_vector_retrieval_planner"] = preserved_prompts.get("learned_prompt_vector_retrieval_planner", "")

            # Also preserve critiques and prompt templates for reference
            new_state["hyperparameters_graph_agent_critique"] = preserved_prompts.get("hyperparameters_graph_agent_critique", "")
            new_state["graph_builder_agent_critique"] = preserved_prompts.get("graph_builder_agent_critique", "")
            new_state["retrieval_planner_agent_critique"] = preserved_prompts.get("retrieval_planner_agent_critique", "")
            new_state["answer_generation_critique"] = preserved_prompts.get("answer_generation_critique", "")
            new_state["community_summarizer_critique"] = preserved_prompts.get("community_summarizer_critique", "")
            new_state["graph_builder_prompt"] = preserved_prompts.get("graph_builder_prompt", "")
            new_state["graph_refinement_prompt"] = preserved_prompts.get("graph_refinement_prompt", "")
            new_state["retrieval_prompt"] = preserved_prompts.get("retrieval_prompt", "")

        # Update iteration tracking
        self.current_iteration = iteration
        self.processing_state.update({
            "current_iteration": iteration,
            "iteration_start_time": datetime.now().isoformat()
        })

        # Save state
        self.save_state(new_state, dataset, setting, batch_id)

        print(f"ITERATION RESET: Starting iteration {iteration} of QA pair {qa_pair_id} (preserving system prompts)")
        return new_state

    def update_qa_pair_prompts(self, qa_pair_id: str, optimized_prompts: Dict[str, str]):
        """
        Update the system prompts for a QA pair.

        Args:
            qa_pair_id (str): QA pair identifier
            optimized_prompts (Dict[str, str]): New optimized prompts
        """
        if qa_pair_id not in self.qa_pair_prompts:
            self.qa_pair_prompts[qa_pair_id] = {}

        self.qa_pair_prompts[qa_pair_id].update(optimized_prompts)
        print(f"Updated system prompts for QA pair {qa_pair_id}")

    def detect_transition_type(self, qa_pair_id: str, iteration: int) -> str:
        """
        Detect whether this is a new QA pair or new iteration.

        Args:
            qa_pair_id (str): QA pair identifier
            iteration (int): Iteration number

        Returns:
            str: 'new_qa_pair', 'new_iteration', or 'same_state'
        """
        if self.current_qa_pair_id is None or self.current_qa_pair_id != qa_pair_id:
            return 'new_qa_pair'
        elif self.current_iteration != iteration:
            return 'new_iteration'
        else:
            return 'same_state'

    def _archive_qa_pair_data(self, qa_pair_id: str, dataset: str, setting: str, batch_id: int = None):
        """
        Archive data for completed QA pair.
        """
        archive_dir = self.state_dir / "archives" / "qa_pairs" / qa_pair_id
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Archive final state
        current_state = self.load_state(dataset, setting, batch_id)
        archive_file = archive_dir / f"final_state_{dataset}_{setting}.json"

        try:
            with open(archive_file, 'w', encoding='utf-8') as f:
                json.dump(current_state, f, indent=2, ensure_ascii=False)
            print(f"Archived QA pair {qa_pair_id} data to {archive_file}")
        except Exception as e:
            print(f"Error archiving QA pair data: {e}")

    def _archive_iteration_data(self, qa_pair_id: str, iteration: int, state: Dict[str, Any], dataset: str, setting: str, batch_id: int = None):
        """
        Archive data for completed iteration.
        """
        archive_dir = self.state_dir / "archives" / "iterations" / qa_pair_id / f"iteration_{iteration}"
        archive_dir.mkdir(parents=True, exist_ok=True)

        archive_file = archive_dir / f"state_{dataset}_{setting}.json"

        try:
            with open(archive_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            print(f"Archived iteration {iteration} data for QA pair {qa_pair_id}")
        except Exception as e:
            print(f"Error archiving iteration data: {e}")

    def validate_reset_state(self, expected_reset_type: str, qa_pair_id: str, iteration: int) -> bool:
        """
        Validate that reset was performed correctly.

        Args:
            expected_reset_type (str): Expected reset type ('complete' or 'partial')
            qa_pair_id (str): QA pair identifier
            iteration (int): Iteration number

        Returns:
            bool: True if state is valid for reset type
        """
        if expected_reset_type == 'complete':
            # For complete reset, system prompts should be empty
            return (
                self.current_qa_pair_id == qa_pair_id and
                self.current_iteration == 0 and
                qa_pair_id in self.qa_pair_prompts
            )
        elif expected_reset_type == 'partial':
            # For partial reset, system prompts should be preserved
            return (
                self.current_qa_pair_id == qa_pair_id and
                self.current_iteration == iteration and
                qa_pair_id in self.qa_pair_prompts
            )
        return False

    def get_processing_state(self) -> Dict[str, Any]:
        """
        Get current processing state for monitoring.

        Returns:
            Dict[str, Any]: Current processing state
        """
        return self.processing_state.copy()

    def clear_graph_data(self, dataset: str, setting: str):
        """
        Clear graph-related data and files.

        Args:
            dataset (str): Dataset name
            setting (str): Setting name
        """
        # Clear graph files if they exist
        graph_dir = Path(f"graphs_{dataset}_{setting}")
        if graph_dir.exists():
            try:
                shutil.rmtree(graph_dir)
                print(f"Cleared graph directory: {graph_dir}")
            except Exception as e:
                print(f"Error clearing graph directory: {e}")

        print(f"Graph data cleared for {dataset}_{setting}")