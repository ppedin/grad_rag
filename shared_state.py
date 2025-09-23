"""
Shared state management for AutoGen agents.
"""

import json
import os
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
            "graph_statistics": {}
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