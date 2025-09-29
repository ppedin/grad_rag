"""
Standardized Evaluation Logging System for GraphRAG and VectorRAG Analysis.

This module provides a unified logging format for automatic analysis and comparison
of GraphRAG and VectorRAG systems. Each system gets its own folder with consistent
data structures for seamless automated processing.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from enum import Enum


class SystemType(Enum):
    """RAG system types."""
    GRAPHRAG = "graphrag"
    VECTORRAG = "vectorrag"


class StandardizedEvaluationLogger:
    """
    Standardized evaluation logger for consistent analysis across RAG systems.

    Features:
    - System-specific folder structure (graphrag/, vectorrag/)
    - Unified JSON schema for automatic analysis
    - Comprehensive intermediate outputs tracking
    - ROUGE score standardization
    - Execution time and performance metrics
    """

    def __init__(self,
                 system_type: SystemType,
                 dataset: str,
                 setting: str,
                 base_dir: str = "standardized_evaluation_logs"):
        """
        Initialize standardized evaluation logger.

        Args:
            system_type: Type of RAG system (GraphRAG or VectorRAG)
            dataset: Dataset name (e.g., "squality", "musique")
            setting: Setting type (e.g., "test", "dev")
            base_dir: Base directory for all evaluation logs
        """
        self.system_type = system_type
        self.dataset = dataset
        self.setting = setting

        # Create system-specific directory structure
        self.base_dir = Path(base_dir)
        self.system_dir = self.base_dir / system_type.value
        self.dataset_dir = self.system_dir / f"{dataset}_{setting}"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{system_type.value}_{dataset}_{setting}_{self.timestamp}"

        # Log files
        self.qa_pair_log = self.dataset_dir / f"qa_pairs_{self.timestamp}.jsonl"
        self.iteration_log = self.dataset_dir / f"iterations_{self.timestamp}.jsonl"
        self.session_summary = self.dataset_dir / f"session_summary_{self.timestamp}.json"

        # Session tracking
        self.session_metadata = {
            "session_id": self.session_id,
            "system_type": system_type.value,
            "dataset": dataset,
            "setting": setting,
            "start_time": datetime.now().isoformat(),
            "total_qa_pairs": 0,
            "total_iterations": 0,
            "avg_rouge_scores": {},
            "performance_metrics": {}
        }

        # Current context
        self.current_qa_pair_id = None
        self.qa_pair_count = 0
        self.iteration_count = 0

    def start_qa_pair_evaluation(self,
                                qa_pair_id: str,
                                question: str,
                                reference_answers: List[str],
                                document_text: str,
                                total_iterations: int,
                                metadata: Dict[str, Any] = None):
        """
        Log the start of QA pair evaluation with unified format.

        Args:
            qa_pair_id: Unique QA pair identifier
            question: The question being asked
            reference_answers: Ground truth answers
            document_text: Source document
            total_iterations: Expected number of iterations
            metadata: Additional metadata
        """
        self.current_qa_pair_id = qa_pair_id
        self.qa_pair_count += 1

        qa_pair_entry = {
            # Standard metadata
            "session_id": self.session_id,
            "system_type": self.system_type.value,
            "dataset": self.dataset,
            "setting": self.setting,
            "timestamp": datetime.now().isoformat(),
            "entry_type": "qa_pair_start",

            # QA pair specific data
            "qa_pair_id": qa_pair_id,
            "qa_pair_number": self.qa_pair_count,
            "question": question,
            "question_length": len(question),
            "reference_answers": reference_answers,
            "reference_answer_count": len(reference_answers),
            "reference_answer_lengths": [len(ans) for ans in reference_answers],
            "document_text": document_text,
            "document_length": len(document_text),
            "total_iterations_planned": total_iterations,

            # Additional metadata
            "metadata": metadata or {}
        }

        self._write_qa_pair_log(qa_pair_entry)

    def log_iteration_evaluation(self,
                                qa_pair_id: str,
                                iteration: int,
                                generated_answer: str,
                                rouge_scores: Dict[str, float],
                                intermediate_outputs: Dict[str, Any],
                                hyperparameters: Dict[str, Any] = None,
                                execution_time_seconds: float = None,
                                system_specific_metrics: Dict[str, Any] = None):
        """
        Log iteration evaluation with standardized format for automatic analysis.

        Args:
            qa_pair_id: QA pair identifier
            iteration: Iteration number (0-indexed)
            generated_answer: Generated answer text
            rouge_scores: Standardized ROUGE scores dict
            intermediate_outputs: All intermediate agent outputs
            hyperparameters: System hyperparameters used
            execution_time_seconds: Total execution time
            system_specific_metrics: GraphRAG graph stats or VectorRAG vector stats
        """
        self.iteration_count += 1

        # Standardize ROUGE scores format
        standardized_rouge = {
            "rouge_1": rouge_scores.get("rouge-1", rouge_scores.get("rouge_1", 0.0)),
            "rouge_2": rouge_scores.get("rouge-2", rouge_scores.get("rouge_2", 0.0)),
            "rouge_l": rouge_scores.get("rouge-l", rouge_scores.get("rouge_l", 0.0))
        }

        iteration_entry = {
            # Standard metadata
            "session_id": self.session_id,
            "system_type": self.system_type.value,
            "dataset": self.dataset,
            "setting": self.setting,
            "timestamp": datetime.now().isoformat(),
            "entry_type": "iteration_evaluation",

            # Iteration specific data
            "qa_pair_id": qa_pair_id,
            "iteration": iteration,
            "iteration_number": self.iteration_count,

            # Generated response
            "generated_answer": generated_answer,
            "generated_answer_length": len(generated_answer),

            # Standardized performance metrics
            "rouge_scores": standardized_rouge,
            "execution_time_seconds": execution_time_seconds,

            # System configuration
            "hyperparameters": hyperparameters or {},

            # Standardized intermediate outputs
            "intermediate_outputs": self._standardize_intermediate_outputs(intermediate_outputs),

            # System-specific metrics (GraphRAG: graph stats, VectorRAG: vector stats)
            "system_specific_metrics": system_specific_metrics or {}
        }

        self._write_iteration_log(iteration_entry)

    def complete_qa_pair_evaluation(self,
                                   qa_pair_id: str,
                                   final_rouge_score: float,
                                   rouge_progression: List[float],
                                   best_iteration: int,
                                   total_iterations_completed: int,
                                   best_answer: str = None):
        """
        Log QA pair completion with summary statistics.

        Args:
            qa_pair_id: QA pair identifier
            final_rouge_score: Final ROUGE-L score
            rouge_progression: ROUGE scores across iterations
            best_iteration: Iteration with highest ROUGE score
            total_iterations_completed: Total iterations actually completed
            best_answer: Answer from best performing iteration
        """
        rouge_improvement = (rouge_progression[-1] - rouge_progression[0]) if len(rouge_progression) > 1 else 0.0

        completion_entry = {
            # Standard metadata
            "session_id": self.session_id,
            "system_type": self.system_type.value,
            "dataset": self.dataset,
            "setting": self.setting,
            "timestamp": datetime.now().isoformat(),
            "entry_type": "qa_pair_completion",

            # QA pair completion data
            "qa_pair_id": qa_pair_id,
            "final_rouge_score": final_rouge_score,
            "rouge_progression": rouge_progression,
            "rouge_improvement": rouge_improvement,
            "best_iteration": best_iteration,
            "best_rouge_score": max(rouge_progression) if rouge_progression else 0.0,
            "total_iterations_completed": total_iterations_completed,
            "best_answer": best_answer,
            "best_answer_length": len(best_answer) if best_answer else 0
        }

        self._write_qa_pair_log(completion_entry)

    def finalize_session(self) -> str:
        """
        Finalize the evaluation session and generate summary statistics.

        Returns:
            str: Path to session summary file
        """
        # Calculate session statistics
        session_stats = self._calculate_session_statistics()

        # Update session metadata
        self.session_metadata.update({
            "end_time": datetime.now().isoformat(),
            "total_qa_pairs": self.qa_pair_count,
            "total_iterations": self.iteration_count,
            **session_stats
        })

        # Write session summary
        with open(self.session_summary, 'w', encoding='utf-8') as f:
            json.dump(self.session_metadata, f, indent=2, ensure_ascii=False)

        return str(self.session_summary)

    def _standardize_intermediate_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize intermediate outputs format across systems.

        Args:
            outputs: Raw intermediate outputs from system

        Returns:
            Dict with standardized intermediate output format
        """
        standardized = {}

        # Common fields across both systems
        if "hyperparameters" in outputs:
            standardized["hyperparameters"] = outputs["hyperparameters"]

        if "retrieval" in outputs:
            standardized["retrieval"] = {
                "query": outputs["retrieval"].get("query", ""),
                "retrieved_context_length": outputs["retrieval"].get("retrieved_context_length", 0),
                "processing_time_ms": outputs["retrieval"].get("processing_time_ms", 0)
            }

        if "answer_generation" in outputs:
            standardized["answer_generation"] = {
                "generated_answer_length": outputs["answer_generation"].get("generated_answer_length", 0),
                "processing_time_ms": outputs["answer_generation"].get("processing_time_ms", 0)
            }

        if "evaluation" in outputs:
            standardized["evaluation"] = {
                "rouge_score": outputs["evaluation"].get("rouge_score", 0.0),
                "processing_time_ms": outputs["evaluation"].get("processing_time_ms", 0)
            }

        # System-specific fields
        if self.system_type == SystemType.GRAPHRAG:
            if "graph_building" in outputs:
                standardized["graph_building"] = outputs["graph_building"]
        else:  # VectorRAG
            if "vector_building" in outputs:
                standardized["vector_building"] = outputs["vector_building"]

        # Include all other fields as-is for flexibility
        for key, value in outputs.items():
            if key not in standardized:
                standardized[key] = value

        return standardized

    def _calculate_session_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive session statistics."""
        stats = {
            "avg_rouge_scores": {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0},
            "performance_metrics": {
                "avg_execution_time": 0.0,
                "total_execution_time": 0.0,
                "avg_iterations_per_qa": 0.0
            }
        }

        # Read iteration logs to calculate statistics
        if self.iteration_log.exists():
            rouge_1_scores = []
            rouge_2_scores = []
            rouge_l_scores = []
            execution_times = []

            with open(self.iteration_log, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("entry_type") == "iteration_evaluation":
                        rouge_scores = entry.get("rouge_scores", {})
                        rouge_1_scores.append(rouge_scores.get("rouge_1", 0.0))
                        rouge_2_scores.append(rouge_scores.get("rouge_2", 0.0))
                        rouge_l_scores.append(rouge_scores.get("rouge_l", 0.0))

                        if entry.get("execution_time_seconds"):
                            execution_times.append(entry["execution_time_seconds"])

            # Calculate averages
            if rouge_1_scores:
                stats["avg_rouge_scores"]["rouge_1"] = sum(rouge_1_scores) / len(rouge_1_scores)
                stats["avg_rouge_scores"]["rouge_2"] = sum(rouge_2_scores) / len(rouge_2_scores)
                stats["avg_rouge_scores"]["rouge_l"] = sum(rouge_l_scores) / len(rouge_l_scores)

            if execution_times:
                stats["performance_metrics"]["avg_execution_time"] = sum(execution_times) / len(execution_times)
                stats["performance_metrics"]["total_execution_time"] = sum(execution_times)

            if self.qa_pair_count > 0:
                stats["performance_metrics"]["avg_iterations_per_qa"] = self.iteration_count / self.qa_pair_count

        return stats

    def _write_qa_pair_log(self, entry: Dict[str, Any]):
        """Write QA pair log entry."""
        with open(self.qa_pair_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def _write_iteration_log(self, entry: Dict[str, Any]):
        """Write iteration log entry."""
        with open(self.iteration_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# Global instances for each system type
_global_loggers: Dict[SystemType, StandardizedEvaluationLogger] = {}


def get_standardized_logger(system_type: SystemType,
                          dataset: str,
                          setting: str) -> StandardizedEvaluationLogger:
    """
    Get or create standardized evaluation logger for a system.

    Args:
        system_type: GraphRAG or VectorRAG
        dataset: Dataset name
        setting: Setting type

    Returns:
        StandardizedEvaluationLogger instance
    """
    global _global_loggers

    key = (system_type, dataset, setting)
    if key not in _global_loggers:
        _global_loggers[key] = StandardizedEvaluationLogger(system_type, dataset, setting)

    return _global_loggers[key]


def initialize_standardized_logging(system_type: SystemType,
                                  dataset: str,
                                  setting: str,
                                  base_dir: str = "standardized_evaluation_logs") -> StandardizedEvaluationLogger:
    """
    Initialize standardized evaluation logging for a system.

    Args:
        system_type: GraphRAG or VectorRAG
        dataset: Dataset name
        setting: Setting type
        base_dir: Base directory for logs

    Returns:
        StandardizedEvaluationLogger instance
    """
    global _global_loggers

    logger = StandardizedEvaluationLogger(system_type, dataset, setting, base_dir)
    _global_loggers[(system_type, dataset, setting)] = logger

    print(f"Standardized evaluation logging initialized for {system_type.value}")
    print(f"Logs directory: {logger.dataset_dir}")
    print(f"Session ID: {logger.session_id}")

    return logger


def finalize_standardized_logging(system_type: SystemType,
                                dataset: str,
                                setting: str) -> Optional[str]:
    """
    Finalize standardized evaluation logging and generate summary.

    Args:
        system_type: GraphRAG or VectorRAG
        dataset: Dataset name
        setting: Setting type

    Returns:
        Optional[str]: Path to session summary file
    """
    global _global_loggers

    key = (system_type, dataset, setting)
    if key in _global_loggers:
        summary_path = _global_loggers[key].finalize_session()
        print(f"Standardized evaluation session finalized: {summary_path}")
        return summary_path

    return None