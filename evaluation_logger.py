"""
Evaluation logging system for tracking intermediate outputs, reference examples, and ROUGE scores.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path


class EvaluationLogger:
    """Logger for evaluation data including intermediate outputs and ROUGE scores."""

    def __init__(self, log_dir: str = "evaluation_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"evaluation_{timestamp}.jsonl"

        self.session_id = timestamp
        self.evaluation_count = 0

        # Track current context
        self.current_qa_pair_id = None
        self.current_iteration = None

    def start_qa_pair_evaluation(self, qa_pair_id: str, question: str, reference_answers: List[str],
                                document_text: str, total_iterations: int, metadata: Dict[str, Any] = None):
        """Start evaluation tracking for a new QA pair."""
        self.current_qa_pair_id = qa_pair_id

        log_entry = {
            "session_id": self.session_id,
            "evaluation_id": f"qa_pair_{qa_pair_id}_start",
            "timestamp": datetime.now().isoformat(),
            "entry_type": "qa_pair_start",
            "qa_pair_id": qa_pair_id,
            "question": question,
            "reference_answers": reference_answers,
            "document_text": document_text,
            "document_length": len(document_text),
            "total_iterations": total_iterations,
            "metadata": metadata or {}
        }

        self._write_log_entry(log_entry)

    def log_iteration_evaluation(self,
                                qa_pair_id: str,
                                iteration: int,
                                intermediate_outputs: Dict[str, Any],
                                generated_answer: str,
                                rouge_scores: Dict[str, float],
                                hyperparameters: Dict[str, Any] = None,
                                graph_metrics: Dict[str, Any] = None,
                                retrieval_context: str = None,
                                execution_time_seconds: float = None,
                                additional_metrics: Dict[str, Any] = None):
        """
        Log evaluation data for a single iteration.

        Args:
            qa_pair_id: QA pair identifier
            iteration: Iteration number
            intermediate_outputs: All intermediate outputs from agents
            generated_answer: Final generated answer
            rouge_scores: ROUGE scores (rouge-1, rouge-2, rouge-l)
            hyperparameters: RAG hyperparameters used
            graph_metrics: Graph connectivity metrics
            retrieval_context: Retrieved context from graph
            execution_time_seconds: Total execution time for this iteration
            additional_metrics: Any additional evaluation metrics
        """
        self.current_iteration = iteration
        self.evaluation_count += 1

        log_entry = {
            "session_id": self.session_id,
            "evaluation_id": f"qa_pair_{qa_pair_id}_iter_{iteration}",
            "timestamp": datetime.now().isoformat(),
            "entry_type": "iteration_evaluation",
            "qa_pair_id": qa_pair_id,
            "iteration": iteration,
            "generated_answer": generated_answer,
            "generated_answer_length": len(generated_answer),
            "rouge_scores": rouge_scores,
            "rouge_1": rouge_scores.get("rouge-1", 0.0),
            "rouge_2": rouge_scores.get("rouge-2", 0.0),
            "rouge_l": rouge_scores.get("rouge-l", 0.0),
            "execution_time_seconds": execution_time_seconds,
            "hyperparameters": hyperparameters or {},
            "graph_metrics": graph_metrics or {},
            "retrieval_context": retrieval_context,
            "retrieval_context_length": len(retrieval_context) if retrieval_context else 0,
            "intermediate_outputs": intermediate_outputs,
            "additional_metrics": additional_metrics or {}
        }

        self._write_log_entry(log_entry)

    def log_intermediate_output(self,
                               qa_pair_id: str,
                               iteration: int,
                               agent_name: str,
                               output_type: str,
                               output_data: Any,
                               processing_time_ms: float = None,
                               metadata: Dict[str, Any] = None):
        """
        Log intermediate output from a specific agent.

        Args:
            qa_pair_id: QA pair identifier
            iteration: Iteration number
            agent_name: Name of the agent producing output
            output_type: Type of output (graph, hyperparameters, context, etc.)
            output_data: The actual output data
            processing_time_ms: Processing time in milliseconds
            metadata: Additional metadata
        """
        log_entry = {
            "session_id": self.session_id,
            "evaluation_id": f"intermediate_{qa_pair_id}_iter_{iteration}_{agent_name}",
            "timestamp": datetime.now().isoformat(),
            "entry_type": "intermediate_output",
            "qa_pair_id": qa_pair_id,
            "iteration": iteration,
            "agent_name": agent_name,
            "output_type": output_type,
            "output_data": output_data,
            "processing_time_ms": processing_time_ms,
            "metadata": metadata or {}
        }

        self._write_log_entry(log_entry)

    def complete_qa_pair_evaluation(self,
                                   qa_pair_id: str,
                                   final_rouge_score: float,
                                   rouge_progression: List[float],
                                   best_iteration: int,
                                   total_iterations_completed: int,
                                   improvement_gained: float = None,
                                   final_metrics: Dict[str, Any] = None):
        """Complete evaluation tracking for a QA pair."""
        log_entry = {
            "session_id": self.session_id,
            "evaluation_id": f"qa_pair_{qa_pair_id}_complete",
            "timestamp": datetime.now().isoformat(),
            "entry_type": "qa_pair_complete",
            "qa_pair_id": qa_pair_id,
            "final_rouge_score": final_rouge_score,
            "rouge_progression": rouge_progression,
            "best_iteration": best_iteration,
            "total_iterations_completed": total_iterations_completed,
            "improvement_gained": improvement_gained,
            "final_metrics": final_metrics or {}
        }

        self._write_log_entry(log_entry)

    def log_rouge_comparison(self,
                           qa_pair_id: str,
                           iteration: int,
                           current_rouge: float,
                           previous_rouge: float = None,
                           improvement: float = None,
                           is_best_so_far: bool = False):
        """Log ROUGE score comparison data."""
        log_entry = {
            "session_id": self.session_id,
            "evaluation_id": f"rouge_comparison_{qa_pair_id}_iter_{iteration}",
            "timestamp": datetime.now().isoformat(),
            "entry_type": "rouge_comparison",
            "qa_pair_id": qa_pair_id,
            "iteration": iteration,
            "current_rouge": current_rouge,
            "previous_rouge": previous_rouge,
            "improvement": improvement,
            "is_best_so_far": is_best_so_far
        }

        self._write_log_entry(log_entry)

    def _write_log_entry(self, log_entry: Dict[str, Any]):
        """Write a log entry to the file."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get a summary of evaluation results."""
        if not self.log_file.exists():
            return {"error": "No log file found"}

        entries = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                entries.append(json.loads(line))

        # Collect QA pair summaries
        qa_pairs = {}
        iteration_data = {}

        for entry in entries:
            qa_pair_id = entry.get('qa_pair_id')
            if not qa_pair_id:
                continue

            if entry['entry_type'] == 'qa_pair_start':
                qa_pairs[qa_pair_id] = {
                    'question': entry['question'],
                    'reference_answers': entry['reference_answers'],
                    'total_iterations': entry['total_iterations']
                }

            elif entry['entry_type'] == 'iteration_evaluation':
                iteration = entry['iteration']
                key = f"{qa_pair_id}_iter_{iteration}"
                iteration_data[key] = {
                    'rouge_scores': entry['rouge_scores'],
                    'generated_answer': entry['generated_answer'],
                    'execution_time': entry.get('execution_time_seconds')
                }

            elif entry['entry_type'] == 'qa_pair_complete':
                if qa_pair_id in qa_pairs:
                    qa_pairs[qa_pair_id].update({
                        'final_rouge_score': entry['final_rouge_score'],
                        'rouge_progression': entry['rouge_progression'],
                        'best_iteration': entry['best_iteration'],
                        'improvement_gained': entry.get('improvement_gained')
                    })

        # Calculate overall statistics
        final_scores = [data.get('final_rouge_score', 0) for data in qa_pairs.values() if 'final_rouge_score' in data]
        improvements = [data.get('improvement_gained', 0) for data in qa_pairs.values() if data.get('improvement_gained') is not None]

        summary = {
            "session_id": self.session_id,
            "total_qa_pairs": len(qa_pairs),
            "total_iterations_logged": len(iteration_data),
            "average_final_rouge": sum(final_scores) / len(final_scores) if final_scores else 0,
            "average_improvement": sum(improvements) / len(improvements) if improvements else 0,
            "qa_pairs": qa_pairs,
            "log_file": str(self.log_file)
        }

        return summary

    def export_for_analysis(self, output_file: str = None) -> str:
        """Export evaluation data in a format suitable for analysis."""
        if not self.log_file.exists():
            return "No log file found."

        if output_file is None:
            output_file = self.log_dir / f"evaluation_analysis_{self.session_id}.json"

        entries = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                entries.append(json.loads(line))

        # Structure data for analysis
        analysis_data = {
            "session_metadata": {
                "session_id": self.session_id,
                "export_timestamp": datetime.now().isoformat(),
                "total_entries": len(entries)
            },
            "qa_pairs": {},
            "iteration_details": [],
            "rouge_progressions": {},
            "performance_metrics": {}
        }

        for entry in entries:
            qa_pair_id = entry.get('qa_pair_id')

            if entry['entry_type'] == 'qa_pair_start':
                analysis_data["qa_pairs"][qa_pair_id] = {
                    "question": entry['question'],
                    "reference_answers": entry['reference_answers'],
                    "document_length": entry['document_length'],
                    "total_iterations": entry['total_iterations']
                }

            elif entry['entry_type'] == 'iteration_evaluation':
                analysis_data["iteration_details"].append({
                    "qa_pair_id": qa_pair_id,
                    "iteration": entry['iteration'],
                    "rouge_scores": entry['rouge_scores'],
                    "generated_answer_length": entry['generated_answer_length'],
                    "execution_time_seconds": entry.get('execution_time_seconds'),
                    "hyperparameters": entry.get('hyperparameters', {}),
                    "graph_metrics": entry.get('graph_metrics', {})
                })

            elif entry['entry_type'] == 'qa_pair_complete':
                analysis_data["rouge_progressions"][qa_pair_id] = entry['rouge_progression']

        # Save analysis data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)

        return str(output_file)

    def create_evaluation_report(self) -> str:
        """Create a human-readable evaluation report."""
        if not self.log_file.exists():
            return "No log file found."

        summary = self.get_evaluation_summary()
        report_file = self.log_dir / f"evaluation_report_{self.session_id}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Evaluation Report\n\n")
            f.write(f"**Session ID:** {self.session_id}\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")

            f.write("## Overview\n\n")
            f.write(f"- **Total QA Pairs:** {summary['total_qa_pairs']}\n")
            f.write(f"- **Total Iterations:** {summary['total_iterations_logged']}\n")
            f.write(f"- **Average Final ROUGE:** {summary['average_final_rouge']:.4f}\n")
            f.write(f"- **Average Improvement:** {summary['average_improvement']:.4f}\n\n")

            f.write("## QA Pair Results\n\n")

            for qa_pair_id, data in summary['qa_pairs'].items():
                f.write(f"### QA Pair: {qa_pair_id}\n\n")
                f.write(f"**Question:** {data['question']}\n\n")
                f.write(f"**Reference Answer:** {data['reference_answers'][0] if data['reference_answers'] else 'N/A'}\n\n")

                if 'final_rouge_score' in data:
                    f.write(f"**Final ROUGE Score:** {data['final_rouge_score']:.4f}\n")
                    f.write(f"**Best Iteration:** {data['best_iteration']}\n")

                    if data.get('improvement_gained') is not None:
                        f.write(f"**Improvement Gained:** {data['improvement_gained']:.4f}\n")

                    if 'rouge_progression' in data:
                        f.write(f"**ROUGE Progression:** {data['rouge_progression']}\n")

                f.write("\n")

        return str(report_file)


# Global instance for easy access
_global_eval_logger: Optional[EvaluationLogger] = None


def get_global_evaluation_logger() -> EvaluationLogger:
    """Get or create the global evaluation logger."""
    global _global_eval_logger
    if _global_eval_logger is None:
        _global_eval_logger = EvaluationLogger()
    return _global_eval_logger


def initialize_evaluation_logging(log_dir: str = "evaluation_logs"):
    """Initialize the global evaluation logging system."""
    global _global_eval_logger
    _global_eval_logger = EvaluationLogger(log_dir)
    print(f"Evaluation logging initialized. Logs will be saved to: {_global_eval_logger.log_file}")


def finalize_evaluation_logging() -> str:
    """Finalize evaluation logging and create reports."""
    global _global_eval_logger
    if _global_eval_logger is None:
        return "No evaluation logging session active."

    report_file = _global_eval_logger.create_evaluation_report()
    analysis_file = _global_eval_logger.export_for_analysis()

    print(f"Evaluation report created: {report_file}")
    print(f"Analysis data exported: {analysis_file}")

    return report_file