"""
Step execution logging system for tracking when steps are executed and their success status.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from enum import Enum


class StepStatus(Enum):
    """Status of step execution."""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class StepExecutionLogger:
    """Logger for tracking step execution status across the pipeline."""

    def __init__(self, log_dir: str = "step_execution_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"step_execution_{timestamp}.jsonl"

        self.session_id = timestamp
        self.step_count = 0

        # Track pipeline state
        self.current_batch_id = None
        self.current_qa_pair_id = None
        self.current_iteration = None
        self.pipeline_start_time = None

    def start_pipeline(self, dataset: str, setting: str, total_qa_pairs: int = None):
        """Mark the start of a pipeline run."""
        self.pipeline_start_time = datetime.now()

        log_entry = {
            "session_id": self.session_id,
            "step_id": "pipeline_start",
            "timestamp": self.pipeline_start_time.isoformat(),
            "step_type": "pipeline",
            "step_name": "pipeline_initialization",
            "status": StepStatus.STARTED.value,
            "dataset": dataset,
            "setting": setting,
            "total_qa_pairs": total_qa_pairs,
            "metadata": {}
        }

        self._write_log_entry(log_entry)

    def start_batch(self, batch_id: int, qa_pair_id: str, iteration: int, total_iterations: int):
        """Mark the start of processing a batch (QA pair + iteration)."""
        self.current_batch_id = batch_id
        self.current_qa_pair_id = qa_pair_id
        self.current_iteration = iteration

        log_entry = {
            "session_id": self.session_id,
            "step_id": f"batch_{batch_id}_start",
            "timestamp": datetime.now().isoformat(),
            "step_type": "batch",
            "step_name": "batch_processing_start",
            "status": StepStatus.STARTED.value,
            "batch_id": batch_id,
            "qa_pair_id": qa_pair_id,
            "iteration": iteration,
            "total_iterations": total_iterations,
            "metadata": {}
        }

        self._write_log_entry(log_entry)

    def log_step(self,
                 step_name: str,
                 status: StepStatus,
                 agent_name: str = None,
                 execution_time_ms: float = None,
                 error_message: str = None,
                 input_data_summary: str = None,
                 output_data_summary: str = None,
                 additional_metadata: Dict[str, Any] = None):
        """
        Log a step execution.

        Args:
            step_name: Name of the step (e.g., 'hyperparameters_generation', 'graph_building')
            status: Status of the step execution
            agent_name: Name of the agent executing the step
            execution_time_ms: Execution time in milliseconds
            error_message: Error message if step failed
            input_data_summary: Brief summary of input data
            output_data_summary: Brief summary of output data
            additional_metadata: Any additional metadata
        """
        self.step_count += 1

        log_entry = {
            "session_id": self.session_id,
            "step_id": f"step_{self.step_count}",
            "timestamp": datetime.now().isoformat(),
            "step_type": "agent_step",
            "step_name": step_name,
            "status": status.value,
            "batch_id": self.current_batch_id,
            "qa_pair_id": self.current_qa_pair_id,
            "iteration": self.current_iteration,
            "agent_name": agent_name,
            "execution_time_ms": execution_time_ms,
            "error_message": error_message,
            "input_data_summary": input_data_summary,
            "output_data_summary": output_data_summary,
            "metadata": additional_metadata or {}
        }

        self._write_log_entry(log_entry)

    def complete_batch(self, success: bool, final_rouge_score: float = None, error_message: str = None):
        """Mark the completion of a batch."""
        status = StepStatus.COMPLETED if success else StepStatus.FAILED

        log_entry = {
            "session_id": self.session_id,
            "step_id": f"batch_{self.current_batch_id}_complete",
            "timestamp": datetime.now().isoformat(),
            "step_type": "batch",
            "step_name": "batch_processing_complete",
            "status": status.value,
            "batch_id": self.current_batch_id,
            "qa_pair_id": self.current_qa_pair_id,
            "iteration": self.current_iteration,
            "final_rouge_score": final_rouge_score,
            "error_message": error_message,
            "metadata": {}
        }

        self._write_log_entry(log_entry)

    def complete_pipeline(self, success: bool, total_qa_pairs_processed: int = None,
                         total_iterations_completed: int = None, error_message: str = None):
        """Mark the completion of the entire pipeline."""
        execution_time = None
        if self.pipeline_start_time:
            execution_time = (datetime.now() - self.pipeline_start_time).total_seconds()

        status = StepStatus.COMPLETED if success else StepStatus.FAILED

        log_entry = {
            "session_id": self.session_id,
            "step_id": "pipeline_complete",
            "timestamp": datetime.now().isoformat(),
            "step_type": "pipeline",
            "step_name": "pipeline_completion",
            "status": status.value,
            "total_execution_time_seconds": execution_time,
            "total_qa_pairs_processed": total_qa_pairs_processed,
            "total_iterations_completed": total_iterations_completed,
            "total_steps_executed": self.step_count,
            "error_message": error_message,
            "metadata": {}
        }

        self._write_log_entry(log_entry)

    def _write_log_entry(self, log_entry: Dict[str, Any]):
        """Write a log entry to the file."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of execution status."""
        if not self.log_file.exists():
            return {"error": "No log file found"}

        entries = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                entries.append(json.loads(line))

        # Count steps by status
        status_counts = {}
        total_steps = 0
        failed_steps = []

        for entry in entries:
            if entry['step_type'] == 'agent_step':
                total_steps += 1
                status = entry['status']
                status_counts[status] = status_counts.get(status, 0) + 1

                if status == StepStatus.FAILED.value:
                    failed_steps.append({
                        'step_name': entry['step_name'],
                        'agent_name': entry.get('agent_name'),
                        'error_message': entry.get('error_message'),
                        'timestamp': entry['timestamp']
                    })

        # Calculate success rate
        completed_steps = status_counts.get(StepStatus.COMPLETED.value, 0)
        success_rate = (completed_steps / total_steps * 100) if total_steps > 0 else 0

        return {
            "session_id": self.session_id,
            "total_steps": total_steps,
            "status_counts": status_counts,
            "success_rate": round(success_rate, 2),
            "failed_steps": failed_steps,
            "log_file": str(self.log_file)
        }

    def create_execution_report(self) -> str:
        """Create a human-readable execution report."""
        if not self.log_file.exists():
            return "No log file found."

        entries = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                entries.append(json.loads(line))

        report_file = self.log_dir / f"execution_report_{self.session_id}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Step Execution Report\n\n")
            f.write(f"**Session ID:** {self.session_id}\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")

            # Pipeline overview
            pipeline_entries = [e for e in entries if e['step_type'] == 'pipeline']
            if pipeline_entries:
                start_entry = next((e for e in pipeline_entries if 'start' in e['step_name']), None)
                end_entry = next((e for e in pipeline_entries if 'complete' in e['step_name']), None)

                f.write("## Pipeline Overview\n\n")
                if start_entry:
                    f.write(f"- **Started:** {start_entry['timestamp']}\n")
                    f.write(f"- **Dataset:** {start_entry.get('dataset', 'Unknown')}\n")
                    f.write(f"- **Setting:** {start_entry.get('setting', 'Unknown')}\n")

                if end_entry:
                    f.write(f"- **Completed:** {end_entry['timestamp']}\n")
                    f.write(f"- **Status:** {end_entry['status']}\n")
                    if end_entry.get('total_execution_time_seconds'):
                        f.write(f"- **Total Time:** {end_entry['total_execution_time_seconds']:.2f} seconds\n")
                    f.write(f"- **QA Pairs Processed:** {end_entry.get('total_qa_pairs_processed', 'Unknown')}\n")
                    f.write(f"- **Total Steps:** {end_entry.get('total_steps_executed', 'Unknown')}\n")

            # Step execution summary
            step_entries = [e for e in entries if e['step_type'] == 'agent_step']
            f.write(f"\n## Step Execution Summary\n\n")
            f.write(f"**Total Steps:** {len(step_entries)}\n\n")

            # Group by QA pair and iteration
            by_qa_pair = {}
            for entry in step_entries:
                qa_pair = entry.get('qa_pair_id', 'unknown')
                iteration = entry.get('iteration', 0)
                key = f"{qa_pair}_iter_{iteration}"

                if key not in by_qa_pair:
                    by_qa_pair[key] = []
                by_qa_pair[key].append(entry)

            for qa_iter_key, steps in by_qa_pair.items():
                qa_pair_id, iteration_part = qa_iter_key.split('_iter_')
                iteration = iteration_part

                f.write(f"### QA Pair: {qa_pair_id} - Iteration {iteration}\n\n")

                for step in steps:
                    status_emoji = {
                        'started': '🔄',
                        'completed': '✅',
                        'failed': '❌',
                        'skipped': '⏭️',
                        'retrying': '🔁'
                    }.get(step['status'], '❓')

                    f.write(f"{status_emoji} **{step['step_name']}** - {step['status']}\n")
                    f.write(f"   - Agent: {step.get('agent_name', 'Unknown')}\n")
                    f.write(f"   - Time: {step['timestamp']}\n")

                    if step.get('execution_time_ms'):
                        f.write(f"   - Duration: {step['execution_time_ms']:.2f}ms\n")

                    if step.get('error_message'):
                        f.write(f"   - Error: {step['error_message']}\n")

                    f.write("\n")

        return str(report_file)


# Global instance for easy access
_global_step_logger: Optional[StepExecutionLogger] = None


def get_global_step_logger() -> StepExecutionLogger:
    """Get or create the global step execution logger."""
    global _global_step_logger
    if _global_step_logger is None:
        _global_step_logger = StepExecutionLogger()
    return _global_step_logger


def initialize_step_logging(log_dir: str = "step_execution_logs"):
    """Initialize the global step execution logging system."""
    global _global_step_logger
    _global_step_logger = StepExecutionLogger(log_dir)
    print(f"Step execution logging initialized. Logs will be saved to: {_global_step_logger.log_file}")


def finalize_step_logging() -> str:
    """Finalize step logging and create execution report."""
    global _global_step_logger
    if _global_step_logger is None:
        return "No step logging session active."

    report_file = _global_step_logger.create_execution_report()
    print(f"Step execution report created: {report_file}")
    return report_file