"""
Comprehensive prompt and response logging system for GraphRAG debugging.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class PromptResponseLogger:
    """Logger for all prompts and responses in the GraphRAG system."""

    def __init__(self, log_dir: str = "prompt_response_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Session ID for fallback when qa_pair_id/iteration not provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = timestamp
        self.fallback_log_file = self.log_dir / f"prompts_responses_{timestamp}.jsonl"

        # Track interaction counts per file
        self.interaction_counts = {}

    def log_interaction(self,
                       agent_name: str,
                       interaction_type: str,
                       system_prompt: str,
                       user_prompt: str,
                       llm_response: str,
                       batch_id: Optional[int] = None,
                       qa_pair_id: Optional[str] = None,
                       iteration: Optional[int] = None,
                       additional_metadata: Optional[Dict[str, Any]] = None):
        """
        Log a complete LLM interaction.

        Args:
            agent_name: Name of the agent making the call
            interaction_type: Type of interaction (retrieval_planning, answer_generation, etc.)
            system_prompt: System prompt used
            user_prompt: User prompt used
            llm_response: LLM response received
            batch_id: Batch ID if applicable
            qa_pair_id: QA pair ID if applicable
            iteration: Iteration number if applicable
            additional_metadata: Any additional metadata
        """
        # Check if logging is disabled globally
        global _logging_disabled
        if _logging_disabled:
            return

        # Determine which log file to use
        if qa_pair_id is not None and iteration is not None:
            # Use QA pair ID and iteration for filename
            log_file = self.log_dir / f"{qa_pair_id}_iter_{iteration}.jsonl"
            file_key = f"{qa_pair_id}_iter_{iteration}"
        else:
            # Fall back to session-based logging
            log_file = self.fallback_log_file
            file_key = "fallback"

        # Increment interaction count for this file
        if file_key not in self.interaction_counts:
            self.interaction_counts[file_key] = 0
        self.interaction_counts[file_key] += 1

        log_entry = {
            "session_id": self.session_id,
            "interaction_id": self.interaction_counts[file_key],
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "interaction_type": interaction_type,
            "batch_id": batch_id,
            "qa_pair_id": qa_pair_id,
            "iteration": iteration,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "llm_response": llm_response,
            "system_prompt_length": len(system_prompt),
            "user_prompt_length": len(user_prompt),
            "response_length": len(llm_response),
            "metadata": additional_metadata or {}
        }

        # Write to JSONL file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def log_critique_generation(self,
                              agent_name: str,
                              critique_type: str,
                              prompt_with_data: str,
                              critique_response: str,
                              token_limit: int,
                              batch_id: Optional[int] = None):
        """Log critique generation specifically."""
        self.log_interaction(
            agent_name=agent_name,
            interaction_type=f"critique_{critique_type}",
            system_prompt=f"Critique generation with {token_limit} token limit",
            user_prompt=prompt_with_data,
            llm_response=critique_response,
            batch_id=batch_id,
            additional_metadata={
                "critique_type": critique_type,
                "token_limit": token_limit
            }
        )

    def log_prompt_optimization(self,
                               agent_name: str,
                               prompt_type: str,
                               critique_input: str,
                               optimized_prompt: str,
                               is_frozen: bool,
                               batch_id: Optional[int] = None):
        """Log prompt optimization specifically."""
        self.log_interaction(
            agent_name=agent_name,
            interaction_type=f"optimize_{prompt_type}",
            system_prompt="Prompt optimization",
            user_prompt=critique_input,
            llm_response=optimized_prompt,
            batch_id=batch_id,
            additional_metadata={
                "prompt_type": prompt_type,
                "is_frozen": is_frozen,
                "action": "skipped" if is_frozen else "updated"
            }
        )

    def create_summary_report(self) -> str:
        """Create a human-readable summary report from all log files."""
        # Find all log files in the directory
        log_files = list(self.log_dir.glob("*.jsonl"))

        if not log_files:
            return "No log files found."

        interactions = []
        for log_file in log_files:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        interactions.append(json.loads(line))
                    except:
                        continue

        report_file = self.log_dir / f"summary_report_{self.session_id}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# GraphRAG Prompt/Response Summary Report\n\n")
            f.write(f"**Session ID:** {self.session_id}\n")
            f.write(f"**Total Interactions:** {len(interactions)}\n")
            f.write(f"**Total Log Files:** {len(log_files)}\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")

            # Group by interaction type
            by_type = {}
            for interaction in interactions:
                interaction_type = interaction['interaction_type']
                if interaction_type not in by_type:
                    by_type[interaction_type] = []
                by_type[interaction_type].append(interaction)

            f.write("## Interactions by Type\n\n")
            for interaction_type, items in by_type.items():
                f.write(f"### {interaction_type.title()} ({len(items)} interactions)\n\n")

                for item in items:
                    f.write(f"**Interaction {item['interaction_id']}** - {item['timestamp']}\n")
                    f.write(f"- **Agent:** {item['agent_name']}\n")
                    if item['batch_id']:
                        f.write(f"- **Batch:** {item['batch_id']}\n")
                    if item['qa_pair_id']:
                        f.write(f"- **QA Pair:** {item['qa_pair_id']}\n")
                    if item['iteration']:
                        f.write(f"- **Iteration:** {item['iteration']}\n")

                    f.write(f"- **System Prompt ({item['system_prompt_length']} chars):**\n")
                    f.write(f"  ```\n  {item['system_prompt'][:300]}{'...' if len(item['system_prompt']) > 300 else ''}\n  ```\n")

                    f.write(f"- **User Prompt ({item['user_prompt_length']} chars):**\n")
                    f.write(f"  ```\n  {item['user_prompt'][:300]}{'...' if len(item['user_prompt']) > 300 else ''}\n  ```\n")

                    f.write(f"- **LLM Response ({item['response_length']} chars):**\n")
                    f.write(f"  ```\n  {item['llm_response'][:300]}{'...' if len(item['llm_response']) > 300 else ''}\n  ```\n\n")

        return str(report_file)


# Global instance for easy access
_global_logger: Optional[PromptResponseLogger] = None
_logging_disabled: bool = False


def get_global_prompt_logger() -> PromptResponseLogger:
    """Get or create the global prompt response logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = PromptResponseLogger()
    return _global_logger


def initialize_prompt_logging(log_dir: str = "prompt_response_logs"):
    """Initialize the global prompt logging system."""
    global _global_logger, _logging_disabled

    if _logging_disabled:
        print("Prompt response logging initialization skipped (disabled for performance).")
        return

    _global_logger = PromptResponseLogger(log_dir)
    print(f"Prompt response logging initialized. Logs will be saved to: {_global_logger.log_dir}")


def finalize_prompt_logging() -> str:
    """Finalize logging and create summary report."""
    global _global_logger
    if _global_logger is None:
        return "No logging session active."

    report_file = _global_logger.create_summary_report()
    print(f"Prompt response summary report created: {report_file}")
    return report_file


def disable_prompt_logging():
    """Disable prompt and response logging for better performance."""
    global _logging_disabled
    _logging_disabled = True
    print("Prompt response logging disabled for performance optimization.")


def enable_prompt_logging():
    """Re-enable prompt and response logging."""
    global _logging_disabled
    _logging_disabled = False
    print("Prompt response logging re-enabled.")


def is_prompt_logging_disabled() -> bool:
    """Check if prompt logging is currently disabled."""
    global _logging_disabled
    return _logging_disabled