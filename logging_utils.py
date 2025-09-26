"""
Logging utilities for GraphRAG/VectorRAG systems.
Provides consistent, readable logging with message truncation.
"""

import logging
from typing import Any, Dict, List, Optional

class LoggingUtils:
    """Utility class for consistent, readable logging throughout the system."""

    # Configuration for different message types
    MAX_LENGTHS = {
        'generated_answer': 100,
        'evaluation_result': 100,
        'prompt_content': 150,
        'critique': 120,
        'reasoning': 100,
        'context': 200,
        'question': 80,
        'graph_description': 150,
        'function_call': 80,
        'default': 100
    }

    @classmethod
    def truncate_message(cls, message: str, msg_type: str = 'default', max_length: Optional[int] = None) -> str:
        """
        Truncate a message to a reasonable length for logging.

        Args:
            message: The message to truncate
            msg_type: Type of message for specific length limits
            max_length: Override the default max length

        Returns:
            Truncated message with ellipsis if needed
        """
        if not message:
            return ""

        max_len = max_length or cls.MAX_LENGTHS.get(msg_type, cls.MAX_LENGTHS['default'])

        if len(message) <= max_len:
            return message

        # Find a good breaking point (prefer word boundaries)
        truncated = message[:max_len]
        if ' ' in truncated:
            # Break at last word boundary
            truncated = truncated.rsplit(' ', 1)[0]

        return f"{truncated}..."

    @classmethod
    def format_agent_message(cls, agent_name: str, action: str, details: Dict[str, Any] = None) -> str:
        """
        Format a consistent agent action message.

        Args:
            agent_name: Name of the agent
            action: Action being performed
            details: Optional details dictionary

        Returns:
            Formatted log message
        """
        base_msg = f"[{agent_name}] {action}"

        if not details:
            return base_msg

        detail_parts = []
        for key, value in details.items():
            if isinstance(value, str) and key in cls.MAX_LENGTHS:
                value = cls.truncate_message(value, key)
            elif isinstance(value, (list, dict)) and len(str(value)) > 50:
                value = f"<{type(value).__name__} with {len(value) if hasattr(value, '__len__') else '?'} items>"
            detail_parts.append(f"{key}={value}")

        return f"{base_msg} | {' | '.join(detail_parts)}"

    @classmethod
    def format_batch_progress(cls, batch_id: int, repetition: int, total_reps: int, action: str, details: str = "") -> str:
        """
        Format batch progress messages consistently.

        Args:
            batch_id: Current batch ID
            repetition: Current repetition
            total_reps: Total repetitions
            action: Action being performed
            details: Additional details

        Returns:
            Formatted progress message
        """
        progress = f"[Batch {batch_id}, Rep {repetition}/{total_reps}]"
        details_str = f" | {cls.truncate_message(details)}" if details else ""
        return f"{progress} {action}{details_str}"

    @classmethod
    def format_qa_processing(cls, qa_pair_id: str, action: str, content: str = "", score: float = None) -> str:
        """
        Format QA pair processing messages.

        Args:
            qa_pair_id: QA pair identifier
            action: Action being performed
            content: Content being processed
            score: Optional score value

        Returns:
            Formatted QA processing message
        """
        base_msg = f"[QA: {qa_pair_id}] {action}"

        if content:
            content_str = cls.truncate_message(content, 'default')
            base_msg += f" | Content: {content_str}"

        if score is not None:
            base_msg += f" | Score: {score:.4f}"

        return base_msg

    @classmethod
    def format_critique_message(cls, agent_type: str, critique: str, is_frozen: bool = False) -> str:
        """
        Format critique generation messages.

        Args:
            agent_type: Type of agent being critiqued
            critique: Generated critique
            is_frozen: Whether the prompt is frozen

        Returns:
            Formatted critique message
        """
        status = "FROZEN - SKIPPED" if is_frozen else "UPDATED"
        critique_preview = cls.truncate_message(critique, 'critique')
        return f"[{agent_type.upper()}] Critique generated | Status: {status} | Preview: {critique_preview}"

    @classmethod
    def setup_concise_logging(cls, logger: logging.Logger, log_level: int = logging.INFO) -> None:
        """
        Configure a logger for concise, readable output.

        Args:
            logger: Logger to configure
            log_level: Logging level to set
        """
        logger.setLevel(log_level)

        # Create formatter for concise output
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )

        # If logger doesn't have handlers, add a console handler
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)


# Convenience functions for common logging patterns
def log_agent_action(logger: logging.Logger, agent_name: str, action: str, **details) -> None:
    """Log an agent action with consistent formatting."""
    message = LoggingUtils.format_agent_message(agent_name, action, details)
    logger.info(message)

def log_batch_progress(logger: logging.Logger, batch_id: int, repetition: int, total_reps: int, action: str, details: str = "") -> None:
    """Log batch progress with consistent formatting."""
    message = LoggingUtils.format_batch_progress(batch_id, repetition, total_reps, action, details)
    logger.info(message)

def log_qa_processing(logger: logging.Logger, qa_pair_id: str, action: str, content: str = "", score: float = None) -> None:
    """Log QA processing with consistent formatting."""
    message = LoggingUtils.format_qa_processing(qa_pair_id, action, content, score)
    logger.info(message)

def log_critique_result(logger: logging.Logger, agent_type: str, critique: str, is_frozen: bool = False) -> None:
    """Log critique generation results with consistent formatting."""
    message = LoggingUtils.format_critique_message(agent_type, critique, is_frozen)
    logger.info(message)