# GraphRAG Multi-Agent System - Comprehensive Logging Guide

This document describes the three-tiered logging system implemented for the Multi-Agent GraphRAG System with Complete Reset Logic.

## Overview

The logging system consists of three complementary mechanisms:

1. **Prompt Response Logging** - Communication flow monitoring
2. **Step Execution Logging** - Pipeline execution status tracking
3. **Evaluation Logging** - Performance evaluation and intermediate outputs

## 1. Prompt Response Logging

**Purpose**: Monitor all LLM communications to ensure proper information flow between agents.

**Location**: `prompt_response_logs/`

**Files**:
- `prompts_responses_YYYYMMDD_HHMMSS.jsonl` - Main log file
- `summary_report_YYYYMMDD_HHMMSS.md` - Human-readable summary

**Implementation**: `prompt_response_logger.py`

**What is logged**:
- System prompts sent to LLMs
- User prompts with context
- Complete LLM responses
- Agent names and interaction types
- Batch/QA pair/iteration context
- Token counts and metadata

**Sample log entry**:
```json
{
  "session_id": "20250928_143022",
  "interaction_id": 1,
  "timestamp": "2025-09-28T14:30:22.123456",
  "agent_name": "HyperparametersGraphAgent",
  "interaction_type": "hyperparameters_generation",
  "batch_id": 1,
  "qa_pair_id": "qa_001",
  "iteration": 0,
  "system_prompt": "You are an expert...",
  "user_prompt": "Given the document text...",
  "llm_response": "Based on the analysis...",
  "system_prompt_length": 156,
  "user_prompt_length": 2341,
  "response_length": 892,
  "metadata": {}
}
```

## 2. Step Execution Logging

**Purpose**: Track when each step executes, its success/failure status, and execution timing.

**Location**: `step_execution_logs/`

**Files**:
- `step_execution_YYYYMMDD_HHMMSS.jsonl` - Main log file
- `execution_report_YYYYMMDD_HHMMSS.md` - Human-readable execution report

**Implementation**: `step_execution_logger.py`

**What is logged**:
- Pipeline start/completion events
- Batch processing start/completion per iteration
- Individual agent step execution status
- Execution timing for each step
- Error messages and recovery attempts
- Success rates and failure patterns

**Step statuses**:
- `started` - Step has begun execution
- `completed` - Step finished successfully
- `failed` - Step failed with error
- `skipped` - Step was skipped
- `retrying` - Step is being retried after failure

**Sample log entry**:
```json
{
  "session_id": "20250928_143022",
  "step_id": "step_5",
  "timestamp": "2025-09-28T14:32:15.789012",
  "step_type": "agent_step",
  "step_name": "graph_building",
  "status": "completed",
  "batch_id": 1,
  "qa_pair_id": "qa_001",
  "iteration": 0,
  "agent_name": "GraphBuilderAgent",
  "execution_time_ms": 2341.56,
  "input_data_summary": "chunk_size: 512, 15 text chunks",
  "output_data_summary": "123 entities, 89 relationships",
  "metadata": {}
}
```

## 3. Evaluation Logging

**Purpose**: Comprehensive evaluation data for analysis including intermediate outputs, reference examples, and ROUGE score progressions.

**Location**: `evaluation_logs/`

**Files**:
- `evaluation_YYYYMMDD_HHMMSS.jsonl` - Main log file
- `evaluation_report_YYYYMMDD_HHMMSS.md` - Human-readable evaluation report
- `evaluation_analysis_YYYYMMDD_HHMMSS.json` - Structured data for analysis

**Implementation**: `evaluation_logger.py`

**What is logged**:
- QA pair metadata (question, reference answers, document text)
- Complete intermediate outputs from each agent per iteration
- Generated answers and their quality metrics
- ROUGE score progression across iterations
- Hyperparameters and graph connectivity metrics
- Retrieved contexts and execution timing
- Final performance summaries per QA pair

**Sample log entry (iteration evaluation)**:
```json
{
  "session_id": "20250928_143022",
  "evaluation_id": "qa_pair_qa_001_iter_0",
  "timestamp": "2025-09-28T14:33:45.012345",
  "entry_type": "iteration_evaluation",
  "qa_pair_id": "qa_001",
  "iteration": 0,
  "generated_answer": "Machine learning is a method...",
  "generated_answer_length": 245,
  "rouge_scores": {
    "rouge-1": 0.7234,
    "rouge-2": 0.6511,
    "rouge-l": 0.6890
  },
  "execution_time_seconds": 15.67,
  "hyperparameters": {"chunk_size": 512},
  "graph_metrics": {"density": 0.4, "entities": 123},
  "retrieval_context": "Retrieved relevant context...",
  "intermediate_outputs": {
    "hyperparameters": {"chunk_size": 512, "processing_time_ms": 234.5},
    "graph_building": {"graph_description": "Complex network...", "connectivity_metrics": {}},
    "retrieval": {"retrieved_context": "Context...", "context_length": 1456},
    "answer_generation": {"generated_answer": "Answer...", "answer_length": 245},
    "evaluation": {"score": 0.85, "confidence": 0.92},
    "backward_pass": {"optimized_prompts": {}}
  }
}
```

## File Locations Summary

```
bb_graphrag/
├── prompt_response_logs/          # Communication flow logs
│   ├── prompts_responses_20250928_143022.jsonl
│   └── summary_report_20250928_143022.md
├── step_execution_logs/           # Pipeline execution logs
│   ├── step_execution_20250928_143022.jsonl
│   └── execution_report_20250928_143022.md
├── evaluation_logs/               # Performance evaluation logs
│   ├── evaluation_20250928_143022.jsonl
│   ├── evaluation_report_20250928_143022.md
│   └── evaluation_analysis_20250928_143022.json
└── [existing logs]
    ├── training_logs/             # Training progress logs
    └── results/                   # Iteration-specific results
```

## Integration with Multi-Agent System

The logging systems are integrated into the `BatchOrchestratorAgent` in `multi_agent_system.py`:

### Initialization
```python
from prompt_response_logger import get_global_prompt_logger
from step_execution_logger import get_global_step_logger, StepStatus
from evaluation_logger import get_global_evaluation_logger

step_logger = get_global_step_logger()
eval_logger = get_global_evaluation_logger()
```

### Pipeline-level logging
- Pipeline start/completion
- Total QA pairs processed
- Overall success metrics

### QA pair-level logging
- Question and reference answer capture
- Multi-iteration progression tracking
- Final performance summaries

### Iteration-level logging
- Agent step execution status
- Intermediate outputs from each agent
- ROUGE score computation and comparison
- Error recovery attempts

## Usage for Analysis

### 1. Communication Analysis
Use prompt response logs to:
- Debug agent communication issues
- Analyze prompt effectiveness
- Monitor token usage patterns
- Identify communication bottlenecks

### 2. Execution Analysis
Use step execution logs to:
- Monitor pipeline reliability
- Identify performance bottlenecks
- Track error patterns and recovery success
- Measure step execution timing

### 3. Performance Analysis
Use evaluation logs to:
- Track ROUGE score improvements
- Analyze intermediate output quality
- Compare hyperparameter effectiveness
- Generate performance reports
- Export data for statistical analysis

## Report Generation

Each logging system provides report generation methods:

```python
# Generate all reports
from prompt_response_logger import finalize_prompt_logging
from step_execution_logger import finalize_step_logging
from evaluation_logger import finalize_evaluation_logging

prompt_report = finalize_prompt_logging()
execution_report = finalize_step_logging()
evaluation_report = finalize_evaluation_logging()
```

## Best Practices

1. **Monitor all three logs** during development to get complete system visibility
2. **Use step execution logs** to identify pipeline failures quickly
3. **Use evaluation logs** for performance optimization and comparison
4. **Use prompt response logs** for debugging agent communication issues
5. **Archive logs regularly** as they can grow large during extensive testing
6. **Export evaluation data** for statistical analysis and visualization

## Integration Example

```python
# Initialize all logging systems at program start
from prompt_response_logger import initialize_prompt_logging
from step_execution_logger import initialize_step_logging
from evaluation_logger import initialize_evaluation_logging

initialize_prompt_logging()
initialize_step_logging()
initialize_evaluation_logging()

# Run your pipeline...

# Generate reports at program end
finalize_prompt_logging()
finalize_step_logging()
finalize_evaluation_logging()
```

This comprehensive logging system provides complete visibility into the Multi-Agent GraphRAG System's operation, enabling effective debugging, performance analysis, and system optimization.