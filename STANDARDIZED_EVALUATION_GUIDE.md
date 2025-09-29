# Standardized Evaluation Logging System - Usage Guide

## Overview

The Standardized Evaluation Logging System provides a unified format for automatic analysis and comparison of GraphRAG, VectorRAG, and All-Context systems. Each system gets its own folder with consistent data structures for seamless automated processing.

## Features

- **System-specific folder structure**: Separate directories for GraphRAG, VectorRAG, and All-Context
- **Unified JSON schema**: Consistent format across all systems for automatic analysis
- **Comprehensive intermediate outputs tracking**: All agent outputs and processing steps
- **ROUGE score standardization**: Consistent metric calculation and formatting
- **Execution time and performance metrics**: Detailed timing and performance data
- **System-specific metrics**: Graph statistics for GraphRAG, vector statistics for VectorRAG, prompt optimization for All-Context

## Directory Structure

```
standardized_evaluation_logs/
├── graphrag/
│   └── {dataset}_{setting}/
│       ├── qa_pairs_{timestamp}.jsonl          # QA pair start/completion entries
│       ├── iterations_{timestamp}.jsonl        # Iteration evaluation entries
│       └── session_summary_{timestamp}.json    # Session summary statistics
├── vectorrag/
│   └── {dataset}_{setting}/
│       ├── qa_pairs_{timestamp}.jsonl
│       ├── iterations_{timestamp}.jsonl
│       └── session_summary_{timestamp}.json
└── allcontext/
    └── {dataset}_{setting}/
        ├── qa_pairs_{timestamp}.jsonl
        ├── iterations_{timestamp}.jsonl
        └── session_summary_{timestamp}.json
```

## Data Formats

### QA Pair Log Entries (`qa_pairs_{timestamp}.jsonl`)

#### QA Pair Start Entry
```json
{
    "session_id": "graphrag_squality_test_20250929_030657",
    "system_type": "graphrag",
    "dataset": "squality",
    "setting": "test",
    "timestamp": "2025-09-29T03:06:57.123456",
    "entry_type": "qa_pair_start",
    "qa_pair_id": "qa_1",
    "qa_pair_number": 1,
    "question": "What is the main theme of the document?",
    "question_length": 42,
    "reference_answers": ["The main theme is..."],
    "reference_answer_count": 1,
    "reference_answer_lengths": [85],
    "document_text": "Document content...",
    "document_length": 15000,
    "total_iterations_planned": 3,
    "metadata": {"batch_id": 0, "dataset": "squality", "setting": "test"}
}
```

#### QA Pair Completion Entry
```json
{
    "session_id": "graphrag_squality_test_20250929_030657",
    "system_type": "graphrag",
    "dataset": "squality",
    "setting": "test",
    "timestamp": "2025-09-29T03:08:45.654321",
    "entry_type": "qa_pair_completion",
    "qa_pair_id": "qa_1",
    "final_rouge_score": 0.75,
    "rouge_progression": [0.45, 0.62, 0.75],
    "rouge_improvement": 0.30,
    "best_iteration": 2,
    "best_rouge_score": 0.75,
    "total_iterations_completed": 3,
    "best_answer": "The main theme of the document is...",
    "best_answer_length": 156
}
```

### Iteration Log Entries (`iterations_{timestamp}.jsonl`)

```json
{
    "session_id": "graphrag_squality_test_20250929_030657",
    "system_type": "graphrag",
    "dataset": "squality",
    "setting": "test",
    "timestamp": "2025-09-29T03:07:32.987654",
    "entry_type": "iteration_evaluation",
    "qa_pair_id": "qa_1",
    "iteration": 0,
    "iteration_number": 1,
    "generated_answer": "Based on the document analysis...",
    "generated_answer_length": 245,
    "rouge_scores": {
        "rouge_1": 0.45,
        "rouge_2": 0.38,
        "rouge_l": 0.45
    },
    "execution_time_seconds": 12.34,
    "hyperparameters": {
        "chunk_size": 512,
        "graph_hyperparameters": {...}
    },
    "intermediate_outputs": {
        "hyperparameters": {...},
        "graph_building": {...},
        "retrieval": {...},
        "answer_generation": {...},
        "evaluation": {...}
    },
    "system_specific_metrics": {
        "total_nodes": 1250,
        "total_edges": 3400,
        "community_count": 45,
        "avg_node_degree": 5.44
    }
}
```

#### All-Context System Example

```json
{
    "session_id": "allcontext_squality_test_20250929_030657",
    "system_type": "allcontext",
    "dataset": "squality",
    "setting": "test",
    "timestamp": "2025-09-29T03:07:32.987654",
    "entry_type": "iteration_evaluation",
    "qa_pair_id": "qa_1",
    "iteration": 0,
    "iteration_number": 1,
    "generated_answer": "Based on the full document provided...",
    "generated_answer_length": 312,
    "rouge_scores": {
        "rouge_1": 0.58,
        "rouge_2": 0.44,
        "rouge_l": 0.58
    },
    "execution_time_seconds": 15.67,
    "hyperparameters": {
        "system_prompt": "You are an expert analyst. Focus on accuracy and completeness.",
        "document_length": 15000
    },
    "intermediate_outputs": {
        "answer_generation": {
            "system_prompt": "You are an expert analyst...",
            "system_prompt_length": 85,
            "document_length": 15000,
            "generated_answer": "Based on the full document provided...",
            "processing_time_ms": 5400
        },
        "evaluation": {
            "evaluation_feedback": "The answer demonstrates good comprehension...",
            "rouge_score": 0.58,
            "processing_time_ms": 3200
        },
        "prompt_critique": {
            "current_prompt": "You are an expert analyst...",
            "optimized_prompt": "You are a detail-oriented analyst. When answering questions...",
            "critique_feedback": "The current prompt could be more specific about...",
            "processing_time_ms": 4100
        }
    },
    "system_specific_metrics": {
        "system_prompt_length": 85,
        "document_length": 15000,
        "prompt_optimization_performed": true,
        "iteration_type": "with_prompt_optimization"
    }
}
```

### Session Summary (`session_summary_{timestamp}.json`)

```json
{
    "session_id": "graphrag_squality_test_20250929_030657",
    "system_type": "graphrag",
    "dataset": "squality",
    "setting": "test",
    "start_time": "2025-09-29T03:06:57.000000",
    "end_time": "2025-09-29T03:12:45.000000",
    "total_qa_pairs": 5,
    "total_iterations": 15,
    "avg_rouge_scores": {
        "rouge_1": 0.62,
        "rouge_2": 0.54,
        "rouge_l": 0.65
    },
    "performance_metrics": {
        "avg_execution_time": 8.7,
        "total_execution_time": 130.5,
        "avg_iterations_per_qa": 3.0
    }
}
```

## Usage Examples

### Basic Integration

All three systems (GraphRAG, VectorRAG, and All-Context) are already integrated with the standardized logger. The logger is automatically initialized when batch processing starts:

```python
from standardized_evaluation_logger import SystemType, initialize_standardized_logging

# Automatic initialization in all systems
logger = initialize_standardized_logging(
    SystemType.GRAPHRAG,  # or SystemType.VECTORRAG or SystemType.ALLCONTEXT
    dataset="squality",
    setting="test"
)
```

### Manual Usage (Advanced)

If you need to use the logger in a custom context:

```python
from standardized_evaluation_logger import StandardizedEvaluationLogger, SystemType

# Create logger instance
logger = StandardizedEvaluationLogger(
    system_type=SystemType.GRAPHRAG,
    dataset="squality",
    setting="test",
    base_dir="custom_logs"
)

# Start QA pair evaluation
logger.start_qa_pair_evaluation(
    qa_pair_id="custom_qa_1",
    question="What is the plot of the story?",
    reference_answers=["The plot involves..."],
    document_text="Story content...",
    total_iterations=2,
    metadata={"custom_field": "value"}
)

# Log iteration evaluation
logger.log_iteration_evaluation(
    qa_pair_id="custom_qa_1",
    iteration=0,
    generated_answer="Generated response...",
    rouge_scores={"rouge_1": 0.5, "rouge_2": 0.4, "rouge_l": 0.5},
    intermediate_outputs={...},
    hyperparameters={"chunk_size": 512},
    execution_time_seconds=5.2,
    system_specific_metrics={...}
)

# Complete QA pair evaluation
logger.complete_qa_pair_evaluation(
    qa_pair_id="custom_qa_1",
    final_rouge_score=0.65,
    rouge_progression=[0.5, 0.65],
    best_iteration=1,
    total_iterations_completed=2,
    best_answer="Final best answer..."
)

# Finalize session
summary_path = logger.finalize_session()
print(f"Session summary saved to: {summary_path}")
```

## Analysis Scripts

### Reading Iteration Data

```python
import json
from pathlib import Path

def analyze_system_performance(system_type, dataset, setting):
    """Analyze performance metrics for a specific system configuration."""

    base_dir = Path("standardized_evaluation_logs")
    system_dir = base_dir / system_type / f"{dataset}_{setting}"

    # Find latest iteration log
    iteration_files = list(system_dir.glob("iterations_*.jsonl"))
    if not iteration_files:
        print(f"No iteration logs found for {system_type}")
        return

    latest_file = max(iteration_files, key=lambda p: p.stat().st_mtime)

    rouge_scores = []
    execution_times = []

    with open(latest_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("entry_type") == "iteration_evaluation":
                rouge_scores.append(entry["rouge_scores"]["rouge_l"])
                if entry.get("execution_time_seconds"):
                    execution_times.append(entry["execution_time_seconds"])

    print(f"System: {system_type}")
    print(f"Average ROUGE-L: {sum(rouge_scores) / len(rouge_scores):.4f}")
    if execution_times:
        print(f"Average execution time: {sum(execution_times) / len(execution_times):.2f}s")

# Usage
analyze_system_performance("graphrag", "squality", "test")
analyze_system_performance("vectorrag", "squality", "test")
```

### Comparing Systems

```python
def compare_systems(dataset, setting):
    """Compare GraphRAG vs VectorRAG vs All-Context performance."""

    systems = ["graphrag", "vectorrag", "allcontext"]
    results = {}

    for system in systems:
        base_dir = Path("standardized_evaluation_logs")
        system_dir = base_dir / system / f"{dataset}_{setting}"

        # Read session summary
        summary_files = list(system_dir.glob("session_summary_*.json"))
        if summary_files:
            latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
            with open(latest_summary, 'r', encoding='utf-8') as f:
                results[system] = json.load(f)

    # Compare results
    if len(results) >= 2:
        print("System Comparison:")
        print("-" * 50)
        for metric in ["avg_rouge_scores", "performance_metrics"]:
            print(f"\n{metric.upper()}:")
            for system in systems:
                if system in results:
                    print(f"  {system}: {results[system].get(metric, {})}")
                else:
                    print(f"  {system}: No data available")

# Usage
compare_systems("squality", "test")
```

### QA Pair Progress Analysis

```python
def analyze_qa_pair_progression(system_type, dataset, setting):
    """Analyze ROUGE score progression for QA pairs."""

    base_dir = Path("standardized_evaluation_logs")
    system_dir = base_dir / system_type / f"{dataset}_{setting}"

    qa_files = list(system_dir.glob("qa_pairs_*.jsonl"))
    if not qa_files:
        return

    latest_file = max(qa_files, key=lambda p: p.stat().st_mtime)

    qa_progressions = {}

    with open(latest_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("entry_type") == "qa_pair_completion":
                qa_id = entry["qa_pair_id"]
                progression = entry["rouge_progression"]
                improvement = entry["rouge_improvement"]
                qa_progressions[qa_id] = {
                    "progression": progression,
                    "improvement": improvement,
                    "final_score": entry["final_rouge_score"]
                }

    # Analysis
    total_improvement = sum(data["improvement"] for data in qa_progressions.values())
    avg_final_score = sum(data["final_score"] for data in qa_progressions.values()) / len(qa_progressions)

    print(f"QA Pair Analysis for {system_type}:")
    print(f"Total QA pairs: {len(qa_progressions)}")
    print(f"Average final ROUGE score: {avg_final_score:.4f}")
    print(f"Total improvement: {total_improvement:.4f}")
    print(f"Average improvement per QA: {total_improvement / len(qa_progressions):.4f}")

# Usage
analyze_qa_pair_progression("graphrag", "squality", "test")
```

## Key Benefits

1. **Automatic Integration**: All three systems (GraphRAG, VectorRAG, All-Context) automatically use the standardized logger
2. **Consistent Format**: All data follows the same schema for easy automated analysis
3. **Comprehensive Data**: Captures all intermediate outputs, metrics, and timing information
4. **System Separation**: Clean separation between GraphRAG, VectorRAG, and All-Context data
5. **Performance Tracking**: Built-in calculation of averages and progression metrics
6. **Extensible**: Easy to add new metrics or analysis scripts

## File Locations

- **GraphRAG logs**: `standardized_evaluation_logs/graphrag/{dataset}_{setting}/`
- **VectorRAG logs**: `standardized_evaluation_logs/vectorrag/{dataset}_{setting}/`
- **All-Context logs**: `standardized_evaluation_logs/allcontext/{dataset}_{setting}/`
- **Session summaries**: Auto-generated at the end of each evaluation session

## All-Context System Features

The All-Context system has unique characteristics that are captured in the standardized logs:

- **No RAG**: Uses entire document in prompt, no retrieval step
- **Prompt Evolution**: Tracks system prompt optimization across iterations
- **Direct Context**: Full document provided to answer generation
- **Iterative Improvement**: Prompt critique and optimization for better performance

The standardized evaluation logging system provides a robust foundation for automated analysis and comparison of GraphRAG, VectorRAG, and All-Context performance across different datasets and settings.