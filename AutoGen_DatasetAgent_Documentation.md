# AutoGen DatasetAgent Documentation

## Overview

The AutoGen DatasetAgent is a **fully compliant** implementation using the AutoGen Core API that follows the `RoutedAgent` pattern with proper message handlers, direct messaging, and AutoGen logging. It manages dataset processing, batch coordination, and evaluation metrics computation in a multi-agent GraphRAG system.

## Implementation Files

### Core Implementation
- **`autogen_dataset_agent.py`** - Main AutoGen DatasetAgent using RoutedAgent (✅ Compliant)
- **`test_autogen_dataset_agent.py`** - AutoGen runtime integration test with trace logging
- **`shared_state.py`** - Shared state management

### Dependencies
- **`datasets_schema.py`** - Pydantic schemas for dataset structure
- **`eval_functions.py`** - Evaluation functions (ROUGE, F1)
- **`autogen-core`** - AutoGen Core API package

### Removed Legacy Files
- ~~`dataset_agent.py`~~ - Replaced by AutoGen implementation
- ~~`test_dataset_agent.py`~~ - Replaced by AutoGen test

## AutoGen Architecture

### Message Types (Pydantic Models)

1. **`DatasetProcessingRequest`**
   ```python
   class DatasetProcessingRequest(BaseModel):
       dataset_name: str
       setting: str
       repetitions: int = 3
   ```

2. **`BatchStartMessage`**
   ```python
   class BatchStartMessage(BaseModel):
       batch_id: int
       repetition: int
       dataset: str
       setting: str
       shared_state: Dict[str, Any]
   ```

3. **`BatchReadyMessage`**
   ```python
   class BatchReadyMessage(BaseModel):
       batch_id: int
       repetition: int
       status: str
       metrics: Optional[Dict[str, Any]] = None
   ```

4. **`MetricsComputedMessage`**
   ```python
   class MetricsComputedMessage(BaseModel):
       batch_id: int
       repetition: int
       mean_rouge: float
       connectivity_metrics: Dict[str, float]
       continue_processing: bool
   ```

### DatasetAgent Class

```python
class DatasetAgent(RoutedAgent):
    def __init__(self, name: str, dataset_name: str, setting: str, repetitions: int = 3)

    @message_handler
    async def handle_dataset_processing_request(self, message: DatasetProcessingRequest, ctx: MessageContext)

    @message_handler
    async def handle_batch_ready_message(self, message: BatchReadyMessage, ctx: MessageContext)
```

## How It Works

### 1. Agent Registration & Runtime Setup

```python
# Create runtime
runtime = SingleThreadedAgentRuntime()

# Register agent type with factory function
await DatasetAgent.register(
    runtime,
    "dataset_agent",
    lambda: create_dataset_agent("quality", "train", repetitions=3)
)

# Start runtime
runtime.start()
```

### 2. Message Flow

1. **Initiation**: Send `DatasetProcessingRequest` to agent
2. **Batch Processing**: Agent sends `BatchStartMessage` (with complete shared state)
3. **Other Agents**: Process batch and update shared state
4. **Completion**: Other agents send `BatchReadyMessage`
5. **Metrics**: Agent computes ROUGE/graph metrics and logs results
6. **Repetition**: Repeats same batch n times before moving to next

### 3. Processing Workflow

```
DatasetProcessingRequest → handle_dataset_processing_request()
    ↓
_process_dataset() → _process_next_batch()
    ↓
generate_batch() → update shared_state → send BatchStartMessage
    ↓
[Other agents process batch]
    ↓
BatchReadyMessage → handle_batch_ready_message()
    ↓
compute_metrics() → check_repetitions() → next_batch_or_complete()
```

## Test Results

The test demonstrates successful AutoGen integration:

```
AutoGen DatasetAgent Testing Suite
==================================================

Testing Message Types
==================================================
OK DatasetProcessingRequest: dataset_name='quality' setting='train' repetitions=3
OK BatchStartMessage: batch_id=0, repetition=1
OK BatchReadyMessage: status=completed
OK MetricsComputedMessage: mean_rouge=0.75

Testing AutoGen DatasetAgent Implementation
==================================================
Created test dataset in test_quality
OK DatasetAgent registered with runtime
OK Runtime started
Sending DatasetProcessingRequest...

[Processing logs showing:]
- DatasetAgent initialized for test_quality (train)
- Found 2 questions in dataset
- Processing batch 0, repetition 1/2
- Mean ROUGE score: 0.5073
- Graph connectivity metrics: density=0.5495, clustering_coefficient=0.3621
- Moving to batch 1
- All batches completed

Checking generated files:
  State files created: 3
    - test_quality_train_batch_0_state.json (Batch ID: 0, QA pairs: 2)
    - test_quality_train_batch_1_state.json (Batch ID: 1, QA pairs: 2)
    - test_quality_train_latest_state.json
  Log files created: 1
    - test_quality_train_processing.log (45 entries, 4 processed batches)

OK Runtime closed
OK Test cleanup completed
```

## File Structure & Persistence

### State Files (`agent_states/`)
- **Individual batch states**: `{dataset}_{setting}_batch_{id}_state.json`
- **Latest state**: `{dataset}_{setting}_latest_state.json`
- **Content**: Complete shared state with batch info, metrics, graph statistics

### Log Files (`agent_logs/`)
- **Processing log**: `{dataset}_{setting}_processing.log`
- **Content**: Timestamped entries, PROCESSED tracking, metrics computation

### State Schema
```json
{
  "batch_information": {
    "batch_id": 0,
    "document_text": "...",
    "qa_pairs": [...]
  },
  "rouge_scores": [0.75, 0.68, 0.82],
  "graph_statistics": {
    "density": [0.3],
    "clustering_coefficient": [0.7],
    "avg_path_length": [2.5]
  },
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
  "_metadata": {
    "last_updated": "2025-09-22T14:53:18.501308",
    "dataset": "test_quality",
    "setting": "train",
    "batch_id": 1
  }
}
```

## Integration with Other Agents

The DatasetAgent is designed to work with:

1. **GraphBuilderAgent** - Receives BatchStart → builds knowledge graphs
2. **RetrievalPlannerAgent** - Plans retrieval strategies
3. **AnswerGenerationAgent** - Generates answers using retrieved context
4. **EvaluationAgent** - Computes evaluation metrics

### Multi-Agent Communication Pattern

```
DatasetAgent --BatchStartMessage--> GraphBuilderAgent
                                       ↓
GraphBuilderAgent --GraphReady--> RetrievalPlannerAgent
                                       ↓
RetrievalPlannerAgent --PlanReady--> AnswerGenerationAgent
                                       ↓
AnswerGenerationAgent --AnswersReady--> EvaluationAgent
                                       ↓
EvaluationAgent --BatchReadyMessage--> DatasetAgent
```

## Usage Example

```python
import asyncio
from autogen_core import AgentId, SingleThreadedAgentRuntime
from autogen_dataset_agent import DatasetAgent, DatasetProcessingRequest, create_dataset_agent

async def run_dataset_processing():
    # Setup runtime
    runtime = SingleThreadedAgentRuntime()

    # Register agent
    await DatasetAgent.register(
        runtime,
        "dataset_agent",
        lambda: create_dataset_agent("quality", "train", repetitions=3)
    )

    # Start runtime
    runtime.start()

    # Create agent ID and send request
    agent_id = AgentId("dataset_agent", "default")
    request = DatasetProcessingRequest(
        dataset_name="quality",
        setting="train",
        repetitions=3
    )

    # Send processing request
    await runtime.send_message(request, agent_id)

    # Wait for completion
    await runtime.stop_when_idle()
    await runtime.close()

# Run
asyncio.run(run_dataset_processing())
```

## Key Features

### ✅ **AutoGen Core Compliance**
- Proper `RoutedAgent` inheritance
- Pydantic message models (not dataclasses)
- `@message_handler` decorators with type hints
- Runtime registration with factory functions
- Async message handling

### ✅ **Dataset Processing**
- Batch generation from dataset files
- Shared state management with persistence
- ROUGE score computation and averaging
- Graph connectivity metrics averaging
- Repetition management (hyperparameter n)

### ✅ **Robust Error Handling**
- File I/O error handling
- JSON parsing error handling
- Missing dataset file handling
- Processing log corruption recovery

### ✅ **Production Ready**
- Comprehensive logging with timestamps
- State file persistence and recovery
- Agent lifecycle management
- Resource cleanup

## ✅ AutoGen Documentation Compliance

### Message and Communication Compliance
- **✅ Pydantic Message Types**: All messages use `pydantic.BaseModel` (not dataclasses)
- **✅ Message Handlers**: Uses `@message_handler` decorators with proper type hints
- **✅ Direct Messaging**: Uses `await self.send_message(message, agent_id)` for inter-agent communication
- **✅ RoutedAgent**: Properly inherits from `RoutedAgent` with message routing
- **✅ AutoGen Logging**: Uses `TRACE_LOGGER_NAME` for proper AutoGen logging
- **✅ Runtime Registration**: Uses factory functions with `await Agent.register()`
- **✅ Async Handlers**: All message handlers are properly async with `MessageContext`

### Test Output Showing Compliance
```
INFO:autogen_core.trace.dataset_agent:DatasetAgent initialized for test_quality (train)
INFO:autogen_core.trace.dataset_agent:Received dataset processing request for test_quality (train)
INFO:autogen_core.trace.dataset_agent:Starting dataset processing for test_quality (train)
INFO:autogen_core.trace.dataset_agent:Sending BatchStart message for batch 0
INFO:autogen_core.trace.dataset_agent:Received BatchReady for batch 0, repetition 0
INFO:autogen_core.trace.dataset_agent:Mean ROUGE score: 0.6030
```

The AutoGen DatasetAgent provides a complete, production-ready foundation for dataset processing in multi-agent GraphRAG systems using the AutoGen Core API with **full compliance** to AutoGen documentation standards.