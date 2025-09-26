# Complete Testing Guide for GraphRAG/VectorRAG Systems

This guide provides detailed instructions for testing the multi-agent GraphRAG and VectorRAG systems with different configurations and frozen prompts.

## Table of Contents
1. [System Overview](#system-overview)
2. [Output Files and Locations](#output-files-and-locations)
3. [Basic Testing Scenarios](#basic-testing-scenarios)
4. [Frozen Prompts Testing](#frozen-prompts-testing)
5. [Advanced Testing Configurations](#advanced-testing-configurations)
6. [Monitoring and Analysis](#monitoring-and-analysis)
7. [Troubleshooting](#troubleshooting)

## System Overview

The system supports two types of RAG architectures:
- **GraphRAG**: Uses knowledge graphs for information retrieval
- **VectorRAG**: Uses vector embeddings and FAISS indexing

Each system has multiple agents that can be trained or frozen:

### GraphRAG Agents
1. **Hyperparameters Agent**: Determines optimal chunk size for graph construction
2. **Graph Builder Agent**: Extracts entities and relationships from text chunks
3. **Graph Retrieval Planner Agent**: Plans graph exploration strategies
4. **Answer Generator Agent**: Generates final answers from retrieved graph information

### VectorRAG Agents
1. **Hyperparameters Agent**: Determines optimal chunk size for vector indexing
2. **Vector Retrieval Planner Agent**: Refines queries for vector similarity search
3. **Answer Generator Agent**: Generates final answers from retrieved vector information

## Output Files and Locations

### Directory Structure After Training
```
bb_graphrag/
├── training_logs/                    # Main training logs
│   ├── training_graph_[dataset].log  # GraphRAG training logs
│   └── training_vector_[dataset].log # VectorRAG training logs
├── agent_logs/                       # Individual agent logs
│   ├── [agent_name]_[timestamp].log  # Per-agent detailed logs
├── agent_states/                     # Shared state storage
│   └── [dataset]_[setting]_batch_[id]_state.json
├── graphs/                           # GraphRAG outputs
│   └── [dataset]_[setting]_batch_[id]_graph.json
├── vector_indexes/                   # VectorRAG outputs
│   ├── [dataset]_[setting]_batch_[id].index
│   └── [dataset]_[setting]_batch_[id]_metadata.json
└── frozen_prompts/                   # Frozen prompt configurations
    ├── hyperparameters_graph_prompt.txt
    ├── answer_generator_vector_prompt.txt
    └── [other_prompt_files].txt
```

### Key Output Files

#### Training Logs
- **Location**: `training_logs/training_[system]_[dataset].log`
- **Content**: High-level training progress, batch completion, final metrics
- **Purpose**: Monitor overall training progress and results

#### Agent States
- **Location**: `agent_states/[dataset]_[setting]_batch_[id]_state.json`
- **Content**: Learned prompts, critiques, conversation histories, metrics
- **Purpose**: Analyze learning progress and agent behavior evolution

#### GraphRAG Specific Files
- **Graph Files**: `graphs/[dataset]_[setting]_batch_[id]_graph.json`
  - Contains extracted entities, relationships, and graph structure
- **Graph Descriptions**: Stored in agent state files
  - High-level descriptions of constructed graphs

#### VectorRAG Specific Files
- **FAISS Indexes**: `vector_indexes/[dataset]_[setting]_batch_[id].index`
  - Binary FAISS index files for similarity search
- **Metadata**: `vector_indexes/[dataset]_[setting]_batch_[id]_metadata.json`
  - Text chunks and associated metadata for vector search

## Basic Testing Scenarios

### Scenario 1: GraphRAG Baseline Training
**Purpose**: Test basic GraphRAG functionality with default learning

```bash
# Basic GraphRAG training with 3 repetitions
python train_system.py \
    --dataset squality \
    --system graph \
    --repetitions 3 \
    --setting train

# Expected outputs:
# - training_logs/training_graph_squality.log
# - graphs/squality_train_batch_[1-3]_graph.json
# - agent_states/squality_train_batch_[1-3]_state.json
```

**What to Check**:
- Training completes successfully
- 3 graph files are created
- Agent states contain learned prompts that evolve over batches
- Final metrics show performance improvement

### Scenario 2: VectorRAG Baseline Training
**Purpose**: Test basic VectorRAG functionality with default learning

```bash
# Basic VectorRAG training with 3 repetitions
python train_system.py \
    --dataset narrativeqa \
    --system vector \
    --repetitions 3 \
    --setting train

# Expected outputs:
# - training_logs/training_vector_narrativeqa.log
# - vector_indexes/narrativeqa_train_batch_[1-3].index
# - vector_indexes/narrativeqa_train_batch_[1-3]_metadata.json
# - agent_states/narrativeqa_train_batch_[1-3]_state.json
```

**What to Check**:
- FAISS indexes are created successfully
- Vector metadata contains text chunks
- Agent states show prompt evolution
- Performance metrics improve over repetitions

### Scenario 3: Console Output Only (No File Logging)
**Purpose**: Test without file logging for debugging

```bash
python train_system.py \
    --dataset hotpotqa \
    --system graph \
    --repetitions 2 \
    --no-log-file

# Expected outputs:
# - Only console output, no log files
# - Graph and state files still created
```

## Frozen Prompts Testing

### Scenario 4: Single Frozen Prompt (GraphRAG)
**Purpose**: Test freezing one specific agent prompt

```bash
# Create a frozen prompt file first
mkdir -p frozen_prompts
cat > frozen_prompts/hyperparameters_graph_prompt.txt << 'EOF'
You are an expert in determining optimal chunk sizes for GraphRAG systems.
Always consider semantic coherence and entity completeness when selecting chunk sizes.
Prioritize chunks that can capture complete entity contexts and relationships.
EOF

# Run training with frozen hyperparameters agent
python train_system.py \
    --dataset squality \
    --system graph \
    --repetitions 3 \
    --freeze-prompts hyperparameters_graph

# Expected outputs:
# - Hyperparameters agent uses fixed prompt throughout all repetitions
# - Other agents learn and evolve normally
# - Training log shows "frozen prompt" messages
```

**What to Check**:
- Console shows "Frozen prompt initialized" messages
- Agent state files show constant hyperparameters prompt
- Other agents' prompts still evolve
- Backward pass logs show "skipping update" for frozen agents

### Scenario 5: Multiple Frozen Prompts (VectorRAG)
**Purpose**: Test freezing multiple agent prompts

```bash
# Create multiple frozen prompt files
cat > frozen_prompts/hyperparameters_vector_prompt.txt << 'EOF'
You are an expert in vector RAG chunk size optimization.
Focus on semantic density and retrieval effectiveness.
EOF

cat > frozen_prompts/answer_generator_vector_prompt.txt << 'EOF'
You are an expert answer generator for vector RAG systems.
Synthesize information comprehensively and accurately.
EOF

# Run training with multiple frozen prompts
python train_system.py \
    --dataset narrativeqa \
    --system vector \
    --repetitions 3 \
    --freeze-prompts hyperparameters_vector,answer_generator_vector

# Expected outputs:
# - Two agents use fixed prompts
# - Vector retrieval planner learns normally
# - Clear logging of frozen vs learning agents
```

### Scenario 6: All Prompts Frozen (GraphRAG)
**Purpose**: Test complete prompt freezing (no learning)

```bash
# Create all GraphRAG prompt files
cat > frozen_prompts/graph_builder_prompt.txt << 'EOF'
You are an expert entity and relationship extractor.
Focus on precise entity identification and meaningful relationship detection.
EOF

cat > frozen_prompts/graph_retrieval_planner_prompt.txt << 'EOF'
You are an expert graph exploration strategist.
Use systematic exploration patterns for comprehensive information retrieval.
EOF

cat > frozen_prompts/answer_generator_graph_prompt.txt << 'EOF'
You are an expert answer generator for graph-based information.
Synthesize graph information to provide comprehensive, accurate answers.
EOF

# Run with all prompts frozen
python train_system.py \
    --dataset squality \
    --system graph \
    --repetitions 3 \
    --freeze-prompts hyperparameters_graph,graph_builder,graph_retrieval_planner,answer_generator_graph

# Expected behavior:
# - No prompt learning occurs
# - All agents use fixed behaviors
# - Useful for testing with expert-designed prompts
```

### Scenario 7: Custom Frozen Prompts Directory
**Purpose**: Test using custom prompt directory

```bash
# Create custom directory with prompts
mkdir -p my_expert_prompts
cp frozen_prompts/hyperparameters_graph_prompt.txt my_expert_prompts/

# Run with custom directory
python train_system.py \
    --dataset squality \
    --system graph \
    --repetitions 2 \
    --freeze-prompts hyperparameters_graph \
    --frozen-prompts-dir my_expert_prompts
```

## Advanced Testing Configurations

### Scenario 8: Different Datasets Comparison
**Purpose**: Compare system performance across different datasets

```bash
# Test GraphRAG on multiple datasets
for dataset in squality narrativeqa hotpotqa; do
    echo "Testing GraphRAG on $dataset"
    python train_system.py \
        --dataset $dataset \
        --system graph \
        --repetitions 2 \
        --setting train
done

# Compare outputs:
# - Check graph complexity across datasets
# - Compare final performance metrics
# - Analyze prompt evolution patterns
```

### Scenario 9: Train vs Test Setting Comparison
**Purpose**: Compare training on different data splits

```bash
# Train on train set
python train_system.py \
    --dataset squality \
    --system graph \
    --repetitions 3 \
    --setting train

# Train on test set
python train_system.py \
    --dataset squality \
    --system graph \
    --repetitions 3 \
    --setting test

# Compare agent_states files to see learning differences
```

### Scenario 10: GraphRAG vs VectorRAG Comparison
**Purpose**: Direct comparison of both systems on same dataset

```bash
# Train GraphRAG
python train_system.py \
    --dataset squality \
    --system graph \
    --repetitions 3

# Train VectorRAG
python train_system.py \
    --dataset squality \
    --system vector \
    --repetitions 3

# Compare:
# - Final performance metrics
# - Processing time and resource usage
# - Output file sizes and complexity
```

## Monitoring and Analysis

### Real-time Monitoring
1. **Console Output**: Watch for agent progression and completion messages
2. **File Creation**: Monitor output directories for new files
3. **Log Files**: Tail training logs for detailed progress
   ```bash
   tail -f training_logs/training_graph_squality.log
   ```

### Post-Training Analysis

#### 1. Agent State Analysis
```bash
# View learned prompts evolution
python -c "
import json
with open('agent_states/squality_train_batch_3_state.json', 'r') as f:
    state = json.load(f)
    for key in state:
        if 'learned_prompt' in key:
            print(f'{key}: {len(state[key])} characters')
            print(f'Content preview: {state[key][:100]}...')
            print('-' * 50)
"
```

#### 2. Performance Metrics Analysis
```bash
# Extract ROUGE scores from agent states
python -c "
import json
with open('agent_states/squality_train_batch_3_state.json', 'r') as f:
    state = json.load(f)
    rouge_scores = state.get('rouge_scores', [])
    print(f'ROUGE Scores: {rouge_scores}')
    if rouge_scores:
        print(f'Average: {sum(rouge_scores)/len(rouge_scores):.4f}')
        print(f'Best: {max(rouge_scores):.4f}')
"
```

#### 3. Graph Analysis (GraphRAG)
```bash
# Analyze generated graphs
python -c "
import json
with open('graphs/squality_train_batch_3_graph.json', 'r') as f:
    graph = json.load(f)
    nodes = [item for item in graph if item['type'] == 'node']
    edges = [item for item in graph if item['type'] == 'relationship']
    print(f'Graph Statistics:')
    print(f'  Nodes: {len(nodes)}')
    print(f'  Edges: {len(edges)}')
    print(f'  Density: {len(edges)/(len(nodes)*(len(nodes)-1)) if len(nodes) > 1 else 0:.4f}')
"
```

#### 4. Vector Index Analysis (VectorRAG)
```bash
# Analyze vector indexes
python -c "
import json
import faiss
with open('vector_indexes/narrativeqa_train_batch_3_metadata.json', 'r') as f:
    metadata = json.load(f)
index = faiss.read_index('vector_indexes/narrativeqa_train_batch_3.index')
print(f'Vector Index Statistics:')
print(f'  Total vectors: {index.ntotal}')
print(f'  Dimension: {index.d}')
print(f'  Chunks: {len(metadata)}')
print(f'  Avg chunk length: {sum(len(item[\"text\"]) for item in metadata)/len(metadata):.0f} chars')
"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Module Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'autogen_core'
# Solution: Install required dependencies
pip install autogen-core autogen-ext
```

#### 2. Memory Issues with Large Datasets
```bash
# Error: Out of memory during graph construction
# Solution: Reduce repetitions or use smaller chunk sizes
python train_system.py --dataset large_dataset --repetitions 1
```

#### 3. Frozen Prompt File Not Found
```bash
# Error: No frozen prompt found for [agent_name]
# Solution: Check file exists and has correct name
ls frozen_prompts/
cat frozen_prompts/hyperparameters_graph_prompt.txt
```

#### 4. Cross-System Prompt Warnings
```bash
# Warning: 'hyperparameters_vector' is a vector prompt but system is 'graph'
# Solution: Use correct prompt names for your system type
```

### Validation Commands

#### Test All Scenarios
```bash
#!/bin/bash
# Complete testing script
echo "=== GraphRAG Baseline ==="
python train_system.py --dataset squality --system graph --repetitions 2

echo "=== VectorRAG Baseline ==="
python train_system.py --dataset squality --system vector --repetitions 2

echo "=== GraphRAG with Frozen Prompts ==="
python train_system.py --dataset squality --system graph --repetitions 2 \
    --freeze-prompts hyperparameters_graph

echo "=== VectorRAG with Frozen Prompts ==="
python train_system.py --dataset squality --system vector --repetitions 2 \
    --freeze-prompts hyperparameters_vector,answer_generator_vector

echo "=== All tests completed ==="
```

This guide provides comprehensive testing coverage for both basic functionality and advanced frozen prompts features. Each scenario includes expected outputs and analysis methods to validate system behavior.