# Frozen Prompts Feature - Usage Guide

The training script now supports freezing specific agent prompts to fixed values loaded from text files. This allows you to control which prompts are updated during training and which remain constant.

## New Command Line Arguments

### `--freeze-prompts`
Comma-separated list of prompts to freeze during training.

**GraphRAG System Options:**
- `hyperparameters_graph` - Freeze the hyperparameters selection agent
- `graph_builder` - Freeze the graph construction agent
- `graph_retrieval_planner` - Freeze the graph retrieval planning agent
- `answer_generator_graph` - Freeze the answer generation agent

**VectorRAG System Options:**
- `hyperparameters_vector` - Freeze the hyperparameters selection agent
- `vector_retrieval_planner` - Freeze the vector retrieval planning agent
- `answer_generator_vector` - Freeze the answer generation agent

### `--frozen-prompts-dir`
Directory containing the frozen prompt files (default: `frozen_prompts`)

## Usage Examples

### Example 1: Freeze GraphRAG Hyperparameters Agent
```bash
python train_system.py --dataset squality --system graph --repetitions 3 \
    --freeze-prompts hyperparameters_graph
```

### Example 2: Freeze Multiple VectorRAG Agents
```bash
python train_system.py --dataset narrativeqa --system vector --repetitions 5 \
    --freeze-prompts hyperparameters_vector,answer_generator_vector
```

### Example 3: Use Custom Frozen Prompts Directory
```bash
python train_system.py --dataset hotpotqa --system graph \
    --freeze-prompts graph_builder,answer_generator_graph \
    --frozen-prompts-dir ./my_custom_prompts
```

### Example 4: Freeze All Prompts for GraphRAG
```bash
python train_system.py --dataset squality --system graph \
    --freeze-prompts hyperparameters_graph,graph_builder,graph_retrieval_planner,answer_generator_graph
```

## Prompt File Structure

Each frozen prompt should be stored in a text file with the following naming convention:

- `hyperparameters_graph_prompt.txt` - GraphRAG hyperparameters agent
- `graph_builder_prompt.txt` - Graph builder agent
- `graph_retrieval_planner_prompt.txt` - Graph retrieval planner agent
- `answer_generator_graph_prompt.txt` - GraphRAG answer generator agent
- `hyperparameters_vector_prompt.txt` - VectorRAG hyperparameters agent
- `vector_retrieval_planner_prompt.txt` - Vector retrieval planner agent
- `answer_generator_vector_prompt.txt` - VectorRAG answer generator agent

## How It Works

1. **Initialization**: At the start of training, frozen prompts are loaded from text files and stored in the shared state
2. **Agent Behavior**: Agents use frozen prompts as their system prompts instead of starting with empty prompts
3. **Training Loop**: During the backward pass, the system skips updating any prompts marked as frozen
4. **Persistence**: Frozen prompts remain constant throughout all training repetitions

## Benefits

- **Controlled Experimentation**: Fix certain agent behaviors while allowing others to adapt
- **Incremental Improvement**: Focus training on specific components by freezing others
- **Baseline Comparison**: Compare performance with and without specific prompt optimizations
- **Expert Knowledge**: Inject domain-specific expertise through carefully crafted frozen prompts
- **Debugging**: Isolate issues by freezing known-good prompts

## Best Practices

1. **Start Simple**: Begin by freezing one prompt at a time to understand its impact
2. **Document Changes**: Keep track of which prompts are frozen in each experiment
3. **Version Control**: Store frozen prompt files in version control for reproducibility
4. **Validate Compatibility**: Ensure frozen prompts are compatible with your system type (GraphRAG vs VectorRAG)
5. **Monitor Performance**: Compare results with and without frozen prompts to measure impact

## Troubleshooting

### Warning: Cross-System Prompts
If you specify a VectorRAG prompt for a GraphRAG system (or vice versa), you'll see a warning but training will continue.

### Missing Prompt Files
If a specified frozen prompt file doesn't exist, the system will fall back to normal learning behavior for that agent.

### Empty Prompt Files
Empty prompt files are ignored and the agent will use normal learning behavior.

## Implementation Details

- Frozen prompts are initialized in the shared state before any agent processing begins
- The backward pass agents check for frozen status before updating learned prompts
- Frozen prompts take precedence over any existing learned prompts from previous training runs
- The system maintains backward compatibility - if no frozen prompts are specified, behavior is unchanged