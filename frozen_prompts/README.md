# Frozen Prompts Directory

This directory contains frozen system prompts that can be used to fix the behavior of specific agents during training.

## Available Prompt Files

### GraphRAG System:
- `hyperparameters_graph_prompt.txt` - System prompt for the GraphRAG hyperparameters agent
- `graph_builder_prompt.txt` - System prompt for the graph builder agent
- `graph_retrieval_planner_prompt.txt` - System prompt for the graph retrieval planner agent
- `answer_generator_graph_prompt.txt` - System prompt for the GraphRAG answer generator agent

### VectorRAG System:
- `hyperparameters_vector_prompt.txt` - System prompt for the VectorRAG hyperparameters agent
- `vector_retrieval_planner_prompt.txt` - System prompt for the vector retrieval planner agent
- `answer_generator_vector_prompt.txt` - System prompt for the VectorRAG answer generator agent

## Usage

To use frozen prompts during training:

```bash
# Freeze specific prompts for GraphRAG system
python train_system.py --dataset squality --system graph --freeze-prompts hyperparameters_graph,answer_generator_graph

# Freeze prompts for VectorRAG system
python train_system.py --dataset narrativeqa --system vector --freeze-prompts hyperparameters_vector,vector_retrieval_planner

# Use custom frozen prompts directory
python train_system.py --dataset squality --system graph --freeze-prompts hyperparameters_graph --frozen-prompts-dir ./my_custom_prompts
```

## Creating Custom Prompts

1. Create a text file with the appropriate name in this directory
2. Write your system prompt content in the file
3. Use the `--freeze-prompts` argument to specify which prompts to freeze
4. The system will use your frozen prompt instead of learning/updating it during training

## Notes

- Frozen prompts are loaded at the start of training and remain constant throughout all repetitions
- If a prompt file doesn't exist, the system will use the default learned prompt behavior
- Empty prompt files are ignored
- Frozen prompts take precedence over any learned prompts from previous training runs