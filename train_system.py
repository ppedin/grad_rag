#!/usr/bin/env python3
"""
Execution script for training GraphRAG and VectorRAG systems.

This script handles dataset training with configurable repetitions for both
GraphRAG and VectorRAG multi-agent systems with meta-learning capabilities.

Usage:
    python train_system.py --dataset <dataset_name> --repetitions <num> --system <graph|vector>

Example:
    python train_system.py --dataset squality --repetitions 3 --system graph
    python train_system.py --dataset narrativeqa --repetitions 5 --system vector --setting test
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, Dict

from logging_utils import LoggingUtils

from autogen_core import AgentId, SingleThreadedAgentRuntime, TRACE_LOGGER_NAME, EVENT_LOGGER_NAME

# Import dataset agent
from autogen_dataset_agent import DatasetAgent, DatasetProcessingRequest, create_dataset_agent

# Import system-specific modules
from multi_agent_system import (
    BatchOrchestratorAgent as GraphBatchOrchestratorAgent,
    create_batch_orchestrator_agent as create_graph_batch_orchestrator_agent,
    create_hyperparameters_graph_agent,
    create_graph_builder_agent,
    create_graph_retrieval_planner_agent,
    create_answer_generator_agent as create_graph_answer_generator_agent,
    create_response_evaluator_agent as create_graph_response_evaluator_agent,
    create_backward_pass_agent as create_graph_backward_pass_agent
)

from multi_agent_system_vector import (
    BatchOrchestratorAgent as VectorBatchOrchestratorAgent,
    create_batch_orchestrator_agent as create_vector_batch_orchestrator_agent,
    create_hyperparameters_vector_agent,
    create_vector_builder_agent,
    create_vector_retrieval_planner_agent,
    create_answer_generator_agent as create_vector_answer_generator_agent,
    create_response_evaluator_agent as create_vector_response_evaluator_agent,
    create_backward_pass_agent as create_vector_backward_pass_agent
)


def setup_logging(system_type: str, dataset_name: str, log_to_file: bool = True) -> None:
    """Setup logging configuration."""
    log_level = logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if log_to_file:
        # Create logs directory
        log_dir = Path("training_logs")
        log_dir.mkdir(exist_ok=True)

        # Setup file handlers for different loggers
        trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
        event_logger = logging.getLogger(EVENT_LOGGER_NAME)

        # Main training log
        main_log_file = log_dir / f"training_{system_type}_{dataset_name}.log"
        file_handler = logging.FileHandler(main_log_file, mode='w')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))

        # Add file handlers
        trace_logger.addHandler(file_handler)
        event_logger.addHandler(file_handler)

        # Individual agent logs
        agent_log_dir = Path("agent_logs")
        agent_log_dir.mkdir(exist_ok=True)

        print(f"Logging to {main_log_file} and individual agent logs in {agent_log_dir}")

    print(f"Logging configured for {system_type} system training on {dataset_name}")


async def setup_graph_system(runtime: SingleThreadedAgentRuntime, dataset_name: str, setting: str, repetitions: int, frozen_prompts: Dict[str, str] = None) -> AgentId:
    """Setup and register all GraphRAG system agents."""
    print("Setting up GraphRAG multi-agent system...")

    if frozen_prompts:
        print("Frozen prompts will be applied to GraphRAG agents:")
        for agent_name in frozen_prompts:
            if "graph" in agent_name or agent_name.startswith("hyperparameters_graph") or agent_name.startswith("answer_generator_graph"):
                print(f"  - {agent_name}")

    # Register DatasetAgent
    from autogen_dataset_agent import DatasetAgent
    await DatasetAgent.register(
        runtime,
        "dataset_agent",
        lambda: create_dataset_agent(dataset_name, setting, repetitions)
    )
    print("  ✓ DatasetAgent registered")

    # Register BatchOrchestratorAgent
    from multi_agent_system import BatchOrchestratorAgent, create_batch_orchestrator_agent
    await BatchOrchestratorAgent.register(
        runtime,
        "batch_orchestrator_agent",
        lambda: create_batch_orchestrator_agent()
    )
    print("  ✓ BatchOrchestratorAgent registered")

    # Register HyperparametersGraphAgent
    from multi_agent_system import HyperparametersGraphAgent, create_hyperparameters_graph_agent
    await HyperparametersGraphAgent.register(
        runtime,
        "hyperparameters_graph_agent",
        lambda: create_hyperparameters_graph_agent()
    )
    print("  ✓ HyperparametersGraphAgent registered")

    # Register GraphBuilderAgent
    from multi_agent_system import GraphBuilderAgent, create_graph_builder_agent
    await GraphBuilderAgent.register(
        runtime,
        "graph_builder_agent",
        lambda: create_graph_builder_agent()
    )
    print("  ✓ GraphBuilderAgent registered")

    # Register GraphRetrievalPlannerAgent
    from multi_agent_system import GraphRetrievalPlannerAgent, create_graph_retrieval_planner_agent
    await GraphRetrievalPlannerAgent.register(
        runtime,
        "graph_retrieval_planner_agent",
        lambda: create_graph_retrieval_planner_agent()
    )
    print("  ✓ GraphRetrievalPlannerAgent registered")

    # Register AnswerGeneratorAgent
    from multi_agent_system import AnswerGeneratorAgent, create_answer_generator_agent
    await AnswerGeneratorAgent.register(
        runtime,
        "answer_generator_agent",
        lambda: create_answer_generator_agent()
    )
    print("  ✓ AnswerGeneratorAgent registered")

    # Register ResponseEvaluatorAgent
    from multi_agent_system import ResponseEvaluatorAgent, create_response_evaluator_agent
    await ResponseEvaluatorAgent.register(
        runtime,
        "response_evaluator_agent",
        lambda: create_response_evaluator_agent()
    )
    print("  ✓ ResponseEvaluatorAgent registered")

    # Register BackwardPassAgent
    from multi_agent_system import BackwardPassAgent, create_backward_pass_agent
    await BackwardPassAgent.register(
        runtime,
        "backward_pass_agent",
        lambda: create_backward_pass_agent()
    )
    print("  ✓ BackwardPassAgent registered")

    return AgentId("dataset_agent", "default")


async def setup_vector_system(runtime: SingleThreadedAgentRuntime, dataset_name: str, setting: str, repetitions: int, frozen_prompts: Dict[str, str] = None) -> AgentId:
    """Setup and register all VectorRAG system agents."""
    print("Setting up VectorRAG multi-agent system...")

    if frozen_prompts:
        print("Frozen prompts will be applied to VectorRAG agents:")
        for agent_name in frozen_prompts:
            if "vector" in agent_name or agent_name.startswith("hyperparameters_vector") or agent_name.startswith("answer_generator_vector"):
                print(f"  - {agent_name}")

    # Register DatasetAgent
    from autogen_dataset_agent import DatasetAgent
    await DatasetAgent.register(
        runtime,
        "dataset_agent",
        lambda: create_dataset_agent(dataset_name, setting, repetitions)
    )
    print("  ✓ DatasetAgent registered")

    # Register BatchOrchestratorAgent
    from multi_agent_system_vector import BatchOrchestratorAgent, create_batch_orchestrator_agent
    await BatchOrchestratorAgent.register(
        runtime,
        "batch_orchestrator_agent",
        lambda: create_batch_orchestrator_agent()
    )
    print("  ✓ BatchOrchestratorAgent registered")

    # Register HyperparametersVectorAgent
    from multi_agent_system_vector import HyperparametersVectorAgent, create_hyperparameters_vector_agent
    await HyperparametersVectorAgent.register(
        runtime,
        "hyperparameters_vector_agent",
        lambda: create_hyperparameters_vector_agent()
    )
    print("  ✓ HyperparametersVectorAgent registered")

    # Register VectorBuilderAgent
    from multi_agent_system_vector import VectorBuilderAgent, create_vector_builder_agent
    await VectorBuilderAgent.register(
        runtime,
        "vector_builder_agent",
        lambda: create_vector_builder_agent()
    )
    print("  ✓ VectorBuilderAgent registered")

    # Register VectorRetrievalPlannerAgent
    from multi_agent_system_vector import VectorRetrievalPlannerAgent, create_vector_retrieval_planner_agent
    await VectorRetrievalPlannerAgent.register(
        runtime,
        "vector_retrieval_planner_agent",
        lambda: create_vector_retrieval_planner_agent()
    )
    print("  ✓ VectorRetrievalPlannerAgent registered")

    # Register AnswerGeneratorAgent
    from multi_agent_system_vector import AnswerGeneratorAgent, create_answer_generator_agent
    await AnswerGeneratorAgent.register(
        runtime,
        "answer_generator_agent",
        lambda: create_answer_generator_agent()
    )
    print("  ✓ AnswerGeneratorAgent registered")

    # Register ResponseEvaluatorAgent
    from multi_agent_system_vector import ResponseEvaluatorAgent, create_response_evaluator_agent
    await ResponseEvaluatorAgent.register(
        runtime,
        "response_evaluator_agent",
        lambda: create_response_evaluator_agent()
    )
    print("  ✓ ResponseEvaluatorAgent registered")

    # Register BackwardPassAgent
    from multi_agent_system_vector import BackwardPassAgent, create_backward_pass_agent
    await BackwardPassAgent.register(
        runtime,
        "backward_pass_agent",
        lambda: create_backward_pass_agent()
    )
    print("  ✓ BackwardPassAgent registered")

    return AgentId("dataset_agent", "default")


async def run_training(system_type: str, dataset_name: str, setting: str, repetitions: int, log_to_file: bool = True, frozen_prompts: Dict[str, str] = None) -> None:
    """Run the training process."""
    print(f"\nStarting {system_type.upper()}RAG Training")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Setting: {setting}")
    print(f"Repetitions: {repetitions}")
    print(f"System: {system_type}")
    if frozen_prompts:
        print(f"Frozen prompts: {len(frozen_prompts)} agents")
    print("=" * 60)

    # Setup logging
    setup_logging(system_type, dataset_name, log_to_file)

    # Create runtime
    runtime = SingleThreadedAgentRuntime()

    try:
        # Setup system-specific agents
        if system_type == "graph":
            dataset_agent_id = await setup_graph_system(runtime, dataset_name, setting, repetitions, frozen_prompts)
        elif system_type == "vector":
            dataset_agent_id = await setup_vector_system(runtime, dataset_name, setting, repetitions, frozen_prompts)
        else:
            raise ValueError(f"Unsupported system type: {system_type}")

        print(f"\nAll {system_type.upper()}RAG agents registered successfully!")

        # Start the runtime
        runtime.start()
        print("Runtime started")

        # Create processing request
        processing_request = DatasetProcessingRequest(
            dataset_name=dataset_name,
            setting=setting,
            repetitions=repetitions
        )

        print(f"\nSending processing request to DatasetAgent...")
        print(f"  Dataset: {dataset_name}")
        print(f"  Setting: {setting}")
        print(f"  Repetitions: {repetitions}")

        # Initialize frozen prompts in shared state before starting
        if frozen_prompts:
            print("Initializing frozen prompts in shared state...")
            # Initialize for batch 1 (first batch) - frozen prompts will be used for all batches
            initialize_frozen_prompts_in_shared_state(frozen_prompts, dataset_name, setting, 1)

            # Also store frozen prompts info in shared state for agents to access
            from shared_state import SharedState
            shared_state = SharedState("agent_states")
            current_state = shared_state.load_state(dataset_name, setting, 1)
            current_state["frozen_prompts"] = list(frozen_prompts.keys())
            shared_state.save_state(current_state, dataset_name, setting, 1)

        # Send processing request and wait for completion
        response = await runtime.send_message(processing_request, dataset_agent_id)

        print(f"\nTraining completed successfully!")
        print(f"Final metrics: {response.final_metrics}")
        print(f"Status: {response.status}")

        # Print training summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"System Type: {system_type.upper()}RAG")
        print(f"Dataset: {dataset_name}")
        print(f"Setting: {setting}")
        print(f"Repetitions: {repetitions}")
        print(f"Status: {response.status}")
        print(f"Metrics: {response.final_metrics}")

        if log_to_file:
            print(f"Logs saved to training_logs/training_{system_type}_{dataset_name}.log")
            print("Individual agent logs saved to agent_logs/")

        return response

    except Exception as e:
        print(f"\nERROR: Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        try:
            await runtime.stop()
            print("Runtime stopped")
        except:
            pass


def create_frozen_prompts_checker(frozen_prompts: Dict[str, str]) -> callable:
    """Create a function to check if a prompt should be frozen."""
    if not frozen_prompts:
        return lambda prompt_type: False

    # Map shared state keys back to frozen prompt names
    reverse_prompt_mapping = {
        "learned_prompt_hyperparameters_graph": "hyperparameters_graph",
        "learned_prompt_graph_builder": "graph_builder",
        "learned_prompt_graph_retrieval_planner": "graph_retrieval_planner",
        "learned_prompt_answer_generator_graph": "answer_generator_graph",
        "learned_prompt_hyperparameters_vector": "hyperparameters_vector",
        "learned_prompt_vector_retrieval_planner": "vector_retrieval_planner",
        "learned_prompt_answer_generator_vector": "answer_generator_vector"
    }

    frozen_state_keys = {reverse_prompt_mapping.get(key, key) for key in reverse_prompt_mapping if reverse_prompt_mapping[key] in frozen_prompts}

    def is_frozen(prompt_type: str) -> bool:
        """Check if a prompt type is frozen."""
        return prompt_type in frozen_state_keys

    return is_frozen


def initialize_frozen_prompts_in_shared_state(frozen_prompts: Dict[str, str], dataset_name: str, setting: str, batch_id: int) -> None:
    """Initialize frozen prompts in shared state before training starts."""
    if not frozen_prompts:
        return

    from shared_state import SharedState
    shared_state = SharedState("agent_states")

    # Load current state or create empty state
    current_state = shared_state.load_state(dataset_name, setting, batch_id)

    # Map frozen prompt names to shared state keys
    prompt_mapping = {
        "hyperparameters_graph": "learned_prompt_hyperparameters_graph",
        "graph_builder": "learned_prompt_graph_builder",
        "graph_retrieval_planner": "learned_prompt_graph_retrieval_planner",
        "answer_generator_graph": "learned_prompt_answer_generator_graph",
        "hyperparameters_vector": "learned_prompt_hyperparameters_vector",
        "vector_retrieval_planner": "learned_prompt_vector_retrieval_planner",
        "answer_generator_vector": "learned_prompt_answer_generator_vector"
    }

    # Set frozen prompts in shared state
    for prompt_name, prompt_content in frozen_prompts.items():
        if prompt_name in prompt_mapping:
            state_key = prompt_mapping[prompt_name]
            current_state[state_key] = prompt_content
            print(f"  ✓ Initialized frozen prompt for {prompt_name}")

    # Save updated state
    shared_state.save_state(current_state, dataset_name, setting, batch_id)


def load_frozen_prompts(frozen_prompts_dir: Path) -> Dict[str, str]:
    """Load frozen prompts from text files in the specified directory."""
    frozen_prompts = {}

    if not frozen_prompts_dir.exists():
        print(f"Warning: Frozen prompts directory '{frozen_prompts_dir}' does not exist")
        return frozen_prompts

    # Define mapping of agent names to their prompt files
    prompt_file_mapping = {
        # GraphRAG agents
        "hyperparameters_graph": "hyperparameters_graph_prompt.txt",
        "graph_builder": "graph_builder_prompt.txt",
        "graph_retrieval_planner": "graph_retrieval_planner_prompt.txt",
        "answer_generator_graph": "answer_generator_graph_prompt.txt",
        # VectorRAG agents
        "hyperparameters_vector": "hyperparameters_vector_prompt.txt",
        "vector_retrieval_planner": "vector_retrieval_planner_prompt.txt",
        "answer_generator_vector": "answer_generator_vector_prompt.txt"
    }

    for agent_name, filename in prompt_file_mapping.items():
        file_path = frozen_prompts_dir / filename
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    prompt_content = f.read().strip()
                    if prompt_content:
                        frozen_prompts[agent_name] = prompt_content
                        print(f"  ✓ Loaded frozen prompt for {agent_name} from {filename}")
                    else:
                        print(f"  ⚠ Empty prompt file: {filename}")
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
        else:
            print(f"  - No frozen prompt file found: {filename}")

    return frozen_prompts


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check system type
    if args.system not in ["graph", "vector"]:
        raise ValueError(f"System must be 'graph' or 'vector', got: {args.system}")

    # Check repetitions
    if args.repetitions < 1:
        raise ValueError(f"Repetitions must be >= 1, got: {args.repetitions}")

    # Check setting
    if args.setting not in ["train", "test", "validation"]:
        print(f"Warning: Setting '{args.setting}' is not standard. Standard settings are: train, test, validation")

    # Validate frozen prompts configuration
    if hasattr(args, 'freeze_prompts') and args.freeze_prompts:
        freeze_list = [p.strip() for p in args.freeze_prompts.split(',')]
        valid_graph_prompts = ["hyperparameters_graph", "graph_builder", "graph_retrieval_planner", "answer_generator_graph"]
        valid_vector_prompts = ["hyperparameters_vector", "vector_retrieval_planner", "answer_generator_vector"]
        valid_prompts = valid_graph_prompts + valid_vector_prompts

        for prompt_name in freeze_list:
            if prompt_name not in valid_prompts:
                raise ValueError(f"Invalid prompt name '{prompt_name}'. Valid options: {', '.join(valid_prompts)}")

            # Check system compatibility
            if args.system == "graph" and prompt_name in valid_vector_prompts:
                print(f"Warning: '{prompt_name}' is a vector prompt but system is 'graph'")
            elif args.system == "vector" and prompt_name in valid_graph_prompts:
                print(f"Warning: '{prompt_name}' is a graph prompt but system is 'vector'")

    print(f"Arguments validated:")
    print(f"  Dataset: {args.dataset}")
    print(f"  System: {args.system}")
    print(f"  Setting: {args.setting}")
    print(f"  Repetitions: {args.repetitions}")
    print(f"  Log to file: {not args.no_log_file}")
    if hasattr(args, 'freeze_prompts') and args.freeze_prompts:
        print(f"  Frozen prompts: {args.freeze_prompts}")
    if hasattr(args, 'frozen_prompts_dir'):
        print(f"  Frozen prompts directory: {args.frozen_prompts_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train GraphRAG or VectorRAG multi-agent systems with meta-learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train GraphRAG system on squality dataset with 3 repetitions
  python train_system.py --dataset squality --repetitions 3 --system graph

  # Train VectorRAG system on narrativeqa dataset with 5 repetitions
  python train_system.py --dataset narrativeqa --repetitions 5 --system vector

  # Train with test setting and no file logging
  python train_system.py --dataset squality --system graph --setting test --no-log-file
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to train on (e.g., 'squality', 'narrativeqa', 'hotpotqa')"
    )

    parser.add_argument(
        "--system",
        type=str,
        choices=["graph", "vector"],
        required=True,
        help="Type of RAG system to train: 'graph' for GraphRAG, 'vector' for VectorRAG"
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of training repetitions (default: 3)"
    )

    parser.add_argument(
        "--setting",
        type=str,
        default="train",
        help="Dataset setting to use (default: 'train'). Options: train, test, validation"
    )

    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable logging to files (only console output)"
    )

    parser.add_argument(
        "--freeze-prompts",
        type=str,
        default="",
        help="Comma-separated list of prompts to freeze. Options for GraphRAG: hyperparameters_graph,graph_builder,graph_retrieval_planner,answer_generator_graph. Options for VectorRAG: hyperparameters_vector,vector_retrieval_planner,answer_generator_vector"
    )

    parser.add_argument(
        "--frozen-prompts-dir",
        type=str,
        default="frozen_prompts",
        help="Directory containing frozen prompt files (default: 'frozen_prompts')"
    )

    args = parser.parse_args()

    try:
        # Validate arguments
        validate_arguments(args)

        # Load frozen prompts if specified
        frozen_prompts = {}
        if args.freeze_prompts:
            print(f"\nLoading frozen prompts from: {args.frozen_prompts_dir}")
            frozen_prompts_dir = Path(args.frozen_prompts_dir)
            all_frozen_prompts = load_frozen_prompts(frozen_prompts_dir)

            # Filter to only the prompts specified in --freeze-prompts
            freeze_list = [p.strip() for p in args.freeze_prompts.split(',')]
            for prompt_name in freeze_list:
                if prompt_name in all_frozen_prompts:
                    frozen_prompts[prompt_name] = all_frozen_prompts[prompt_name]
                else:
                    print(f"  ⚠ No frozen prompt found for {prompt_name}")

            if frozen_prompts:
                print(f"Successfully loaded {len(frozen_prompts)} frozen prompts")
            else:
                print("No frozen prompts were loaded")

        # Run training
        asyncio.run(run_training(
            system_type=args.system,
            dataset_name=args.dataset,
            setting=args.setting,
            repetitions=args.repetitions,
            log_to_file=not args.no_log_file,
            frozen_prompts=frozen_prompts
        ))

        print("\n🎉 Training completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()