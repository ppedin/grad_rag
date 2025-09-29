"""
Run All-Context system using the DatasetAgent framework.
"""

import asyncio
from autogen_core import SingleThreadedAgentRuntime
from autogen_dataset_agent import create_dataset_agent
from multi_agent_all_context import (
    create_batch_orchestrator_agent,
    create_answer_generator_agent,
    create_response_evaluator_agent,
    create_critique_agent,
    create_prompt_optimization_agent
)


async def run_all_context_evaluation():
    """Run All-Context evaluation on a dataset."""

    print("🚀 Starting All-Context Multi-Agent System...")

    # Create runtime
    runtime = SingleThreadedAgentRuntime()

    try:
        # Register All-Context agents
        await runtime.register("batch_orchestrator_agent", create_batch_orchestrator_agent)
        await runtime.register("answer_generator_agent", create_answer_generator_agent)
        await runtime.register("response_evaluator_agent", create_response_evaluator_agent)
        await runtime.register("critique_agent", create_critique_agent)
        await runtime.register("prompt_optimization_agent", create_prompt_optimization_agent)

        # Register DatasetAgent
        await runtime.register("dataset_agent", create_dataset_agent)

        print("✅ All agents registered successfully!")

        # Configuration
        config = {
            "dataset": "squality",           # Dataset to use
            "setting": "test",               # Always "test" for all systems
            "iterations": 2,                 # Number of iterations per QA pair
            "agent_system": "allcontext",    # Specify all-context system
            "repetitions": 5,                # Number of QA pairs to process
            "agent_type": "batch_orchestrator_agent"
        }

        print(f"📊 Configuration:")
        print(f"   Dataset: {config['dataset']}")
        print(f"   Setting: {config['setting']}")
        print(f"   Iterations: {config['iterations']}")
        print(f"   QA Pairs: {config['repetitions']}")
        print(f"   System: {config['agent_system']}")

        # Start evaluation
        print(f"\n🔄 Starting evaluation...")

        # The DatasetAgent will handle the execution
        # You would typically call this through the main execution script
        print("💡 To run this configuration, use:")
        print(f"   python main_execution_script.py --config '{config}'")
        print("\n   OR modify the main script to use these settings.")

        return config

    finally:
        await runtime.stop()


if __name__ == "__main__":
    config = asyncio.run(run_all_context_evaluation())
    print(f"\n📋 Use this configuration in your main execution script:")
    for key, value in config.items():
        print(f"   {key}: {value}")