"""
Example main execution script showing how to integrate All-Context system.
"""

import asyncio
from autogen_core import SingleThreadedAgentRuntime, AgentId
from autogen_dataset_agent import create_dataset_agent, DatasetStartMessage

# Import GraphRAG system agents
from multi_agent_system import create_batch_orchestrator_agent as create_graphrag_orchestrator

# Import VectorRAG system agents
from multi_agent_system_vector import (
    create_batch_orchestrator_agent as create_vectorrag_orchestrator,
    create_hyperparameters_vector_agent,
    create_vector_builder_agent,
    create_vector_retrieval_planner_agent,
    create_answer_generator_agent as create_vector_answer_generator,
    create_response_evaluator_agent as create_vector_response_evaluator,
    create_backward_pass_agent as create_vector_backward_pass,
)

# Import All-Context system agents
from multi_agent_all_context import (
    create_batch_orchestrator_agent as create_allcontext_orchestrator,
    create_answer_generator_agent,
    create_response_evaluator_agent,
    create_critique_agent,
    create_prompt_optimization_agent
)


async def main():
    """Main execution function."""

    # Choose system: "graphrag", "vectorrag", or "allcontext"
    SYSTEM = "vectorrag"  # Change this to run different systems

    # Configuration
    config = {
        "dataset": "qmsum",
        "setting": "test",
        "iterations": 2,
        "repetitions": 3,  # Number of QA pairs
        "agent_system": SYSTEM
    }

    print(f"🚀 Running {SYSTEM.upper()} evaluation...")
    print(f"📊 Config: {config}")

    # Create runtime
    runtime = SingleThreadedAgentRuntime()

    try:
        # Register DatasetAgent (always needed)
        await runtime.register("dataset_agent", create_dataset_agent)

        # Register system-specific agents
        if SYSTEM == "graphrag":
            await runtime.register("batch_orchestrator_agent", create_graphrag_orchestrator)
            # Register other GraphRAG agents...

        elif SYSTEM == "vectorrag":
            # Register all VectorRAG agents
            await runtime.register("batch_orchestrator_agent", create_vectorrag_orchestrator)
            await runtime.register("hyperparameters_vector_agent", create_hyperparameters_vector_agent)
            await runtime.register("vector_builder_agent", create_vector_builder_agent)
            await runtime.register("vector_retrieval_planner_agent", create_vector_retrieval_planner_agent)
            await runtime.register("answer_generator_vector_agent", create_vector_answer_generator)
            # Pass dataset_name to ResponseEvaluatorAgent for learned patterns loading
            # Construct dataset_name as "dataset_setting" (e.g., "qmsum_test")
            dataset_name = f"{config['dataset']}_{config['setting']}"
            await runtime.register("response_evaluator_agent",
                                  lambda: create_vector_response_evaluator(dataset_name=dataset_name))
            await runtime.register("backward_pass_vector_agent", create_vector_backward_pass)

        elif SYSTEM == "allcontext":
            # Register All-Context agents
            await runtime.register("batch_orchestrator_agent", create_allcontext_orchestrator)
            await runtime.register("answer_generator_agent", create_answer_generator_agent)
            await runtime.register("response_evaluator_agent", create_response_evaluator_agent)
            await runtime.register("critique_agent", create_critique_agent)
            await runtime.register("prompt_optimization_agent", create_prompt_optimization_agent)

        print("✅ All agents registered!")

        # Create DatasetStartMessage
        start_message = DatasetStartMessage(
            dataset=config["dataset"],
            setting=config["setting"],
            repetitions=config["repetitions"],
            iterations=config["iterations"],
            agent_type="batch_orchestrator_agent",
            agent_system=config["agent_system"]
        )

        # Send to DatasetAgent
        dataset_agent_id = AgentId("dataset_agent", "default")
        print("🔄 Starting evaluation...")

        result = await runtime.send_message(start_message, dataset_agent_id)

        print(f"✅ Evaluation completed!")
        print(f"📊 Results: {result}")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())