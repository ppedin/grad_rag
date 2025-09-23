#!/usr/bin/env python3
"""
Quick test to verify message serialization fixes.
"""

import asyncio
import json
import logging
from pathlib import Path
from autogen_core import AgentId, SingleThreadedAgentRuntime, TRACE_LOGGER_NAME

from autogen_dataset_agent import DatasetAgent, DatasetProcessingRequest, create_dataset_agent
from multi_agent_system import (
    BatchOrchestratorAgent, HyperparametersGraphAgent,
    create_batch_orchestrator_agent, create_hyperparameters_graph_agent
)

async def create_minimal_test_dataset():
    """Create a minimal test dataset."""
    demo_data_dir = Path("demo_squality")
    demo_data_dir.mkdir(exist_ok=True)

    demo_document = {
        "id": "test_doc",
        "text": "This is a test document for the multi-agent system. It contains some basic text to process.",
        "questions": [
            {
                "id": "test_q1",
                "question": "What is this document about?",
                "answers": ["This document is about testing the multi-agent system."],
                "metadata": {"difficulty": "easy"}
            }
        ],
        "metadata": {"source": "test", "length": "short"}
    }

    # Save test dataset in the correct path expected by DatasetAgent
    output_file = demo_data_dir / "demo_squality_train.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([demo_document], f, indent=2)

    return demo_data_dir

async def quick_test():
    """Quick test of basic message flow."""

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
    logger = logging.getLogger("QUICK_TEST")

    # Enable AutoGen tracing
    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
    trace_logger.setLevel(logging.INFO)
    trace_logger.addHandler(logging.StreamHandler())

    # Create test dataset
    test_data_dir = await create_minimal_test_dataset()

    try:
        logger.info("Creating runtime...")
        runtime = SingleThreadedAgentRuntime()

        # Register minimal agents for test
        logger.info("Registering DatasetAgent...")
        await DatasetAgent.register(
            runtime,
            "dataset_agent",
            lambda: create_dataset_agent("demo_squality", "train", repetitions=1)
        )

        logger.info("Registering BatchOrchestratorAgent...")
        await BatchOrchestratorAgent.register(
            runtime,
            "batch_orchestrator_agent",
            lambda: create_batch_orchestrator_agent()
        )

        logger.info("Registering HyperparametersGraphAgent...")
        await HyperparametersGraphAgent.register(
            runtime,
            "hyperparameters_graph_agent",
            lambda: create_hyperparameters_graph_agent()
        )

        logger.info("Starting runtime...")
        runtime.start()

        # Create a simple processing request
        logger.info("Sending processing request...")
        dataset_agent_id = AgentId("dataset_agent", "default")
        request = DatasetProcessingRequest(
            dataset_name="demo_squality",
            setting="train",
            repetitions=1
        )

        await runtime.send_message(request, dataset_agent_id)

        logger.info("Waiting for completion...")
        await runtime.stop_when_idle()
        await runtime.close()

        logger.info("✅ TEST COMPLETED SUCCESSFULLY - No serialization errors!")

    except Exception as e:
        logger.error(f"❌ TEST FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())

    finally:
        # Cleanup
        try:
            import shutil
            if test_data_dir.exists():
                shutil.rmtree(test_data_dir)
            for cleanup_dir in ["agent_states", "agent_logs"]:
                cleanup_path = Path(cleanup_dir)
                if cleanup_path.exists():
                    shutil.rmtree(cleanup_path)
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(quick_test())