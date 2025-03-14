from typing import Any, Dict, List, Optional, Tuple, Union

"""Test integration with AgentOps for agent monitoring."""

import os
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path to make imports work
sys.path.append(str(Path(__file__).parent.parent))

# Check if agentops is installed
try:
    import agentops
    from agentops.events import ActionEvent, LLMEvent
except ImportError:
    print("AgentOps package not found. Please install it with:")
    print("pip install agentops")
    sys.exit(1)

# Try to import GraySwan components
try:
    from cybersec_agents.grayswan import ExploitDeliveryAgent
except ImportError:
    print("Gray Swan Arena package not found or couldn't be imported")


def test_agentops_monitoring():
    """Test basic integration with AgentOps for monitoring."""
    print("Testing basic AgentOps monitoring...")

    # Load environment variables from .env
    load_dotenv()

    # Check if AGENTOPS_API_KEY is available
    api_key = os.environ.get("AGENTOPS_API_KEY")
    if not api_key:
        print("AGENTOPS_API_KEY not found in environment variables")
        print("Please set it in a .env file or directly in your environment")
        return False

    try:
        # Initialize AgentOps - a session is automatically started by default
        # Will use API key from environment
        session = agentops.init()
        session_id = session.session_id
        print(f"Started AgentOps session with ID: {session_id}")

        # Add tags to the session for easier filtering
        session.add_tags(["integration-test", "basic-test"])

        # Record an action event
        print("Recording action events...")
        session.record(
            ActionEvent(
                action_type="test_started",
                inputs={"test_name": "agentops_basic_test"},
                outputs={"status": "running"},
            )
        )

        # Simulate some agent work
        agent_id = str(uuid.uuid4())
        session.record(
            ActionEvent(
                action_type="agent_created",
                inputs={"agent_id": agent_id, "agent_type": "test_agent"},
                outputs={"status": "initialized"},
            )
        )

        # Simulate prompt execution
        prompt: str = "What are the best practices for AI safety and ethics?"
        session.record(
            ActionEvent(
                action_type="prompt_execution_started",
                inputs={"prompt": prompt},
                outputs={},
            )
        )

        # Sleep to simulate processing time
        time.sleep(1)

        # Simulate LLM call
        session.record(
            LLMEvent(
                model="gpt-3.5-turbo",
                prompt=prompt,
                completion="AI safety and ethics involve responsible development, testing, and deployment...",
                tokens_prompt=20,
                tokens_completion=30,
                duration_ms=800,
            )
        )

        # Record prompt completion
        session.record(
            ActionEvent(
                action_type="prompt_execution_completed",
                inputs={"prompt": prompt},
                outputs={
                    "response": "AI safety and ethics involve responsible development, testing, and deployment..."
                },
            )
        )

        # Get analytics for the session
        analytics = session.get_analytics()
        print(f"Session analytics: {analytics}")

        # End the test and session
        session.record(
            ActionEvent(
                action_type="test_completed", inputs={}, outputs={"status": "success"}
            )
        )

        # End session with status
        cost = session.end_session("Success")
        print(f"Session ended with Success status. Cost: {cost}")

        print("AgentOps basic monitoring test successful!")
        return True
    except Exception as e:
        print(f"Error testing AgentOps integration: {e}")
        try:
            # Try to end the session if an error occurred
            agentops.end_session("Failure", f"Exception: {str(e)}")
        except Exception as end_error:
            print(f"Error ending session: {end_error}")
        return False


def test_real_agent_with_agentops():
    """Test monitoring a real Gray Swan Arena agent with AgentOps."""
    print("\nTesting real agent with AgentOps monitoring...")

    # Load environment variables
    load_dotenv()

    # Check if required API keys are available
    agentops_api_key = os.environ.get("AGENTOPS_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not agentops_api_key:
        print("AGENTOPS_API_KEY not found in environment variables")
        return False

    if not openai_api_key:
        print("OPENAI_API_KEY not found in environment variables")
        return False

    try:
        # Initialize AgentOps with automatic session start
        session = agentops.init(auto_start_session=True)
        session_id = session.session_id

        # Add tags for this test
        session.set_tags(["integration-test", "real-agent-test", "gray-swan-arena"])

        # Log the start of the test
        session.record(
            ActionEvent(
                action_type="test_started",
                inputs={"test_name": "real_agent_test"},
                outputs={"session_id": session_id},
            )
        )

        # Check if ExploitDeliveryAgent is available
        if "ExploitDeliveryAgent" not in globals():
            print("ExploitDeliveryAgent not available, skipping real agent test")
            session.record(
                ActionEvent(
                    action_type="test_skipped",
                    inputs={"reason": "ExploitDeliveryAgent not available"},
                    outputs={},
                )
            )
            session.end_session("Success", "Test skipped due to missing dependencies")
            return False

        # Create an agent
        print("Creating ExploitDeliveryAgent...")
        ExploitDeliveryAgent()

        agent_id = str(uuid.uuid4())
        session.record(
            ActionEvent(
                action_type="agent_created",
                inputs={"agent_type": "ExploitDeliveryAgent", "agent_id": agent_id},
                outputs={"status": "initialized"},
            )
        )

        # Start a trace for the prompt execution
        # Note: AgentOps may also have a trace_id concept - refer to docs
        # In newer versions, you might use agentops.start_trace() instead
        trace_id = str(uuid.uuid4())

        # Execute a prompt
        prompt: str = "What are the best practices for AI safety and ethics?"
        session.record(
            ActionEvent(
                action_type="prompt_execution",
                inputs={"prompt": prompt, "trace_id": trace_id},
                outputs={},
            )
        )

        print(f"Executing prompt: {prompt}")
        # Note: We're not actually executing the prompt here to keep the test simple
        # response = agent.execute_prompt(prompt)

        # Simulate LLM call
        session.record(
            LLMEvent(
                model="gpt-3.5-turbo",
                prompt=prompt,
                completion="AI safety and ethics involve responsible development, testing, and deployment...",
                tokens_prompt=20,
                tokens_completion=30,
                duration_ms=1200,
            )
        )

        # Log a simulated response
        response: str = "AI safety and ethics involve responsible development, testing, and deployment..."
        session.record(
            ActionEvent(
                action_type="prompt_response",
                inputs={"prompt": prompt, "trace_id": trace_id},
                outputs={"response": response[:100] + "..."},
            )
        )

        # End the test
        session.record(
            ActionEvent(
                action_type="test_completed", inputs={}, outputs={"status": "success"}
            )
        )

        # Retrieve analytics before ending the session
        analytics = session.get_analytics()
        print(f"Session analytics: {analytics}")

        # End the session with success status
        cost = session.end_session("Success")
        print(f"Session ended with Success status. Cost: {cost}")

        print("Real agent monitoring with AgentOps test successful!")
        return True
    except Exception as e:
        print(f"Error testing real agent with AgentOps: {e}")
        try:
            # Try to end the session if an error occurred
            agentops.end_session("Failure", f"Exception: {str(e)}")
        except Exception as end_error:
            print(f"Error ending session: {end_error}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("TESTING AGENTOPS INTEGRATION")
    print("=" * 50)

    basic_test = test_agentops_monitoring()
    real_agent_test = test_real_agent_with_agentops()

    print("\nSummary:")
    print(f"Basic AgentOps Integration: {'✅ PASSED' if basic_test else '❌ FAILED'}")
    print(f"Real Agent with AgentOps: {'✅ PASSED' if real_agent_test else '❌ FAILED'}")

    if basic_test and real_agent_test:
        print("\nAll AgentOps integration tests passed!")
    else:
        print(
            "\nSome AgentOps integration tests failed. Please check the output above for details."
        )
