"""End-to-end tests using google/gemini-2.5-flash-lite.

These tests make real API calls and require OPENROUTER_API_KEY to be set.
Run with: uv run pytest e2e_tests/ -v
"""

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from mcp_minion.agent import Agent
from mcp_minion.cli import load_run_folder
from mcp_minion.logging import RunLogger
from mcp_minion.mcp_client import MCPManager
from mcp_minion.tools import create_default_registry

# Load environment variables
load_dotenv()

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)

# Path to e2e test run folders
E2E_RUNS = Path(__file__).parent / "runs"


def run_agent_from_folder(folder: Path) -> tuple:
    """Run agent from a folder and return (result, logger)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    config, mcp_servers, prompt = load_run_folder(folder)
    logger = RunLogger(log_dir=folder)
    tools = create_default_registry()
    if mcp_servers:
        return _run_with_mcp(api_key, config, mcp_servers, prompt, tools, logger)
    agent = Agent(api_key=api_key, tools=tools, config=config, logger=logger)
    result = agent.run(prompt)
    return result, logger


async def _run_with_mcp_async(api_key, config, mcp_servers, prompt, tools, logger):
    """Run agent with MCP servers."""
    mcp_manager = MCPManager(mcp_servers)
    async with mcp_manager:
        agent = Agent(
            api_key=api_key,
            tools=tools,
            config=config,
            logger=logger,
            mcp_manager=mcp_manager,
        )
        result = await agent.run_async(prompt)
    return result, logger


def _run_with_mcp(api_key, config, mcp_servers, prompt, tools, logger):
    """Sync wrapper for MCP agent run."""
    import asyncio

    return asyncio.run(
        _run_with_mcp_async(api_key, config, mcp_servers, prompt, tools, logger)
    )


class TestSimpleAdd:
    """Test simple addition using the add tool."""

    def test_simple_add(self) -> None:
        folder = E2E_RUNS / "simple_add"
        result, logger = run_agent_from_folder(folder)

        # Check that the answer contains 42 (15 + 27)
        assert "42" in result.answer

        # Check that at least one tool call was made
        assert result.tool_calls_made >= 1

        # Verify log file was created and is valid
        log_data = json.loads(logger.log_path.read_text())
        assert log_data["status"] == "completed"


class TestMultiStep:
    """Test multi-step calculation requiring multiple tool calls."""

    def test_multi_step_calculation(self) -> None:
        folder = E2E_RUNS / "multi_step"
        result, logger = run_agent_from_folder(folder)

        # (8 * 7) + (12 - 5) = 56 + 7 = 63
        assert "63" in result.answer

        # Should make multiple tool calls
        assert result.tool_calls_made >= 2

        # Verify log contains multiple steps
        log_data = json.loads(logger.log_path.read_text())
        assert log_data["status"] == "completed"

        # Count total tool calls in log
        total_tool_calls = sum(
            len(step.get("tool_calls", [])) for step in log_data["steps"]
        )
        assert total_tool_calls >= 2


class TestNoTools:
    """Test question that model can answer directly without tools."""

    def test_no_tools_needed(self) -> None:
        folder = E2E_RUNS / "no_tools"
        result, logger = run_agent_from_folder(folder)

        # Should answer with "4" (or use a tool anyway - both are acceptable)
        assert "4" in result.answer

        # Verify log
        log_data = json.loads(logger.log_path.read_text())
        assert log_data["status"] == "completed"


class TestLogIntegrity:
    """Test that logs are complete and well-formed."""

    def test_log_has_all_fields(self) -> None:
        folder = E2E_RUNS / "simple_add"
        result, logger = run_agent_from_folder(folder)

        log_data = json.loads(logger.log_path.read_text())

        # Check required top-level fields
        assert "run_id" in log_data
        assert "started_at" in log_data
        assert "completed_at" in log_data
        assert "status" in log_data
        assert "config" in log_data
        assert "prompt" in log_data
        assert "steps" in log_data
        assert "result" in log_data

        # Check config was logged
        assert "model" in log_data["config"]
        assert "max_steps" in log_data["config"]

        # Check prompt was logged
        assert len(log_data["prompt"]) > 0

        # Check steps have required fields
        for step in log_data["steps"]:
            assert "step" in step
            assert "started_at" in step


class TestMCPIPython:
    """Test MCP integration with the IPython code execution server."""

    def test_ipython_code_execution(self) -> None:
        folder = E2E_RUNS / "mcp_ipython"
        result, logger = run_agent_from_folder(folder)

        # sum(range(1, 11)) = 55
        assert "55" in result.answer

        # Should have made at least one tool call
        assert result.tool_calls_made >= 1

        # Verify log
        log_data = json.loads(logger.log_path.read_text())
        assert log_data["status"] == "completed"
