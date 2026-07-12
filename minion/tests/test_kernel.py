"""Unit tests for automatic python_exec kernel setup."""

import json

from mcp_minion.agent import Agent, AgentConfig
from mcp_minion.tools import create_default_registry


class FakeMCPManager:
    """Minimal stand-in for MCPManager that records tool calls."""

    def __init__(self, tools: set[str], reset_result: str) -> None:
        self._tools = tools
        self._reset_result = reset_result
        self.calls: list[tuple[str, dict]] = []

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        # Record a copy so later mutation of the dict is not observed.
        self.calls.append((name, dict(arguments)))
        if name == "python_reset":
            return self._reset_result
        return json.dumps({"result": "ok"})


def make_agent(mcp: FakeMCPManager, packages: list[str]) -> Agent:
    return Agent(
        api_key="test-key",
        tools=create_default_registry(),
        config=AgentConfig(packages=packages),
        mcp_manager=mcp,
    )


async def test_auto_reset_before_first_exec_flat_kernel_id() -> None:
    mcp = FakeMCPManager(
        tools={"python_exec", "python_reset"},
        reset_result=json.dumps({"kernel_id": "k1"}),
    )
    agent = make_agent(mcp, packages=["z3-solver"])

    await agent._execute_tool("python_exec", {"code": "print(1)"})

    names = [c[0] for c in mcp.calls]
    assert names == ["python_reset", "python_exec"]
    # Packages forwarded to python_reset.
    assert mcp.calls[0][1] == {"packages": ["z3-solver"]}
    # kernel_id injected into the exec call.
    assert mcp.calls[1][1]["kernel_id"] == "k1"


async def test_auto_reset_double_encoded_kernel_id() -> None:
    mcp = FakeMCPManager(
        tools={"python_exec", "python_reset"},
        reset_result=json.dumps({"result": json.dumps({"kernel_id": "k9"})}),
    )
    agent = make_agent(mcp, packages=["numpy"])

    await agent._execute_tool("python_exec", {"code": "x=1"})
    assert agent._kernel_id == "k9"
    assert mcp.calls[1][1]["kernel_id"] == "k9"


async def test_reset_happens_once_and_kernel_reused() -> None:
    mcp = FakeMCPManager(
        tools={"python_exec", "python_reset"},
        reset_result=json.dumps({"kernel_id": "k1"}),
    )
    agent = make_agent(mcp, packages=["z3-solver"])

    await agent._execute_tool("python_exec", {"code": "a"})
    await agent._execute_tool("python_exec", {"code": "b"})

    names = [c[0] for c in mcp.calls]
    # Only one reset, two execs.
    assert names == ["python_reset", "python_exec", "python_exec"]
    assert mcp.calls[2][1]["kernel_id"] == "k1"


async def test_no_packages_means_no_auto_reset() -> None:
    mcp = FakeMCPManager(
        tools={"python_exec", "python_reset"},
        reset_result=json.dumps({"kernel_id": "k1"}),
    )
    agent = make_agent(mcp, packages=[])

    await agent._execute_tool("python_exec", {"code": "a"})

    names = [c[0] for c in mcp.calls]
    assert names == ["python_exec"]
    assert "kernel_id" not in mcp.calls[0][1]


async def test_explicit_reset_suppresses_auto_reset() -> None:
    mcp = FakeMCPManager(
        tools={"python_exec", "python_reset"},
        reset_result=json.dumps({"kernel_id": "explicit"}),
    )
    agent = make_agent(mcp, packages=["z3-solver"])

    # LLM calls python_reset itself first.
    await agent._execute_tool("python_reset", {"packages": ["custom"]})
    await agent._execute_tool("python_exec", {"code": "a"})

    names = [c[0] for c in mcp.calls]
    # No second (auto) reset was inserted.
    assert names == ["python_reset", "python_exec"]
    assert agent._kernel_id == "explicit"
    assert mcp.calls[1][1]["kernel_id"] == "explicit"


async def test_no_reset_tool_available() -> None:
    mcp = FakeMCPManager(
        tools={"python_exec"},
        reset_result=json.dumps({"kernel_id": "k1"}),
    )
    agent = make_agent(mcp, packages=["z3-solver"])

    await agent._execute_tool("python_exec", {"code": "a"})
    names = [c[0] for c in mcp.calls]
    assert names == ["python_exec"]
    assert agent._kernel_id is None
