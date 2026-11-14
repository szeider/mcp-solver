"""Unit tests for the kernel convenience (eager python_reset, kernel_id)."""

import json
from types import SimpleNamespace

from mcp_minion import RunLogger, ToolOutcome
from mcp_minion.agent import Agent, AgentConfig


def _ok(kernel_id: str) -> ToolOutcome:
    return ToolOutcome(
        text=json.dumps({"success": True, "kernel_id": kernel_id, "message": "m"})
    )


class FakeMCPManager:
    """Minimal stand-in for MCPManager that records tool calls."""

    def __init__(self, tools: set[str], reset_results: list[ToolOutcome]) -> None:
        self._tools = tools
        self._reset_results = list(reset_results)
        self.calls: list[tuple[str, dict]] = []

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tools(self):
        return []

    def get_resources(self):
        return []

    def get_instructions(self):
        return {}

    def get_openai_tools(self):
        return [
            {"type": "function", "function": {"name": n, "parameters": {}}}
            for n in sorted(self._tools)
        ]

    async def call_tool(self, name: str, arguments: dict) -> ToolOutcome:
        # Record a copy so later mutation of the dict is not observed.
        self.calls.append((name, dict(arguments)))
        if name == "python_reset":
            return self._reset_results.pop(0)
        return ToolOutcome(text=json.dumps({"success": True}))


class CountingClient:
    """Fake AsyncOpenAI client that answers 'done' and counts calls."""

    def __init__(self) -> None:
        self.count = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.count += 1
        message = SimpleNamespace(content="done", tool_calls=None)
        return SimpleNamespace(
            id="r",
            model="m",
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=None,
        )


ALL_KERNEL_TOOLS = {"python_exec", "python_reset", "python_status", "python_interrupt"}


def make_agent(mcp: FakeMCPManager, packages: list[str], logger=None) -> Agent:
    agent = Agent(
        api_key="test-key",
        tools=None,
        config=AgentConfig(packages=packages),
        mcp_manager=mcp,
        logger=logger,
    )
    agent.client = CountingClient()
    return agent


async def test_eager_reset_before_first_llm_call() -> None:
    mcp = FakeMCPManager(ALL_KERNEL_TOOLS, [_ok("k1")])
    agent = make_agent(mcp, packages=["z3-solver"])

    result = await agent.run_async("task")

    assert result.answer == "done"
    assert mcp.calls == [("python_reset", {"packages": ["z3-solver"]})]
    assert agent._kernel_id == "k1"
    assert agent.client.count == 1


async def test_kernel_setup_logged(tmp_path) -> None:
    logger = RunLogger(log_dir=tmp_path)
    mcp = FakeMCPManager(ALL_KERNEL_TOOLS, [_ok("k1")])
    await make_agent(mcp, packages=["numpy"], logger=logger).run_async("task")
    log = json.loads(logger.log_path.read_text())
    assert log["kernel_setup"]["arguments"] == {"packages": ["numpy"]}
    assert log["kernel_setup"]["is_error"] is False


async def test_failed_reset_success_false_ends_run_without_llm_call() -> None:
    failed = ToolOutcome(text=json.dumps({"success": False, "error": "uv failed"}))
    mcp = FakeMCPManager(ALL_KERNEL_TOOLS, [failed])
    agent = make_agent(mcp, packages=["no-such-pkg"])

    result = await agent.run_async("task")

    assert agent.client.count == 0
    assert result.error is not None
    assert "kernel setup failed" in result.error
    assert "no-such-pkg" in result.error and "uv failed" in result.error
    assert result.steps == []


async def test_failed_reset_is_error_ends_run(tmp_path) -> None:
    logger = RunLogger(log_dir=tmp_path)
    mcp = FakeMCPManager(
        ALL_KERNEL_TOOLS,
        [ToolOutcome(text="Tool 'python_reset' timed out", is_error=True)],
    )
    agent = make_agent(mcp, packages=["numpy"], logger=logger)

    result = await agent.run_async("task")

    assert agent.client.count == 0
    assert "timed out" in result.error
    assert json.loads(logger.log_path.read_text())["status"] == "error"


async def test_kernel_id_injected_into_all_kernel_tools() -> None:
    mcp = FakeMCPManager(ALL_KERNEL_TOOLS, [_ok("k1"), _ok("k1")])
    agent = make_agent(mcp, packages=["z3-solver"])
    await agent.run_async("task")

    for name in ("python_exec", "python_status", "python_interrupt", "python_reset"):
        await agent._execute_tool(name, {})
    assert [c[1].get("kernel_id") for c in mcp.calls[1:]] == ["k1"] * 4
    # The model's argument-less python_reset restarts the SAME kernel and
    # passes no packages (the server keeps that kernel's packages).
    assert mcp.calls[-1] == ("python_reset", {"kernel_id": "k1"})


async def test_explicit_kernel_id_not_overridden() -> None:
    mcp = FakeMCPManager(ALL_KERNEL_TOOLS, [_ok("k1")])
    agent = make_agent(mcp, packages=["z3-solver"])
    await agent.run_async("task")
    await agent._execute_tool("python_exec", {"code": "a", "kernel_id": "other"})
    assert mcp.calls[-1][1]["kernel_id"] == "other"


async def test_new_kernel_id_from_model_reset_is_tracked() -> None:
    mcp = FakeMCPManager(ALL_KERNEL_TOOLS, [_ok("k1"), _ok("k2")])
    agent = make_agent(mcp, packages=["z3-solver"])
    await agent.run_async("task")
    await agent._execute_tool("python_reset", {"kernel_id": "fresh"})
    assert agent._kernel_id == "k2"
    await agent._execute_tool("python_exec", {"code": "a"})
    assert mcp.calls[-1][1]["kernel_id"] == "k2"


async def test_failed_model_reset_keeps_tracked_kernel() -> None:
    failed = ToolOutcome(text=json.dumps({"success": False, "kernel_id": "bad"}))
    mcp = FakeMCPManager(ALL_KERNEL_TOOLS, [_ok("k1"), failed])
    agent = make_agent(mcp, packages=["z3-solver"])
    await agent.run_async("task")
    await agent._execute_tool("python_reset", {})
    assert agent._kernel_id == "k1"


async def test_no_packages_means_no_auto_reset_but_tracking() -> None:
    mcp = FakeMCPManager(ALL_KERNEL_TOOLS, [_ok("k7")])
    agent = make_agent(mcp, packages=[])
    await agent.run_async("task")
    assert mcp.calls == []

    await agent._execute_tool("python_exec", {"code": "a"})
    assert "kernel_id" not in mcp.calls[0][1]
    await agent._execute_tool("python_reset", {})
    await agent._execute_tool("python_exec", {"code": "b"})
    assert mcp.calls[-1][1]["kernel_id"] == "k7"


async def test_no_reset_tool_available() -> None:
    mcp = FakeMCPManager({"python_exec"}, [])
    agent = make_agent(mcp, packages=["z3-solver"])
    await agent.run_async("task")
    await agent._execute_tool("python_exec", {"code": "a"})
    assert [c[0] for c in mcp.calls] == ["python_exec"]
    assert agent._kernel_id is None


async def test_non_kernel_tools_get_no_kernel_id() -> None:
    mcp = FakeMCPManager(ALL_KERNEL_TOOLS | {"submit_code"}, [_ok("k1")])
    agent = make_agent(mcp, packages=["z3-solver"])
    await agent.run_async("task")
    await agent._execute_tool("submit_code", {"code": "x"})
    assert mcp.calls[-1] == ("submit_code", {"code": "x"})


# --- one Agent, several runs: no kernel leak (review R2) ---------------------


async def test_second_run_restarts_tracked_kernel(tmp_path) -> None:
    mcp = FakeMCPManager(ALL_KERNEL_TOOLS, [_ok("k1"), _ok("k1")])
    logger = RunLogger(log_dir=tmp_path)
    agent = make_agent(mcp, packages=["z3-solver"], logger=logger)

    await agent.run_async("first")
    await agent.run_async("second")

    assert mcp.calls == [
        ("python_reset", {"packages": ["z3-solver"]}),
        ("python_reset", {"packages": ["z3-solver"], "kernel_id": "k1"}),
    ]
    assert agent._kernel_id == "k1"
    setup = json.loads(logger.log_path.read_text())["kernel_setup"]
    assert setup["arguments"]["kernel_id"] == "k1"
    assert "fallback_from" not in setup


async def test_unknown_tracked_kernel_falls_back_to_new_kernel(tmp_path) -> None:
    not_found = ToolOutcome(
        text=json.dumps(
            {
                "success": False,
                "error": "Kernel k1 not found. Omit kernel_id to create a new kernel.",
            }
        )
    )
    mcp = FakeMCPManager(ALL_KERNEL_TOOLS, [_ok("k1"), not_found, _ok("k2")])
    logger = RunLogger(log_dir=tmp_path)
    agent = make_agent(mcp, packages=["numpy"], logger=logger)

    await agent.run_async("first")
    result = await agent.run_async("second")

    assert result.error is None and result.answer == "done"
    assert mcp.calls[1:] == [
        ("python_reset", {"packages": ["numpy"], "kernel_id": "k1"}),
        ("python_reset", {"packages": ["numpy"]}),
    ]
    assert agent._kernel_id == "k2"
    setup = json.loads(logger.log_path.read_text())["kernel_setup"]
    assert setup["arguments"] == {"packages": ["numpy"]}
    assert "not found" in setup["fallback_from"]


async def test_other_failure_of_tracked_reset_does_not_fall_back() -> None:
    failed = ToolOutcome(text=json.dumps({"success": False, "error": "uv failed"}))
    mcp = FakeMCPManager(ALL_KERNEL_TOOLS, [_ok("k1"), failed])
    agent = make_agent(mcp, packages=["numpy"])

    await agent.run_async("first")
    result = await agent.run_async("second")

    assert len(mcp.calls) == 2
    assert result.error is not None and "uv failed" in result.error


# --- a model reset with its own packages keeps the configured ones ----------

CONFIGURED = ["python-sat[pb]", "/path/to/checkout"]


async def _model_reset(packages_cfg, args):
    mcp = FakeMCPManager(ALL_KERNEL_TOOLS, [_ok("k1"), _ok("k1")])
    agent = make_agent(mcp, packages=packages_cfg)
    await agent.run_async("task")
    await agent._execute_tool("python_reset", args)
    return mcp.calls[-1][1], args


async def test_model_subset_list_replaced_by_configured() -> None:
    sent, recorded = await _model_reset(CONFIGURED, {"packages": ["python-sat"]})
    assert sent == {"packages": CONFIGURED, "kernel_id": "k1"}
    assert recorded["packages"] == CONFIGURED  # step record shows what was sent


async def test_model_extra_package_appended() -> None:
    sent, _ = await _model_reset(
        CONFIGURED, {"packages": ["Python_SAT>=1.8", "numpy", "/path/to/checkout"]}
    )
    assert sent["packages"] == [*CONFIGURED, "numpy"]


async def test_model_reset_without_packages_untouched() -> None:
    sent, _ = await _model_reset(CONFIGURED, {})
    assert sent == {"kernel_id": "k1"}
    sent, _ = await _model_reset(CONFIGURED, {"packages": []})
    assert sent == {"packages": [], "kernel_id": "k1"}


async def test_no_configured_packages_passes_model_list_through() -> None:
    mcp = FakeMCPManager(ALL_KERNEL_TOOLS, [_ok("k1")])
    agent = make_agent(mcp, packages=[])
    await agent.run_async("task")
    await agent._execute_tool("python_reset", {"packages": ["python-sat"]})
    assert mcp.calls == [("python_reset", {"packages": ["python-sat"]})]


def test_merge_packages_cover_rule() -> None:
    from mcp_minion.agent import merge_packages

    assert merge_packages(["cpmpy"], ["cpmpy>=1", "z3.solver"]) == [
        "cpmpy",
        "z3.solver",
    ]
    assert merge_packages(["z3-solver"], ["Z3_Solver[x]==4"]) == ["z3-solver"]
    # Paths and URLs compare as plain strings.
    assert merge_packages(["./helpers"], ["./helpers", "./other"]) == [
        "./helpers",
        "./other",
    ]
    assert merge_packages(["helpers"], ["https://x.org/helpers.whl"]) == [
        "helpers",
        "https://x.org/helpers.whl",
    ]
