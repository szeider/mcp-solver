"""Unit tests for the v4.1 toolkit MCP server (no LLM, no network).

The round-trip tests drive the real MCPServer over an in-memory client
session; the engine (ipython_mcp) is replaced by a fake MCPManager that
mimics its public contract: flat-JSON tool results (``success``/``kernel_id``
for kernel tools, ``ok`` for submit_code) returned as ``ToolOutcome`` text.
"""

import asyncio
import json
import re

import pytest
from mcp.client import Client
from mcp_minion import ToolOutcome

from mcp_solver.agent import engine as engine_mod
from mcp_solver.agent import mcp_server

BACKENDS = ("pysat", "maxsat", "z3", "cpmpy", "clingo", "didp")

ENGINE_TOOLS = {
    "python_exec",
    "python_reset",
    "python_status",
    "python_interrupt",
    "submit_code",
}


class FakeEngine:
    """Mimics mcp_minion.MCPManager connected to the ipython_mcp engine."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.kernels: set[str] = set()
        self.next_id = 0
        self.break_reset = False  # make python_reset return malformed JSON
        # python_exec codes answered with the engine's timeout reply
        self.timeout_codes: set[str] = set()
        # python_exec codes that wait for `gate` before replying
        self.gated_codes: set[str] = set()
        self.gate = asyncio.Event()
        self.events: list[tuple[str, str]] = []  # (start|end, code)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    def has_tool(self, name):
        return name in ENGINE_TOOLS

    async def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        code = arguments.get("code", name)
        self.events.append(("start", code))
        if name == "python_exec":
            if code in self.gated_codes:
                await self.gate.wait()
            await asyncio.sleep(0.01)  # let concurrent calls interleave
        self.events.append(("end", code))
        return ToolOutcome(text=json.dumps(self._respond(name, arguments)))

    def _respond(self, name, args):
        if name == "python_reset":
            if self.break_reset:
                return {}
            kid = args.get("kernel_id")
            if kid:
                if kid not in self.kernels:
                    return {"success": False, "error": f"Kernel '{kid}' not found"}
                return {"success": True, "kernel_id": kid, "message": "reset"}
            self.next_id += 1
            kid = f"k{self.next_id}"
            self.kernels.add(kid)
            return {"success": True, "kernel_id": kid, "message": "started"}
        if name == "python_exec":
            if args.get("code") in self.timeout_codes:
                return {
                    "success": False,
                    "kernel_id": args.get("kernel_id", "_default"),
                    "error": f"Execution timed out after {float(args['timeout'])}s."
                    " Code interrupted but session state preserved.",
                }
            return {
                "success": True,
                "kernel_id": args.get("kernel_id", "_default"),
                "stdout": "ran\n",
                "stderr": "",
                "result": None,
                "error": None,
            }
        if name == "submit_code":
            code = args.get("code", "")
            try:
                compile(code, "<submission>", "exec")
            except SyntaxError as e:
                return {
                    "ok": False,
                    "error": f"Syntax error at line {e.lineno}: {e.msg}",
                }
            return {"ok": True}
        return {"success": True, "kernel_id": args.get("kernel_id")}


@pytest.fixture()
def engine(monkeypatch):
    """Fresh fake engine wired into the server's lifespan + clean state."""
    import mcp_minion

    fake = FakeEngine()
    monkeypatch.setattr(mcp_minion, "MCPManager", lambda *a, **k: fake)
    monkeypatch.delenv("MCP_SOLVER_DEV", raising=False)
    monkeypatch.delenv("MCP_SOLVER_STATS", raising=False)
    monkeypatch.setattr(mcp_server, "_kernel_id", None)
    monkeypatch.setattr(mcp_server, "_episode", None)
    monkeypatch.setattr(mcp_server, "_timed_out", set())
    monkeypatch.setattr(mcp_server, "_bare_kernel", None)
    monkeypatch.setattr(mcp_server, "_exec_lock", asyncio.Lock())
    mcp_server._submissions.clear()
    return fake


def _session():
    """In-process client session (runs the server lifespan)."""
    return Client(mcp_server.mcp)


def _engine_payload(result) -> dict:
    """Decode the engine's flat JSON from a proxied tool result."""
    return json.loads(result.content[0].text)


# --- Tier 0 budget ----------------------------------------------------------


def test_select_backend_description_within_budget():
    # Claude Code truncates tool descriptions at 2KB; spec budget is 1.2KB.
    assert len(mcp_server.SELECT_BACKEND_DESCRIPTION.encode()) <= 1200


def test_select_backend_result_within_budget():
    # Claude Code (2.1.325, measured 2026-11-14) replaces an MCP tool result
    # of more than 50,000 characters by a 2 KB preview, so the host would
    # solve without the template. The result is the template plus a kernel
    # note; 500 characters cover the note.
    for solver in BACKENDS:
        assert len(mcp_server.get_template(solver)) + 500 <= 49_000, solver


EXAMPLE_RULE_MARKER = "fixes the output format"


def _template_rules_problems(text: str) -> list[str]:
    """Return the shared rules a template fails: the python_exec time limit
    (30 s default, 300 s maximum, in one sentence) and the example rule."""
    flat = " ".join(text.split())
    problems = []
    if not any(
        "python_exec" in s
        and "timeout" in s
        and re.search(r"(?<!\d)30(?!\d)", s)
        and re.search(r"(?<!\d)300(?!\d)", s)
        for s in re.split(r"(?<=\.)\s+", flat)
    ):
        problems.append("time limit")
    if EXAMPLE_RULE_MARKER not in flat:
        problems.append("example rule")
    return problems


def test_templates_share_time_limit_and_example_rule():
    # Rules that are not backend-specific must read the same in all six
    # templates: python_exec stops after 30 s by default and takes a timeout of
    # at most 300 s (mcp_server._MAX_EXEC_TIMEOUT), and an example output fixes
    # the format; its values count only if the problem says they are valid.
    assert mcp_server._MAX_EXEC_TIMEOUT == 300
    for solver in BACKENDS:
        assert _template_rules_problems(mcp_server.get_template(solver)) == [], solver


def test_guide_and_description_mention_every_backend():
    for solver in BACKENDS:
        assert solver in mcp_server.GUIDE
        assert solver in mcp_server.SELECT_BACKEND_DESCRIPTION


# --- resource functions -----------------------------------------------------


def test_template_resource_returns_backend_instructions():
    assert "z3" in mcp_server.template_resource("z3").lower()


def test_template_resource_rejects_unknown_solver():
    with pytest.raises(mcp_server.ResourceError, match="unknown solver"):
        mcp_server.template_resource("gurobi")


def test_submission_resource_rejects_unknown_id():
    with pytest.raises(mcp_server.ResourceError, match="unknown submission"):
        mcp_server.submission_resource("nope")


# --- engine envelope helpers ------------------------------------------------


def test_parse_engine_json():
    assert mcp_server._parse_engine_json('{"ok": true}') == {"ok": True}
    assert mcp_server._parse_engine_json("not json") == {}
    assert mcp_server._parse_engine_json("[1, 2]") == {}


async def test_engine_call_without_engine(monkeypatch):
    monkeypatch.setattr(mcp_server, "_engine", None)
    with pytest.raises(Exception, match="engine not connected"):
        await mcp_server._engine_call("python_exec", {"code": "1"})


async def test_engine_call_surfaces_error_outcome(monkeypatch):
    class Broken:
        async def call_tool(self, name, arguments):
            return ToolOutcome(text="Tool 'python_exec' timed out", is_error=True)

    monkeypatch.setattr(mcp_server, "_engine", Broken())
    with pytest.raises(mcp_server.ToolError, match="timed out"):
        await mcp_server._engine_call("python_exec", {"code": "1"})


async def test_engine_call_passes_text_through(monkeypatch):
    # Non-JSON text and engine-level failures (success false) are not
    # errors at this layer: the host receives the engine's text unchanged.
    class Plain:
        async def call_tool(self, name, arguments):
            return ToolOutcome(text='{"success": false, "error": "boom"}')

    monkeypatch.setattr(mcp_server, "_engine", Plain())
    text = await mcp_server._engine_call("python_exec", {"code": "1"})
    assert text == '{"success": false, "error": "boom"}'


async def test_lifespan_fails_on_engine_without_submit_code(monkeypatch):
    import mcp_minion

    class OldEngine(FakeEngine):
        def has_tool(self, name):
            return name != "submit_code"

    monkeypatch.setattr(mcp_minion, "MCPManager", lambda *a, **k: OldEngine())
    with pytest.raises(engine_mod.EngineError, match="submit_code") as info:
        async with _session():
            pass
    assert engine_mod.MIN_ENGINE_VERSION in str(info.value)


# --- full in-memory round trip ----------------------------------------------


async def test_toolkit_round_trip(engine):
    async with _session() as session:
        # Tier 0: all six tools are listed, budgets hold at protocol level.
        tools = await session.list_tools()
        names = {t.name for t in tools.tools}
        assert names == ENGINE_TOOLS | {"select_backend"}
        for t in tools.tools:
            assert len((t.description or "").encode()) <= 1200

        # Tier 1: the guide resource is readable.
        guide = await session.read_resource("mcp-solver://guide")
        assert "cpmpy" in guide.contents[0].text

        # select_backend: fresh kernel, template + note as PLAIN TEXT (the
        # template is the payload; structuredContent would hide it in
        # clients that prefer it over text).
        result = await session.call_tool("select_backend", {"solver": "z3"})
        assert not result.is_error
        assert result.structured_content is None
        text = result.content[0].text
        assert "fresh solving kernel" in text
        assert "submit_code" in text
        assert "z3" in text.lower()  # the template made it into the result
        name, args = engine.calls[0]
        assert name == "python_reset" and "kernel_id" not in args
        assert args["packages"] == ["z3-solver"]

        # Bare python_exec is routed to the solving kernel...
        result = await session.call_tool("python_exec", {"code": "print(1)"})
        assert _engine_payload(result)["kernel_id"] == "k1"
        assert engine.calls[-1][1]["kernel_id"] == "k1"
        # ...and an explicit kernel_id overrides the routing.
        await session.call_tool(
            "python_exec", {"code": "print(1)", "kernel_id": "other"}
        )
        assert engine.calls[-1][1]["kernel_id"] == "other"

        # Re-selecting recycles the same kernel (state cleared).
        result = await session.call_tool("select_backend", {"solver": "pysat"})
        text = result.content[0].text
        assert "recycled" in text
        name, args = engine.calls[-1]
        assert name == "python_reset" and args["kernel_id"] == "k1"
        assert args["packages"] == ["python-sat[pb]"]

        # A stale kernel falls back to a fresh one, and routing follows.
        engine.kernels.clear()
        result = await session.call_tool("select_backend", {"solver": "cpmpy"})
        assert "fresh solving kernel" in result.content[0].text
        name, args = engine.calls[-1]
        assert name == "python_reset" and "kernel_id" not in args
        await session.call_tool("python_exec", {"code": "print(1)"})
        assert engine.calls[-1][1]["kernel_id"] == "k2"


async def test_select_backend_rejects_unknown_solver(engine):
    async with _session() as session:
        result = await session.call_tool("select_backend", {"solver": "gurobi"})
        assert result.is_error
        assert "unknown solver" in result.content[0].text


async def test_select_backend_fails_closed_on_malformed_reset(engine):
    # An engine reply without a "success" field must surface as an error,
    # not as a phantom "fresh kernel ready" with kernel_id None.
    engine.break_reset = True
    async with _session() as session:
        result = await session.call_tool("select_backend", {"solver": "z3"})
        assert result.is_error
        assert "kernel setup" in result.content[0].text
    assert mcp_server._kernel_id is None


async def test_submit_code_round_trip(engine):
    async with _session() as session:
        # Syntax errors are rejected as an error result carrying the engine's
        # reason; no resource_link, ok=false rides along.
        result = await session.call_tool("submit_code", {"code": "def f(:"})
        assert result.is_error
        assert result.content[0].text.startswith("Syntax error at line 1")
        assert result.structured_content["ok"] is False
        assert all(c.type != "resource_link" for c in result.content)

        # A valid program is stored and linked as a fetchable resource.
        program = "print('{\"answer\": 42}')"
        result = await session.call_tool("submit_code", {"code": program})
        assert not result.is_error
        assert json.loads(result.content[0].text) == {"ok": True}
        assert result.structured_content == {"ok": True}
        links = [c for c in result.content if c.type == "resource_link"]
        assert len(links) == 1
        assert links[0].size == len(program.encode())
        fetched = await session.read_resource(links[0].uri)
        assert fetched.contents[0].text == program


async def test_submission_store_is_bounded(engine, monkeypatch):
    monkeypatch.setattr(mcp_server, "_MAX_SUBMISSIONS", 2)
    async with _session() as session:
        for i in range(3):
            result = await session.call_tool("submit_code", {"code": f"print({i})"})
            assert result.structured_content == {"ok": True}
    assert len(mcp_server._submissions) == 2
    assert list(mcp_server._submissions.values()) == ["print(1)", "print(2)"]


async def test_submit_stats_ride_along(engine):
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        await session.call_tool("python_exec", {"code": "1"})
        await session.call_tool("python_exec", {"code": "2"})
        result = await session.call_tool("submit_code", {"code": "print(1)"})
        sc = result.structured_content
        assert sc["ok"] is True
        assert sc["stats"]["exec_calls"] == 2
        assert sc["stats"]["exec_failures"] == 0
        assert sc["stats"]["submit_attempts"] == 1
        assert "wall_seconds" in sc["stats"]


async def test_exec_timeout_clamped_to_valid_range(engine):
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        await session.call_tool("python_exec", {"code": "1", "timeout": -5})
        assert engine.calls[-1][1]["timeout"] == 1.0
        await session.call_tool("python_exec", {"code": "1", "timeout": 9999})
        assert engine.calls[-1][1]["timeout"] == 300


async def test_failed_submit_attempt_survives_in_snapshot(
    engine, monkeypatch, tmp_path
):
    # A failed submission never syncs again later; the attempt must already
    # be in the .open snapshot when the host kills the server afterwards.
    stats_file = tmp_path / "stats.jsonl"
    open_file = tmp_path / "stats.jsonl.open"
    monkeypatch.setenv("MCP_SOLVER_STATS", str(stats_file))
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        await session.call_tool("submit_code", {"code": "def broken(:"})
        snap = json.loads(open_file.read_text())
        assert snap["submit_attempts"] == 1
        assert snap["submit_ok"] == 0


async def test_episode_stats_jsonl(engine, monkeypatch, tmp_path):
    stats_file = tmp_path / "stats.jsonl"
    open_file = tmp_path / "stats.jsonl.open"
    monkeypatch.setenv("MCP_SOLVER_STATS", str(stats_file))
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        await session.call_tool("python_exec", {"code": "1"})
        # The open episode is snapshotted eagerly: a host that kills the
        # server without a clean shutdown must not lose its stats.
        snap = json.loads(open_file.read_text())
        assert snap["end"] == "open" and snap["solver"] == "z3"
        assert snap["tool_calls"] == {"select_backend": 1, "python_exec": 1}
        # A second selection closes the first episode...
        await session.call_tool("select_backend", {"solver": "pysat"})
    # ...and server shutdown closes the second, removing the snapshot.
    assert not open_file.exists()
    lines = [json.loads(ln) for ln in stats_file.read_text().splitlines()]
    assert len(lines) == 2
    first, second = lines
    assert first["solver"] == "z3" and first["end"] == "superseded"
    assert first["tool_calls"] == {"select_backend": 1, "python_exec": 1}
    assert first["submit_ok"] == 0 and "wall_seconds" in first
    assert second["solver"] == "pysat" and second["end"] == "shutdown"


async def test_manual_reset_retargets_routing(engine):
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        # A manual python_reset creates a second kernel and takes over routing.
        result = await session.call_tool("python_reset", {})
        assert _engine_payload(result)["kernel_id"] == "k2"
        await session.call_tool("python_exec", {"code": "print(1)"})
        assert engine.calls[-1][1]["kernel_id"] == "k2"


# --- timeout blame: earlier code that ignores interrupts ----------------------


def _exec_codes(engine) -> list[str]:
    return [args["code"] for name, args in engine.calls if name == "python_exec"]


async def test_exec_after_timeout_reports_busy_kernel(engine):
    # The probe times out too: the earlier code still holds the kernel, so the
    # new code is not sent and the reply says so instead of blaming it.
    engine.timeout_codes = {"hog()", "pass"}
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        await session.call_tool("python_exec", {"code": "hog()", "timeout": 3})
        assert "k1" in mcp_server._timed_out
        result = await session.call_tool("python_exec", {"code": "print(1)"})
        assert not result.is_error
        payload = _engine_payload(result)
        assert payload["success"] is False and payload["kernel_id"] == "k1"
        assert "still running" in payload["error"]
        assert "NOT executed" in payload["error"]
        assert 'python_reset(kernel_id="k1")' in payload["error"]
        assert _exec_codes(engine) == ["hog()", "pass"]
        assert engine.calls[-1][1]["timeout"] == mcp_server._PROBE_TIMEOUT
        assert engine.calls[-1][1]["kernel_id"] == "k1"
        assert "k1" in mcp_server._timed_out


async def test_exec_after_timeout_runs_when_probe_succeeds(engine):
    engine.timeout_codes = {"slow()"}
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        await session.call_tool("python_exec", {"code": "slow()", "timeout": 3})
        result = await session.call_tool("python_exec", {"code": "print(1)"})
        assert _engine_payload(result)["success"] is True
        assert _exec_codes(engine) == ["slow()", "pass", "print(1)"]
        assert mcp_server._timed_out == set()


async def test_no_probe_without_previous_timeout(engine):
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        await session.call_tool("python_exec", {"code": "x = 1"})
        await session.call_tool("python_exec", {"code": "print(x)"})
        assert _exec_codes(engine) == ["x = 1", "print(x)"]


async def test_python_reset_clears_timeout_flag(engine):
    engine.timeout_codes = {"hog()", "pass"}
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        await session.call_tool("python_exec", {"code": "hog()"})
        await session.call_tool("python_reset", {"kernel_id": "k1"})
        assert mcp_server._timed_out == set()
        result = await session.call_tool("python_exec", {"code": "print(1)"})
        assert _engine_payload(result)["success"] is True
        assert _exec_codes(engine) == ["hog()", "print(1)"]


async def test_select_backend_clears_timeout_flag(engine):
    engine.timeout_codes = {"hog()", "pass"}
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        await session.call_tool("python_exec", {"code": "hog()"})
        await session.call_tool("select_backend", {"solver": "pysat"})
        assert mcp_server._timed_out == set()


async def test_concurrent_execs_do_not_interleave(engine):
    engine.timeout_codes = {"slow()"}
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        await session.call_tool("python_exec", {"code": "slow()"})
        engine.events.clear()
        await asyncio.gather(
            session.call_tool("python_exec", {"code": "a()"}),
            session.call_tool("python_exec", {"code": "b()"}),
        )
    # Each engine call ends before the next starts: probe, first code, then
    # the second call (no probe: the first cleared the flag).
    codes = [c for _, c in engine.events]
    assert [ph for ph, _ in engine.events] == ["start", "end"] * 3
    assert codes[:2] == ["pass", "pass"]
    assert {codes[2], codes[4]} == {"a()", "b()"}
    assert codes[2:4] == [codes[2]] * 2


async def test_reset_not_blocked_by_running_exec(engine):
    engine.gated_codes = {"long()"}
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        running = asyncio.create_task(
            session.call_tool("python_exec", {"code": "long()"})
        )
        while ("start", "long()") not in engine.events:
            await asyncio.sleep(0.01)
        try:
            await asyncio.wait_for(
                session.call_tool("python_reset", {"kernel_id": "k1"}), 2
            )
            await asyncio.wait_for(session.call_tool("python_interrupt", {}), 2)
            assert not running.done()
            assert ("end", "long()") not in engine.events
        finally:
            engine.gate.set()
            await running
    names = [name for name, _ in engine.calls]
    assert names[-3:] == ["python_exec", "python_reset", "python_interrupt"]


async def test_timeout_flag_shared_by_bare_and_explicit_id(engine):
    # No backend selected: a bare call runs in the engine's default kernel,
    # and a later call naming it explicitly must hit the same flag.
    engine.timeout_codes = {"hog()", "pass"}
    async with _session() as session:
        result = await session.call_tool("python_exec", {"code": "hog()"})
        kid = _engine_payload(result)["kernel_id"]
        assert kid == "_default"
        result = await session.call_tool(
            "python_exec", {"code": "print(1)", "kernel_id": kid}
        )
        assert "NOT executed" in _engine_payload(result)["error"]
        assert _exec_codes(engine) == ["hog()", "pass"]


async def test_resource_error_text_reaches_client(engine):
    # mcp>=2.1 hides the text of a plain exception raised in a resource;
    # ResourceError keeps it visible.
    async with _session() as session:
        with pytest.raises(Exception, match="unknown solver 'gurobi'"):
            await session.read_resource("mcp-solver://template/gurobi")


# --- dev mode: same error as the CLI ------------------------------------------


async def test_select_backend_dev_env_without_checkout_errors(engine, monkeypatch):
    # MCP_SOLVER_DEV=1 without a checkout is an error (formerly ignored).
    monkeypatch.setenv("MCP_SOLVER_DEV", "1")
    monkeypatch.setattr(engine_mod, "local_checkout", lambda: None)
    async with _session() as session:
        result = await session.call_tool("select_backend", {"solver": "z3"})
        assert result.is_error
        assert "no source checkout" in result.content[0].text
    assert engine.calls == []


def test_template_resource_dev_env_without_checkout_errors(monkeypatch):
    monkeypatch.setenv("MCP_SOLVER_DEV", "auto")
    monkeypatch.setattr(engine_mod, "local_checkout", lambda: None)
    with pytest.raises(mcp_server.ResourceError, match="no source checkout"):
        mcp_server.template_resource("z3")


async def test_select_backend_unpublished_version_without_checkout(engine, monkeypatch):
    # Kernels get only the solver library, so a version that is not on PyPI
    # needs no dev mode (formerly an error).
    import mcp_solver

    monkeypatch.delenv("MCP_SOLVER_DEV", raising=False)
    monkeypatch.setattr(mcp_solver, "__version__", "4.0.0a0")
    monkeypatch.setattr(engine_mod, "local_checkout", lambda: None)
    async with _session() as session:
        result = await session.call_tool("select_backend", {"solver": "z3"})
        assert not result.is_error
    assert engine.calls[0][1]["packages"] == ["z3-solver"]
