"""Unit tests for the v4.1 toolkit MCP server (no LLM, no network).

The round-trip tests drive the real MCPServer over an in-memory client
session; the engine (ipython_mcp) is replaced by a fake MCPManager that
mimics its public contract (agentic-python-coder >=4 kernel-id rules):
flat-JSON tool results (``success``/``kernel_id`` for kernel tools, ``ok``
for submit_code) returned as ``ToolOutcome`` text.
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import pytest
from mcp.client import Client
from mcp_minion import ToolOutcome

from mcp_solver.agent import engine as engine_mod
from mcp_solver.agent import mcp_server
from mcp_solver.agent.cli import ProgramRun

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
        self.kernels: dict[str, list[str]] = {}  # live kernel id -> packages
        self.crashed: dict[str, list[str]] = {}  # crash records -> packages
        self.break_reset = False  # make python_reset return malformed JSON
        # a reset that asks for one of these packages fails to start
        self.unresolvable: set[str] = set()
        # canned python_exec replies by code
        self.exec_replies: dict[str, dict] = {}
        # python_exec and submit_code codes that wait for `gate` before replying
        self.gated_codes: set[str] = set()
        self.gate = asyncio.Event()
        self.events: list[tuple[str, str]] = []  # (start|end, code)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    def has_tool(self, name):
        return name in ENGINE_TOOLS

    def crash(self, kid):
        """The kernel's process died: keep a crash record (engine >=4)."""
        self.crashed[kid] = self.kernels.pop(kid)

    async def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        code = arguments.get("code", name)
        self.events.append(("start", code))
        if name in ("python_exec", "submit_code") and code in self.gated_codes:
            await self.gate.wait()
        self.events.append(("end", code))
        return ToolOutcome(text=json.dumps(self._respond(name, arguments)))

    def _respond(self, name, args):
        kid = args.get("kernel_id") or "_default"
        if name == "python_reset":
            # Engine >=4: restart a live or crashed id, else create it under
            # that id; a non-empty packages list replaces the kernel's
            # packages; a failed restart leaves a crash record with the old
            # packages.
            if self.break_reset:
                return {}
            created = kid not in self.kernels and kid not in self.crashed
            old = self.kernels.pop(kid, None) or self.crashed.pop(kid, [])
            new = list(args.get("packages") or old)
            if self.unresolvable & set(new):
                self.crashed[kid] = old
                return {
                    "success": False,
                    "kernel_id": kid,
                    "error": "Failed to start kernel: unresolvable package",
                }
            self.kernels[kid] = new
            message = "Session started."
            if created and kid != "_default":
                # rc1's sentence for a created named kernel, verbatim.
                message += (
                    f' Created kernel "{kid}". Pass kernel_id="{kid}" on every'
                    " python_exec, python_status and python_interrupt call;"
                    " calls without kernel_id use the default kernel."
                )
            return {
                "success": True,
                "kernel_id": kid,
                "message": message,
                "created": created,
            }
        if name == "python_exec":
            if kid in self.crashed:
                return {
                    "success": False,
                    "kernel_id": kid,
                    "error": f'Kernel "{kid}" crashed, its state is lost.',
                }
            if kid != "_default" and kid not in self.kernels:
                return {
                    "success": False,
                    "kernel_id": kid,
                    "error": f'Kernel "{kid}" not found.',
                }
            self.kernels.setdefault(kid, [])
            if args.get("code") in self.exec_replies:
                return self.exec_replies[args["code"]]
            return {
                "success": True,
                "kernel_id": kid,
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
        return {"success": True, "kernel_id": kid}


@pytest.fixture()
def engine(monkeypatch, tmp_path):
    """Fresh fake engine wired into the server's lifespan + clean state.

    A fake ``uv`` on PATH stands in for the standalone run of submit_code:
    it logs its arguments to $MCP_SOLVER_TEST_UV_LOG (the prefix passes the
    kernel environment's filter) and runs the program, its last argument,
    under this interpreter.
    """
    import mcp_minion

    fake = FakeEngine()
    monkeypatch.setattr(mcp_minion, "MCPManager", lambda *a, **k: fake)
    monkeypatch.delenv("MCP_SOLVER_DEV", raising=False)
    monkeypatch.delenv("MCP_SOLVER_STATS", raising=False)
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    fake_uv = bindir / "uv"
    fake_uv.write_text(
        '#!/bin/sh\necho "$@" >> "$MCP_SOLVER_TEST_UV_LOG"\n'
        f'for last; do :; done\nexec "{sys.executable}" "$last"\n'
    )
    fake_uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    fake.uv_log = tmp_path / "uv.log"
    monkeypatch.setenv("MCP_SOLVER_TEST_UV_LOG", str(fake.uv_log))
    monkeypatch.setattr(mcp_server, "_backend_packages", None)
    monkeypatch.setattr(mcp_server, "_kernel_packages", None)
    monkeypatch.setattr(mcp_server, "_episode", None)
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


def test_templates_and_guide_state_the_submit_time_limit():
    # submit_code runs the program with STANDALONE_TIMEOUT, below the 300 s
    # python_exec maximum, so every text that advises raising the python_exec
    # timeout also states the limit of the submitted program's own run.
    limit = re.compile(rf"(?<!\d){mcp_server.STANDALONE_TIMEOUT}(?!\d)")
    texts = {"guide": mcp_server.GUIDE, "submit_code": mcp_server.submit_code.__doc__}
    texts.update({solver: mcp_server.get_template(solver) for solver in BACKENDS})
    for name, text in texts.items():
        flat = " ".join(text.replace("`", "").split())
        assert any(
            "submit" in s and limit.search(s) for s in re.split(r"(?<=\.)\s+", flat)
        ), name


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


async def test_server_info_reports_package_version(engine):
    import mcp_solver

    assert mcp_solver.__version__
    async with _session() as session:
        info = session.server_info
        assert info is not None
        assert info.name == "mcp-solver"
        assert info.version == mcp_solver.__version__


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
        assert engine.calls[0] == (
            "python_reset",
            {"kernel_id": "solver", "packages": ["z3-solver"]},
        )

        # Bare python_exec is routed to the solving kernel...
        result = await session.call_tool("python_exec", {"code": "print(1)"})
        assert _engine_payload(result)["kernel_id"] == "solver"
        assert engine.calls[-1][1]["kernel_id"] == "solver"
        # ...and an explicit kernel_id is passed through unchanged.
        await session.call_tool(
            "python_exec", {"code": "print(1)", "kernel_id": "other"}
        )
        assert engine.calls[-1][1]["kernel_id"] == "other"

        # Re-selecting restarts the same kernel (state cleared), one reset.
        n = len(engine.calls)
        result = await session.call_tool("select_backend", {"solver": "pysat"})
        assert "recycled" in result.content[0].text
        assert engine.calls[n:] == [
            ("python_reset", {"kernel_id": "solver", "packages": ["python-sat[pb]"]})
        ]
        assert engine.kernels["solver"] == ["python-sat[pb]"]


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
        assert mcp_server._backend_packages is None
        assert mcp_server._episode is None
        # Nothing is selected, so calls without kernel_id are not routed.
        await session.call_tool("python_exec", {"code": "1"})
        assert "kernel_id" not in engine.calls[-1][1]


async def test_failed_backend_switch_keeps_previous_backend(engine):
    # A switch whose reset fails must not claim the new backend: the previous
    # backend's packages (kept by the engine's crash record) and episode stay.
    engine.unresolvable = {"z3-solver"}
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "pysat"})
        result = await session.call_tool("select_backend", {"solver": "z3"})
        assert result.is_error
        assert "kernel setup for 'z3' failed" in result.content[0].text
        assert mcp_server._backend_packages == ["python-sat[pb]"]
        assert mcp_server._episode["solver"] == "pysat"
        assert engine.crashed["solver"] == ["python-sat[pb]"]
        # A routed reset with packages still merges the previous backend's.
        await session.call_tool("python_reset", {"packages": ["numpy"]})
        assert engine.kernels["solver"] == ["python-sat[pb]", "numpy"]


async def test_submit_code_round_trip(engine):
    async with _session() as session:
        # Syntax errors are rejected as an error result carrying the engine's
        # reason; no resource_link, ok=false rides along.
        result = await session.call_tool("submit_code", {"code": "def f(:"})
        assert result.is_error
        assert result.content[0].text.startswith("Syntax error at line 1")
        assert result.structured_content["ok"] is False
        assert all(c.type != "resource_link" for c in result.content)

        # A valid program is stored and linked as a fetchable resource;
        # without a selected backend it is compiled, not run.
        program = "print('{\"answer\": 42}')"
        result = await session.call_tool("submit_code", {"code": program})
        assert not result.is_error
        expected = {
            "ok": True,
            "executed": False,
            "message": mcp_server.NOT_EXECUTED_MESSAGE,
        }
        assert json.loads(result.content[0].text) == expected
        assert result.structured_content == expected
        assert "not executed: no backend selected" in expected["message"]
        assert not engine.uv_log.exists()
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
            assert result.structured_content["ok"] is True
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
        assert sc["stats"]["exec_before_first_submit"] == 2
        assert "wall_seconds" in sc["stats"]


async def test_exec_before_first_submit(engine, monkeypatch, tmp_path):
    # Counts the python_exec calls before the FIRST submit_code of an
    # episode (0: submitted untested); later calls do not change it.
    stats_file = tmp_path / "stats.jsonl"
    monkeypatch.setenv("MCP_SOLVER_STATS", str(stats_file))
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        snap = json.loads((tmp_path / "stats.jsonl.open").read_text())
        assert snap["exec_before_first_submit"] is None
        result = await session.call_tool("submit_code", {"code": "print(1)"})
        assert result.structured_content["stats"]["exec_before_first_submit"] == 0
        await session.call_tool("python_exec", {"code": "1"})
        result = await session.call_tool("submit_code", {"code": "print(1)"})
        assert result.structured_content["stats"]["exec_before_first_submit"] == 0
        # A new episode starts over.
        await session.call_tool("select_backend", {"solver": "z3"})
        await session.call_tool("python_exec", {"code": "1"})
        await session.call_tool("submit_code", {"code": "print(1)"})
    rows = [json.loads(ln) for ln in stats_file.read_text().splitlines()]
    assert [r["exec_before_first_submit"] for r in rows] == [0, 1]
    assert [r["submit_attempts"] for r in rows] == [2, 1]


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


async def test_select_backend_after_crash_restarts_named_kernel(engine):
    # After a crash the engine keeps a crash record for "solver"; the same
    # single reset restarts it, the reply has created=false, and the note
    # says recycled (as in the live run against rc1).
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        engine.crash("solver")
        result = await session.call_tool("python_exec", {"code": "print(1)"})
        assert "crashed" in _engine_payload(result)["error"]
        n = len(engine.calls)
        result = await session.call_tool("select_backend", {"solver": "cpmpy"})
        assert "recycled" in result.content[0].text
        assert engine.calls[n:] == [
            ("python_reset", {"kernel_id": "solver", "packages": ["cpmpy"]})
        ]
        assert engine.kernels["solver"] == ["cpmpy"]
        assert "solver" not in engine.crashed


async def test_select_backend_after_lost_kernel_reports_fresh(engine):
    # An id without kernel and crash record is created: created=true.
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        engine.kernels.clear()
        result = await session.call_tool("select_backend", {"solver": "cpmpy"})
        assert "fresh solving kernel" in result.content[0].text


async def test_calls_without_kernel_id_before_select_backend_unrouted(engine):
    # Before any select_backend, calls without kernel_id reach the engine
    # unchanged (its default kernel).
    async with _session() as session:
        for tool, args in (
            ("python_exec", {"code": "1"}),
            ("python_status", {}),
            ("python_interrupt", {}),
            ("python_reset", {}),
        ):
            await session.call_tool(tool, args)
            assert "kernel_id" not in engine.calls[-1][1], tool
        await session.call_tool("python_reset", {"packages": ["numpy"]})
        assert engine.calls[-1][1] == {"packages": ["numpy"]}


async def test_calls_without_kernel_id_routed_after_select_backend(engine):
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        for tool, args in (
            ("python_exec", {"code": "1"}),
            ("python_status", {}),
            ("python_interrupt", {}),
            ("python_reset", {}),
            ("python_exec", {"code": "2", "kernel_id": ""}),
        ):
            await session.call_tool(tool, args)
            assert engine.calls[-1][1]["kernel_id"] == "solver", tool
        # A reset without packages forwards none: the engine keeps them.
        reset_args = [a for n, a in engine.calls if n == "python_reset"][-1]
        assert reset_args == {"kernel_id": "solver"}
        assert engine.kernels["solver"] == ["z3-solver"]
        assert set(engine.kernels) == {"solver"}  # no second kernel


async def test_explicit_kernel_id_passes_through_and_keeps_routing(engine):
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        result = await session.call_tool(
            "python_reset", {"kernel_id": "scratch", "packages": ["numpy"]}
        )
        payload = _engine_payload(result)
        assert payload["kernel_id"] == "scratch" and payload["created"] is True
        assert engine.calls[-1][1] == {"kernel_id": "scratch", "packages": ["numpy"]}
        # The engine's last sentence is true only for the engine itself; the
        # server appends where calls without kernel_id go behind it.
        assert payload["message"].startswith(
            'Session started. Created kernel "scratch".'
            ' Pass kernel_id="scratch" on every python_exec, python_status and'
            " python_interrupt call; calls without kernel_id use the default"
            " kernel. "
        )
        assert payload["message"].endswith(ROUTING_NOTE.format(kid="scratch"))
        # Also on a restart of that kernel (created false).
        result = await session.call_tool("python_reset", {"kernel_id": "scratch"})
        payload = _engine_payload(result)
        assert payload["created"] is False
        assert payload["message"].endswith(ROUTING_NOTE.format(kid="scratch"))
        for tool, args in (
            ("python_status", {"kernel_id": "scratch"}),
            ("python_interrupt", {"kernel_id": "_default"}),
        ):
            await session.call_tool(tool, args)
            assert engine.calls[-1][1]["kernel_id"] == args["kernel_id"]
        # Routing of calls without kernel_id is unchanged.
        await session.call_tool("python_exec", {"code": "print(1)"})
        assert engine.calls[-1][1]["kernel_id"] == "solver"
        assert engine.kernels["scratch"] == ["numpy"]


ROUTING_NOTE = (
    "Through mcp-solver-serve, calls without kernel_id go to the solving kernel"
    ' "solver"; pass kernel_id="{kid}" to use this kernel.'
)


async def test_explicit_reset_before_select_backend_unchanged(engine):
    async with _session() as session:
        result = await session.call_tool("python_reset", {"kernel_id": "scratch"})
        # Byte for byte what the engine sent: no routing note before a
        # backend is selected.
        message = (
            'Session started. Created kernel "scratch". Pass kernel_id="scratch"'
            " on every python_exec, python_status and python_interrupt call;"
            " calls without kernel_id use the default kernel."
        )
        assert result.content[0].text == json.dumps(
            {
                "success": True,
                "kernel_id": "scratch",
                "message": message,
                "created": True,
            }
        )


def test_routing_note_passes_other_replies_through():
    for text in ("not json", '{"success": false, "message": "x"}', '{"success": true}'):
        assert mcp_server._with_routing_note(text, "k") == text


async def test_explicit_solver_reset_keeps_backend_packages(engine):
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "pysat"})
        result = await session.call_tool(
            "python_reset", {"kernel_id": "solver", "packages": ["numpy"]}
        )
        assert engine.calls[-1][1] == {
            "kernel_id": "solver",
            "packages": ["python-sat[pb]", "numpy"],
        }
        assert engine.kernels["solver"] == ["python-sat[pb]", "numpy"]
        # The solving kernel's reply carries no routing note.
        assert "mcp-solver-serve" not in result.content[0].text


async def test_routed_reset_with_packages_keeps_backend_packages(engine):
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "pysat"})
        await session.call_tool(
            "python_reset", {"packages": ["numpy", "python-sat[pb]", "numpy"]}
        )
        assert engine.calls[-1][1] == {
            "kernel_id": "solver",
            "packages": ["python-sat[pb]", "numpy"],
        }
        assert engine.kernels["solver"] == ["python-sat[pb]", "numpy"]
        # A later routed reset with other packages replaces the host's extras,
        # never the backend's.
        await session.call_tool("python_reset", {"packages": ["networkx"]})
        assert engine.kernels["solver"] == ["python-sat[pb]", "networkx"]
        # An empty list counts as no packages.
        await session.call_tool("python_reset", {"packages": []})
        assert engine.calls[-1][1] == {"kernel_id": "solver"}
        # A backend switch installs exactly the new backend's packages.
        await session.call_tool("select_backend", {"solver": "z3"})
        assert engine.kernels["solver"] == ["z3-solver"]
        await session.call_tool("python_reset", {"packages": ["numpy"]})
        assert engine.kernels["solver"] == ["z3-solver", "numpy"]


def test_merge_packages():
    assert mcp_server._merge_packages(["a", "b"], ["b", "c", "c"]) == ["a", "b", "c"]
    assert mcp_server._merge_packages(["a"], []) == ["a"]


# --- engine replies pass through; reset is not blocked by a running exec ----


async def test_refused_exec_reply_reaches_client_unchanged(engine):
    # The engine (>=3.6.0) refuses code behind code that ignored the
    # interrupt; the reply is a normal result (is_error false) and reaches
    # the host as the engine wrote it.
    refused = {
        "success": False,
        "kernel_id": "solver",
        "error": "Kernel busy with earlier code; this code was not executed.",
        "executed": False,
        "kernel_busy": True,
        "timed_out": False,
    }
    engine.exec_replies["print(1)"] = refused
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        result = await session.call_tool("python_exec", {"code": "print(1)"})
        assert not result.is_error
        assert result.content[0].text == json.dumps(refused)


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
            await asyncio.wait_for(session.call_tool("python_reset", {}), 2)
            await asyncio.wait_for(session.call_tool("python_interrupt", {}), 2)
            assert not running.done()
            assert ("end", "long()") not in engine.events
        finally:
            engine.gate.set()
            await running
    names = [name for name, _ in engine.calls]
    assert names[-3:] == ["python_exec", "python_reset", "python_interrupt"]
    assert all(args["kernel_id"] == "solver" for _, args in engine.calls[-3:])


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


# --- submit_code: standalone run of the accepted program --------------------


def _shaped(solve_body: str = "    return {'x': 3}\n", check: str = "x == 3") -> str:
    """A program in the templates' shape: solve(), verify(), main(check).

    Its output differs between the plain run and the self-check ("checked"
    is false, resp. true), so a test can tell which one was recorded.
    """
    return (
        "import json\n"
        "def solve():\n"
        f"{solve_body}"
        "def verify(sol):\n"
        f"    x = sol['x']\n"
        f"    assert {check}, 'wrong x'\n"
        "def main(check: bool = False):\n"
        "    sol = solve()\n"
        "    if check:\n"
        "        verify(sol)\n"
        "    print(json.dumps(dict(sol, checked=check)))\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


VERIFIED = _shaped()
PLAIN_OUT = '{"x": 3, "checked": false}\n'


def _links(result):
    return [c for c in result.content if c.type == "resource_link"]


async def test_submit_runs_program_standalone(engine):
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "cpmpy"})
        result = await session.call_tool("submit_code", {"code": VERIFIED})
        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload == {
            "ok": True,
            "executed": True,
            "exit_code": 0,
            "stdout": PLAIN_OUT,
            "message": mcp_server.RUN_OK_MESSAGE,
            "self_check": {
                "status": "passed",
                "message": "Self-check passed: main(check=True) ran and returned"
                " without error.",
            },
        }
        sc = result.structured_content
        assert sc["stdout"] == PLAIN_OUT
        assert sc["stats"]["submit_attempts"] == 1
        assert sc["stats"]["self_check"] == "passed"
        assert len(_links(result)) == 1
        assert list(mcp_server._submissions.values()) == [VERIFIED]
        assert mcp_server._episode["submit_ok"] == 1
    # The plain run, then the self-check in the same kind of environment.
    plain, check = (ln.split() for ln in engine.uv_log.read_text().splitlines())
    assert plain[:4] == ["run", "--no-project", "--isolated", "--python"]
    assert plain[5:7] == ["--with", "cpmpy"] and plain[-1].endswith("submission.py")
    assert check[:-1] == plain[:-1] and check[-1].endswith("self_check.py")


async def test_submit_failing_standalone_is_rejected_not_stored(
    engine, monkeypatch, tmp_path
):
    # The engine accepts PLACEHOLDER (it compiles); run on its own it raises
    # NameError, as a program that relies on kernel variables does.
    stats_file = tmp_path / "stats.jsonl"
    monkeypatch.setenv("MCP_SOLVER_STATS", str(stats_file))
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        result = await session.call_tool("submit_code", {"code": "PLACEHOLDER"})
        assert result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["ok"] is False and payload["executed"] is True
        assert payload["exit_code"] == 1 and payload["timed_out"] is False
        assert "NameError" in payload["stderr"]
        assert payload["error"].startswith(
            "NameError: name 'PLACEHOLDER' is not defined."
        )
        assert "kernel session" in payload["error"]
        assert payload["error"].endswith(mcp_server._FIX)
        assert result.structured_content["ok"] is False
        assert _links(result) == []
        assert mcp_server._submissions == {}
        snap = json.loads((tmp_path / "stats.jsonl.open").read_text())
        assert snap["submit_attempts"] == 1 and snap["submit_ok"] == 0
        assert snap["code_bytes"] is None


async def test_submit_without_output_is_rejected(engine):
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        result = await session.call_tool("submit_code", {"code": "x = 1"})
        assert result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["exit_code"] == 0
        assert "printed nothing" in payload["error"]
        assert mcp_server._submissions == {}


@pytest.mark.skipif(os.name != "posix", reason="process groups, /bin/sh")
async def test_submit_standalone_timeout(engine, monkeypatch):
    monkeypatch.setattr(mcp_server, "STANDALONE_TIMEOUT", 3)
    # 3 s leave room for interpreter start-up under load before the print.
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        start = time.monotonic()
        result = await session.call_tool(
            "submit_code",
            {"code": "import time\nprint(1, flush=True)\ntime.sleep(600)"},
        )
        assert time.monotonic() - start < 10
        assert result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["timed_out"] is True and payload["exit_code"] is None
        assert payload["stdout"] == "1\n"
        assert "did not finish within 3 s" in payload["error"]
        assert mcp_server._submissions == {}


async def test_submit_without_uv_is_rejected_as_not_executed(
    engine, monkeypatch, tmp_path
):
    # Without uv on PATH the program never started: the rejection must not
    # claim that it was executed.
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv("PATH", str(empty))
        result = await session.call_tool("submit_code", {"code": VERIFIED})
    assert result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["ok"] is False and payload["executed"] is False
    assert payload["exit_code"] is None and payload["timed_out"] is False
    assert "uv not found on PATH" in payload["error"]
    assert mcp_server._submissions == {}


async def test_submit_output_capped_keeping_the_end(engine):
    program = "print('x' * 30000)\nprint('{\"end\": 1}')\n"
    failing = "import sys\nprint('y' * 30000)\nsys.stderr.write('e' * 5000 + 'LAST')\nsys.exit(3)\n"
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        payload = json.loads(
            (await session.call_tool("submit_code", {"code": program})).content[0].text
        )
        assert payload["ok"] is True
        assert len(payload["stdout"]) == mcp_server._STDOUT_CAP == 20_000
        assert payload["stdout"].endswith('x\n{"end": 1}\n')
        assert "stdout cut" in payload["note"]
        payload = json.loads(
            (await session.call_tool("submit_code", {"code": failing})).content[0].text
        )
        assert payload["ok"] is False and payload["exit_code"] == 3
        assert len(payload["stdout"]) == 20_000
        assert len(payload["stderr"]) == mcp_server._STDERR_CAP == 3_000
        assert payload["stderr"].endswith("eLAST")
        assert "stderr cut" in payload["note"]


async def test_submit_run_uses_packages_added_by_routed_reset(engine):
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "pysat"})
        await session.call_tool("python_reset", {"packages": ["numpy"]})
        # A failed routed reset keeps the kernel's previous packages.
        engine.unresolvable = {"bogus"}
        await session.call_tool("python_reset", {"packages": ["bogus"]})
        assert mcp_server._kernel_packages == ["python-sat[pb]", "numpy"]
        await session.call_tool("submit_code", {"code": VERIFIED})
        # A backend switch resets the list to the new backend's packages.
        await session.call_tool("select_backend", {"solver": "z3"})
        await session.call_tool("submit_code", {"code": VERIFIED})
    # Each submission: the plain run and the self-check, same packages.
    runs = engine.uv_log.read_text().splitlines()
    assert len(runs) == 4
    assert all("--with python-sat[pb] --with numpy python" in r for r in runs[:2])
    assert all("--with z3-solver python" in r for r in runs[2:])


async def test_standalone_run_does_not_block_kernel_tools(engine, tmp_path):
    # The program signals its start, then waits (at most 20 s) until the
    # test releases it, so the checks below run while it is in flight.
    marker = tmp_path / "started"
    release = tmp_path / "release"
    program = (
        "import os, time\n"
        f"open({str(marker)!r}, 'w').close()\n"
        "deadline = time.monotonic() + 20\n"
        f"while not os.path.exists({str(release)!r}) and time.monotonic() < deadline:\n"
        "    time.sleep(0.02)\n"
        "print(1)\n"
    )
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        running = asyncio.create_task(
            session.call_tool("submit_code", {"code": program})
        )
        try:
            deadline = time.monotonic() + 10
            while not marker.exists():
                assert time.monotonic() < deadline
                await asyncio.sleep(0.02)
            # The event loop serves other calls, and the kernel lock is free.
            assert not mcp_server._kernel_lock.locked()
            await asyncio.wait_for(session.call_tool("python_status", {}), 2)
            await asyncio.wait_for(session.call_tool("python_reset", {}), 2)
            assert not running.done()
        finally:
            release.touch()
            result = await running
        assert not result.is_error
        assert result.structured_content["stdout"] == "1\n"


# --- submit_code: the self-check main(check=True) is information only -------

INFO = mcp_server._SELF_CHECK_INFO


def _self_check(result) -> dict:
    return json.loads(result.content[0].text)["self_check"]


def _not_available(what: str) -> dict:
    return {
        "status": "not_available",
        "message": "Self-check not available: " + what + INFO,
    }


async def test_self_check_failure_is_reported_not_rejected(
    engine, monkeypatch, tmp_path
):
    # The plain run prints a solution; verify() rejects it. The submission
    # is accepted and stored, the failure reported with the assertion line.
    stats_file = tmp_path / "stats.jsonl"
    monkeypatch.setenv("MCP_SOLVER_STATS", str(stats_file))
    code = _shaped(check="x == 4")
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        result = await session.call_tool("submit_code", {"code": code})
        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["ok"] is True and payload["stdout"] == PLAIN_OUT
        check = payload["self_check"]
        assert check["status"] == "failed"
        assert check["message"] == (
            "Self-check failed: main(check=True) raised AssertionError: wrong x."
            " Either the solution or verify() is wrong: check both against the"
            " problem statement, and if the solution is wrong, fix the program"
            " and submit it again." + INFO
        )
        assert result.structured_content["stats"]["self_check"] == "failed"
        assert len(_links(result)) == 1
        assert list(mcp_server._submissions.values()) == [code]
        snap = json.loads((tmp_path / "stats.jsonl.open").read_text())
        assert snap["submit_ok"] == 1 and snap["self_check"] == "failed"
    row = json.loads(stats_file.read_text())
    assert row["self_check"] == "failed"


async def test_self_check_not_available_without_main_accepting_check(engine):
    no_main = "print('{\"x\": 3}')\n"
    main_without_check = (
        "def main():\n    print('{\"x\": 3}')\nif __name__ == '__main__':\n    main()\n"
    )
    expected = _not_available("the program has no main() that accepts check=True.")
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        for code in (no_main, main_without_check):
            result = await session.call_tool("submit_code", {"code": code})
            assert not result.is_error
            assert _self_check(result) == expected
            assert result.structured_content["stats"]["self_check"] == "not_available"
    assert len(mcp_server._submissions) == 2


async def test_exit_during_import_is_not_a_passed_check(engine):
    # Exit code 0 before main(check=True) was called (here: no main at all,
    # the import itself exits) is no evidence of a check.
    exits_on_import = "import sys\nprint('{\"x\": 3}')\nsys.exit(0)\n"
    # main(check=True) that ends with sys.exit(0) did run: passed.
    exits_in_main = (
        "import sys\n"
        "def main(check=False):\n"
        "    print('{\"x\": 3}')\n"
        "    sys.exit(0)\n"
        "if __name__ == '__main__':\n    main()\n"
    )
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        result = await session.call_tool("submit_code", {"code": exits_on_import})
        assert not result.is_error
        assert _self_check(result) == _not_available(
            "the program exited with code 0 before main(check=True) returned"
            " (for example during its import)."
        )
        result = await session.call_tool("submit_code", {"code": exits_in_main})
        assert not result.is_error
        assert _self_check(result)["status"] == "passed"


async def test_failing_plain_run_is_rejected_without_self_check(engine):
    code = _shaped(solve_body="    raise ValueError('no model')\n")
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        result = await session.call_tool("submit_code", {"code": code})
        assert result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["ok"] is False and "self_check" not in payload
        assert payload["error"].startswith("ValueError: no model.")
        assert result.structured_content["stats"]["self_check"] is None
    assert mcp_server._submissions == {}
    # Only the plain run was started.
    assert len(engine.uv_log.read_text().splitlines()) == 1


async def test_rejection_keeps_the_last_accepted_self_check(engine):
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        result = await session.call_tool("submit_code", {"code": VERIFIED})
        assert result.structured_content["stats"]["self_check"] == "passed"
        result = await session.call_tool("submit_code", {"code": "PLACEHOLDER"})
        assert result.is_error
        assert result.structured_content["stats"]["self_check"] == "passed"
        assert mcp_server._episode["self_check"] == "passed"
        result = await session.call_tool(
            "submit_code", {"code": _shaped(check="x == 4")}
        )
        assert not result.is_error
        assert result.structured_content["stats"]["self_check"] == "failed"
        assert mcp_server._episode["submit_ok"] == 2


async def test_self_check_setup_error_keeps_the_submission(engine, monkeypatch):
    # Any failure to set up the check (here PermissionError when starting
    # it) is reported; the plain run's acceptance stands.
    original = mcp_server._run_standalone

    def broken_check(*args, **kwargs):
        if kwargs.get("check"):
            raise PermissionError("denied")
        return original(*args, **kwargs)

    monkeypatch.setattr(mcp_server, "_run_standalone", broken_check)
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        result = await session.call_tool("submit_code", {"code": VERIFIED})
        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["ok"] is True and payload["stdout"] == PLAIN_OUT
        assert payload["self_check"] == _not_available(
            "its run could not start (PermissionError: denied)."
        )
        assert len(_links(result)) == 1
        assert list(mcp_server._submissions.values()) == [VERIFIED]
        assert mcp_server._episode["submit_ok"] == 1
        assert mcp_server._episode["self_check"] == "not_available"


async def test_self_check_skipped_when_plain_run_used_the_budget(engine, monkeypatch):
    # The plain run takes about 2 s of a 4 s budget; with 2.5 s required for
    # the check, none is started.
    monkeypatch.setattr(mcp_server, "STANDALONE_TIMEOUT", 4)
    monkeypatch.setattr(mcp_server, "_MIN_CHECK_SECONDS", 2.5)
    code = _shaped(solve_body="    import time; time.sleep(2)\n    return {'x': 3}\n")
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        result = await session.call_tool("submit_code", {"code": code})
    assert not result.is_error
    assert _self_check(result) == _not_available(
        "less than 2.5 s of the 4 s budget were left for main(check=True) after"
        " the plain run."
    )
    assert len(engine.uv_log.read_text().splitlines()) == 1


@pytest.mark.skipif(os.name != "posix", reason="process groups, /bin/sh")
async def test_self_check_gets_only_the_rest_of_the_budget(engine, monkeypatch):
    # The plain run takes about 2 s of a 4 s budget and verify() hangs: the
    # check is killed when the 4 s are over (a check with a fresh budget
    # would end only after about 6 s) and reported as not available; the
    # submission stands.
    monkeypatch.setattr(mcp_server, "STANDALONE_TIMEOUT", 4)
    monkeypatch.setattr(mcp_server, "_MIN_CHECK_SECONDS", 0.5)
    code = _shaped(
        solve_body="    import time; time.sleep(2)\n    return {'x': 3}\n",
        check="__import__('time').sleep(600)",
    )
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        start = time.monotonic()
        result = await session.call_tool("submit_code", {"code": code})
        elapsed = time.monotonic() - start
    assert elapsed < 5.2, elapsed
    assert not result.is_error
    assert _self_check(result) == _not_available(
        "main(check=True) did not finish before the 4 s budget ran out."
    )


def test_check_time_is_measured_right_before_the_start(monkeypatch):
    # The check gets the time left until the deadline when its process
    # starts, not a value computed before the worker thread ran; past the
    # deadline (less than _MIN_CHECK_SECONDS left) it does not start.
    timeouts = []

    def fake_run(program, packages, timeout, **kwargs):
        timeouts.append(timeout)
        return ProgramRun(0, "", "", False)

    monkeypatch.setattr(mcp_server, "run_isolated", fake_run)
    deadline = time.monotonic() + 30
    time.sleep(0.3)
    run = mcp_server._run_standalone("print(1)", [], deadline=deadline, check=True)
    assert run is not None and 29 - 0.5 < timeouts[0] <= 29.7 + 0.01
    late = time.monotonic() + mcp_server._MIN_CHECK_SECONDS - 0.1
    assert mcp_server._run_standalone("print(1)", [], deadline=late, check=True) is None
    assert len(timeouts) == 1


@pytest.mark.skipif(os.name != "posix", reason="process groups, /bin/sh")
async def test_self_check_over_output_budget_is_failed_even_on_exit_0(engine):
    # verify() writes 5 MB to stdout and returns: the check exits 0 (or is
    # killed for its output), and either way it is no passed check.
    code = _shaped(check="__import__('sys').stdout.write('x' * 5_000_000)")
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        result = await session.call_tool("submit_code", {"code": code})
    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["stdout"] == PLAIN_OUT
    check = payload["self_check"]
    assert check["status"] == "failed"
    assert check["message"].startswith(
        "Self-check failed: main(check=True) wrote more than 4,000,000 bytes"
    )


@pytest.mark.skipif(os.name != "posix", reason="process groups, /bin/sh")
async def test_cancelled_submission_kills_its_self_check(engine, tmp_path):
    pid_file = tmp_path / "pid"
    # verify() writes its pid, then hangs (a list evaluates both).
    hang = (
        f"[open({str(pid_file)!r}, 'w').write(str(__import__('os').getpid())),"
        " __import__('time').sleep(600)]"
    )
    pids = []
    try:
        async with _session() as session:
            await session.call_tool("select_backend", {"solver": "z3"})
            task = asyncio.create_task(mcp_server.submit_code(_shaped(check=hang)))
            pids.append(await _read_pid(pid_file))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert _pid_gone(pids[0])
            assert not mcp_server._standalone_runs
    finally:
        _kill(pids)


def test_self_check_outcomes_of_other_runs():
    outcome = mcp_server._self_check_outcome
    done = mcp_server._CHECK_DONE_MARKER
    status, text = outcome(ProgramRun(2, "", "warning\nboom\n", False))
    assert status == "failed"
    assert text.startswith(
        "Self-check failed: main(check=True) ended with exit code 2 (last line"
        " of stderr: boom)."
    )
    assert text.endswith(INFO)
    # Over the output budget is a failure, also when the run exited 0.
    for exit_code in (None, 0):
        status, text = outcome(ProgramRun(exit_code, "", done, False, True))
        assert status == "failed" and "wrote more than 4,000,000 bytes" in text
    status, text = outcome(ProgramRun(None, None, "uv not found on PATH", False))
    assert (status, text) == (
        "not_available",
        "Self-check not available: its run could not start (uv not found on"
        " PATH)." + INFO,
    )
    # A program that exits 86 on its own is not mistaken for a missing main.
    status, _ = outcome(ProgramRun(86, "", "", False))
    assert status == "failed"
    # Exit 0 counts as passed only with the driver's completion marker.
    assert outcome(ProgramRun(0, "", "", False))[0] == "not_available"
    assert outcome(ProgramRun(0, "", f"x\n{done}\n", False))[0] == "passed"
    # A huge last line of stderr (sys.exit("e" * 100_000)) is cut.
    status, text = outcome(ProgramRun(1, "", "e" * 100_000, False))
    assert status == "failed" and len(text) < 1_000
    assert "e" * mcp_server._MAX_EXC_LINE + " ...)." in text


def test_self_check_messages_end_alike():
    # Every not-available and failed message ends with the same sentence.
    runs = [
        None,
        ProgramRun(None, "", "", True),
        ProgramRun(None, None, "uv not found on PATH", False),
        ProgramRun(86, "", mcp_server._NO_CHECK_MARKER, False),
        ProgramRun(0, "", "", False),
        ProgramRun(1, "", "Traceback (most recent call last):\nValueError: v\n", False),
        ProgramRun(None, "", "", False, True),
    ]
    for run in runs:
        status, text = mcp_server._self_check_outcome(run)
        assert status in ("not_available", "failed"), run
        assert text.endswith(INFO), run


def test_submit_description_states_the_self_check_is_information():
    flat = " ".join(mcp_server.submit_code.__doc__.split())
    assert "main(check=True)" in flat
    assert "the self-check does not decide acceptance" in flat
    assert "separate fresh isolated environment" in flat
    assert "same environment" not in flat


# --- submit_code: the rejection text names the actual failure ----------------

# The 16 rejected submissions of a 458-session benchmark run (round 2,
# 2026-11): the standalone run's exit code, stdout and stderr as the server
# returned them, environment paths shortened. All 16 were told "the usual
# cause is that it relies on variables or imports from the kernel session";
# none was.
REJECTIONS = json.loads(
    (Path(__file__).parent / "data" / "submit_rejections.json").read_text()
)
CLINGO_PARSE = "RuntimeError: parsing failed. Clingo reported: "
VERIFY_REJECTED = "The program's own verify() rejected its solution"
EXPECTED_START = {
    "01": CLINGO_PARSE + "<block>:1:14-15: error: syntax error, unexpected \\.",
    "02": "AssertionError. " + VERIFY_REJECTED,
    "03": "Timed out: the program did not finish within 240 s",
    "04": CLINGO_PARSE + "<block>:5:53-55: error: syntax error, unexpected =",
    "05": CLINGO_PARSE + "<block>:1:8-9: error: syntax error, unexpected"
    " <VARIABLE>, expecting <IDENTIFIER> or default or override; <block>:10:28-32:"
    " error: syntax error, unexpected #max.",
    "06": "AttributeError: 'int' object has no attribute '_rep'. The program"
    " raised AttributeError",
    "07": CLINGO_PARSE + "<block>:5:34-38: error: syntax error, unexpected #sum;",
    "08": CLINGO_PARSE + "<block>:7:38-44: error: syntax error, unexpected #count.",
    "09": CLINGO_PARSE + "<block>:8:38-44: error: syntax error, unexpected #count.",
    "10": CLINGO_PARSE + "<block>:9:19-25: error: syntax error, unexpected #count.",
    "11": CLINGO_PARSE + "<block>:3:56-57: error: syntax error, unexpected +",
    "12": CLINGO_PARSE + "<block>:1:8-9: error: syntax error, unexpected <VARIABLE>",
    "13": "AssertionError. " + VERIFY_REJECTED,
    "14": "Exception: Unexpected argument '0' to 'all'. The program raised Exception",
    "15": "Exception: Unexpected argument '0' to 'all'. The program raised Exception",
    "16": "Exception: Unexpected argument '1' to 'all'. The program raised Exception",
}


def _failure(exit_code, stderr="", stdout="", timed_out=False, packages=("z3-solver",)):
    run = ProgramRun(exit_code, stdout, stderr, timed_out)
    return mcp_server._run_failure(run, list(packages))


def test_rejection_texts_of_the_real_rejections():
    assert len(REJECTIONS) == 16
    for r in REJECTIONS:
        text = _failure(
            r["exit_code"], r["stderr"], r["stdout"], r["timed_out"], ("clingo",)
        )
        assert text.startswith(EXPECTED_START[r["id"][:2]]), (r["id"], text)
        assert text.endswith(" " + mcp_server._FIX), r["id"]
        assert "kernel" not in text, (r["id"], text)


def _traceback(*frames_and_exc: str) -> str:
    return "Traceback (most recent call last):\n" + "\n".join(frames_and_exc) + "\n"


def test_rejection_succeeds_without_text():
    assert _failure(0, stdout='{"x": 1}\n') is None


def test_rejection_empty_stdout():
    text = _failure(0, stdout="  \n")
    assert text.startswith("No output: the program exited with code 0 but printed")
    assert "kernel" not in text


def test_rejection_assert_outside_verify():
    stderr = _traceback(
        '  File "/tmp/run/submission.py", line 9, in <module>',
        "    assert solve() is not None, 'no model'",
        "AssertionError: no model",
    )
    text = _failure(1, stderr)
    assert text.startswith("AssertionError: no model. An assert in the program")
    assert "verify()" not in text


def test_rejection_name_error_names_kernel_state_and_the_name():
    stderr = _traceback(
        '  File "/tmp/run/submission.py", line 4, in solve',
        "    return {'n': N_ITEMS}",
        "NameError: name 'N_ITEMS' is not defined",
    )
    text = _failure(1, stderr)
    assert text.startswith("NameError: name 'N_ITEMS' is not defined. The program")
    assert "'N_ITEMS' without defining or importing it" in text
    assert "kernel session" in text


def test_rejection_unbound_local_is_not_blamed_on_the_kernel():
    stderr = _traceback(
        '  File "/tmp/run/submission.py", line 4, in solve',
        "UnboundLocalError: cannot access local variable 'best' where it is"
        " not associated with a value",
    )
    text = _failure(1, stderr)
    assert text.startswith("UnboundLocalError: cannot access local variable 'best'")
    assert "kernel" not in text


def test_rejection_module_not_found_names_module_and_packages():
    stderr = _traceback(
        '  File "/tmp/run/submission.py", line 2, in <module>',
        "    import networkx as nx",
        "ModuleNotFoundError: No module named 'networkx'",
    )
    text = _failure(1, stderr, packages=("cpmpy", "numpy"))
    assert text.startswith("ModuleNotFoundError: No module named 'networkx'. ")
    assert "only the packages cpmpy, numpy, without 'networkx'" in text
    assert "kernel session" in text and "python_reset" in text


def test_rejection_chained_exception_reports_the_last_one():
    stderr = (
        _traceback(
            '  File "/tmp/run/submission.py", line 5, in verify',
            "    total = solution['cost']",
            "KeyError: 'cost'",
        )
        + "\nDuring handling of the above exception, another exception occurred:\n\n"
        + _traceback(
            '  File "/tmp/run/submission.py", line 9, in <module>',
            "    raise ValueError('bad solution') from None",
            "ValueError: bad solution",
        )
    )
    text = _failure(1, stderr)
    assert text.startswith("ValueError: bad solution. The program raised ValueError")
    # The verify frame belongs to the earlier exception, and only an
    # AssertionError is reported as verify() rejecting the solution.
    assert "verify()" not in text


def test_rejection_assertion_in_verify_after_other_output():
    # Output after the traceback (here a clingo info line) does not hide it.
    stderr = (
        _traceback(
            '  File "/tmp/run/submission.py", line 30, in <module>',
            "    verify(solution)",
            '  File "/tmp/run/submission.py", line 22, in verify',
            '    assert len(s) == 12, "wrong length"',
            "AssertionError: wrong length",
        )
        + "<block>:2:1-9: info: atom does not occur in any rule head:\n  b\n"
    )
    text = _failure(1, stderr)
    assert text.startswith("AssertionError: wrong length. " + VERIFY_REJECTED)


def test_rejection_exit_without_traceback():
    text = _failure(3, "warming up\nno feasible schedule\n")
    assert text.startswith(
        "Exit code 3 when the server ran the program on its own. Last line of"
        " stderr: no feasible schedule"
    )
    assert _failure(2).startswith("Exit code 2 when the server ran the program")
    assert _failure(None, "uv not found on PATH").startswith("The run could not start")


def test_rejection_strips_color_codes():
    stderr = _traceback(
        '  File "/tmp/run/submission.py", line 2, in <module>',
        "\x1b[1;35mZeroDivisionError\x1b[0m: \x1b[35mdivision by zero\x1b[0m",
    )
    assert _failure(1, stderr).startswith("ZeroDivisionError: division by zero.")


def test_rejection_long_exception_line_is_cut():
    text = _failure(1, _traceback("ValueError: " + "x" * 2000))
    assert len(text) < 1000 and "x ..." in text


# Reduced versions of the real programs, run end to end under the fake uv.
PROGRAM_VERIFY_FAILS = """\
import json
SLOTS = 12

def solve():
    return {"sequence": list(range(SLOTS - 1))}

def verify(solution):
    assert len(solution["sequence"]) == SLOTS

if __name__ == "__main__":
    solution = solve()
    verify(solution)
    print(json.dumps(solution))
"""
PROGRAM_CLINGO_PARSE = """\
import sys
sys.stderr.write("<block>:3:5-11: error: syntax error, unexpected #count\\n\\n")
raise RuntimeError("parsing failed")
"""
PROGRAM_MISSING_MODULE = "import json\nimport no_such_module_mcp_solver\nprint(1)\n"


async def test_submit_rejection_texts_end_to_end(engine):
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "clingo"})
        texts = []
        for code in (
            PROGRAM_VERIFY_FAILS,
            PROGRAM_CLINGO_PARSE,
            PROGRAM_MISSING_MODULE,
        ):
            result = await session.call_tool("submit_code", {"code": code})
            assert result.is_error
            payload = json.loads(result.content[0].text)
            assert payload["ok"] is False and payload["exit_code"] == 1
            assert "Traceback" in payload["stderr"]
            texts.append(payload["error"])
    verify_text, clingo_text, module_text = texts
    assert verify_text.startswith("AssertionError. " + VERIFY_REJECTED)
    assert clingo_text.startswith(
        CLINGO_PARSE + "<block>:3:5-11: error: syntax error, unexpected #count."
    )
    assert module_text.startswith(
        "ModuleNotFoundError: No module named 'no_such_module_mcp_solver'."
        " The run has only the packages clingo, without"
    )


TEST_FIRST_RULE = (
    "run the complete program with python_exec, calling main(check=True), and"
    " check its output;"
)


def test_test_first_rule_stated_everywhere():
    # One rule in every text the solving host reads: run the complete program
    # in the kernel before submitting; submit_code is not for a first try.
    texts = {
        "description": mcp_server.SELECT_BACKEND_DESCRIPTION,
        "guide": mcp_server.GUIDE,
        "submit_code": mcp_server.submit_code.__doc__,
        **{solver: mcp_server.get_template(solver) for solver in BACKENDS},
    }
    for name, text in texts.items():
        flat = " ".join(text.replace("*", "").replace("`", "").split())
        assert TEST_FIRST_RULE in flat, name
        assert "not for a first try" in flat, name
        assert "submit again" not in flat, name


SELF_CHECK_SENTENCES = (
    "The self-check is a heuristic: when it fails, either the solution or"
    " verify() is wrong, so check both against the problem statement; a wrong"
    " verify() does not make a correct solution wrong.",
    "The guard stays main(): to run the check in the kernel, call"
    " main(check=True) in a separate statement after the program text, never"
    " by changing the guard; the submitted program is the text with the guard"
    " main().",
)


def test_templates_state_self_check_heuristic_and_guard_rule():
    # The self-check does not gate a correct solution, and the guard is not
    # edited to get the check in the kernel (a live solve submitted a guard
    # calling main(check=True)).
    for solver in BACKENDS:
        text = mcp_server.get_template(solver)
        flat = " ".join(text.replace("`", "").split())
        for sentence in SELF_CHECK_SENTENCES:
            assert sentence in flat, (solver, sentence[:30])
        assert "Verification Requirement" not in flat, solver


VERIFY_METHOD_SENTENCE = (
    "verify() checks the solution against the statement's own definitions by"
    " direct computation in plain Python (simulate the process, recount,"
    " recompute), not by repeating the derivation the model used; a verify()"
    " that mirrors the encoding passes whatever the encoding gets wrong."
)


def test_templates_state_verify_tests_by_direct_computation():
    # A verify() that repeats the model's derivation passes the model's error
    # (live runs: a pooled reachability fixpoint in both model and verify()).
    for solver in BACKENDS:
        flat = " ".join(mcp_server.get_template(solver).replace("`", "").split())
        assert VERIFY_METHOD_SENTENCE in flat, solver


PROOF_STATUS_SENTENCE = (
    "A program that stops the solver before it finishes (time limit, cancel,"
    " solution or conflict limit) reads the solver's proof status and reports"
    " a result as optimal, or the problem as unsatisfiable, only if the solver"
    " proved it; otherwise it raises, or reports the status if the output"
    " format has a place for it."
)
PROOF_STATUS_CALLS = {
    "clingo": ("exhausted", "interrupted"),
    "cpmpy": ("ExitStatus.OPTIMAL", "ExitStatus.UNKNOWN"),
    "z3": ("unknown",),
    "pysat": ("solve_limited()", "None"),
    "maxsat": ("expect_interrupt=True", "interrupted"),
    "didp": ("time_out", "is_optimal", "is_infeasible"),
}


def test_templates_state_proof_status_rule():
    # Programs that stopped the solver reported unproven optima (five of 229
    # benchmark programs); each template states the rule and its status call.
    assert set(PROOF_STATUS_CALLS) == set(BACKENDS)
    for solver in BACKENDS:
        flat = " ".join(mcp_server.get_template(solver).replace("`", "").split())
        assert PROOF_STATUS_SENTENCE in flat, solver
        for call in PROOF_STATUS_CALLS[solver]:
            assert call in flat, (solver, call)


async def test_kernel_note_states_test_first_rule(engine):
    async with _session() as session:
        result = await session.call_tool("select_backend", {"solver": "z3"})
        note = result.content[0].text.split("\n\n---\n\n")[-1]
        assert TEST_FIRST_RULE in note and "not for a first try" in note


# --- submit_code: cancellation, backend switch, output budget ----------------


def _pid_gone(pid, wait=3.0):
    """True once *pid* no longer exists."""
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        time.sleep(0.05)
    return False


def _kill(pids):
    """Kill whatever a failed test left running."""
    import contextlib
    import signal

    for pid in pids:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


async def _read_pid(path, limit=10):
    deadline = time.monotonic() + limit
    while not path.exists() or not path.read_text():
        assert time.monotonic() < deadline, "program did not start"
        await asyncio.sleep(0.02)
    return int(path.read_text())


def _sleeper(pid_file):
    return (
        "import os, time\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(600)\n"
        "print(1)\n"
    )


@pytest.mark.skipif(os.name != "posix", reason="process groups, /bin/sh")
async def test_cancelled_submission_kills_its_program(engine, tmp_path):
    pid_file = tmp_path / "pid"
    pids = []
    try:
        async with _session() as session:
            await session.call_tool("select_backend", {"solver": "z3"})
            # The tool coroutine, as the server's request handler runs it.
            task = asyncio.create_task(mcp_server.submit_code(_sleeper(pid_file)))
            pids.append(await _read_pid(pid_file))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert _pid_gone(pids[0])
            assert not mcp_server._standalone_runs
    finally:
        _kill(pids)


@pytest.mark.skipif(os.name != "posix", reason="process groups, /bin/sh")
async def test_disconnect_during_submission_kills_its_program(engine, tmp_path):
    pid_file = tmp_path / "pid"
    pids = []
    try:
        async with _session() as session:
            await session.call_tool("select_backend", {"solver": "z3"})
            task = asyncio.create_task(
                session.call_tool("submit_code", {"code": _sleeper(pid_file)})
            )
            pids.append(await _read_pid(pid_file))
            task.cancel()
        assert _pid_gone(pids[0])
    finally:
        _kill(pids)


@pytest.mark.skipif(os.name != "posix", reason="process groups")
async def test_shutdown_kills_registered_runs(engine):
    import subprocess

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"],
        start_new_session=True,
    )
    try:
        async with _session():
            mcp_server._standalone_runs.add(proc)
        assert proc.wait(timeout=5) is not None
        assert not mcp_server._standalone_runs
    finally:
        proc.kill()
        proc.wait()


async def test_backend_switch_during_compile_check_keeps_packages(engine):
    # The engine's compile check of a z3 submission is held while the host
    # selects cpmpy: the program still runs with z3's packages.
    code = "print('{\"z3 program\": 1}')"
    engine.gated_codes = {code}
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        task = asyncio.create_task(session.call_tool("submit_code", {"code": code}))
        deadline = time.monotonic() + 10
        while ("start", code) not in engine.events:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)
        await session.call_tool("select_backend", {"solver": "cpmpy"})
        engine.gate.set()
        result = await task
    assert not result.is_error
    assert "--with z3-solver python" in engine.uv_log.read_text()


async def test_backend_switch_during_run_credits_the_submitting_episode(
    engine, tmp_path
):
    # A z3 run is in flight when the host selects cpmpy: its success is not
    # credited to the cpmpy episode (which has no submission).
    started, release = tmp_path / "started", tmp_path / "release"
    code = (
        "import os, time\n"
        f"open({str(started)!r}, 'w').close()\n"
        "deadline = time.monotonic() + 20\n"
        f"while not os.path.exists({str(release)!r}) and time.monotonic() < deadline:\n"
        "    time.sleep(0.02)\n"
        "print('{}')\n"
    )
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        z3_episode = mcp_server._episode
        task = asyncio.create_task(session.call_tool("submit_code", {"code": code}))
        deadline = time.monotonic() + 10
        while not started.exists():
            assert time.monotonic() < deadline
            await asyncio.sleep(0.02)
        await session.call_tool("select_backend", {"solver": "cpmpy"})
        release.touch()
        result = await task
        assert not result.is_error
        cpmpy_episode = mcp_server._episode
        assert cpmpy_episode["solver"] == "cpmpy"
        assert cpmpy_episode["submit_attempts"] == 0
        assert cpmpy_episode["submit_ok"] == 0
        assert z3_episode["submit_attempts"] == 1 and z3_episode["submit_ok"] == 1
        assert result.structured_content["stats"]["submit_attempts"] == 1


@pytest.mark.skipif(os.name != "posix", reason="process groups, /bin/sh")
async def test_submit_output_over_budget_is_killed_and_rejected(engine, monkeypatch):
    from mcp_solver.agent import cli

    # 1 MB chunks, at most about 100 MB/s, 40 MB in all, then it waits (so
    # without the budget the run would end only at the time limit).
    monkeypatch.setattr(mcp_server, "STANDALONE_TIMEOUT", 20)
    code = (
        "import sys, time\nchunk = 'x' * 1_000_000 + '\\n'\n"
        "for _ in range(40):\n    sys.stdout.write(chunk)\n    sys.stdout.flush()\n"
        "    time.sleep(0.01)\ntime.sleep(600)\n"
    )
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        start = time.monotonic()
        result = await session.call_tool("submit_code", {"code": code})
        assert time.monotonic() - start < 30
    assert result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["ok"] is False and payload["executed"] is True
    assert payload["timed_out"] is False
    assert payload["error"].startswith("Too much output:")
    assert f"{cli.CAPTURE_BUDGET_BYTES:,} bytes" in payload["error"]
    assert mcp_server._submissions == {}


def test_submit_description_states_the_output_budget():
    from mcp_solver.agent import cli

    assert cli.CAPTURE_BUDGET_BYTES == 4 * cli.CAPTURE_TAIL_BYTES
    flat = " ".join(mcp_server.submit_code.__doc__.split())
    assert "4 MB of output" in flat


async def test_submit_cut_output_is_not_called_the_result(engine):
    code = "import json\nprint(json.dumps({'schedule': list(range(10000))}))\n"
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        result = await session.call_tool("submit_code", {"code": code})
    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert len(payload["stdout"]) == mcp_server._STDOUT_CAP
    message = payload["message"]
    assert mcp_server.RUN_OK_MESSAGE not in message
    assert "is the result" not in message
    assert f"longer than {mcp_server._STDOUT_CAP:,} characters" in message
    assert "only its end" in message
    assert "run the program" in message


def test_rejection_without_uv_names_the_setup_not_the_program():
    text = _failure(None, "uv not found on PATH")
    assert text.startswith("The run could not start: uv not found on PATH.")
    assert "python_exec" not in text
    assert "install uv" in text
