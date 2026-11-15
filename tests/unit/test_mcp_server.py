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
        self.kernels: dict[str, list[str]] = {}  # live kernel id -> packages
        self.crashed: dict[str, list[str]] = {}  # crash records -> packages
        self.break_reset = False  # make python_reset return malformed JSON
        # a reset that asks for one of these packages fails to start
        self.unresolvable: set[str] = set()
        # canned python_exec replies by code
        self.exec_replies: dict[str, dict] = {}
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

    def crash(self, kid):
        """The kernel's process died: keep a crash record (engine >=4)."""
        self.crashed[kid] = self.kernels.pop(kid)

    async def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        code = arguments.get("code", name)
        self.events.append(("start", code))
        if name == "python_exec" and code in self.gated_codes:
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

VERIFIED = (
    "import json\n"
    "def verify(sol):\n"
    "    return sol['x'] == 3\n"
    "sol = {'x': 3}\n"
    "assert verify(sol)\n"
    "print(json.dumps(sol))\n"
)


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
            "stdout": '{"x": 3}\n',
            "message": mcp_server.RUN_OK_MESSAGE,
        }
        sc = result.structured_content
        assert sc["stdout"] == '{"x": 3}\n'
        assert sc["stats"]["submit_attempts"] == 1
        assert len(_links(result)) == 1
        assert list(mcp_server._submissions.values()) == [VERIFIED]
        assert mcp_server._episode["submit_ok"] == 1
    args = engine.uv_log.read_text().split()
    assert args[:4] == ["run", "--no-project", "--isolated", "--python"]
    assert args[5:7] == ["--with", "cpmpy"] and args[-1].endswith("submission.py")


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
        assert "compiles but failed when run on its own" in payload["error"]
        assert "self-contained" in payload["error"]
        assert "python_exec" in payload["error"]
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
    first, second = engine.uv_log.read_text().splitlines()
    assert "--with python-sat[pb] --with numpy python" in first
    assert "--with z3-solver python" in second


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
