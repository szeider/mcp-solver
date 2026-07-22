"""Unit tests for the v4.1 toolkit MCP server (no LLM, no network).

The round-trip tests drive the real FastMCP server over an in-memory client
session; the engine (ipython_mcp) is replaced by a fake MCPManager that
mimics its public contract: flat-JSON tool results (``success``/``kernel_id``
for kernel tools, ``ok`` for submit_code) inside the client's
``{"result"|"error"}`` envelope.
"""

import json

import pytest

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

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    def has_tool(self, name):
        return name in ENGINE_TOOLS

    async def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        return json.dumps({"result": json.dumps(self._respond(name, arguments))})

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
    mcp_server._submissions.clear()
    return fake


def _session():
    from mcp.shared.memory import create_connected_server_and_client_session

    return create_connected_server_and_client_session(mcp_server.mcp._mcp_server)


def _engine_payload(result) -> dict:
    """Decode the engine's flat JSON from a proxied tool result."""
    return json.loads(result.content[0].text)


# --- Tier 0 budget ----------------------------------------------------------


def test_select_backend_description_within_budget():
    # Claude Code truncates tool descriptions at 2KB; spec budget is 1.2KB.
    assert len(mcp_server.SELECT_BACKEND_DESCRIPTION.encode()) <= 1200


def test_guide_and_description_mention_every_backend():
    for solver in BACKENDS:
        assert solver in mcp_server.GUIDE
        assert solver in mcp_server.SELECT_BACKEND_DESCRIPTION


# --- resource functions -----------------------------------------------------


def test_template_resource_returns_backend_instructions():
    assert "z3" in mcp_server.template_resource("z3").lower()


def test_template_resource_rejects_unknown_solver():
    with pytest.raises(ValueError, match="unknown solver"):
        mcp_server.template_resource("gurobi")


def test_submission_resource_rejects_unknown_id():
    with pytest.raises(ValueError, match="unknown submission"):
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


async def test_engine_call_surfaces_client_error(monkeypatch):
    class Broken:
        async def call_tool(self, name, arguments):
            return json.dumps({"error": "Tool 'python_exec' timed out"})

    monkeypatch.setattr(mcp_server, "_engine", Broken())
    with pytest.raises(Exception, match="timed out"):
        await mcp_server._engine_call("python_exec", {"code": "1"})


async def test_engine_call_rejects_malformed_envelope(monkeypatch):
    class Broken:
        async def call_tool(self, name, arguments):
            return "not json at all"

    monkeypatch.setattr(mcp_server, "_engine", Broken())
    with pytest.raises(Exception, match="malformed envelope"):
        await mcp_server._engine_call("python_exec", {"code": "1"})


async def test_engine_call_rejects_non_object_envelope(monkeypatch):
    class Broken:
        async def call_tool(self, name, arguments):
            return json.dumps([1, 2, 3])

    monkeypatch.setattr(mcp_server, "_engine", Broken())
    with pytest.raises(Exception, match="malformed envelope"):
        await mcp_server._engine_call("python_exec", {"code": "1"})


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
        assert not result.isError
        assert result.structuredContent is None
        text = result.content[0].text
        assert "fresh solving kernel" in text
        assert "submit_code" in text
        assert "z3" in text.lower()  # the template made it into the result
        name, args = engine.calls[0]
        assert name == "python_reset" and "kernel_id" not in args
        assert args["packages"][0] == "z3-solver"

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
        assert args["packages"][0] == "python-sat"

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
        assert result.isError
        assert "unknown solver" in result.content[0].text


async def test_select_backend_fails_closed_on_malformed_reset(engine):
    # An engine reply without a "success" field must surface as an error,
    # not as a phantom "fresh kernel ready" with kernel_id None.
    engine.break_reset = True
    async with _session() as session:
        result = await session.call_tool("select_backend", {"solver": "z3"})
        assert result.isError
        assert "kernel setup" in result.content[0].text
    assert mcp_server._kernel_id is None


async def test_submit_code_round_trip(engine):
    async with _session() as session:
        # Syntax errors are rejected: no resource_link, ok=false rides along.
        result = await session.call_tool("submit_code", {"code": "def f(:"})
        assert not result.isError
        assert result.structuredContent["ok"] is False
        assert all(c.type != "resource_link" for c in result.content)

        # A valid program is stored and linked as a fetchable resource.
        program = "print('{\"answer\": 42}')"
        result = await session.call_tool("submit_code", {"code": program})
        assert result.structuredContent == {"ok": True}
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
            assert result.structuredContent == {"ok": True}
    assert len(mcp_server._submissions) == 2
    assert list(mcp_server._submissions.values()) == ["print(1)", "print(2)"]


async def test_submit_stats_ride_along(engine):
    async with _session() as session:
        await session.call_tool("select_backend", {"solver": "z3"})
        await session.call_tool("python_exec", {"code": "1"})
        await session.call_tool("python_exec", {"code": "2"})
        result = await session.call_tool("submit_code", {"code": "print(1)"})
        sc = result.structuredContent
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
