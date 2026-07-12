"""Unit tests for the v4.1 MCP server (no LLM, no network).

The full-round-trip tests drive the real FastMCP server over an in-memory
client session, with the CLI subprocess replaced by a fake that mimics its
observable contract (stderr step lines, stdout solution JSON, artifact file).
"""

import asyncio
import json

import pytest

from mcp_solver.agent import mcp_server

# --- Tier 0 budget ----------------------------------------------------------


def test_tool_description_within_budget():
    # Claude Code truncates tool descriptions at 2KB; spec budget is 1.2KB.
    assert len(mcp_server.SOLVE_DESCRIPTION.encode()) <= 1200


def test_guide_mentions_every_backend():
    for solver in ("pysat", "maxsat", "z3", "cpmpy", "clingo"):
        assert solver in mcp_server.GUIDE
        assert solver in mcp_server.SOLVE_DESCRIPTION


# --- run artifact resource guards ------------------------------------------


def test_run_file_unknown_run():
    with pytest.raises(ValueError, match="unknown run"):
        mcp_server.run_file("nope", "problem_code.py")


def test_run_file_no_traversal(tmp_path, monkeypatch):
    (tmp_path / "problem_code.py").write_text("print(1)", encoding="utf-8")
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("s", encoding="utf-8")
    monkeypatch.setitem(mcp_server._runs, "r1", tmp_path)
    assert mcp_server.run_file("r1", "problem_code.py") == "print(1)"
    with pytest.raises(ValueError, match="unknown artifact"):
        mcp_server.run_file("r1", "../secret.txt")
    with pytest.raises(ValueError, match="unknown artifact"):
        mcp_server.run_file("r1", "absent.py")


# --- full in-memory round trip ---------------------------------------------


def _reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


class FakeProc:
    """Mimics the CLI subprocess: step lines, solution JSON, artifact file."""

    def __init__(self, cmd: list[str]):
        from pathlib import Path

        workdir = Path(cmd[cmd.index("--workdir") + 1])
        (workdir / "problem_code.py").write_text(
            "print('{\"satisfiable\": true}')", encoding="utf-8"
        )
        (workdir / "run_20260712_000000.json").write_text("{}", encoding="utf-8")
        self.stdout = _reader(b'{"satisfiable": true, "x": 2}\n')
        self.stderr = _reader(
            b"mcp-solver: step 1: python_exec\n"
            b"mcp-solver: step 2: submit_code\n"
            b"mcp-solver: step 3: final answer\n"
        )

    async def wait(self) -> int:
        return 0

    def kill(self) -> None:  # pragma: no cover - not hit on success
        pass


async def _fake_subprocess(*cmd, **kwargs):
    return FakeProc(list(cmd))


async def test_solve_round_trip(monkeypatch):
    from mcp.shared.memory import create_connected_server_and_client_session

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_subprocess)

    async with create_connected_server_and_client_session(
        mcp_server.mcp._mcp_server
    ) as session:
        # Tier 0: the tool is listed with its full description.
        tools = await session.list_tools()
        assert [t.name for t in tools.tools] == ["solve"]

        # Tier 1: the guide resource is readable.
        guide = await session.read_resource("mcp-solver://guide")
        assert "cpmpy" in guide.contents[0].text

        # The solve itself.
        result = await session.call_tool(
            "solve", {"solver": "z3", "problem": "toy problem"}
        )
        assert not result.isError
        text = result.content[0]
        assert json.loads(text.text) == {"satisfiable": True, "x": 2}

        # Tier 2: resource links to the program and the run log...
        links = [c for c in result.content if c.type == "resource_link"]
        assert [link.name for link in links] == [
            "problem_code.py",
            "run_20260712_000000.json",
        ]
        # ...and the linked program is fetchable through the resource.
        program = await session.read_resource(links[0].uri)
        assert "satisfiable" in program.contents[0].text


async def test_solve_rejects_bad_input():
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(
        mcp_server.mcp._mcp_server
    ) as session:
        result = await session.call_tool("solve", {"solver": "gurobi", "problem": "x"})
        assert result.isError
        assert "unknown solver" in result.content[0].text

        result = await session.call_tool("solve", {"solver": "z3", "problem": " "})
        assert result.isError


class FailingProc:
    def __init__(self):
        self.stdout = _reader(b"")
        self.stderr = _reader(b"mcp-solver: something broke\n")

    async def wait(self) -> int:
        return 3

    def kill(self) -> None:  # pragma: no cover
        pass


async def test_solve_surfaces_cli_failure(monkeypatch):
    from mcp.shared.memory import create_connected_server_and_client_session

    async def fake(*cmd, **kwargs):
        return FailingProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake)

    async with create_connected_server_and_client_session(
        mcp_server.mcp._mcp_server
    ) as session:
        result = await session.call_tool("solve", {"solver": "z3", "problem": "toy"})
        assert result.isError
        assert "exit 3" in result.content[0].text
