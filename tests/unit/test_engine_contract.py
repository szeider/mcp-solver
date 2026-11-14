"""Contract tests against the REAL engine kernel server (no LLM, no network).

mcp-solver relies on the result shapes of agentic-python-coder's
``ipython_mcp`` server: the CLI drives it through ``mcp_minion.MCPManager``
and persists the last accepted ``submit_code`` call, and the toolkit server
proxies its kernel tools. These tests start the server exactly as
``cli.solve`` does and pin those shapes, so an engine release that changes
them fails here rather than in a paid solve. Kernels are started without
``--with`` packages, so no package download is needed.
"""

import asyncio
import json
import sys

import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio(loop_scope="module")

ENGINE_TOOLS = {
    "python_exec",
    "python_reset",
    "python_status",
    "python_interrupt",
    "submit_code",
}

EXEC_KEYS = {"stdout", "stderr", "result", "error", "kernel_id", "success"}


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def engine():
    """One engine connection for the module, built like ``cli.solve``.

    The manager is entered and exited inside one dedicated task: its anyio
    cancel scopes must be exited by the task that entered them, and
    pytest-asyncio runs fixture setup and teardown in different tasks.
    """
    from mcp_minion import MCPManager

    servers = {
        "ipython": {
            "command": sys.executable,
            "args": ["-m", "agentic_python_coder.mcp_server"],
        }
    }
    ready: asyncio.Future = asyncio.get_running_loop().create_future()
    stop = asyncio.Event()

    async def own_manager():
        try:
            async with MCPManager(servers) as manager:
                ready.set_result(manager)
                await stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise

    owner = asyncio.create_task(own_manager())
    try:
        yield await ready
    finally:
        stop.set()
        await owner


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def kernel_id(engine):
    """A kernel started by python_reset without packages."""
    payload = await _call(engine, "python_reset", {})
    assert payload["success"] is True, payload
    assert payload["kernel_id"]
    return payload["kernel_id"]


def _unwrap(raw: str) -> dict:
    """Parse the manager's ``{"result": "<engine JSON text>"}`` envelope."""
    envelope = json.loads(raw)
    assert set(envelope) == {"result"}, envelope
    return json.loads(envelope["result"])


async def _call(engine, name: str, arguments: dict) -> dict:
    return _unwrap(await engine.call_tool(name, arguments))


async def test_tool_list(engine):
    names = {tool.name for tool in engine.get_tools()}
    assert names >= ENGINE_TOOLS


async def test_reset_without_packages(kernel_id):
    assert isinstance(kernel_id, str)


async def test_reset_unknown_kernel_fails(engine):
    payload = await _call(engine, "python_reset", {"kernel_id": "no-such-kernel"})
    assert payload["success"] is False


async def test_exec_success_shape(engine, kernel_id):
    payload = await _call(
        engine, "python_exec", {"code": "print(1)", "kernel_id": kernel_id}
    )
    assert set(payload) >= EXEC_KEYS, payload
    assert payload["success"] is True
    assert payload["stdout"] == "1\n"
    assert payload["kernel_id"] == kernel_id


async def test_exec_error_shape(engine, kernel_id):
    payload = await _call(
        engine, "python_exec", {"code": "1/0", "kernel_id": kernel_id}
    )
    assert set(payload) >= EXEC_KEYS, payload
    assert payload["success"] is False
    assert payload["error"]


async def test_submit_code_accepts_valid_code(engine):
    payload = await _call(engine, "submit_code", {"code": "print('ok')\n"})
    assert payload == {"ok": True}


async def test_submit_code_rejects_syntax_error(engine):
    payload = await _call(engine, "submit_code", {"code": "def f(:\n"})
    assert payload["ok"] is False
    assert payload["error"]


async def test_rejected_submit_after_good_keeps_good(engine):
    # The CLI persists the LAST ACCEPTED submission: a later submit that the
    # engine rejects must not replace an earlier accepted one. Steps are
    # recorded the way mcp_minion's Agent records tool calls.
    from mcp_minion import extract_last_submission

    good = "print('good')\n"
    bad = "def f(:\n"
    steps = []
    for step, code in enumerate((good, bad), start=1):
        arguments = {"code": code}
        result = await engine.call_tool("submit_code", arguments)
        steps.append(
            {
                "step": step,
                "content": None,
                "tool_calls": [
                    {"name": "submit_code", "arguments": arguments, "result": result}
                ],
            }
        )
    assert extract_last_submission(steps) == good
