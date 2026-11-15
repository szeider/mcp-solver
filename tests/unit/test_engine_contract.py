"""Contract tests against the REAL engine kernel server (no LLM, no network).

mcp-solver relies on the result shapes of agentic-python-coder's
``ipython_mcp`` server: the CLI drives it through ``mcp_minion.MCPManager``
and persists the last accepted ``submit_code`` call, and the toolkit server
proxies its kernel tools. These tests start the server exactly as
``cli.solve`` and the toolkit server do (``engine.engine_server_config`` and
``engine.require_engine``) and pin those shapes, so an engine release that
changes them fails here rather than in a paid solve. Kernels are started
without ``--with`` packages, so no package download is needed.
"""

import asyncio
import json
import os

import pytest
import pytest_asyncio

from mcp_solver.agent import engine as engine_mod

pytestmark = pytest.mark.asyncio(loop_scope="module")

# Set in the process before the engine starts: the secret must not reach
# kernels, the UV_ variable must (see engine.kernel_env).
SECRET_VAR = "OPENROUTER_API_KEY"
PASSED_VAR = "UV_MCP_SOLVER_CONTRACT_PROBE"

EXEC_KEYS = {"stdout", "stderr", "result", "error", "kernel_id", "success"}


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def engine():
    """One engine connection for the module, built like ``cli.solve``.

    The manager is entered and exited inside one dedicated task: its anyio
    cancel scopes must be exited by the task that entered them, and
    pytest-asyncio runs fixture setup and teardown in different tasks.
    """
    from mcp_minion import MCPManager

    saved = {name: os.environ.get(name) for name in (SECRET_VAR, PASSED_VAR)}
    os.environ[SECRET_VAR] = "sk-contract-test"
    os.environ[PASSED_VAR] = "1"
    try:
        servers = engine_mod.engine_server_config()
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    ready: asyncio.Future = asyncio.get_running_loop().create_future()
    stop = asyncio.Event()

    async def own_manager():
        try:
            async with MCPManager(servers) as manager:
                engine_mod.require_engine(manager)
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


async def _call(engine, name: str, arguments: dict) -> dict:
    """Call an engine tool; its outcome must be a non-error JSON object."""
    outcome = await engine.call_tool(name, arguments)
    assert not outcome.is_error, outcome.text
    payload = json.loads(outcome.text)
    assert isinstance(payload, dict), payload
    return payload


async def test_tool_list(engine):
    names = {tool.name for tool in engine.get_tools()}
    assert names >= set(engine_mod.ENGINE_TOOLS)


async def test_reset_without_packages(kernel_id):
    assert isinstance(kernel_id, str)


async def test_reset_without_kernel_id_targets_default(kernel_id):
    # Engine >=4: an id-less reset acts on the default kernel.
    assert kernel_id == "_default"


async def test_reset_unknown_kernel_creates_it(engine):
    # mcp-solver-serve relies on this: select_backend creates or restarts
    # the solving kernel with one python_reset(kernel_id="solver").
    first = await _call(engine, "python_reset", {"kernel_id": "contract-k"})
    assert first["success"] is True, first
    assert first["kernel_id"] == "contract-k"
    assert first["created"] is True
    second = await _call(engine, "python_reset", {"kernel_id": "contract-k"})
    assert second["success"] is True, second
    assert second["created"] is False


async def test_exec_unknown_kernel_fails(engine):
    payload = await _call(
        engine, "python_exec", {"code": "print(1)", "kernel_id": "no-such-kernel"}
    )
    assert payload["success"] is False
    assert 'Kernel "no-such-kernel" not found' in payload["error"]


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
    # A rejection is a normal (non-error) result with ok=false; the toolkit
    # server turns it into an error result for its host.
    payload = await _call(engine, "submit_code", {"code": "def f(:\n"})
    assert payload["ok"] is False
    assert payload["error"]


async def test_kernel_env_is_filtered(engine, kernel_id):
    code = (
        "import os, json; print(json.dumps("
        f"[{SECRET_VAR!r} in os.environ, os.environ.get({PASSED_VAR!r})]))"
    )
    payload = await _call(engine, "python_exec", {"code": code, "kernel_id": kernel_id})
    assert payload["success"] is True, payload
    assert json.loads(payload["stdout"]) == [False, "1"]


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
        outcome = await engine.call_tool("submit_code", arguments)
        steps.append(
            {
                "step": step,
                "content": None,
                "tool_calls": [
                    {
                        "name": "submit_code",
                        "arguments": arguments,
                        "result": outcome.text,
                        "is_error": outcome.is_error,
                    }
                ],
            }
        )
    assert extract_last_submission(steps) == good
