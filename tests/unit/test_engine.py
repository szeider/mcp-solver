"""Unit tests for mcp_solver.agent.engine (no engine process, no network)."""

import re
import sys
import tomllib
from pathlib import Path

import pytest

from mcp_solver.agent import engine


def _engine_pin() -> str:
    """The agentic-python-coder minimum in the [agent] extra of pyproject.toml.

    The unit tests run from a source checkout, so the pyproject is the source
    of truth (installed metadata can lag behind an edit until a reinstall).
    """
    checkout = engine.local_checkout()
    assert checkout is not None, "unit tests must run from a source checkout"
    data = tomllib.loads((Path(checkout) / "pyproject.toml").read_text())
    for req in data["project"]["optional-dependencies"]["agent"]:
        match = re.fullmatch(r"agentic-python-coder\s*>=\s*([\w.]+)", req.strip())
        if match:
            return match.group(1)
    raise AssertionError("no agentic-python-coder>=X pin in the [agent] extra")


def test_min_engine_version_matches_pyproject_pin():
    assert _engine_pin() == engine.MIN_ENGINE_VERSION


# --- kernel environment ----------------------------------------------------


def test_kernel_env_filters_secrets(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-other")
    monkeypatch.setenv("UV_INDEX_URL", "https://example.invalid/simple")
    monkeypatch.setenv("https_proxy", "http://proxy.invalid:3128")
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ca.pem")
    monkeypatch.setenv("MCP_SOLVER_DEV", "/some/checkout")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    env = engine.kernel_env()
    assert "OPENROUTER_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert env["UV_INDEX_URL"] == "https://example.invalid/simple"
    assert env["https_proxy"] == "http://proxy.invalid:3128"
    assert env["SSL_CERT_FILE"] == "/etc/ca.pem"
    assert env["MCP_SOLVER_DEV"] == "/some/checkout"
    assert env["LC_ALL"] == "C.UTF-8"


def test_kernel_env_keeps_mcp_default_whitelist():
    from mcp.client.stdio import get_default_environment

    env = engine.kernel_env()
    assert get_default_environment().items() <= env.items()
    assert "PATH" in env


def test_engine_server_config():
    config = engine.engine_server_config()
    assert list(config) == [engine.ENGINE_SERVER_NAME]
    server = config[engine.ENGINE_SERVER_NAME]
    assert server["command"] == sys.executable
    assert server["args"] == ["-m", "agentic_python_coder.mcp_server"]
    assert server["env"] == engine.kernel_env()


# --- engine check ----------------------------------------------------------


class _Manager:
    def __init__(self, tools):
        self.tools = set(tools)
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.exited = True

    def has_tool(self, name):
        return name in self.tools


def test_require_engine_passes_with_all_tools():
    engine.require_engine(_Manager(engine.ENGINE_TOOLS))


def test_require_engine_names_missing_tools_and_version():
    tools = set(engine.ENGINE_TOOLS) - {"submit_code"}
    with pytest.raises(engine.EngineError) as info:
        engine.require_engine(_Manager(tools))
    assert "submit_code" in str(info.value)
    assert f">={engine.MIN_ENGINE_VERSION}" in str(info.value)


async def test_start_engine_closes_manager_on_failed_check(monkeypatch):
    import mcp_minion

    manager = _Manager(["python_exec"])
    seen = {}

    def factory(servers, **kwargs):
        seen["servers"] = servers
        return manager

    monkeypatch.setattr(mcp_minion, "MCPManager", factory)
    with pytest.raises(engine.EngineError):
        await engine.start_engine()
    assert manager.exited
    assert engine.ENGINE_SERVER_NAME in seen["servers"]


# --- dev mode ----------------------------------------------------------------


def test_resolve_dev_path_order(monkeypatch):
    monkeypatch.delenv("MCP_SOLVER_DEV", raising=False)
    monkeypatch.setattr(engine, "local_checkout", lambda: "/repo")
    assert engine.resolve_dev_path() == "/repo"
    assert engine.resolve_dev_path(no_dev=True) is None
    assert engine.resolve_dev_path("/arg") == "/arg"
    assert engine.resolve_dev_path("auto") == "/repo"
    monkeypatch.setenv("MCP_SOLVER_DEV", "/env")
    assert engine.resolve_dev_path() == "/env"
    assert engine.resolve_dev_path("/arg") == "/arg"


@pytest.mark.parametrize(
    ("dev", "env"), [("auto", None), (None, "1"), (None, "true"), (None, "auto")]
)
def test_resolve_dev_path_auto_without_checkout_errors(monkeypatch, dev, env):
    monkeypatch.delenv("MCP_SOLVER_DEV", raising=False)
    if env is not None:
        monkeypatch.setenv("MCP_SOLVER_DEV", env)
    monkeypatch.setattr(engine, "local_checkout", lambda: None)
    with pytest.raises(engine.DevModeError, match="no source checkout"):
        engine.resolve_dev_path(dev)


def test_kernel_env_drops_credential_like_names(monkeypatch):
    monkeypatch.setenv("UV_INDEX_FOO_PASSWORD", "hunter2")
    monkeypatch.setenv("UV_PUBLISH_TOKEN", "pypi-token")
    monkeypatch.setenv("UV_ENV_FILE", "/home/u/.env")
    monkeypatch.setenv("CODER_API_KEY", "sk-coder")
    monkeypatch.setenv("XDG_secret_dir", "/x")
    monkeypatch.setenv("UV_INDEX_URL", "https://example.invalid/simple")
    monkeypatch.setenv("UV_CACHE_DIR", "/tmp/uv-cache")
    env = engine.kernel_env()
    for name in (
        "UV_INDEX_FOO_PASSWORD",
        "UV_PUBLISH_TOKEN",
        "UV_ENV_FILE",
        "CODER_API_KEY",
        "XDG_secret_dir",
    ):
        assert name not in env
    assert env["UV_INDEX_URL"] == "https://example.invalid/simple"
    assert env["UV_CACHE_DIR"] == "/tmp/uv-cache"


def test_kernel_env_passes_native_build_variables(monkeypatch):
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/opt/z3/lib")
    monkeypatch.setenv("CFLAGS", "-O2")
    env = engine.kernel_env()
    assert env["DYLD_LIBRARY_PATH"] == "/opt/z3/lib"
    assert env["CFLAGS"] == "-O2"
