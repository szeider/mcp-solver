"""The engine: starting and checking the ipython_mcp kernel server.

Shared by the CLI (``mcp-solver``) and the toolkit server
(``mcp-solver-serve``): both talk to agentic-python-coder's ``ipython_mcp``
server over stdio through ``mcp_minion.MCPManager``. This module owns

- the server config handed to MCPManager (:func:`engine_server_config`),
- the check that the engine offers the tools mcp-solver relies on
  (:func:`require_engine`, with the minimum engine version defined once),
- the environment for the engine, its kernels and the CLI's re-execution of
  the submitted program (:func:`kernel_env`),
- dev-mode resolution (templates from a source checkout,
  :func:`resolve_dev_path`).
"""

from __future__ import annotations

import os
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import mcp_solver

# Minimum agentic-python-coder release; must equal the pin in the [agent]
# extra of pyproject.toml (a unit test checks this).
MIN_ENGINE_VERSION = "4.0.0"

# Name of the engine server in the MCPManager config.
ENGINE_SERVER_NAME = "ipython"

# Engine tools mcp-solver relies on.
ENGINE_TOOLS = (
    "python_exec",
    "python_reset",
    "python_status",
    "python_interrupt",
    "submit_code",
)

# Environment variables passed through to the engine, its kernels and the
# re-executed program, on top of the mcp library's default whitelist
# (HOME, LOGNAME, PATH, SHELL, TERM, USER on Unix; the Windows system
# variables there). This is hygiene that keeps API keys and tokens out of
# the environment of model-written code; it is not a sandbox (that code can
# still read files in the home directory, for instance). Values pass
# unchanged, so a credential inside a passed URL (UV_INDEX_URL or a proxy
# URL with user and password) reaches that code; uv needs it to install from
# a private index.
_ENV_PREFIXES = ("UV_", "CODER_", "MCP_SOLVER_", "XDG_", "LC_")
# Prefix-matched variables are dropped when their name contains one of
# these words (case-insensitive), e.g. UV_PUBLISH_TOKEN or
# UV_INDEX_<NAME>_PASSWORD, or when they are listed here: UV_ENV_FILE would
# make uv load a dotenv file, which may hold an API key.
_SECRET_WORDS = ("PASSWORD", "TOKEN", "SECRET", "KEY", "CREDENTIAL")
_ENV_DENY = frozenset({"UV_ENV_FILE"})
_ENV_NAMES = frozenset(
    {
        "LANG",
        "TMPDIR",
        "TEMP",
        "TMP",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        # Native libraries and source builds (solver wheels, sdists).
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "Z3_LIBRARY_PATH",
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "CPPFLAGS",
        "PKG_CONFIG_PATH",
        "INCLUDE",
        "LIB",
        "LIBPATH",
    }
    | {
        name
        for proxy in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY")
        for name in (proxy, proxy.lower())
    }
)

_DEV_ENV = "MCP_SOLVER_DEV"


class EngineError(RuntimeError):
    """The engine is unavailable or lacks a tool mcp-solver needs."""


class DevModeError(ValueError):
    """Dev mode was requested but no source checkout can be found."""


# --- engine server -----------------------------------------------------------


def kernel_env() -> dict[str, str]:
    """The environment for the engine, its kernels and the re-execution.

    A filtered copy of ``os.environ``: the mcp library's default whitelist,
    the names in ``_ENV_NAMES`` (locale, temp dirs, proxies, CA bundles,
    native-library and build settings), and variables with the prefixes in
    ``_ENV_PREFIXES`` unless their name looks like a credential. Hygiene,
    not a sandbox: API keys such as OPENROUTER_API_KEY and variables named
    like tokens stay out of the environment of model-written code, but a
    credential inside a passed URL value (an index or proxy URL with user
    and password) reaches it, and that code can still read files the user
    can read.
    """
    # Imported here so that the CLI's --help and its install guidance work
    # in a base install without the [agent] extra (no mcp).
    from mcp.client.stdio import get_default_environment

    env = get_default_environment()
    for key, value in os.environ.items():
        if key in _ENV_NAMES or (
            key.startswith(_ENV_PREFIXES) and not _looks_secret(key)
        ):
            env[key] = value
    return env


def _looks_secret(name: str) -> bool:
    upper = name.upper()
    return upper in _ENV_DENY or any(word in upper for word in _SECRET_WORDS)


def engine_server_config() -> dict[str, dict[str, Any]]:
    """The MCPManager servers config that starts the engine.

    The engine runs under this interpreter (it is a dependency of the
    [agent] extra) with :func:`kernel_env` as its environment.
    """
    return {
        ENGINE_SERVER_NAME: {
            "command": sys.executable,
            "args": ["-m", "agentic_python_coder.mcp_server"],
            "env": kernel_env(),
        }
    }


def require_engine(manager: Any) -> None:
    """Raise EngineError unless *manager* offers every tool in ENGINE_TOOLS."""
    missing = [name for name in ENGINE_TOOLS if not manager.has_tool(name)]
    if missing:
        raise EngineError(
            "the ipython_mcp engine lacks the tools"
            f" {', '.join(missing)}; install agentic-python-coder"
            f" >={MIN_ENGINE_VERSION} in this environment"
        )


def _too_old(installed: str) -> bool:
    """True if *installed* is below MIN_ENGINE_VERSION in PEP 440 order
    (pre-releases count: 4.0.0rc0 < 4.0.0rc1 < 4.0.0), or is no version."""
    # packaging is a dependency of agentic-python-coder, so it is present
    # wherever the engine is installed.
    from packaging.version import InvalidVersion, Version

    try:
        return Version(installed) < Version(MIN_ENGINE_VERSION)
    except InvalidVersion:
        return True


def require_engine_version(installed: str | None = None) -> None:
    """Raise EngineError unless agentic-python-coder >= MIN_ENGINE_VERSION.

    The engine runs under this interpreter, so the installed distribution is
    the one that starts. Older releases offer the same tools, so the tool
    check alone does not catch them.
    """
    hint = f'install "agentic-python-coder>={MIN_ENGINE_VERSION}" in this environment'
    if installed is None:
        try:
            installed = metadata.version("agentic-python-coder")
        except metadata.PackageNotFoundError:
            raise EngineError(
                f"agentic-python-coder is not installed; {hint}"
            ) from None
    if _too_old(installed):
        raise EngineError(
            f"agentic-python-coder {installed} is too old (mcp-solver needs"
            f" >={MIN_ENGINE_VERSION}); {hint}"
        )


async def start_engine(errlog: Any = None) -> Any:
    """Start the engine and check it; return the entered MCPManager.

    The caller exits the manager (``await manager.__aexit__(None, None,
    None)``). On a failed check the manager is closed before EngineError
    propagates.
    """
    import mcp_minion

    require_engine_version()
    kwargs = {} if errlog is None else {"errlog": errlog}
    manager = mcp_minion.MCPManager(engine_server_config(), **kwargs)
    await manager.__aenter__()
    try:
        require_engine(manager)
    except BaseException:
        await manager.__aexit__(None, None, None)
        raise
    return manager


# --- dev mode -----------------------------------------------------------------


def local_checkout() -> str | None:
    """Return the repo root when mcp_solver runs from a source checkout.

    An editable install resolves ``mcp_solver`` to ``<root>/src/mcp_solver``;
    a regular install resolves it inside site-packages, where the checkout
    markers are absent.
    """
    root = Path(mcp_solver.__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "src" / "mcp_solver").is_dir():
        return str(root)
    return None


def _auto_checkout(source: str) -> str:
    checkout = local_checkout()
    if checkout is None:
        raise DevModeError(
            f"dev mode was requested ({source}) but no source checkout of"
            " mcp-solver was detected; pass --dev PATH or set"
            f" {_DEV_ENV} to a checkout path"
        )
    return checkout


def resolve_dev_path(dev: str | None = None, no_dev: bool = False) -> str | None:
    """Return the effective dev-mode path (a source checkout root) or None.

    Dev mode means that the solver templates come from a local checkout.
    Resolution order:

    1. *no_dev* → None (packaged templates), overrides everything.
    2. *dev* is a path → that path; ``"auto"`` → the detected checkout.
    3. ``$MCP_SOLVER_DEV`` set to ``1``/``true``/``auto`` → the detected
       checkout; any other non-empty value → treated as a path.
    4. Nothing set → a detected checkout, else None.

    Raises DevModeError when an explicit auto request (2 or 3) finds no
    checkout. The CLI passes its ``--dev``/``--no-dev`` flags; the toolkit
    server has none and passes nothing.
    """
    if no_dev:
        return None
    if dev is not None:
        return _auto_checkout("--dev") if dev == "auto" else dev
    env = os.environ.get(_DEV_ENV)
    if env:
        if env.lower() in ("1", "true", "auto"):
            return _auto_checkout(f"{_DEV_ENV}={env}")
        return env
    return local_checkout()
