"""CLI entry point for the ReAct agent."""

import argparse
import asyncio
import contextlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from mcp_minion import __version__
from mcp_minion.agent import Agent, AgentConfig
from mcp_minion.artifacts import extract_last_submission
from mcp_minion.env import find_api_key
from mcp_minion.logging import RunLogger
from mcp_minion.mcp_client import MCPManager
from mcp_minion.tools import create_default_registry

# Use AgentConfig defaults as single source of truth
_DEFAULTS = AgentConfig()


def strip_html_comments(text: str) -> str:
    """Remove HTML comments (<!-- ... -->) from text."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def save_submission(steps: Any, folder: Path) -> Path | None:
    """Write the last successfully submitted program to <folder>/submission.py.

    Returns the path when a successful submission exists in *steps*, else
    None. Overwritten on every run; earlier submissions remain recoverable
    from the timestamped run_*.json logs.
    """
    code = extract_last_submission(steps)
    if code is None:
        return None
    path = folder / "submission.py"
    path.write_text(code, encoding="utf-8")
    return path


# --- verbose step rendering --------------------------------------------------

_RESULT_TEXT_LIMIT = 500  # chars of plain-text result shown on the console
_STREAM_LIMIT = 2000  # chars of a kernel stdout/stderr stream shown
_ARG_LIMIT = 120  # chars of a non-code argument value shown


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n… ({len(text):,} chars — full text in run log)"


def render_tool_call(name: str, arguments: dict) -> str:
    """One tool call as 'Tool: name' plus indented arguments.

    Multi-line string arguments (code) are shown as real code blocks;
    everything else is a truncated repr on one line.
    """
    lines = [f"Tool: {name}"]
    for key, val in (arguments or {}).items():
        if isinstance(val, str) and "\n" in val:
            lines.append(f"  {key}:")
            lines.append(_indent(val.rstrip()))
        else:
            r = repr(val)
            if len(r) > _ARG_LIMIT:
                r = r[:_ARG_LIMIT] + f"… ({len(r):,} chars)"
            lines.append(f"  {key}: {r}")
    return "\n".join(lines)


def render_tool_result(text: str, is_error: bool = False) -> str:
    """A tool result (step-record text), shortened for the console.

    Error outcomes are marked; kernel payloads (stdout/stderr/success) are
    labeled blocks; long plain text (e.g. a modeling template) collapses to
    its first line plus a size note. The complete text is always in the
    run_*.json log.
    """
    inner = text if isinstance(text, str) else str(text)
    if is_error:
        return "Result: ERROR\n" + _indent(_truncate(inner.strip(), _STREAM_LIMIT))

    payload = None
    with contextlib.suppress(json.JSONDecodeError, TypeError):
        payload = json.loads(inner)

    if isinstance(payload, dict) and ("success" in payload or "ok" in payload):
        lines = []
        if "success" in payload:
            status = "ok" if payload.get("success") else "FAILED"
            kid = payload.get("kernel_id")
            lines.append(f"Result: {status}" + (f" (kernel {kid})" if kid else ""))
        else:
            ok = payload.get("ok")
            lines.append(
                "Result: accepted"
                if ok
                else f"Result: rejected — {payload.get('error', '?')}"
            )
        for stream in ("stdout", "stderr"):
            value = payload.get(stream)
            if value and value.strip():
                lines.append(f"  {stream}:")
                lines.append(_indent(_truncate(value.rstrip(), _STREAM_LIMIT)))
        error = payload.get("error")
        if "success" in payload and error:
            lines.append(f"  error: {error}")
        return "\n".join(lines)

    plain = inner.strip()
    if len(plain) > _RESULT_TEXT_LIMIT:
        first_line = plain.splitlines()[0] if plain.splitlines() else ""
        return (
            f"Result: text, {len(plain):,} chars (full text in run log)\n  {first_line}"
        )
    return f"Result: {plain}" if "\n" not in plain else "Result:\n" + _indent(plain)


# Top-level keys of config.json (the nested format).
CONFIG_KEYS = ("model", "agent", "mcpServers", "files", "packages")
# model.* keys that go into the request's extra_body.
EXTRA_BODY_KEYS = ("reasoning", "provider", "top_k")


def _config_error(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def build_agent_config(data: dict[str, Any]) -> AgentConfig:
    """AgentConfig from a nested config.json (``model``/``agent`` objects).

    ``model.name`` is the model; ``model.reasoning``, ``model.provider``,
    ``model.top_k`` and the entries of an explicit ``model.extra_body``
    object go into ``extra_body`` (the named keys win on a clash);
    ``model.request_timeout`` becomes ``request_timeout``; every other
    ``model.*`` key is passed to the API call as is (``api_params``).
    A flat config (pre-0.3 format) or any unknown top-level key is an error.
    """
    unknown = sorted(set(data) - set(CONFIG_KEYS))
    model_cfg = data.get("model", {})
    agent_cfg = data.get("agent", {})
    if unknown or not isinstance(model_cfg, dict) or not isinstance(agent_cfg, dict):
        detail = (
            f"unknown top-level keys {unknown}"
            if unknown
            else "'model' and 'agent' must be objects"
        )
        _config_error(
            f"config.json is not in the nested format ({detail}). Use"
            ' {"model": {"name": ..., "temperature": ..., "max_tokens": ...},'
            ' "agent": {"max_steps": ..., "max_total_tokens": ...},'
            ' "mcpServers": {...}, "files": {"system": ...}, "packages": [...]};'
            " the flat keys modelstring/max_steps/temperature are no longer read."
        )

    model_cfg = dict(model_cfg)
    model_name = model_cfg.pop("name", _DEFAULTS.model)
    extra_body = model_cfg.pop("extra_body", None) or {}
    if not isinstance(extra_body, dict):
        _config_error("config.json: model.extra_body must be an object")
    extra_body = dict(extra_body)
    for key in EXTRA_BODY_KEYS:
        if key in model_cfg:
            extra_body[key] = model_cfg.pop(key)
    request_timeout = model_cfg.pop("request_timeout", None)
    max_result = agent_cfg.get("max_tool_result_chars", _DEFAULTS.max_tool_result_chars)
    if (
        isinstance(max_result, bool)
        or not isinstance(max_result, int)
        or max_result < 0
    ):
        _config_error(
            "config.json: agent.max_tool_result_chars must be an integer >= 0"
            f" (0 disables the cut), got {max_result!r}"
        )

    return AgentConfig(
        model=model_name,
        max_steps=agent_cfg.get("max_steps", _DEFAULTS.max_steps),
        api_params=model_cfg,  # temperature, max_tokens, etc.
        max_total_tokens=agent_cfg.get("max_total_tokens", _DEFAULTS.max_total_tokens),
        max_tool_result_chars=max_result,
        extra_body=extra_body or None,
        request_timeout=request_timeout,
    )


def load_run_folder(folder: Path) -> tuple[AgentConfig, dict[str, Any] | None, str]:
    """Load config and prompt from a run folder (nested config format only).

    Args:
        folder: Path to the run folder containing config.json, project.md, and task.md

    Returns:
        Tuple of (AgentConfig, mcp_servers config or None, combined prompt string)
    """
    config_path = folder / "config.json"
    project_path = folder / "project.md"
    task_path = folder / "task.md"

    # Load config
    if not config_path.exists():
        print(f"Error: config.json not found in {folder}", file=sys.stderr)
        sys.exit(1)

    try:
        with config_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {config_path}: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, dict):
        _config_error(f"{config_path} must hold a JSON object")

    config = build_agent_config(data)
    mcp_servers = data.get("mcpServers") or None
    files_cfg = data.get("files") or {}
    packages = data.get("packages") or []
    system_rel = files_cfg.get("system")

    # Optional system prompt file (files.system), resolved relative to folder.
    if system_rel:
        system_path = folder / system_rel
        if not system_path.exists():
            print(
                f"Error: system prompt file not found: {system_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        config.system_prompt = system_path.read_text(encoding="utf-8")

    # Optional kernel packages for python_exec auto-setup.
    config.packages = list(packages)

    # Load project.md (optional - general instructions)
    project_content = ""
    if project_path.exists():
        project_content = project_path.read_text(encoding="utf-8").strip()

    # Load task.md (required - specific task)
    if not task_path.exists():
        print(f"Error: task.md not found in {folder}", file=sys.stderr)
        sys.exit(1)

    task_content = task_path.read_text(encoding="utf-8").strip()
    if not task_content:
        print("Error: task.md is empty", file=sys.stderr)
        sys.exit(1)

    # Combine prompts: project instructions first, then task
    if project_content:
        prompt = f"{project_content}\n\n---\n\n{task_content}"
    else:
        prompt = task_content

    # Strip HTML comments (<!-- ... -->) from prompt
    prompt = strip_html_comments(prompt)

    return config, mcp_servers, prompt


async def run_agent_async(
    args: argparse.Namespace,
    config: AgentConfig,
    mcp_servers: dict[str, Any] | None,
    prompt: str,
    api_key: str,
) -> int:
    """Run the agent with MCP support; return the process exit code.

    The exit code is 1 when the run ended with an LLM API error
    (``AgentResult.error``), else 0. A valid submission is saved either way.
    """
    # Create logger (saves to run folder)
    logger = None if args.no_log else RunLogger(log_dir=args.folder)

    # The arithmetic demo tools only when no MCP server is configured.
    tools = None if mcp_servers else create_default_registry()

    # Start MCP servers if configured; their stderr (and their children's,
    # e.g. kernel processes) goes to a per-folder log, not the console.
    mcp_manager = None
    server_log = None
    if mcp_servers:
        server_log_path = args.folder / "server.log"
        # Deliberately no context manager: the file must outlive this block
        # (servers write to it until the finally below closes it).
        server_log = open(server_log_path, "w", encoding="utf-8")  # noqa: SIM115
        print(f"server stderr → {server_log_path}", file=sys.stderr)
        mcp_manager = MCPManager(mcp_servers, errlog=server_log)

    try:
        if mcp_manager:
            await mcp_manager.__aenter__()
            if args.verbose:
                mcp_tools = mcp_manager.get_tools()
                print(f"MCP tools: {[t.name for t in mcp_tools]}")

        agent = Agent(
            api_key=api_key,
            tools=tools,
            config=config,
            logger=logger,
            mcp_manager=mcp_manager,
        )

        # Run agent
        if args.verbose:
            print(f"Folder: {args.folder.resolve()}")
            print(f"Model: {config.model}")
            print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            if logger:
                print(f"Log: {logger.log_path}")
            print("-" * 40)

        result = await agent.run_async(prompt)

        # Output results
        if args.verbose:
            print("\n=== Agent Steps ===")
            for step in result.steps:
                print(f"\n--- Step {step['step']} ---")
                if step.get("content"):
                    print(f"Thought: {step['content']}")
                if step.get("tool_calls"):
                    for tc in step["tool_calls"]:
                        print(render_tool_call(tc["name"], tc["arguments"]))
                        print(
                            render_tool_result(tc["result"], tc.get("is_error", False))
                        )
            print("\n" + "=" * 40)
            print(f"Tool calls made: {result.tool_calls_made}")
            print("=" * 40)

        print("\n=== Final Answer ===")
        print(result.answer)

        # Persist the submitted program (if any) next to the run log.
        submission_path = save_submission(result.steps, args.folder)
        if submission_path:
            print(f"\nSubmission: {submission_path}")

        # Show log file location
        if logger:
            print(
                f"Log: {logger.log_path}"
                if submission_path
                else f"\nLog: {logger.log_path}"
            )

        if result.error:
            print(f"Error: agent run failed: {result.error}", file=sys.stderr)
            return 1
        return 0

    except Exception as e:
        print(f"Error running agent: {e}", file=sys.stderr)
        if logger:
            print(f"Partial log saved to: {logger.log_path}", file=sys.stderr)
        raise
    finally:
        if mcp_manager:
            await mcp_manager.__aexit__(None, None, None)
        if server_log:
            server_log.close()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mcp-minion",
        description="A minimal ReAct agent over MCP servers using OpenRouter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Run Folder Structure:
  config.json   Model and agent configuration
                - model.name: OpenRouter model ID (e.g., "google/gemini-3-flash-preview")
                - model.temperature, model.max_tokens: API parameters
                - model.reasoning, model.provider, model.top_k,
                  model.extra_body: sent in the request body (extra_body)
                - model.request_timeout: LLM request timeout in seconds
                - agent.max_steps: Maximum reasoning steps (default: 10)
                - agent.max_tool_result_chars: Longer tool results are cut
                  (head + tail) for the model (default: 64000, 0 = never)
                - agent.max_total_tokens: Per-run cap on cumulative
                  input+output tokens; the run stops after the step that
                  exceeds it (default: absent = unlimited)
                - mcpServers: Optional MCP servers, each with command, args,
                  env, cwd (a server that fails to start aborts the run;
                  without servers the arithmetic demo tools are offered)
                - files.system: Optional path to a system-prompt markdown file
                - packages: Optional list of packages for python_exec kernels
  project.md    General instructions (optional, shared context)
  task.md       Specific task for this run (required)
  run_*.json    Log files (created automatically, gitignored)
  submission.py Last accepted submit_code program (if any)
  server.log    MCP servers' stderr (when servers are configured)

Environment:
  OPENROUTER_API_KEY  Required. An already-set environment variable wins;
                      otherwise it is loaded from FOLDER/.env (run folder),
                      then ~/.mcp-minion.

Model Parameters (model.* in config.json):
  temperature    0 for deterministic (recommended for agents)
  max_tokens     Max output tokens (model-dependent)

Examples:
  mcp-minion                      Run agent in current directory
  mcp-minion runs/math_task       Run agent from specific folder
  mcp-minion runs/math_task -v    Show step-by-step reasoning
  mcp-minion --no-log .           Run without saving logs

More info: https://github.com/szeider/mcp-solver""",
    )
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=Path("."),
        metavar="FOLDER",
        help="run folder with config.json + task.md (default: .)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show detailed step-by-step output",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="disable logging to run_*.json",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    # Validate folder exists
    if not args.folder.is_dir():
        print(f"Error: {args.folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Environment first, then FOLDER/.env, then ~/.mcp-minion.
    api_key = find_api_key(folder=args.folder)
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found", file=sys.stderr)
        print(
            "Create ~/.mcp-minion with: OPENROUTER_API_KEY=sk-...",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load config and prompt from folder
    config, mcp_servers, prompt = load_run_folder(args.folder)

    # Run async
    try:
        exit_code = asyncio.run(
            run_agent_async(args, config, mcp_servers, prompt, api_key)
        )
    except Exception:
        sys.exit(1)
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
