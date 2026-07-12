"""CLI entry point for the ReAct agent."""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mcp_minion import __version__
from mcp_minion.agent import Agent, AgentConfig
from mcp_minion.logging import RunLogger
from mcp_minion.mcp_client import MCPManager
from mcp_minion.tools import create_default_registry

# Use AgentConfig defaults as single source of truth
_DEFAULTS = AgentConfig()


def strip_html_comments(text: str) -> str:
    """Remove HTML comments (<!-- ... -->) from text."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def load_run_folder(folder: Path) -> tuple[AgentConfig, dict[str, Any] | None, str]:
    """Load config and prompt from a run folder.

    Supports both old flat format and new nested format.

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

    # Optional top-level keys shared by both formats.
    files_cfg = data.get("files") or {}
    packages = data.get("packages") or []
    system_rel = files_cfg.get("system")

    # Detect format: new has nested "model" dict, old has flat "modelstring"
    if "model" in data and isinstance(data["model"], dict):
        # New format
        model_cfg = data.get("model", {})
        agent_cfg = data.get("agent", {})
        mcp_servers = data.get("mcpServers")

        # Extract model name, rest goes to api_params
        model_name = model_cfg.pop("name", _DEFAULTS.model)
        config = AgentConfig(
            model=model_name,
            max_steps=agent_cfg.get("max_steps", _DEFAULTS.max_steps),
            api_params=model_cfg,  # temperature, max_tokens, etc.
        )
    else:
        # Old flat format (backward compat)
        model = data.pop("modelstring", _DEFAULTS.model)
        max_steps = data.pop("max_steps", _DEFAULTS.max_steps)
        # Keep shared/structural keys out of api_params — everything left in
        # `data` goes verbatim into the API request.
        for key in ("files", "packages", "mcpServers", "model", "agent"):
            data.pop(key, None)
        config = AgentConfig(model=model, max_steps=max_steps, api_params=data)
        mcp_servers = None

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
) -> None:
    """Run the agent with MCP support."""
    # Create logger (saves to run folder)
    logger = None if args.no_log else RunLogger(log_dir=args.folder)

    # Create tool registry with built-in tools
    tools = create_default_registry()

    # Start MCP servers if configured
    mcp_manager = None
    if mcp_servers:
        mcp_manager = MCPManager(mcp_servers)

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
                        print(f"Tool: {tc['name']}({tc['arguments']})")
                        print(f"Result: {tc['result']}")
            print("\n" + "=" * 40)
            print(f"Tool calls made: {result.tool_calls_made}")
            print("=" * 40)

        print("\n=== Final Answer ===")
        print(result.answer)

        # Show log file location
        if logger:
            print(f"\nLog: {logger.log_path}")

    except Exception as e:
        print(f"Error running agent: {e}", file=sys.stderr)
        if logger:
            print(f"Partial log saved to: {logger.log_path}", file=sys.stderr)
        raise
    finally:
        if mcp_manager:
            await mcp_manager.__aexit__(None, None, None)


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
                - agent.max_steps: Maximum reasoning steps (default: 10)
                - mcpServers: Optional MCP server configurations
                - files.system: Optional path to a system-prompt markdown file
                - packages: Optional list of packages for python_exec kernels
  project.md    General instructions (optional, shared context)
  task.md       Specific task for this run (required)
  run_*.json    Log files (created automatically, gitignored)

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

    # Load environment variables (check run folder, then ~/.mcp-minion)
    load_dotenv(args.folder / ".env")  # Run folder .env
    load_dotenv(Path.home() / ".mcp-minion")  # Fallback to ~/.mcp-minion

    api_key = os.getenv("OPENROUTER_API_KEY")
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
        asyncio.run(run_agent_async(args, config, mcp_servers, prompt, api_key))
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
