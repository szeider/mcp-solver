"""Helpers for extracting artifacts from an agent run.

The main entry point, :func:`extract_last_submission`, recovers the code that
an agent submitted through a designated tool (e.g. ``submit_code``) so the CLI
can write it to a ``<basename>_code.py`` file next to the run folder.
"""

from __future__ import annotations

import json
from typing import Any


def _is_error_result(result: Any) -> bool:
    """Return True if a tool result represents an error.

    Tool results are the JSON strings produced by ``MCPManager.call_tool`` /
    ``Tool.execute``: ``{"result": ...}`` on success, ``{"error": ...}`` on
    failure. A missing result counts as unsuccessful; a non-JSON string is
    treated as a (successful) plain result.
    """
    if result is None:
        return True
    if isinstance(result, dict):
        return "error" in result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return False
        return isinstance(parsed, dict) and "error" in parsed
    return False


def _iter_steps(log_or_steps: Any) -> list[dict[str, Any]]:
    """Normalise the input into a list of step dicts."""
    if isinstance(log_or_steps, dict):
        steps = log_or_steps.get("steps", [])
    else:
        steps = log_or_steps
    return list(steps or [])


def extract_last_submission(
    log_or_steps: Any,
    tool_name: str = "submit_code",
    code_key: str = "code",
) -> str | None:
    """Return the code argument of the LAST successful call to ``tool_name``.

    Args:
        log_or_steps: Either a run-log dict (with a ``"steps"`` list) or a list
            of step dicts, as found in ``AgentResult.steps`` or a run log's
            ``steps`` field. Each step may carry a ``"tool_calls"`` list whose
            entries have ``"name"``, ``"arguments"`` and ``"result"`` keys.
        tool_name: Name of the submission tool to look for.
        code_key: The argument key holding the submitted code.

    Returns:
        The code string from the last successful (non-error) call, or None if
        there was no such call.
    """
    last: str | None = None
    for step in _iter_steps(log_or_steps):
        for call in step.get("tool_calls") or []:
            if call.get("name") != tool_name:
                continue
            if _is_error_result(call.get("result")):
                continue
            args = call.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(args, dict):
                continue
            code = args.get(code_key)
            if code is not None:
                last = code
    return last
