"""Helpers for extracting artifacts from an agent run.

The main entry point, :func:`extract_last_submission`, recovers the code that
an agent submitted through a designated tool (e.g. ``submit_code``) so the CLI
can write it to a ``<basename>_code.py`` file next to the run folder.
"""

from __future__ import annotations

import json
from typing import Any


def _json_loads(value: Any) -> Any:
    """``value`` parsed as JSON if it is a string, else unchanged; None on failure."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, RecursionError):
        return None


def _rejected(call: dict[str, Any]) -> bool:
    """Return True if a recorded tool call did not succeed.

    Two record shapes exist, told apart by the presence of the
    ``is_error`` key:

    * current (mcp-minion >= 0.3): ``{"result": <text>, "is_error": <bool>}``.
      The call failed if ``is_error`` is true, or if the text is a JSON
      object with ``"ok": false`` (the kernel server's ``submit_code``
      reports a rejected program as a normal result
      ``{"ok": false, "error": "Syntax error ..."}``).
    * old (run logs of mcp-minion <= 0.2, no ``is_error`` key): ``result`` is
      a JSON envelope ``{"result": <text>}`` or ``{"error": ...}``. The call
      failed on an ``error`` envelope or when the inner text is a JSON object
      with ``"ok": false``; a non-JSON string counts as success.

    A missing result counts as unsuccessful in both shapes.
    """
    result = call.get("result")
    if result is None:
        return True
    if "is_error" in call:
        if call["is_error"]:
            return True
        payload = _json_loads(result)
        return isinstance(payload, dict) and payload.get("ok") is False

    envelope = _json_loads(result)
    if not isinstance(envelope, dict):
        return False
    if "error" in envelope:
        return True
    inner = _json_loads(envelope.get("result"))
    return isinstance(inner, dict) and inner.get("ok") is False


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
            entries have ``"name"``, ``"arguments"``, ``"result"`` and (since
            0.3) ``"is_error"`` keys; old run logs without ``is_error`` are
            read too (see ``_rejected``).
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
            if _rejected(call):
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
