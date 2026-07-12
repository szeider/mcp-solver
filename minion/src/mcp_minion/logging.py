"""Run logging for the ReAct agent.

Produces JSON log files with complete, non-truncated records of each agent run.
Writes incrementally to preserve partial logs even if the agent crashes.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Keys managed by the agent itself: filtered out of user api_params before an
# API call, and excluded from the logged request's api_params. Shared with
# agent.py so the two views cannot drift.
RESERVED_REQUEST_KEYS = frozenset(
    ("model", "messages", "tools", "tool_choice", "extra_headers", "extra_body")
)


@dataclass
class RunLogger:
    """Logger that writes agent runs to JSON files incrementally.

    Writes after each significant event to preserve partial logs on crash.
    """

    log_dir: Path
    run_id: str = field(default="")
    log_path: Path = field(default=Path())
    _log_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize log directory and file."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if not self.run_id:
            self.run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        self.log_path = self.log_dir / f"run_{self.run_id}.json"
        self._log_data = {
            "run_id": self.run_id,
            "started_at": self._timestamp(),
            "completed_at": None,
            "status": "running",
            "config": {},
            "prompt": None,
            "steps": [],
            "token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
            },
            "error": None,
            "result": None,
        }
        self._write()

    @staticmethod
    def _timestamp() -> str:
        """Return ISO format timestamp with timezone."""
        return datetime.now(UTC).isoformat()

    def _write(self) -> None:
        """Write current log data to file."""
        self.log_path.write_text(
            json.dumps(self._log_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def log_config(self, config: dict[str, Any]) -> None:
        """Log agent configuration."""
        self._log_data["config"] = config
        self._write()

    def log_prompt(self, prompt: str) -> None:
        """Log the user prompt."""
        self._log_data["prompt"] = prompt
        self._write()

    def log_step_start(self, step_num: int) -> None:
        """Log the start of a new step."""
        self._log_data["steps"].append(
            {
                "step": step_num,
                "started_at": self._timestamp(),
                "completed_at": None,
                "api_request": None,
                "api_response": None,
                "tool_calls": [],
                "content": None,
            }
        )
        self._write()

    def log_api_request(self, request: dict[str, Any]) -> None:
        """Log an API request (for the current step)."""
        if self._log_data["steps"]:
            # Sanitize request - remove sensitive headers but keep structure.
            # The full messages list is NOT stored: it grows by one entry per
            # step, so per-step snapshots would make the log quadratic. The
            # conversation is reconstructable from prompt + per-step content
            # and tool calls.
            sanitized = {
                "model": request.get("model"),
                "message_count": len(request.get("messages") or []),
                "tools": request.get("tools"),
                "tool_choice": request.get("tool_choice"),
                "api_params": {
                    k: v for k, v in request.items() if k not in RESERVED_REQUEST_KEYS
                },
            }
            self._log_data["steps"][-1]["api_request"] = sanitized
            self._write()

    def log_api_response(self, response: dict[str, Any]) -> None:
        """Log an API response (for the current step)."""
        if self._log_data["steps"]:
            self._log_data["steps"][-1]["api_response"] = response
            self._log_data["steps"][-1]["completed_at"] = self._timestamp()
            self._write()

    def log_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate token usage from an API call."""
        self._log_data["token_usage"]["input_tokens"] += input_tokens
        self._log_data["token_usage"]["output_tokens"] += output_tokens
        self._write()

    def log_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: str,
        tool_call_id: str,
    ) -> None:
        """Log a tool call with its input and output."""
        if self._log_data["steps"]:
            self._log_data["steps"][-1]["tool_calls"].append(
                {
                    "timestamp": self._timestamp(),
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "arguments": tool_args,
                    "result": tool_result,
                }
            )
            self._write()

    def log_step_content(self, content: str | None) -> None:
        """Log the assistant's content for the current step."""
        if self._log_data["steps"]:
            self._log_data["steps"][-1]["content"] = content
            self._write()

    def log_error(self, error: str, traceback: str | None = None) -> None:
        """Log an error."""
        self._log_data["error"] = {
            "timestamp": self._timestamp(),
            "message": error,
            "traceback": traceback,
        }
        self._log_data["status"] = "error"
        self._log_data["completed_at"] = self._timestamp()
        self._write()

    def log_completion(self, result: dict[str, Any]) -> None:
        """Log successful completion."""
        self._log_data["result"] = result
        self._log_data["status"] = "completed"
        self._log_data["completed_at"] = self._timestamp()
        self._write()
