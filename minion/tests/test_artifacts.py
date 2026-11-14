"""Unit tests for artifact extraction."""

import json

from mcp_minion.artifacts import extract_last_submission


def _ok(payload: str = "done") -> str:
    return json.dumps({"result": payload})


def _err(msg: str = "boom") -> str:
    return json.dumps({"error": msg})


def _call(name: str, code: str | None, result: str) -> dict:
    args = {} if code is None else {"code": code}
    return {"name": name, "arguments": args, "result": result}


class TestExtractLastSubmission:
    def test_none_when_no_calls(self) -> None:
        steps = [{"tool_calls": []}, {"tool_calls": None}]
        assert extract_last_submission(steps) is None

    def test_returns_last_successful(self) -> None:
        steps = [
            {"tool_calls": [_call("submit_code", "v1", _ok())]},
            {"tool_calls": [_call("submit_code", "v2", _ok())]},
        ]
        assert extract_last_submission(steps) == "v2"

    def test_skips_trailing_error(self) -> None:
        """A failed final submission should not shadow the last good one."""
        steps = [
            {"tool_calls": [_call("submit_code", "good", _ok())]},
            {"tool_calls": [_call("submit_code", "bad", _err())]},
        ]
        assert extract_last_submission(steps) == "good"

    def test_ignores_other_tools(self) -> None:
        steps = [
            {"tool_calls": [_call("python_exec", "noise", _ok())]},
            {"tool_calls": [_call("submit_code", "real", _ok())]},
        ]
        assert extract_last_submission(steps) == "real"

    def test_accepts_log_dict(self) -> None:
        log = {"steps": [{"tool_calls": [_call("submit_code", "x", _ok())]}]}
        assert extract_last_submission(log) == "x"

    def test_custom_tool_name(self) -> None:
        steps = [{"tool_calls": [_call("finish", "z", _ok())]}]
        assert extract_last_submission(steps, tool_name="finish") == "z"
        assert extract_last_submission(steps) is None

    def test_all_errors_returns_none(self) -> None:
        steps = [
            {"tool_calls": [_call("submit_code", "a", _err())]},
            {"tool_calls": [_call("submit_code", "b", _err())]},
        ]
        assert extract_last_submission(steps) is None

    def test_multiple_calls_in_one_step(self) -> None:
        steps = [
            {
                "tool_calls": [
                    _call("submit_code", "first", _ok()),
                    _call("submit_code", "second", _ok()),
                ]
            }
        ]
        assert extract_last_submission(steps) == "second"

    def test_arguments_as_json_string(self) -> None:
        call = {
            "name": "submit_code",
            "arguments": json.dumps({"code": "stringified"}),
            "result": _ok(),
        }
        assert extract_last_submission([{"tool_calls": [call]}]) == "stringified"


# Real result shapes of the kernel server's submit_code, as wrapped by
# MCPManager.call_tool: the server reports a rejected submission as a NORMAL
# result whose text is {"ok": false, "error": ...}.
_SUBMIT_OK = json.dumps({"result": json.dumps({"ok": True})})
_SUBMIT_REJECTED = json.dumps(
    {
        "result": json.dumps(
            {"ok": False, "error": "Syntax error at line 1: invalid syntax"}
        )
    }
)


class TestRejectedSubmissions:
    def test_good_then_rejected_returns_good(self) -> None:
        steps = [
            {"tool_calls": [_call("submit_code", "good", _SUBMIT_OK)]},
            {"tool_calls": [_call("submit_code", "def (:", _SUBMIT_REJECTED)]},
        ]
        assert extract_last_submission(steps) == "good"

    def test_only_rejected_returns_none(self) -> None:
        steps = [{"tool_calls": [_call("submit_code", "def (:", _SUBMIT_REJECTED)]}]
        assert extract_last_submission(steps) is None

    def test_rejected_then_good_returns_good(self) -> None:
        steps = [
            {"tool_calls": [_call("submit_code", "def (:", _SUBMIT_REJECTED)]},
            {"tool_calls": [_call("submit_code", "good", _SUBMIT_OK)]},
        ]
        assert extract_last_submission(steps) == "good"

    def test_result_without_ok_key_counts_as_success(self) -> None:
        result = json.dumps({"result": json.dumps({"status": "stored"})})
        steps = [{"tool_calls": [_call("submit_code", "v", result)]}]
        assert extract_last_submission(steps) == "v"

    def test_plain_text_result_counts_as_success(self) -> None:
        steps = [{"tool_calls": [_call("submit_code", "v", _ok("accepted"))]}]
        assert extract_last_submission(steps) == "v"

    def test_rejected_result_as_dict_is_skipped(self) -> None:
        call = {
            "name": "submit_code",
            "arguments": {"code": "bad"},
            "result": {"result": {"ok": False, "error": "x"}},
        }
        assert extract_last_submission([{"tool_calls": [call]}]) is None
