"""Unit tests for artifact extraction."""

import json

from mcp_minion.artifacts import extract_last_submission

# The tests up to TestNewShape use the OLD record shape (mcp-minion <= 0.2:
# result is a {"result": ...}/{"error": ...} envelope, no is_error key), which
# extract_last_submission still reads so that old run logs extract.


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


class TestRejectedSubmissionsOldShape:
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


def _new(code: str, text: str, is_error: bool = False) -> dict:
    return {
        "name": "submit_code",
        "arguments": {"code": code},
        "result": text,
        "is_error": is_error,
    }


class TestNewShape:
    """Current records: result is the plain text, is_error a bool."""

    def test_accepted_plain_text(self) -> None:
        steps = [{"tool_calls": [_new("v1", "accepted")]}]
        assert extract_last_submission(steps) == "v1"

    def test_ok_true_accepted(self) -> None:
        steps = [{"tool_calls": [_new("v1", json.dumps({"ok": True}))]}]
        assert extract_last_submission(steps) == "v1"

    def test_is_error_skipped(self) -> None:
        steps = [
            {"tool_calls": [_new("good", '{"ok": true}')]},
            {"tool_calls": [_new("bad", "Tool 'submit_code' timed out", True)]},
        ]
        assert extract_last_submission(steps) == "good"

    def test_ok_false_skipped(self) -> None:
        rejected = json.dumps({"ok": False, "error": "Syntax error"})
        steps = [
            {"tool_calls": [_new("good", '{"ok": true}')]},
            {"tool_calls": [_new("def (:", rejected)]},
        ]
        assert extract_last_submission(steps) == "good"

    def test_error_key_in_text_is_not_an_envelope(self) -> None:
        # In the new shape only is_error and ok:false reject; a text that
        # happens to be {"error": ...} is the server's own payload.
        steps = [{"tool_calls": [_new("v", json.dumps({"error": None, "ok": True}))]}]
        assert extract_last_submission(steps) == "v"

    def test_old_and_new_shapes_mixed(self) -> None:
        steps = [
            {"tool_calls": [_call("submit_code", "old", _SUBMIT_OK)]},
            {"tool_calls": [_new("new", '{"ok": true}')]},
            {"tool_calls": [_call("submit_code", "old-bad", _SUBMIT_REJECTED)]},
        ]
        assert extract_last_submission(steps) == "new"

    def test_from_agent_step_records(self) -> None:
        import asyncio
        from types import SimpleNamespace

        from mcp_minion import Agent, ToolOutcome

        class Manager:
            def has_tool(self, name):
                return name == "submit_code"

            def get_tools(self):
                return []

            def get_resources(self):
                return []

            def get_instructions(self):
                return {}

            def get_openai_tools(self):
                return [{"type": "function", "function": {"name": "submit_code"}}]

            async def call_tool(self, name, arguments):
                ok = "def" not in arguments["code"]
                return ToolOutcome(text=json.dumps({"ok": ok}))

        def response(content=None, code=None):
            calls = None
            if code is not None:
                calls = [
                    SimpleNamespace(
                        id="c",
                        function=SimpleNamespace(
                            name="submit_code", arguments=json.dumps({"code": code})
                        ),
                    )
                ]
            msg = SimpleNamespace(content=content, tool_calls=calls)
            return SimpleNamespace(
                id="r",
                model="m",
                choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                usage=None,
            )

        queue = [response(code="print(1)"), response(code="def ("), response("done")]

        async def create(**kwargs):
            return queue.pop(0)

        agent = Agent(api_key="k", tools=None, mcp_manager=Manager())
        agent.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        result = asyncio.run(agent.run_async("go"))
        assert extract_last_submission(result.steps) == "print(1)"
