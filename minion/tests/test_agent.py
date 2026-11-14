"""Unit tests for agent module."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from mcp_minion.agent import Agent, AgentConfig, AgentResult


class TestAgentConfig:
    """Tests for the AgentConfig dataclass."""

    def test_defaults(self) -> None:
        config = AgentConfig()
        assert config.model == "google/gemini-3-flash-preview"
        assert config.max_steps == 10
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.api_params == {}
        # 0 = no per-run token cap.
        assert config.max_total_tokens == 0

    def test_custom_values(self) -> None:
        config = AgentConfig(
            model="openai/gpt-4",
            max_steps=5,
            api_params={"temperature": 0.5},
        )
        assert config.model == "openai/gpt-4"
        assert config.max_steps == 5
        assert config.api_params == {"temperature": 0.5}

    def test_asdict(self) -> None:
        config = AgentConfig(model="test-model", max_steps=3)
        d = asdict(config)
        assert d["model"] == "test-model"
        assert d["max_steps"] == 3
        assert "base_url" in d
        assert "api_params" in d


class TestAgentResult:
    """Tests for the AgentResult dataclass."""

    def test_defaults(self) -> None:
        result = AgentResult(answer="42")
        assert result.answer == "42"
        assert result.steps == []
        assert result.tool_calls_made == 0

    def test_with_steps(self) -> None:
        steps = [
            {"step": 1, "content": "Thinking..."},
            {"step": 2, "content": "Done"},
        ]
        result = AgentResult(answer="Done", steps=steps, tool_calls_made=3)
        assert result.answer == "Done"
        assert len(result.steps) == 2
        assert result.tool_calls_made == 3

    def test_empty_answer(self) -> None:
        result = AgentResult(answer="")
        assert result.answer == ""

    def test_new_field_defaults(self) -> None:
        result = AgentResult(answer="x")
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.max_steps_reached is False
        assert result.token_cap_reached is False


# --- run_async behavior with a fake OpenAI client --------------------------


def _response(
    content=None,
    tool_calls=None,
    prompt=7,
    completion=3,
    cost=None,
    finish_reason="stop",
):
    """Build a minimal chat-completions response object."""
    return SimpleNamespace(
        id="resp-1",
        model="fake-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cost=cost,
        ),
    )


def _tool_call(name="add", arguments='{"a": 1, "b": 2}'):
    return SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeClient:
    """Stands in for the OpenAI client; returns canned responses in order.

    An exception instance in the list is raised instead of returned.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _make_agent(responses, config=None, on_step=None, logger=None):
    agent = Agent(
        api_key="test-key",
        tools=None,
        config=config or AgentConfig(),
        logger=logger,
        on_step=on_step,
    )
    agent.client = FakeClient(responses)
    return agent


async def test_token_accumulation_without_logger() -> None:
    agent = _make_agent(
        [
            _response(tool_calls=[_tool_call()], prompt=10, completion=2),
            _response(content="done", prompt=20, completion=4),
        ]
    )
    result = await agent.run_async("hi")
    assert result.answer == "done"
    assert result.input_tokens == 30
    assert result.output_tokens == 6
    # No cost reported by the provider -> stays 0.0.
    assert result.cost == 0.0


async def test_cost_accumulation_from_usage_accounting() -> None:
    agent = _make_agent(
        [
            _response(tool_calls=[_tool_call()], cost=0.0021),
            _response(content="done", cost=0.0034),
        ]
    )
    result = await agent.run_async("hi")
    assert result.answer == "done"
    assert result.cost == pytest.approx(0.0055)
    assert result.max_steps_reached is False


async def test_on_step_called_per_step() -> None:
    seen: list[dict] = []
    agent = _make_agent(
        [
            _response(tool_calls=[_tool_call()]),
            _response(content="done"),
        ],
        on_step=seen.append,
    )
    await agent.run_async("hi")
    assert [s["step"] for s in seen] == [1, 2]
    assert seen[0]["tool_calls"] is not None
    assert seen[1]["tool_calls"] is None


async def test_max_steps_reached_flag() -> None:
    agent = _make_agent(
        [_response(tool_calls=[_tool_call()])],
        config=AgentConfig(max_steps=1),
    )
    result = await agent.run_async("hi")
    assert result.max_steps_reached is True
    assert "maximum steps" in result.answer
    assert result.input_tokens == 7


# --- per-run token cap -------------------------------------------------------


async def test_token_cap_stops_run_after_step() -> None:
    # Step 1 spends 12 tokens (> cap 10), so the run stops after that step
    # instead of asking the model again.
    agent = _make_agent(
        [
            _response(tool_calls=[_tool_call()], prompt=10, completion=2),
            _response(content="never reached", prompt=99, completion=99),
        ],
        config=AgentConfig(max_steps=5, max_total_tokens=10),
    )
    result = await agent.run_async("hi")
    assert result.token_cap_reached is True
    assert result.max_steps_reached is False
    assert "per-run token cap of 10 exceeded (cumulative 12)" in result.answer
    assert len(result.steps) == 1
    assert result.input_tokens == 10
    assert result.output_tokens == 2


async def test_token_cap_not_reached_lets_run_finish() -> None:
    agent = _make_agent(
        [
            _response(tool_calls=[_tool_call()], prompt=10, completion=2),
            _response(content="done", prompt=10, completion=2),
        ],
        config=AgentConfig(max_steps=5, max_total_tokens=1000),
    )
    result = await agent.run_async("hi")
    assert result.answer == "done"
    assert result.token_cap_reached is False


async def test_token_cap_zero_is_unlimited() -> None:
    agent = _make_agent(
        [
            _response(tool_calls=[_tool_call()], prompt=10_000, completion=10_000),
            _response(content="done", prompt=10_000, completion=10_000),
        ],
        config=AgentConfig(max_steps=5, max_total_tokens=0),
    )
    result = await agent.run_async("hi")
    assert result.answer == "done"
    assert result.token_cap_reached is False


async def test_token_cap_recorded_in_run_log(tmp_path) -> None:
    import json as json_mod

    from mcp_minion.logging import RunLogger

    logger = RunLogger(log_dir=tmp_path)
    agent = _make_agent(
        [
            _response(tool_calls=[_tool_call()], prompt=10, completion=2),
            _response(content="never reached"),
        ],
        config=AgentConfig(max_steps=5, max_total_tokens=10),
        logger=logger,
    )
    await agent.run_async("hi")

    log = json_mod.loads(logger.log_path.read_text())
    # Terminal status is recorded the same way as any other finished run.
    assert log["status"] == "completed"
    assert log["result"]["token_cap_reached"] is True
    assert log["result"]["max_steps_reached"] is False
    assert log["result"]["cumulative_tokens"] == 12
    # The step that tripped the cap carries the details.
    assert log["steps"][-1]["token_cap_reached"] is True
    assert log["steps"][-1]["cumulative_tokens"] == 12
    assert log["steps"][-1]["token_cap"] == 10


# --- read_resource built-in tool --------------------------------------------


class FakeResourceManager:
    """MCP manager stub exposing resources but no read_resource tool."""

    def __init__(self):
        self.read_uris: list[str] = []

    def has_tool(self, name: str) -> bool:
        return False

    def get_tools(self):
        return []

    def get_resources(self):
        from mcp_minion.mcp_client import MCPResourceInfo

        return [
            MCPResourceInfo(
                uri="app://guide",
                name="guide",
                description="How to choose.\nMore detail.",
                mime_type="text/markdown",
                server_name="app",
            )
        ]

    async def read_resource(self, uri: str) -> str:
        self.read_uris.append(uri)
        return "guide text"


async def test_read_resource_tool_routes_to_manager() -> None:
    import json

    mcp = FakeResourceManager()
    agent = Agent(api_key="k", tools=None, mcp_manager=mcp)
    result = await agent._execute_tool("read_resource", {"uri": "app://guide"})
    assert json.loads(result) == {"result": "guide text"}
    assert mcp.read_uris == ["app://guide"]


async def test_read_resource_missing_uri_is_error() -> None:
    import json

    agent = Agent(api_key="k", tools=None, mcp_manager=FakeResourceManager())
    result = await agent._execute_tool("read_resource", {})
    assert "error" in json.loads(result)


def test_tool_sections_list_resources() -> None:
    agent = Agent(api_key="k", tools=None, mcp_manager=FakeResourceManager())
    sections = agent._build_tool_sections()
    assert "## Resources" in sections
    assert "`app://guide` (guide): How to choose." in sections
    assert "read_resource" in sections


# --- LLM error responses, truncation and API failures ------------------------


def _error_envelope():
    """OpenRouter's HTTP-200 error body: no choices, an `error` object."""
    return SimpleNamespace(
        id="resp-err",
        model="fake-model",
        choices=None,
        usage=None,
        model_extra={"error": {"code": 502, "message": "upstream error"}},
    )


@pytest.fixture
def no_backoff(monkeypatch):
    import mcp_minion.agent as agent_mod

    monkeypatch.setattr(agent_mod, "API_ERROR_BACKOFF_SECONDS", 0)


async def test_error_response_is_retried(no_backoff) -> None:
    agent = _make_agent(
        [
            _error_envelope(),
            _response(content="oops", finish_reason="error"),
            _response(content="done"),
        ]
    )
    result = await agent.run_async("hi")
    assert result.answer == "done"
    assert result.error is None
    assert len(agent.client.requests) == 3


async def test_persistent_error_finish_reason_is_not_success(no_backoff) -> None:
    """A reply with finish_reason 'error' must not become the final answer."""
    from mcp_minion.agent import API_ERROR_MAX_RETRIES

    agent = _make_agent(
        [_response(content="partial", finish_reason="error")]
        * (API_ERROR_MAX_RETRIES + 1)
    )
    result = await agent.run_async("hi")
    assert result.error is not None
    assert "LLMResponseError" in result.error
    assert "finish_reason 'error'" in result.error
    assert result.answer != "partial"


async def test_api_error_keeps_steps_so_far(no_backoff, tmp_path) -> None:
    import json as json_mod

    from mcp_minion.agent import API_ERROR_MAX_RETRIES
    from mcp_minion.logging import RunLogger

    logger = RunLogger(log_dir=tmp_path)
    agent = _make_agent(
        [_response(tool_calls=[_tool_call()], prompt=10, completion=2, cost=0.01)]
        + [_error_envelope()] * (API_ERROR_MAX_RETRIES + 1),
        config=AgentConfig(max_steps=5),
        logger=logger,
    )
    result = await agent.run_async("hi")
    assert result.error is not None
    assert result.error.startswith("LLMResponseError: ")
    assert "\n" not in result.error
    assert len(result.steps) == 1
    assert result.tool_calls_made == 1
    assert result.input_tokens == 10
    assert result.output_tokens == 2
    assert result.cost == pytest.approx(0.01)
    log = json_mod.loads(logger.log_path.read_text())
    assert log["status"] == "error"


async def test_openai_exception_returns_partial_result() -> None:
    import httpx
    import openai

    exc = openai.APIConnectionError(
        message="Connection error.",
        request=httpx.Request("POST", "https://openrouter.ai/api/v1"),
    )
    agent = _make_agent(
        [_response(tool_calls=[_tool_call()], prompt=10, completion=2), exc],
        config=AgentConfig(max_steps=5),
    )
    result = await agent.run_async("hi")
    assert result.error == "APIConnectionError: Connection error."
    assert len(result.steps) == 1
    assert result.input_tokens == 10


def test_sync_run_returns_partial_result_on_api_error() -> None:
    import openai

    agent = _make_agent([openai.OpenAIError("no key")])
    result = agent.run("hi")
    assert result.error == "OpenAIError: no key"
    assert result.steps == []


async def test_programming_error_still_raises() -> None:
    agent = _make_agent([TypeError("bug in minion")])
    with pytest.raises(TypeError):
        await agent.run_async("hi")


async def test_truncated_reply_is_not_final_answer() -> None:
    agent = _make_agent(
        [
            _response(content="The answer is", finish_reason="length"),
            _response(content="done"),
        ],
        config=AgentConfig(max_steps=5),
    )
    result = await agent.run_async("hi")
    assert result.answer == "done"
    assert result.error is None
    assert result.steps[0]["truncated"] is True
    assert len(result.steps) == 2
    # The model was told to continue.
    second_messages = agent.client.requests[1]["messages"]
    assert second_messages[-2] == {"role": "assistant", "content": "The answer is"}
    assert "truncated" in second_messages[-1]["content"]


async def test_truncation_until_max_steps_is_not_success() -> None:
    agent = _make_agent(
        [_response(content="cut", finish_reason="length")] * 2,
        config=AgentConfig(max_steps=2),
    )
    result = await agent.run_async("hi")
    assert result.max_steps_reached is True
    assert result.answer != "cut"


# --- malformed tool-call arguments -------------------------------------------


def _recording_registry():
    from mcp_minion.tools import Tool, ToolRegistry

    seen: list[dict] = []

    def ping(**kwargs):
        seen.append(kwargs)
        return "pong"

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="ping",
            description="ping",
            parameters={"type": "object", "properties": {}},
            function=ping,
        )
    )
    return registry, seen


@pytest.mark.parametrize("raw", ["", None, "  "])
async def test_empty_tool_arguments_become_empty_dict(raw) -> None:
    import json

    registry, seen = _recording_registry()
    agent = _make_agent(
        [
            _response(tool_calls=[_tool_call(name="ping", arguments=raw)]),
            _response(content="done"),
        ]
    )
    agent.tools = registry
    result = await agent.run_async("hi")
    assert result.answer == "done"
    assert seen == [{}]
    call = result.steps[0]["tool_calls"][0]
    assert call["arguments"] == {}
    assert json.loads(call["result"]) == {"result": "pong"}
    # The echoed assistant message never carries null arguments.
    echoed = agent.client.requests[1]["messages"][2]["tool_calls"][0]
    assert echoed["function"]["arguments"] == "{}"


@pytest.mark.parametrize("raw", ["[1, 2]", "42", '"text"', "null"])
async def test_non_object_tool_arguments_are_error_result(raw) -> None:
    import json

    registry, seen = _recording_registry()
    agent = _make_agent(
        [
            _response(tool_calls=[_tool_call(name="ping", arguments=raw)]),
            _response(content="done"),
        ]
    )
    agent.tools = registry
    result = await agent.run_async("hi")
    # The run continues and the model sees an error tool result.
    assert result.answer == "done"
    assert seen == []
    call = result.steps[0]["tool_calls"][0]
    assert "must be a JSON object" in json.loads(call["result"])["error"]
    assert call["arguments"] == {"_raw": raw}


# --- undecodable responses and non-string tool arguments ---------------------


async def test_malformed_json_response_returns_partial_result(no_backoff) -> None:
    import json

    from mcp_minion.agent import API_ERROR_MAX_RETRIES

    decode_error = json.JSONDecodeError("Expecting value", "<html>", 0)
    agent = _make_agent(
        [_response(tool_calls=[_tool_call()], prompt=10, completion=2)]
        + [decode_error] * (API_ERROR_MAX_RETRIES + 1),
        config=AgentConfig(max_steps=5),
    )
    result = await agent.run_async("hi")
    assert result.error is not None
    assert result.error.startswith("LLMResponseError: ")
    assert "malformed JSON response" in result.error
    assert len(result.steps) == 1
    assert result.input_tokens == 10


async def test_text_response_returns_partial_result(no_backoff) -> None:
    from mcp_minion.agent import API_ERROR_MAX_RETRIES

    page = "<html><body>502 Bad Gateway</body></html>"
    agent = _make_agent(
        [_response(tool_calls=[_tool_call()], prompt=10, completion=2)]
        + [page] * (API_ERROR_MAX_RETRIES + 1),
        config=AgentConfig(max_steps=5),
    )
    result = await agent.run_async("hi")
    assert result.error is not None
    assert "unexpected response of type str" in result.error
    assert "502 Bad Gateway" in result.error
    assert len(result.steps) == 1


async def test_malformed_response_then_good_is_retried(no_backoff) -> None:
    agent = _make_agent(["<html>oops</html>", _response(content="done")])
    result = await agent.run_async("hi")
    assert result.answer == "done"
    assert result.error is None


async def test_dict_tool_arguments_are_accepted() -> None:
    import json

    registry, seen = _recording_registry()
    agent = _make_agent(
        [
            _response(tool_calls=[_tool_call(name="ping", arguments={"x": 1})]),
            _response(content="done"),
        ]
    )
    agent.tools = registry
    result = await agent.run_async("hi")
    assert result.answer == "done"
    assert seen == [{"x": 1}]
    echoed = agent.client.requests[1]["messages"][2]["tool_calls"][0]
    assert echoed["function"]["arguments"] == json.dumps({"x": 1})


@pytest.mark.parametrize("raw", [42, [1, 2], []])
async def test_non_string_non_dict_tool_arguments_are_error_result(raw) -> None:
    import json

    registry, seen = _recording_registry()
    agent = _make_agent(
        [
            _response(tool_calls=[_tool_call(name="ping", arguments=raw)]),
            _response(content="done"),
        ]
    )
    agent.tools = registry
    result = await agent.run_async("hi")
    assert result.answer == "done"
    assert seen == []
    call = result.steps[0]["tool_calls"][0]
    assert "must be a JSON object" in json.loads(call["result"])["error"]
    echoed = agent.client.requests[1]["messages"][2]["tool_calls"][0]
    assert echoed["function"]["arguments"] == json.dumps(raw)
