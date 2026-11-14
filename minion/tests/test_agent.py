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
    """Stands in for the AsyncOpenAI client; returns canned responses in order.

    An exception instance in the list is raised instead of returned.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        # Snapshot the messages: the agent keeps appending to the same list.
        self.requests.append({**kwargs, "messages": list(kwargs.get("messages", []))})
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

    def get_instructions(self):
        return {}

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
    from mcp_minion import ToolOutcome

    mcp = FakeResourceManager()
    agent = Agent(api_key="k", tools=None, mcp_manager=mcp)
    result = await agent._execute_tool("read_resource", {"uri": "app://guide"})
    assert result == ToolOutcome(text="guide text")
    assert mcp.read_uris == ["app://guide"]


async def test_read_resource_missing_uri_is_error() -> None:
    agent = Agent(api_key="k", tools=None, mcp_manager=FakeResourceManager())
    result = await agent._execute_tool("read_resource", {})
    assert result.is_error
    assert "uri" in result.text


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
    assert call["result"] == "pong"
    assert call["is_error"] is False
    # The echoed assistant message never carries null arguments.
    echoed = agent.client.requests[1]["messages"][2]["tool_calls"][0]
    assert echoed["function"]["arguments"] == "{}"


@pytest.mark.parametrize("raw", ["[1, 2]", "42", '"text"', "null"])
async def test_non_object_tool_arguments_are_error_result(raw) -> None:

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
    assert "must be a JSON object" in call["result"]
    assert call["is_error"] is True
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
    assert "must be a JSON object" in call["result"]
    assert call["is_error"] is True
    echoed = agent.client.requests[1]["messages"][2]["tool_calls"][0]
    assert echoed["function"]["arguments"] == json.dumps(raw)


# --- tool outcomes reach the model as plain text (item A) -------------------


class FakeToolManager:
    """MCP manager stub with fixed tool outcomes and optional instructions."""

    def __init__(self, outcomes, instructions=None, tools=None):
        self._outcomes = outcomes
        self._instructions = instructions or {}
        self._tool_infos = tools or []
        self.calls: list[tuple[str, dict]] = []

    def has_tool(self, name):
        return name in self._outcomes

    def get_tools(self):
        return list(self._tool_infos)

    def get_resources(self):
        return []

    def get_instructions(self):
        return dict(self._instructions)

    def get_openai_tools(self):
        return [
            {
                "type": "function",
                "function": {"name": n, "description": "", "parameters": {}},
            }
            for n in self._outcomes
        ]

    async def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        return self._outcomes[name]


def _agent_with_manager(manager, responses, config=None):
    agent = Agent(
        api_key="k", tools=None, config=config or AgentConfig(), mcp_manager=manager
    )
    agent.client = FakeClient(responses)
    return agent


async def test_multiline_tool_text_reaches_model_unescaped() -> None:
    from mcp_minion import ToolOutcome

    text = 'line "one"\nline two'
    manager = FakeToolManager({"t": ToolOutcome(text=text)})
    agent = _agent_with_manager(
        manager,
        [_response(tool_calls=[_tool_call(name="t", arguments="{}")]), _response("ok")],
    )
    result = await agent.run_async("hi")
    tool_msg = agent.client.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"] == text  # no JSON envelope, no escaping
    call = result.steps[0]["tool_calls"][0]
    assert call == {"name": "t", "arguments": {}, "result": text, "is_error": False}


async def test_error_outcome_prefixed_for_model(tmp_path) -> None:
    import json as json_mod

    from mcp_minion import RunLogger, ToolOutcome

    logger = RunLogger(log_dir=tmp_path)
    manager = FakeToolManager({"t": ToolOutcome(text="it failed", is_error=True)})
    agent = _agent_with_manager(
        manager,
        [_response(tool_calls=[_tool_call(name="t", arguments="{}")]), _response("ok")],
    )
    agent.logger = logger
    result = await agent.run_async("hi")
    assert agent.client.requests[1]["messages"][-1]["content"] == "ERROR: it failed"
    call = result.steps[0]["tool_calls"][0]
    assert call["result"] == "it failed" and call["is_error"] is True
    logged = json_mod.loads(logger.log_path.read_text())["steps"][0]["tool_calls"][0]
    assert logged["result"] == "it failed" and logged["is_error"] is True


async def test_registry_tool_outcome_text() -> None:
    from mcp_minion.tools import create_default_registry

    agent = _make_agent([_response(tool_calls=[_tool_call()]), _response("done")])
    agent.tools = create_default_registry()
    result = await agent.run_async("hi")
    assert result.steps[0]["tool_calls"][0]["result"] == "3"
    assert agent.client.requests[1]["messages"][-1]["content"] == "3"


async def test_unknown_tool_is_error_outcome() -> None:
    agent = Agent(api_key="k", tools=None)
    outcome = await agent._execute_tool("nope", {})
    assert outcome.is_error and "Unknown tool: nope" in outcome.text


# --- request options (item B) and async client (item G) ---------------------


async def test_extra_body_merged_over_usage_include() -> None:
    config = AgentConfig(
        extra_body={
            "reasoning": {"effort": "high"},
            "usage": {"include": False, "other": 1},
        },
        request_timeout=42.0,
    )
    agent = _make_agent([_response(content="done")], config=config)
    await agent.run_async("hi")
    request = agent.client.requests[0]
    assert request["extra_body"] == {
        "reasoning": {"effort": "high"},
        "usage": {"include": True, "other": 1},
    }
    assert request["timeout"] == 42.0


async def test_default_request_has_usage_only_and_no_timeout() -> None:
    agent = _make_agent([_response(content="done")])
    await agent.run_async("hi")
    request = agent.client.requests[0]
    assert request["extra_body"] == {"usage": {"include": True}}
    assert "timeout" not in request


def test_merge_extra_body_does_not_mutate_input() -> None:
    from mcp_minion.agent import merge_extra_body

    user = {"usage": {"include": False}}
    assert merge_extra_body(user) == {"usage": {"include": True}}
    assert user == {"usage": {"include": False}}
    assert merge_extra_body(None) == {"usage": {"include": True}}


# --- per-run client lifetime (review R1) -------------------------------------


def _chat_json(content="done"):
    return {
        "id": "r",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


@pytest.fixture
def recording_factory(monkeypatch):
    """Replace AsyncOpenAI with real clients on an httpx mock transport.

    Each created client is recorded; the handler answers each request with
    the next item of ``factory.replies`` (a dict body, or an int status).
    """
    import httpx
    from openai import AsyncOpenAI

    import mcp_minion.agent as agent_mod

    created = []
    replies: list = []

    def handler(request):
        reply = replies.pop(0) if replies else _chat_json()
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": {"message": "bad"}})
        return httpx.Response(200, json=reply)

    def factory(**kwargs):
        client = AsyncOpenAI(
            **kwargs,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        created.append(client)
        return client

    monkeypatch.setattr(agent_mod, "AsyncOpenAI", factory)
    return SimpleNamespace(created=created, replies=replies)


def test_two_sync_runs_on_one_agent(recording_factory) -> None:
    agent = Agent(api_key="k", tools=None)
    assert agent.client is None
    assert agent.run("one").answer == "done"
    assert agent.run("two").answer == "done"
    assert len(recording_factory.created) == 2
    assert all(c.is_closed() for c in recording_factory.created)
    assert agent.client is None


def test_created_client_closed_after_error_run(recording_factory) -> None:
    recording_factory.replies.append(400)
    agent = Agent(api_key="k", tools=None)
    result = agent.run("hi")
    assert result.error is not None and "BadRequestError" in result.error
    assert recording_factory.created[0].is_closed()


def test_created_client_closed_after_exception(monkeypatch, recording_factory) -> None:
    agent = Agent(api_key="k", tools=None)

    async def boom(prompt, client):
        raise RuntimeError("bug")

    monkeypatch.setattr(agent, "_run", boom)
    with pytest.raises(RuntimeError):
        agent.run("hi")
    assert recording_factory.created[0].is_closed()


def test_injected_client_used_and_not_closed(recording_factory) -> None:
    closed = []
    fake = FakeClient([_response(content="a"), _response(content="b")])

    async def close():
        closed.append(True)

    fake.close = close
    agent = Agent(api_key="k", tools=None)
    agent.client = fake
    assert agent.run("one").answer == "a"
    assert agent.run("two").answer == "b"
    assert closed == []
    assert recording_factory.created == []


# --- tool-result bound in the model context (item C) ------------------------


def test_truncate_for_model_head_and_tail() -> None:
    from mcp_minion.agent import truncate_for_model

    text = "".join(chr(ord("a") + i % 26) for i in range(1000))
    cut = truncate_for_model(text, 100)
    assert cut.startswith(text[:75] + "\n[... 900 characters truncated ...]\n")
    assert cut.endswith(text[-25:])
    assert truncate_for_model(text, 0) == text
    assert truncate_for_model(text, 1000) == text


async def test_long_tool_result_cut_for_model_but_full_in_record() -> None:
    from mcp_minion import ToolOutcome

    text = "x" * 300 + "TAIL"
    manager = FakeToolManager({"t": ToolOutcome(text=text, is_error=True)})
    agent = _agent_with_manager(
        manager,
        [_response(tool_calls=[_tool_call(name="t", arguments="{}")]), _response("ok")],
        config=AgentConfig(max_tool_result_chars=100),
    )
    result = await agent.run_async("hi")
    content = agent.client.requests[1]["messages"][-1]["content"]
    assert content.startswith("ERROR: " + "x" * 75 + "\n[... 204 characters truncated")
    assert content.endswith("TAIL")
    assert result.steps[0]["tool_calls"][0]["result"] == text


# --- server instructions in the system prompt (item D) ----------------------


def test_server_instructions_in_tool_sections() -> None:
    from mcp_minion.mcp_client import MCPToolInfo

    tool = MCPToolInfo(name="t", description="Do it", parameters={}, server_name="s1")
    manager = FakeToolManager(
        {}, instructions={"s1": "Call t first.", "s2": "Only notes."}, tools=[tool]
    )
    sections = Agent(
        api_key="k", tools=None, mcp_manager=manager
    )._build_tool_sections()
    assert "## s1\nServer instructions:\nCall t first.\n" in sections
    assert "- `t`: Do it" in sections
    assert "## s2\nServer instructions:\nOnly notes." in sections


def test_no_instructions_add_nothing() -> None:
    from mcp_minion.mcp_client import MCPToolInfo

    tool = MCPToolInfo(name="t", description="Do it", parameters={}, server_name="s1")
    manager = FakeToolManager({}, tools=[tool])
    sections = Agent(
        api_key="k", tools=None, mcp_manager=manager
    )._build_tool_sections()
    assert "Server instructions" not in sections
    assert (
        Agent(
            api_key="k", tools=None, mcp_manager=FakeToolManager({})
        )._build_tool_sections()
        == ""
    )


async def test_instructions_reach_system_prompt() -> None:
    manager = FakeToolManager({}, instructions={"srv": "Be careful."})
    agent = _agent_with_manager(manager, [_response(content="done")])
    await agent.run_async("hi")
    assert "Be careful." in agent.client.requests[0]["messages"][0]["content"]


# --- reasoning replay (item M) -----------------------------------------------


def _reasoning_response(details=None, reasoning=None, **kwargs):
    response = _response(**kwargs)
    message = response.choices[0].message
    if details is not None:
        message.reasoning_details = details
    if reasoning is not None:
        message.reasoning = reasoning
    return response


async def test_reasoning_details_replayed_unmodified(tmp_path) -> None:
    import json as json_mod

    from mcp_minion import RunLogger

    details = [
        {"type": "reasoning.encrypted", "data": "abc", "id": "r1", "format": "g"},
        {"type": "reasoning.text", "text": "think", "signature": None, "index": 1},
    ]
    logger = RunLogger(log_dir=tmp_path)
    agent = _make_agent(
        [
            _reasoning_response(
                details=details, reasoning="think", tool_calls=[_tool_call()]
            ),
            _response(content="done"),
        ],
        logger=logger,
    )
    from mcp_minion.tools import create_default_registry

    agent.tools = create_default_registry()
    result = await agent.run_async("hi")
    assistant = agent.client.requests[1]["messages"][2]
    assert assistant["role"] == "assistant" and assistant["tool_calls"]
    assert assistant["reasoning_details"] == details
    assert assistant["reasoning_details"] is details  # same list, untouched
    assert "reasoning" not in assistant  # details take precedence
    assert result.steps[0]["reasoning_details"] == details
    assert result.steps[0]["reasoning"] == "think"
    logged = json_mod.loads(logger.log_path.read_text())["steps"][0]["api_response"]
    assert logged["reasoning_details"] == details
    assert logged["reasoning"] == "think"


async def test_no_reasoning_adds_no_keys() -> None:
    from mcp_minion.tools import create_default_registry

    agent = _make_agent([_response(tool_calls=[_tool_call()]), _response("done")])
    agent.tools = create_default_registry()
    result = await agent.run_async("hi")
    assistant = agent.client.requests[1]["messages"][2]
    assert "reasoning_details" not in assistant and "reasoning" not in assistant
    assert "reasoning_details" not in result.steps[0]
    assert "reasoning" not in result.steps[0]
    # The request-level reasoning parameter is not touched.
    assert "reasoning" not in agent.client.requests[0]["extra_body"]


async def test_plain_reasoning_replayed_on_truncated_continuation() -> None:
    agent = _make_agent(
        [
            _reasoning_response(
                reasoning="thinking...", content="cut", finish_reason="length"
            ),
            _response(content="done"),
        ],
        config=AgentConfig(max_steps=3),
    )
    result = await agent.run_async("hi")
    continuation = agent.client.requests[1]["messages"][-2]
    assert continuation == {
        "role": "assistant",
        "content": "cut",
        "reasoning": "thinking...",
    }
    assert result.steps[0]["reasoning"] == "thinking..."


# --- default tool-result bound and logged system prompt ---------------------


def test_default_max_tool_result_chars() -> None:
    assert AgentConfig().max_tool_result_chars == 64000


async def test_backend_template_size_result_reaches_model_uncut() -> None:
    from mcp_minion import ToolOutcome

    text = "".join(f"line {i}\n" for i in range(7000))[:54262]
    assert len(text) == 54262
    manager = FakeToolManager({"select_backend": ToolOutcome(text=text)})
    agent = _agent_with_manager(
        manager,
        [
            _response(tool_calls=[_tool_call(name="select_backend", arguments="{}")]),
            _response("ok"),
        ],
    )
    await agent.run_async("hi")
    assert agent.client.requests[1]["messages"][-1]["content"] == text


async def test_system_prompt_logged_once(tmp_path) -> None:
    import json as json_mod

    from mcp_minion import RunLogger
    from mcp_minion.agent import build_system_prompt
    from mcp_minion.mcp_client import MCPToolInfo

    tool = MCPToolInfo(name="t", description="Do it", parameters={}, server_name="srv")
    manager = FakeToolManager(
        {"t": None}, instructions={"srv": "Server rule: call t."}, tools=[tool]
    )
    logger = RunLogger(log_dir=tmp_path)
    agent = _agent_with_manager(manager, [_response(content="done")])
    agent.logger = logger
    await agent.run_async("hi")

    log = json_mod.loads(logger.log_path.read_text())
    expected = build_system_prompt(None, agent._build_tool_sections())
    assert log["system_prompt"] == expected
    assert "Server rule: call t." in log["system_prompt"]
    assert log["system_prompt"] == agent.client.requests[0]["messages"][0]["content"]
    assert all("system_prompt" not in step for step in log["steps"])
