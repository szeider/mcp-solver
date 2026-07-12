"""Unit tests for agent module."""

from dataclasses import asdict
from types import SimpleNamespace

from mcp_minion.agent import Agent, AgentConfig, AgentResult


class TestAgentConfig:
    """Tests for the AgentConfig dataclass."""

    def test_defaults(self) -> None:
        config = AgentConfig()
        assert config.model == "google/gemini-3-flash-preview"
        assert config.max_steps == 10
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.api_params == {}

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


# --- run_async behavior with a fake OpenAI client --------------------------


def _response(content=None, tool_calls=None, prompt=7, completion=3):
    """Build a minimal chat-completions response object."""
    return SimpleNamespace(
        id="resp-1",
        model="fake-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
    )


def _tool_call(name="add", arguments='{"a": 1, "b": 2}'):
    return SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeClient:
    """Stands in for the OpenAI client; returns canned responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return self._responses.pop(0)


def _make_agent(responses, config=None, on_step=None):
    agent = Agent(
        api_key="test-key",
        tools=None,
        config=config or AgentConfig(),
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
