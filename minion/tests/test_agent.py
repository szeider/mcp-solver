"""Unit tests for agent module."""

from dataclasses import asdict

from mcp_minion.agent import AgentConfig, AgentResult


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
