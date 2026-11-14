"""Unit tests for CLI module."""

import json
import tempfile
from pathlib import Path

import pytest

from mcp_minion.cli import load_run_folder, strip_html_comments


class TestStripHtmlComments:
    """Tests for the strip_html_comments function."""

    def test_single_comment(self) -> None:
        text = "Hello <!-- comment --> World"
        assert strip_html_comments(text) == "Hello  World"

    def test_multiline_comment(self) -> None:
        text = "Hello <!-- multi\nline\ncomment --> World"
        assert strip_html_comments(text) == "Hello  World"

    def test_multiple_comments(self) -> None:
        text = "A <!-- one --> B <!-- two --> C"
        assert strip_html_comments(text) == "A  B  C"

    def test_no_comments(self) -> None:
        text = "No comments here"
        assert strip_html_comments(text) == "No comments here"

    def test_empty_comment(self) -> None:
        text = "Hello <!----> World"
        assert strip_html_comments(text) == "Hello  World"

    def test_nested_dashes(self) -> None:
        text = "Hello <!-- --- --> World"
        assert strip_html_comments(text) == "Hello  World"


class TestLoadRunFolder:
    """Tests for the load_run_folder function."""

    def test_load_valid_folder_with_both_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            # Create config.json
            config = {
                "model": {"name": "test/model", "temperature": 0.7},
                "agent": {"max_steps": 5},
            }
            (folder / "config.json").write_text(json.dumps(config))

            # Create project.md and task.md
            (folder / "project.md").write_text("You are a helpful assistant.")
            (folder / "task.md").write_text("What is 2+2?")

            agent_config, mcp_servers, prompt = load_run_folder(folder)

            assert agent_config.model == "test/model"
            assert agent_config.max_steps == 5
            assert agent_config.api_params == {"temperature": 0.7}
            assert mcp_servers is None
            # Combined prompt should include both
            assert "helpful assistant" in prompt
            assert "2+2" in prompt
            assert "---" in prompt  # Separator

    def test_load_folder_task_only(self) -> None:
        """project.md is optional - should work with just task.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"model": {"name": "test"}}')
            (folder / "task.md").write_text("Do something")

            _agent_config, mcp_servers, prompt = load_run_folder(folder)

            assert prompt == "Do something"
            assert "---" not in prompt  # No separator when no project.md
            assert mcp_servers is None

    def test_missing_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "task.md").write_text("Test task")

            with pytest.raises(SystemExit):
                load_run_folder(folder)

    def test_missing_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"model": {"name": "test"}}')
            (folder / "project.md").write_text("Project instructions")
            # No task.md

            with pytest.raises(SystemExit):
                load_run_folder(folder)

    def test_empty_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"model": {"name": "test"}}')
            (folder / "task.md").write_text("   ")  # Whitespace only

            with pytest.raises(SystemExit):
                load_run_folder(folder)

    def test_default_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            # Minimal config - no modelstring or max_steps
            (folder / "config.json").write_text("{}")
            (folder / "task.md").write_text("Test task")

            agent_config, _, _prompt = load_run_folder(folder)

            # Should use defaults from AgentConfig
            assert agent_config.model == "google/gemini-3-flash-preview"
            assert agent_config.max_steps == 10

    def test_extra_api_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            config = {
                "model": {
                    "name": "test/model",
                    "temperature": 0,
                    "max_tokens": 1000,
                    "top_p": 0.9,
                },
                "agent": {"max_steps": 3},
            }
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "task.md").write_text("Test")

            agent_config, _, _ = load_run_folder(folder)

            assert agent_config.api_params["temperature"] == 0
            assert agent_config.api_params["max_tokens"] == 1000
            assert agent_config.api_params["top_p"] == 0.9

    def test_task_whitespace_stripped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"model": {"name": "test"}}')
            (folder / "task.md").write_text("\n\n  Test task  \n\n")

            _, _, prompt = load_run_folder(folder)

            assert prompt == "Test task"

    def test_project_whitespace_stripped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"model": {"name": "test"}}')
            (folder / "project.md").write_text("\n  Project  \n")
            (folder / "task.md").write_text("Task")

            _, _, prompt = load_run_folder(folder)

            assert prompt.startswith("Project")
            assert prompt.endswith("Task")

    def test_empty_project_treated_as_missing(self) -> None:
        """Empty project.md should be treated as if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"model": {"name": "test"}}')
            (folder / "project.md").write_text("   ")  # Whitespace only
            (folder / "task.md").write_text("Task only")

            _, _, prompt = load_run_folder(folder)

            # Should just be the task, no separator
            assert prompt == "Task only"
            assert "---" not in prompt

    def test_new_config_format(self) -> None:
        """Test new nested config format with mcpServers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            config = {
                "model": {
                    "name": "test/new-model",
                    "temperature": 0.5,
                    "max_tokens": 4096,
                },
                "agent": {
                    "max_steps": 7,
                },
                "mcpServers": {
                    "test": {
                        "command": "uv",
                        "args": ["run", "test-server"],
                    }
                },
            }
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "task.md").write_text("Test task")

            agent_config, mcp_servers, _prompt = load_run_folder(folder)

            assert agent_config.model == "test/new-model"
            assert agent_config.max_steps == 7
            assert agent_config.api_params["temperature"] == 0.5
            assert agent_config.api_params["max_tokens"] == 4096
            assert mcp_servers is not None
            assert "test" in mcp_servers
            assert mcp_servers["test"]["command"] == "uv"

    def test_max_total_tokens_loaded(self) -> None:
        """agent.max_total_tokens reaches the agent config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            config = {
                "model": {"name": "test/model"},
                "agent": {"max_steps": 80, "max_total_tokens": 500000},
            }
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "task.md").write_text("Test task")

            agent_config, _, _ = load_run_folder(folder)

            assert agent_config.max_total_tokens == 500000
            assert "max_total_tokens" not in agent_config.api_params

    def test_max_total_tokens_absent_is_unlimited(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            config = {"model": {"name": "test/model"}, "agent": {"max_steps": 3}}
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "task.md").write_text("Test task")

            agent_config, _, _ = load_run_folder(folder)

            assert agent_config.max_total_tokens == 0

    def test_flat_config_rejected(self, capsys) -> None:
        """The pre-0.3 flat format fails with a message naming the nested keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            config = {"modelstring": "test/model", "max_steps": 3, "temperature": 0}
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "task.md").write_text("Test task")

            with pytest.raises(SystemExit):
                load_run_folder(folder)
            err = capsys.readouterr().err
            assert "nested format" in err
            assert "modelstring" in err
            assert '"model": {"name"' in err and '"agent": {"max_steps"' in err

    def test_model_as_string_rejected(self, capsys) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"model": "x/y"}')
            (folder / "task.md").write_text("Test task")
            with pytest.raises(SystemExit):
                load_run_folder(folder)
            assert "must be objects" in capsys.readouterr().err

    def test_reasoning_provider_top_k_go_to_extra_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            config = {
                "model": {
                    "name": "m",
                    "temperature": 0,
                    "reasoning": {"effort": "high"},
                    "provider": {"order": ["a"]},
                    "top_k": 40,
                    "request_timeout": 120,
                    "extra_body": {"transforms": [], "top_k": 1},
                }
            }
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "task.md").write_text("Test task")

            agent_config, _, _ = load_run_folder(folder)
            assert agent_config.api_params == {"temperature": 0}
            assert agent_config.extra_body == {
                "transforms": [],
                "top_k": 40,  # the named key wins over extra_body's entry
                "reasoning": {"effort": "high"},
                "provider": {"order": ["a"]},
            }
            assert agent_config.request_timeout == 120

    def test_no_extra_body_keys_means_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"model": {"name": "m"}}')
            (folder / "task.md").write_text("Test task")
            agent_config, _, _ = load_run_folder(folder)
            assert agent_config.extra_body is None
            assert agent_config.request_timeout is None

    def test_max_tool_result_chars_from_run_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            config = {"model": {"name": "m"}, "agent": {"max_tool_result_chars": 0}}
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "task.md").write_text("Test task")
            agent_config, _, _ = load_run_folder(folder)
            assert agent_config.max_tool_result_chars == 0
            assert "max_tool_result_chars" not in agent_config.api_params

    def test_max_tool_result_chars_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"model": {"name": "m"}}')
            (folder / "task.md").write_text("Test task")
            agent_config, _, _ = load_run_folder(folder)
            assert agent_config.max_tool_result_chars == 64000

    @pytest.mark.parametrize("bad", [-1, 1.5, "100", True, None])
    def test_max_tool_result_chars_invalid(self, bad, capsys) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            config = {"model": {"name": "m"}, "agent": {"max_tool_result_chars": bad}}
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "task.md").write_text("Test task")
            with pytest.raises(SystemExit):
                load_run_folder(folder)
            assert "max_tool_result_chars must be an integer >= 0" in (
                capsys.readouterr().err
            )

    def test_extra_body_must_be_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text(
                '{"model": {"name": "m", "extra_body": [1]}}'
            )
            (folder / "task.md").write_text("Test task")
            with pytest.raises(SystemExit):
                load_run_folder(folder)

    async def test_reasoning_run_folder_reaches_request(self, tmp_path) -> None:
        """A run folder with model.reasoning no longer fails with TypeError."""
        from types import SimpleNamespace

        from mcp_minion.agent import Agent

        config = {"model": {"name": "m", "reasoning": {"effort": "low"}}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "task.md").write_text("Test task")
        agent_config, _, prompt = load_run_folder(tmp_path)

        agent = Agent(api_key="k", tools=None, config=agent_config)
        requests = []

        async def create(**kwargs):
            requests.append(kwargs)
            message = SimpleNamespace(content="done", tool_calls=None)
            return SimpleNamespace(
                id="r",
                model="m",
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                usage=None,
            )

        agent.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        result = await agent.run_async(prompt)
        assert result.error is None
        assert "reasoning" not in requests[0]
        assert requests[0]["extra_body"] == {
            "reasoning": {"effort": "low"},
            "usage": {"include": True},
        }

    def test_invalid_json_config(self) -> None:
        """Invalid JSON in config.json should exit with clear error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text("{invalid json}")
            (folder / "task.md").write_text("Task")

            with pytest.raises(SystemExit):
                load_run_folder(folder)

    def test_html_comments_stripped_from_task(self) -> None:
        """HTML comments should be stripped from task.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"model": {"name": "test"}}')
            (folder / "task.md").write_text("Do this <!-- secret note --> task")

            _, _, prompt = load_run_folder(folder)

            assert "secret note" not in prompt
            assert "Do this" in prompt
            assert "task" in prompt

    def test_html_comments_stripped_from_project(self) -> None:
        """HTML comments should be stripped from project.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"model": {"name": "test"}}')
            (folder / "project.md").write_text("Instructions <!-- hidden -->")
            (folder / "task.md").write_text("Task")

            _, _, prompt = load_run_folder(folder)

            assert "hidden" not in prompt
            assert "Instructions" in prompt

    def test_multiline_html_comments_stripped(self) -> None:
        """Multiline HTML comments should be stripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"model": {"name": "test"}}')
            (folder / "task.md").write_text(
                "Start <!--\nTODO: fix this\nNOTE: check later\n--> End"
            )

            _, _, prompt = load_run_folder(folder)

            assert "TODO" not in prompt
            assert "NOTE" not in prompt
            assert "Start" in prompt
            assert "End" in prompt


class TestRunFolderExtras:
    """Tests for files.system and packages loading."""

    def test_system_prompt_file_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            config = {
                "model": {"name": "test/model"},
                "files": {"system": "system.md"},
            }
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "system.md").write_text("SOLVER RULES { a : b }")
            (folder / "task.md").write_text("Do it")

            agent_config, _, _ = load_run_folder(folder)
            assert agent_config.system_prompt == "SOLVER RULES { a : b }"

    def test_missing_system_prompt_file_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            config = {
                "model": {"name": "test/model"},
                "files": {"system": "nope.md"},
            }
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "task.md").write_text("Do it")

            with pytest.raises(SystemExit):
                load_run_folder(folder)

    def test_packages_loaded_new_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            config = {
                "model": {"name": "test/model"},
                "packages": ["z3-solver", "numpy"],
            }
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "task.md").write_text("Do it")

            agent_config, _, _ = load_run_folder(folder)
            assert agent_config.packages == ["z3-solver", "numpy"]

    def test_packages_default_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"model": {"name": "m"}}')
            (folder / "task.md").write_text("Do it")

            agent_config, _, _ = load_run_folder(folder)
            assert agent_config.packages == []

    def test_packages_and_files_not_leaked_to_api_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            config = {
                "model": {"name": "test/model", "temperature": 0},
                "packages": ["z3-solver"],
                "files": {"system": "sys.md"},
            }
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "sys.md").write_text("hi")
            (folder / "task.md").write_text("Do it")

            agent_config, _, _ = load_run_folder(folder)
            assert agent_config.packages == ["z3-solver"]
            assert agent_config.system_prompt == "hi"
            assert agent_config.api_params == {"temperature": 0}


class TestVerboseRendering:
    """The -v step renderer: readable code blocks, unwrapped results."""

    def test_code_argument_rendered_as_block(self) -> None:
        from mcp_minion.cli import render_tool_call

        out = render_tool_call(
            "python_exec", {"code": "x = 1\nprint(x)", "timeout": 60}
        )
        assert "Tool: python_exec" in out
        assert "    x = 1\n    print(x)" in out  # real newlines, indented
        assert "\\n" not in out
        assert "timeout: 60" in out

    def test_long_scalar_argument_truncated(self) -> None:
        from mcp_minion.cli import render_tool_call

        out = render_tool_call("t", {"arg": "y" * 500})
        assert "chars)" in out and "y" * 500 not in out

    def test_kernel_result_shows_stdout_block(self) -> None:
        from mcp_minion.cli import render_tool_result

        text = json.dumps(
            {
                "stdout": "54 True\n",
                "stderr": "",
                "result": None,
                "error": None,
                "kernel_id": "k1",
                "success": True,
            }
        )
        out = render_tool_result(text)
        assert "Result: ok (kernel k1)" in out
        assert "stdout:" in out and "54 True" in out
        assert '"stderr"' not in out  # no raw JSON leakage

    def test_failed_exec_marked(self) -> None:
        from mcp_minion.cli import render_tool_result

        text = json.dumps(
            {"stdout": "", "stderr": "boom", "error": "NameError", "success": False}
        )
        out = render_tool_result(text)
        assert "FAILED" in out and "boom" in out and "NameError" in out

    def test_long_text_result_collapses_to_size_note(self) -> None:
        from mcp_minion.cli import render_tool_result

        template = "# DIDP project prompt\n" + "body\n" * 500
        out = render_tool_result(template)
        assert "full text in run log" in out
        assert "# DIDP project prompt" in out  # first line survives
        assert out.count("body") == 0

    def test_short_plain_result_passes_through(self) -> None:
        from mcp_minion.cli import render_tool_result

        assert render_tool_result("hello") == "Result: hello"

    def test_error_outcome_marked(self) -> None:
        from mcp_minion.cli import render_tool_result

        out = render_tool_result("it failed\nbadly", is_error=True)
        assert out == "Result: ERROR\n    it failed\n    badly"


class TestSaveSubmission:
    """save_submission writes the last accepted program to submission.py."""

    def _steps(self, code, ok=True):
        return [
            {
                "step": 1,
                "tool_calls": [
                    {
                        "name": "submit_code",
                        "arguments": {"code": code},
                        "result": json.dumps({"ok": True}) if ok else "rejected",
                        "is_error": not ok,
                    }
                ],
            }
        ]

    def test_writes_last_successful_submission(self, tmp_path) -> None:
        from mcp_minion.cli import save_submission

        path = save_submission(self._steps("print('hi')"), tmp_path)
        assert path == tmp_path / "submission.py"
        assert path.read_text() == "print('hi')"

    def test_no_submission_writes_nothing(self, tmp_path) -> None:
        from mcp_minion.cli import save_submission

        assert save_submission([], tmp_path) is None
        assert save_submission(self._steps("x", ok=False), tmp_path) is None
        assert not (tmp_path / "submission.py").exists()


class TestRunAgentErrorExit:
    """An LLM API failure ends the CLI run non-zero but keeps the submission."""

    @staticmethod
    def _run(monkeypatch, tmp_path, agent_result):
        import argparse
        import asyncio

        import mcp_minion.cli as cli_mod
        from mcp_minion.agent import AgentConfig

        class FakeAgent:
            def __init__(self, **kwargs) -> None:
                created.append(kwargs)

            async def run_async(self, prompt):
                return agent_result

        created: list[dict] = []
        monkeypatch.setattr(cli_mod, "Agent", FakeAgent)
        args = argparse.Namespace(folder=tmp_path, no_log=True, verbose=False)
        code = asyncio.run(
            cli_mod.run_agent_async(args, AgentConfig(), None, "task", "key")
        )
        TestRunAgentErrorExit.created = created
        return code

    @staticmethod
    def _steps():
        return [
            {
                "step": 1,
                "content": None,
                "tool_calls": [
                    {
                        "name": "submit_code",
                        "arguments": {"code": "print(1)"},
                        "result": json.dumps({"ok": True}),
                        "is_error": False,
                    }
                ],
            }
        ]

    def test_error_exits_nonzero_and_saves_submission(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        from mcp_minion.agent import AgentResult

        result = AgentResult(
            answer="[Agent stopped]",
            steps=self._steps(),
            error="APIConnectionError: Connection error.",
        )
        code = self._run(monkeypatch, tmp_path, result)
        assert code == 1
        assert (tmp_path / "submission.py").read_text().startswith("print(1)")
        assert "APIConnectionError: Connection error." in capsys.readouterr().err

    def test_success_exits_zero(self, monkeypatch, tmp_path) -> None:
        from mcp_minion.agent import AgentResult

        result = AgentResult(answer="done", steps=self._steps())
        assert self._run(monkeypatch, tmp_path, result) == 0

    def test_demo_tools_without_servers(self, monkeypatch, tmp_path) -> None:
        from mcp_minion.agent import AgentResult

        self._run(monkeypatch, tmp_path, AgentResult(answer="done"))
        tools = self.created[0]["tools"]
        assert tools is not None and tools.get("add") is not None

    def test_no_demo_tools_with_servers(self, monkeypatch, tmp_path) -> None:
        import argparse
        import asyncio

        import mcp_minion.cli as cli_mod
        from mcp_minion.agent import AgentConfig, AgentResult

        created = []

        class FakeAgent:
            def __init__(self, **kwargs) -> None:
                created.append(kwargs)

            async def run_async(self, prompt):
                return AgentResult(answer="done")

        class FakeManager:
            def __init__(self, servers, errlog=None) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

        monkeypatch.setattr(cli_mod, "Agent", FakeAgent)
        monkeypatch.setattr(cli_mod, "MCPManager", FakeManager)
        args = argparse.Namespace(folder=tmp_path, no_log=True, verbose=False)
        servers = {"s": {"command": "srv"}}
        asyncio.run(cli_mod.run_agent_async(args, AgentConfig(), servers, "t", "k"))
        assert created[0]["tools"] is None
        assert created[0]["mcp_manager"] is not None

    def test_main_exits_nonzero_on_agent_error(self, monkeypatch, tmp_path) -> None:
        import sys

        import mcp_minion.cli as cli_mod

        async def fake_run(*args, **kwargs):
            return 1

        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "task.md").write_text("do it")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setattr(cli_mod, "run_agent_async", fake_run)
        monkeypatch.setattr(sys, "argv", ["mcp-minion", str(tmp_path)])
        with pytest.raises(SystemExit) as exc:
            cli_mod.main()
        assert exc.value.code == 1


class TestFindApiKey:
    """find_api_key: environment, folder/.env, extra files, ~/.mcp-minion."""

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        (tmp_path / "home").mkdir()

    def test_environment_wins(self, monkeypatch, tmp_path) -> None:
        from mcp_minion import find_api_key

        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from-folder\n")
        assert find_api_key(folder=tmp_path) == "from-env"

    def test_order_folder_extra_home(self, tmp_path) -> None:
        import os

        from mcp_minion import find_api_key

        home_file = tmp_path / "home" / ".mcp-minion"
        home_file.write_text("OPENROUTER_API_KEY=from-home\n")
        extra1 = tmp_path / "e1.env"
        extra2 = tmp_path / "e2.env"
        extra2.write_text("OPENROUTER_API_KEY=from-extra2\n")
        assert find_api_key(extra_dotenv=[extra1, extra2]) == "from-extra2"
        assert find_api_key() == "from-home"
        folder = tmp_path / "run"
        folder.mkdir()
        (folder / ".env").write_text("OPENROUTER_API_KEY=from-folder\n")
        assert find_api_key(extra_dotenv=[extra2], folder=folder) == "from-folder"
        # Nothing is written into the process environment.
        assert "OPENROUTER_API_KEY" not in os.environ

    def test_missing_everywhere_and_other_var(self, monkeypatch, tmp_path) -> None:
        from mcp_minion import find_api_key

        assert find_api_key(folder=tmp_path) is None
        (tmp_path / ".env").write_text("OTHER_KEY=x\n")
        assert find_api_key("OTHER_KEY", folder=tmp_path) == "x"

    def test_cli_uses_folder_env(self, monkeypatch, tmp_path) -> None:
        import sys

        import mcp_minion.cli as cli_mod

        seen = []

        async def fake_run(args, config, servers, prompt, api_key):
            seen.append(api_key)
            return 0

        folder = tmp_path / "run"
        folder.mkdir()
        (folder / "config.json").write_text("{}")
        (folder / "task.md").write_text("do it")
        (folder / ".env").write_text("OPENROUTER_API_KEY=sk-folder\n")
        monkeypatch.setattr(cli_mod, "run_agent_async", fake_run)
        monkeypatch.setattr(sys, "argv", ["mcp-minion", str(folder)])
        cli_mod.main()
        assert seen == ["sk-folder"]
