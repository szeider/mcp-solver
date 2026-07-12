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
                "modelstring": "test/model",
                "max_steps": 5,
                "temperature": 0.7,
            }
            (folder / "config.json").write_text(json.dumps(config))

            # Create project.md and task.md
            (folder / "project.md").write_text("You are a helpful assistant.")
            (folder / "task.md").write_text("What is 2+2?")

            agent_config, mcp_servers, prompt = load_run_folder(folder)

            assert agent_config.model == "test/model"
            assert agent_config.max_steps == 5
            assert agent_config.api_params == {"temperature": 0.7}
            assert mcp_servers is None  # Old format has no MCP servers
            # Combined prompt should include both
            assert "helpful assistant" in prompt
            assert "2+2" in prompt
            assert "---" in prompt  # Separator

    def test_load_folder_task_only(self) -> None:
        """project.md is optional - should work with just task.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"modelstring": "test"}')
            (folder / "task.md").write_text("Do something")

            agent_config, mcp_servers, prompt = load_run_folder(folder)

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
            (folder / "config.json").write_text('{"modelstring": "test"}')
            (folder / "project.md").write_text("Project instructions")
            # No task.md

            with pytest.raises(SystemExit):
                load_run_folder(folder)

    def test_empty_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"modelstring": "test"}')
            (folder / "task.md").write_text("   ")  # Whitespace only

            with pytest.raises(SystemExit):
                load_run_folder(folder)

    def test_default_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            # Minimal config - no modelstring or max_steps
            (folder / "config.json").write_text("{}")
            (folder / "task.md").write_text("Test task")

            agent_config, _, prompt = load_run_folder(folder)

            # Should use defaults from AgentConfig
            assert agent_config.model == "google/gemini-3-flash-preview"
            assert agent_config.max_steps == 10

    def test_extra_api_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            config = {
                "modelstring": "test/model",
                "max_steps": 3,
                "temperature": 0,
                "max_tokens": 1000,
                "top_p": 0.9,
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
            (folder / "config.json").write_text('{"modelstring": "test"}')
            (folder / "task.md").write_text("\n\n  Test task  \n\n")

            _, _, prompt = load_run_folder(folder)

            assert prompt == "Test task"

    def test_project_whitespace_stripped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"modelstring": "test"}')
            (folder / "project.md").write_text("\n  Project  \n")
            (folder / "task.md").write_text("Task")

            _, _, prompt = load_run_folder(folder)

            assert prompt.startswith("Project")
            assert prompt.endswith("Task")

    def test_empty_project_treated_as_missing(self) -> None:
        """Empty project.md should be treated as if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"modelstring": "test"}')
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

            agent_config, mcp_servers, prompt = load_run_folder(folder)

            assert agent_config.model == "test/new-model"
            assert agent_config.max_steps == 7
            assert agent_config.api_params["temperature"] == 0.5
            assert agent_config.api_params["max_tokens"] == 4096
            assert mcp_servers is not None
            assert "test" in mcp_servers
            assert mcp_servers["test"]["command"] == "uv"

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
            (folder / "config.json").write_text('{"modelstring": "test"}')
            (folder / "task.md").write_text("Do this <!-- secret note --> task")

            _, _, prompt = load_run_folder(folder)

            assert "secret note" not in prompt
            assert "Do this" in prompt
            assert "task" in prompt

    def test_html_comments_stripped_from_project(self) -> None:
        """HTML comments should be stripped from project.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"modelstring": "test"}')
            (folder / "project.md").write_text("Instructions <!-- hidden -->")
            (folder / "task.md").write_text("Task")

            _, _, prompt = load_run_folder(folder)

            assert "hidden" not in prompt
            assert "Instructions" in prompt

    def test_multiline_html_comments_stripped(self) -> None:
        """Multiline HTML comments should be stripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "config.json").write_text('{"modelstring": "test"}')
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

    def test_old_format_packages_not_leaked_to_api_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            config = {
                "modelstring": "test/model",
                "temperature": 0,
                "packages": ["z3-solver"],
                "files": {"system": "sys.md"},
            }
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "sys.md").write_text("hi")
            (folder / "task.md").write_text("Do it")

            agent_config, _, _ = load_run_folder(folder)
            assert agent_config.packages == ["z3-solver"]
            assert agent_config.system_prompt == "hi"
            assert "packages" not in agent_config.api_params
            assert "files" not in agent_config.api_params
            assert agent_config.api_params == {"temperature": 0}
