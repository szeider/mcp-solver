"""Unit tests for logging module."""

import json
import tempfile
from pathlib import Path

from mcp_minion.logging import RunLogger


class TestRunLogger:
    """Tests for the RunLogger class."""

    def test_creates_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(log_dir=Path(tmpdir))
            assert logger.log_path.exists()
            assert logger.log_path.suffix == ".json"

    def test_initial_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(log_dir=Path(tmpdir))
            data = json.loads(logger.log_path.read_text())
            assert data["status"] == "running"
            assert data["completed_at"] is None
            assert data["steps"] == []
            assert data["token_usage"] == {"input_tokens": 0, "output_tokens": 0}
            assert data["error"] is None

    def test_custom_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(log_dir=Path(tmpdir), run_id="test123")
            assert logger.run_id == "test123"
            assert "test123" in logger.log_path.name

    def test_log_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(log_dir=Path(tmpdir))
            logger.log_config({"model": "test-model", "max_steps": 5})
            data = json.loads(logger.log_path.read_text())
            assert data["config"]["model"] == "test-model"
            assert data["config"]["max_steps"] == 5

    def test_log_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(log_dir=Path(tmpdir))
            logger.log_prompt("What is 2+2?")
            data = json.loads(logger.log_path.read_text())
            assert data["prompt"] == "What is 2+2?"

    def test_log_step_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(log_dir=Path(tmpdir))
            logger.log_step_start(1)
            data = json.loads(logger.log_path.read_text())
            assert len(data["steps"]) == 1
            assert data["steps"][0]["step"] == 1
            assert data["steps"][0]["started_at"] is not None

    def test_log_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(log_dir=Path(tmpdir))
            logger.log_step_start(1)
            logger.log_tool_call(
                tool_name="add",
                tool_args={"a": 2, "b": 3},
                tool_result='{"result": 5}',
                tool_call_id="call_123",
            )
            data = json.loads(logger.log_path.read_text())
            tool_calls = data["steps"][0]["tool_calls"]
            assert len(tool_calls) == 1
            assert tool_calls[0]["name"] == "add"
            assert tool_calls[0]["arguments"] == {"a": 2, "b": 3}
            assert tool_calls[0]["result"] == '{"result": 5}'

    def test_log_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(log_dir=Path(tmpdir))
            logger.log_error("Something went wrong", "Traceback...")
            data = json.loads(logger.log_path.read_text())
            assert data["status"] == "error"
            assert data["error"]["message"] == "Something went wrong"
            assert data["error"]["traceback"] == "Traceback..."
            assert data["completed_at"] is not None

    def test_log_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(log_dir=Path(tmpdir))
            logger.log_completion({"answer": "42", "steps_count": 3})
            data = json.loads(logger.log_path.read_text())
            assert data["status"] == "completed"
            assert data["result"]["answer"] == "42"
            assert data["completed_at"] is not None

    def test_creates_log_dir_if_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "logs"
            logger = RunLogger(log_dir=nested_dir)
            assert nested_dir.exists()
            assert logger.log_path.exists()

    def test_multiple_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(log_dir=Path(tmpdir))
            logger.log_step_start(1)
            logger.log_step_start(2)
            logger.log_step_start(3)
            data = json.loads(logger.log_path.read_text())
            assert len(data["steps"]) == 3
            assert [s["step"] for s in data["steps"]] == [1, 2, 3]

    def test_log_token_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(log_dir=Path(tmpdir))
            logger.log_token_usage(input_tokens=100, output_tokens=50)
            data = json.loads(logger.log_path.read_text())
            assert data["token_usage"]["input_tokens"] == 100
            assert data["token_usage"]["output_tokens"] == 50

    def test_log_token_usage_accumulates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(log_dir=Path(tmpdir))
            logger.log_token_usage(input_tokens=100, output_tokens=50)
            logger.log_token_usage(input_tokens=200, output_tokens=75)
            logger.log_token_usage(input_tokens=150, output_tokens=25)
            data = json.loads(logger.log_path.read_text())
            assert data["token_usage"]["input_tokens"] == 450
            assert data["token_usage"]["output_tokens"] == 150
