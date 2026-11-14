"""Unit tests for the mcp-solver-runfolder tool (no API, no network)."""

import json

import pytest

from mcp_solver.agent import runfolder
from mcp_solver.templates import SOLVERS


def test_markers_cover_all_solvers():
    assert set(SOLVERS) <= set(runfolder.SOLVER_MARKERS)


def test_write_run_folder_contents(tmp_path):
    folder = runfolder.write_run_folder(
        tmp_path / "demo", "cpmpy", "Place 8 queens.", tmp_path / "repo"
    )
    config = json.loads((folder / "config.json").read_text())
    assert config["model"]["name"] == runfolder.DEFAULT_MODEL
    assert config["mcpServers"]["mcp-solver"]["args"][-1] == "mcp-solver-serve"
    task = (folder / "task.md").read_text()
    assert "cpmpy backend ONLY" in task
    assert task.endswith("Place 8 queens.")


def test_write_run_folder_preserves_other_files(tmp_path):
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "run_1.json").write_text("{}")
    runfolder.write_run_folder(dest, "z3", "x", tmp_path / "repo")
    assert (dest / "run_1.json").read_text() == "{}"


def test_cli_list_and_generate(tmp_path, capsys, monkeypatch):
    # Fake checkout with one problem.
    repo = tmp_path / "repo"
    problems = repo / "tests" / "problems" / "z3"
    problems.mkdir(parents=True)
    (problems / "puzzle.md").write_text("Solve me.")
    (problems / "test.md").write_text("smoke")  # excluded from discovery
    monkeypatch.setattr(runfolder, "local_checkout", lambda: str(repo))

    assert runfolder.main(["z3", "list"]) == 0
    assert capsys.readouterr().out.splitlines() == ["puzzle"]

    dest = tmp_path / "out"
    assert runfolder.main(["z3", "puzzle", str(dest)]) == 0
    assert "Solve me." in (dest / "task.md").read_text()


def test_cli_unknown_problem_errors(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "tests" / "problems" / "z3").mkdir(parents=True)
    monkeypatch.setattr(runfolder, "local_checkout", lambda: str(repo))
    with pytest.raises(SystemExit):
        runfolder.main(["z3", "nope", str(tmp_path / "out")])


def test_model_block_full_id_has_no_settings():
    assert runfolder.model_block("openai/gpt-5.6-terra") == {
        "name": "openai/gpt-5.6-terra"
    }


def test_model_block_from_model_file(tmp_path):
    # Resolved like the CLI: reasoning and request_timeout reach the config,
    # in the keys minion's run-folder format reads.
    model_file = tmp_path / "m.json"
    model_file.write_text(
        json.dumps(
            {
                "path": "vendor/model-x",
                "temperature": 0.2,
                "max_tokens": 4096,
                "reasoning": {"effort": "high"},
                "request_timeout": 90,
            }
        )
    )
    folder = runfolder.write_run_folder(
        tmp_path / "demo", "z3", "x", tmp_path / "repo", model=str(model_file)
    )
    config = json.loads((folder / "config.json").read_text())
    assert config["model"] == {
        "name": "vendor/model-x",
        "temperature": 0.2,
        "max_tokens": 4096,
        "reasoning": {"effort": "high"},
        "request_timeout": 90,
    }
    # minion reads it back into the same AgentConfig fields.
    from mcp_minion.cli import build_agent_config

    agent_config = build_agent_config(config)
    assert agent_config.model == "vendor/model-x"
    assert agent_config.api_params == {"temperature": 0.2, "max_tokens": 4096}
    assert agent_config.extra_body == {"reasoning": {"effort": "high"}}
    assert agent_config.request_timeout == 90


def test_model_block_bundled_alias():
    block = runfolder.model_block("gemini31")
    assert block["name"].startswith("google/")
    assert "request_timeout" in block


def test_cli_unknown_model_errors(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    problems = repo / "tests" / "problems" / "z3"
    problems.mkdir(parents=True)
    (problems / "puzzle.md").write_text("Solve me.")
    monkeypatch.setattr(runfolder, "local_checkout", lambda: str(repo))
    with pytest.raises(SystemExit):
        runfolder.main(
            ["z3", "puzzle", str(tmp_path / "out"), "--model", "no-such-alias"]
        )


def test_model_block_rejects_custom_endpoint(tmp_path):
    model_file = tmp_path / "m.json"
    model_file.write_text(
        json.dumps({"path": "local/m", "base_url": "http://localhost:8000/v1"})
    )
    with pytest.raises(ValueError, match="base_url"):
        runfolder.model_block(str(model_file))


def test_clingo_tool_result_cap_fits_template(tmp_path):
    from mcp_solver.templates import get_template

    folder = runfolder.write_run_folder(
        tmp_path / "demo", "clingo", "x", tmp_path / "repo"
    )
    config = json.loads((folder / "config.json").read_text())
    cap = config["agent"]["max_tool_result_chars"]
    assert cap >= len(get_template("clingo"))
    assert cap >= runfolder.MIN_TOOL_RESULT_CHARS
    assert cap % 1000 == 0
    # minion's run-folder loader reads it into the agent config.
    from mcp_minion.cli import load_run_folder

    agent_config, _, _ = load_run_folder(folder)
    assert agent_config.max_tool_result_chars == cap


@pytest.mark.parametrize("solver", SOLVERS)
def test_tool_result_cap_rule(solver):
    from mcp_solver.templates import get_template

    cap = runfolder.tool_result_cap(solver)
    assert cap >= max(runfolder.MIN_TOOL_RESULT_CHARS, 2 * len(get_template(solver)))
    assert cap - 2 * len(get_template(solver)) < 1000 or cap == 64000
