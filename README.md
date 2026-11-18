# MCP Solver

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Python Version](https://img.shields.io/badge/Python-3.13%20%7C%203.14-blue.svg)](https://www.python.org/)

MCP Solver is an MCP server for constraint solving with SAT, MaxSAT, SMT,
constraint programming, answer set programming and dynamic programming. You
describe a problem in plain words to your MCP host, such as Claude Desktop or
Claude Code. The host's language model writes a Python program for one of six
backends, each built on a solver library, and runs it in a kernel, a Python
process that the server starts. When the host submits the program, the server
runs it once more on its own, and you get the solution as JSON together with
the program.

This is version 4. Version 3, described in the SAT 2025 paper, works
differently (see [From v3 to v4](#from-v3-to-v4)). To connect the server to
your host, see [Quick start](#quick-start).

![Architecture of MCP Solver. You give your problem to an MCP host, such as Claude Desktop or Claude Code, or to mcp-minion, the project's scripted agent. Either one calls the tools select_backend, python_exec and submit_code of mcp-solver-serve over MCP. The server runs the code in a Python kernel with the solver library of the chosen backend (pysat, maxsat, z3, cpmpy, clingo or didp), and the accepted program and its solution JSON go back to whichever of the two called it.](https://raw.githubusercontent.com/szeider/mcp-solver/main/assets/architecture.svg)

## Example

The following small scheduling problem is shortened from
[`tests/problems/clingo/shift_assignment.md`](tests/problems/clingo/shift_assignment.md).
The run below solved it with the `clingo` backend (answer set programming).

> Assign two employees, Anna and Ben, to three shifts: morning, afternoon,
> evening. Each shift gets exactly one employee, and nobody works more than
> two shifts. Anna prefers all three shifts, Ben only the morning. Maximize
> the number of shifts that go to an employee who prefers them. Return JSON
> with `satisfiable`, `assignment` and `satisfied_preferences`.

The program printed this solution:

```json
{"satisfiable": true, "assignment": {"afternoon": "anna", "evening": "anna", "morning": "ben"}, "satisfied_preferences": 3}
```

The submitted program has 98 lines. This excerpt shows its shape:

```python
import json
import itertools
import clingo

# ... 6 lines: EMPLOYEES, SHIFTS and PREFERENCES, the instance data
ASP_RULES = """
1 { assigned(Shift,Employee) : employee(Employee) } 1 :- shift(Shift).
:- employee(Employee), 3 { assigned(Shift,Employee) : shift(Shift) }.
satisfied(Shift) :- assigned(Shift,Employee), prefers(Employee,Shift).
preference_score(Score) :- Score = #count { Shift : satisfied(Shift) }.
#maximize { 1,Shift : satisfied(Shift) }.
"""

# ... 2 lines: quote(value) writes a name as a quoted string for clingo

def solve():
    # ... 28 lines: add the data as facts, ground, keep each model in on_model
    result = control.solve(on_model=on_model)
    if not result.satisfiable:
        return {"satisfiable": False}
    if not result.exhausted:
        raise RuntimeError("Optimization did not finish, so optimality is unproven")
    return best

def verify(solution):
    # ... 24 lines: check each constraint and the score, try all assignments
    assert actual_score == max(feasible_scores), "assignment is not preference-optimal"

def main(check=False):
    solution = solve()
    if check:
        verify(solution)
    print(json.dumps(solution))

if __name__ == "__main__":
    main()
```

This example comes from one run of the command
`mcp-solver clingo --problem tests/problems/clingo/shift_assignment.md` (see
[Command line](#command-line-and-mcp-minion)) on version 4.5.0 with the
default model, `openai/gpt-5.6-terra`. The problem's validator accepted the
solution.

## Quick start

You need Python 3.13 or 3.14 and [uv](https://docs.astral.sh/uv/) on the
PATH (the server starts its kernels through uv). The server runs no language
model and needs no API key. The PyPI package is `agentic-mcp-solver`. Do not
install the PyPI package `mcp-solver`, a third-party upload of stale v2.0.0
code (a [PEP 541](https://peps.python.org/pep-0541/) transfer request is
pending). The commands keep the names `mcp-solver` and `mcp-solver-serve`.

**Claude Desktop.** Add the server to `claude_desktop_config.json`. uvx
fetches the package from PyPI, so there is no install step:

```json
{
  "mcpServers": {
    "mcp-solver": {
      "command": "uvx",
      "args": ["--python", "3.14", "--from", "agentic-mcp-solver[agent]", "mcp-solver-serve"]
    }
  }
}
```

Restart Claude Desktop after editing the file. If it cannot find `uvx`, put
the absolute path of `uvx` (the output of `which uvx`) in `command`.

**Claude Code.** Register the server once (add `-s user` to make it
available in all projects):

```bash
claude mcp add mcp-solver -- uvx --python 3.14 --from "agentic-mcp-solver[agent]" mcp-solver-serve
```

**Any other MCP host.** Configure a stdio server with the same command.

Then describe your problem to the host and ask it to use MCP Solver. You do
not write the program yourself. To install the commands instead, run
`uv tool install --python 3.14 "agentic-mcp-solver[agent]"`. For a source
checkout and for troubleshooting, see [INSTALL.md](INSTALL.md).

## Backends

| Backend  | Problem class                          | Library    | Typical use                                      |
| -------- | -------------------------------------- | ---------- | ------------------------------------------------ |
| `pysat`  | Boolean satisfiability (SAT)           | python-sat | placements, colorings, covers                    |
| `maxsat` | Weighted MaxSAT                        | python-sat | hard constraints plus weighted preferences       |
| `z3`     | SMT                                    | z3-solver  | integer and real arithmetic, verification        |
| `cpmpy`  | Constraint programming (CP)            | CPMpy      | rosters, timetables, assignment                  |
| `clingo` | Answer set programming (ASP)           | clingo     | rules with defaults and exceptions, reachability |
| `didp`   | Domain-independent dynamic programming | didppy     | routing with time windows, sequencing, packing   |

The kernel starts with the packages of the chosen backend: `python-sat[pb]`,
`z3-solver`, `cpmpy`, `clingo`, or `didppy>=0.11.1,<0.12`. A `didp` model is
a state transition system ([DIDP](https://didp.ai/), Kuroiwa and Beck). By
default the program solves it with didppy's anytime beam search (CABS).

## How it works

1. **Select a backend.** `select_backend(solver)` takes a backend name such
   as `clingo`. It starts the kernel, an IPython kernel that keeps its state
   between calls, with that backend's library. It returns the backend's
   modeling instructions, called the template.
2. **Write and run.** The host runs code in the kernel with `python_exec`,
   builds the constraint model step by step, and runs the solver on it.
3. **Check.** The template asks for a program in three parts. `solve()`
   builds and solves the constraint model and returns the solution as a
   dict. `verify(solution)` re-checks every constraint and the objective
   value in plain Python and raises `AssertionError` when a check fails.
   `main(check=False)` runs `solve()`, runs `verify()` only when `check` is
   true, and prints the solution as JSON. A `__main__` guard calls `main()`.
   In the kernel, the host runs `main(check=True)` as the final test.
4. **Submit.** `submit_code(code)` takes the final program and passes it
   through the submission gate. The server checks that the program compiles
   and then runs it once on its own, in a fresh temporary directory with
   `uv run --isolated` and the kernel's packages. This standalone run has
   none of the kernel's state, may take at most 240 s and may print at most
   4 MB. If it exits nonzero, times out, prints too much or prints nothing,
   the server rejects the submission and names the reason, for example the
   last traceback line. The host can then fix the program and submit it
   again.
5. **Self-check.** After an accepted run, the server loads the program in
   a second fresh environment, calls `main(check=True)` within what is left
   of the 240 s, and reports the outcome as `self_check`: `passed`,
   `failed` or `not_available` (for example when `main()` takes no `check`
   argument). The self-check is information only. A failed self-check does
   not reject the submission, since a wrong `verify()` can fail a right
   solution.

The server does not compare the solution with the problem. An accepted
submission means that the program ran without the kernel's state, exited
with code 0 and printed output.

The kernels come from `ipython_mcp`, the kernel server of
[agentic-python-coder](https://github.com/szeider/agentic-python-coder) 4
(`>=4,<5`), which `mcp-solver-serve` controls. Below, "engine" means
agentic-python-coder.

## Command line and mcp-minion

**mcp-minion** is the project's scripted agent, a separate PyPI package
developed in [`minion/`](minion/). It does what an MCP host does, driven by a
script instead of a chat, and calls a language model through
[OpenRouter](https://openrouter.ai) with your API key. The `mcp-solver`
command uses it to solve a problem without an MCP host:

```bash
mcp-solver pysat "Place 8 non-attacking queens on an 8x8 board and give one solution."
mcp-solver pysat --problem tests/problems/pysat/n_queens.md
```

Here mcp-minion talks to the engine's kernel server directly, with the
template as its system prompt. The CLI writes the submitted program to
`<basename>_code.py` in the working directory, where `<basename>` is the
problem file's name without `.md` (`n_queens_code.py` above) or `task` for a
task given on the command line. The CLI then makes a standalone run of the
program with `uv run --isolated` (360 s limit) and prints its output, the
solution JSON, to stdout. A run log `run_*.json` goes to the working
directory too.

| Option                | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| `--problem FILE`      | Read the task from a markdown file instead of the command line |
| `--model NAME`        | Agent model: an alias, a model `.json` file, or a full OpenRouter ID (default `gpt56terra`, i.e. `openai/gpt-5.6-terra`) |
| `--workdir DIR`       | Working directory for the run (default: current directory)   |
| `--step-limit N`      | Maximum agent steps (default 30)                             |
| `--token-limit N`     | Maximum input plus output tokens of the run, 0 for none (default 2,000,000) |
| `-q`, `--quiet`       | Suppress progress output                                     |
| `--dev [PATH]`        | Take templates from a source checkout (on by default inside one, or set `MCP_SOLVER_DEV`) |
| `--no-dev`            | Use the packaged templates                                   |
| `--stats-json FILE`   | Write the run statistics, including tokens and cost, as JSON |

mcp-minion can also drive `mcp-solver-serve` the way an MCP host does, from a
run folder that holds `config.json` and `task.md` (see
[minion/README.md](minion/README.md)). In a source checkout,
`mcp-solver-runfolder clingo shift_assignment ~/demos/shift` writes one for a
bundled test problem, and
`uv run --project /path/to/mcp-solver --extra agent mcp-minion -v ~/demos/shift`
runs it.

The `mcp-solver` command looks for `OPENROUTER_API_KEY` in the environment,
in `~/.config/coder/.env` and in `~/.mcp-minion`. The standalone
`mcp-minion` command looks in the environment, in the run folder's `.env`
and in `~/.mcp-minion`. A key in the environment or in `~/.mcp-minion` (a
line `OPENROUTER_API_KEY=...`) serves both.

## The MCP server in detail

`mcp-solver-serve` speaks MCP over stdio, takes no arguments and has six tools:

- **`select_backend(solver)`** takes a backend name and creates the kernel
  or restarts it for a new problem or backend. The engine knows this kernel
  by the id `solver`. The result is the full template as text, followed by a
  note. The note says whether the kernel is fresh or was recycled, names its
  id and packages, and asks the host to run the complete program with
  `python_exec` before calling `submit_code`.
- **`python_exec(code, timeout=30, kernel_id=None)`** runs code in the
  kernel. The server limits its `timeout` to 300 s.
- **`python_reset(packages=None, kernel_id=None)`** restarts the kernel.
  Extra `packages` are installed next to the backend's libraries, which stay.
- **`python_status(kernel_id=None)`** lists kernels, variables and packages.
- **`python_interrupt(kernel_id=None)`** stops running code and keeps the
  kernel's state.
- **`submit_code(code)`** runs the submission gate and the self-check of
  steps 4 and 5 in [How it works](#how-it-works). The reply to an accepted
  submission contains `executed`, `exit_code`, `stdout` (the last 20,000
  characters), `self_check` (its status and a message) and a link to the
  stored program, `mcp-solver://submissions/{id}` (the
  server keeps the last 50). When stdout is longer, the reply says that it
  shows only the end. The server returns a rejected submission as an error
  result and does not store it. Before any `select_backend`, the check is
  compile-only and the reply says so.

After `select_backend`, kernel tool calls without `kernel_id` act on this
kernel. A call with an explicit `kernel_id` goes to the engine unchanged. The
server also offers the resources `mcp-solver://guide` (workflow, backend
choice, time limits), `mcp-solver://template/{solver}` (each template,
without selecting a backend) and the stored submissions.

**Statistics.** Set `MCP_SOLVER_STATS=/path/to/stats.jsonl` in the `env`
block of the host's server configuration (a variable set only in your shell
may not reach the server). The server then writes one JSON line per episode
(from one `select_backend` call to the next, or to server shutdown) with
tool-call counts, execution failures, submit attempts and successes,
`python_exec` calls before the first submission, the self-check status of
the last accepted submission, and wall time.
`submit_code` results contain a compact snapshot as structured content. The
server cannot see how many tokens the host uses, since MCP does not pass that
on, so compare hosts by their tool usage.

**Environment.** Kernels and standalone runs get a filtered environment: a
short whitelist (`HOME`, `PATH`, locale, temp, proxy, CA bundle) and the
prefixes `UV_`, `CODER_`, `MCP_SOLVER_`, `XDG_`, `LC_`, minus names that
contain `KEY`, `TOKEN`, `SECRET`, `PASSWORD` or `CREDENTIAL`. API keys in
the environment therefore do not reach model-written code. That code still
runs with your file permissions and can read credentials stored in files or
inside a passed URL. The filter is not a sandbox.

## Benchmark harness

The repository contains 30 test problems in `tests/problems/<backend>/`,
each with an output-format section and a `*_ground_truth.py` validator that
reads the solution JSON on stdin and returns `{"valid": ..., "message": ...}`.
Validators of optimization problems recompute the optimum.
`mcp-solver-bench` solves the problems through the CLI and validates every
result:

```bash
mcp-solver-bench                               # every backend
mcp-solver-bench pysat z3 --runs 3 --jobs 4    # two backends, 3 runs each, 4 in parallel
```

A run counts as valid only if the validator accepts the solution and the
program imports its backend's library. Results are appended to
`results.jsonl` in the output directory (default
`bench_results/<timestamp>`). The same problems run as pytest end-to-end
tests, for example with `uv run pytest minion/e2e_tests -k didp`.

On two larger benchmarks, Claude Code with Claude Opus 5.5 as the host and
`mcp-solver-serve` 4.5.0 solved all 229 problems, each in one session with
one submission: 101 of 101 in CP-Bench v3 (cpmpy backend) and 128 of 128 in
ASP-Bench v1.2 (clingo backend). The benchmark's validator checked every
solution. A session took 40 s (CP-Bench) and 59 s (ASP-Bench) in the median.
The templates were revised while these benchmarks were run, in two places
after a failure on a single problem, so the figures describe the released
templates on these problems and are not a held-out test.

## From v3 to v4

The SAT 2025 paper, ["Bridging Language Models and Symbolic Solvers via the
Model Context Protocol"](https://doi.org/10.4230/LIPIcs.SAT.2025.30),
describes v3, an MCP server with model-editing tools (`add_item`,
`replace_item`, `solve_model`, ...) through which the host's language model
builds a solver model item by item for the server to solve. The v3 code is
still on the [`v3`](https://github.com/szeider/mcp-solver/tree/v3) branch,
unchanged. v4 replaces these tools with the execution of a complete Python
program and follows the IPython-kernel approach of the
[CP-Agent](https://doi.org/10.1145/3786181.3788711) and
[ASP-Bench](https://doi.org/10.1145/3786168.3788402) papers.

|                   | v3 (SAT 2025 paper)                          | v4                                                     |
| ----------------- | -------------------------------------------- | ------------------------------------------------------ |
| Who solves        | The host's language model, via model-editing tools | The host's language model (or mcp-minion), as a solver programmer |
| Unit of work      | Model items (`add_item`, `solve_model`, ...) | A complete, runnable Python program                    |
| Execution         | Inside the MCP server                        | In a kernel, then once more in a standalone run        |
| Checking          | Left to the host's language model            | `verify()` in the program, run by the server as a self-check |
| Result            | Transient model state                        | A program you keep, plus its solution                  |
| Backends          | MiniZinc, PySAT, MaxSAT, Z3, ASP             | PySAT, MaxSAT, Z3, CPMpy, clingo, DIDP                 |

## Citations

- Stefan Szeider, ["Bridging Language Models and Symbolic Solvers via the
  Model Context Protocol"](https://doi.org/10.4230/LIPIcs.SAT.2025.30), SAT
  2025. Describes v3.
- Stefan Szeider, ["CP-Agent: Agentic Constraint
  Programming"](https://doi.org/10.1145/3786181.3788711), in *Proceedings of
  the 3rd International Workshop on Large Language Models For Code (LLM4Code
  2026)*, pages 6–13, ACM, 2026. The IPython-kernel approach that v4 uses.
- Stefan Szeider, ["ASP-Bench: From Natural Language to Logic
  Programs"](https://doi.org/10.1145/3786168.3788402), in *Proceedings of the
  2026 IEEE/ACM 2nd International Workshop on Neuro-Symbolic Software
  Engineering (NSE '26)*, pages 1–8, ACM, 2026. Its benchmark accepts a
  solution only after a validator has checked it, as the test problems of v4
  do.

## License

MIT. See [LICENSE](LICENSE).
