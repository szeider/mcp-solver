# constants.py
import os
import sys
from pathlib import Path
from datetime import timedelta

try:
    import tomllib
except ImportError:
    import tomli as tomllib

def get_memo_path() -> Path:
    """Get memo file path from environment, pyproject.toml config, or platform default."""
    if env_path := os.environ.get("MCP_SOLVER_MEMO"):
        return Path(env_path).expanduser()

    # Check pyproject.toml in current directory
    if Path("pyproject.toml").exists():
        try:
            with open("pyproject.toml", "rb") as f:
                config = tomllib.load(f)
                if memo_path := config.get("tool", {}).get("mcp-solver", {}).get("memo_file"):
                    return Path(memo_path).expanduser()
        except tomllib.TOMLDecodeError:
            pass

    # Platform-specific defaults
    if sys.platform == "win32":
        base_path = Path(os.environ.get("APPDATA", ""))
        return base_path / "mcp-solver" / "memo.md"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "mcp-solver" / "memo.md"
    else:  # Linux and other Unix-like systems
        return Path.home() / ".config" / "mcp-solver" / "memo.md"

# Timeouts
DEFAULT_SOLVE_TIMEOUT = timedelta(seconds=4)
MAX_SOLVE_TIMEOUT = timedelta(seconds=10)
FLATTEN_TIMEOUT = timedelta(seconds=2)

# Get memo file path
MEMO_FILE = str(get_memo_path())




PROMPT_TEMPLATE = """Welcome to the MiniZinc Constraint Solver!

I'll help you formulate and solve constraint satisfaction problems.

## Available Tools:
- get_model/edit_model: View and edit your constraint model
- validate_model: Check model syntax and semantics
- solve_model: Execute solver with configurable timeout
- get_solution/get_variable: Retrieve results
- get_memo/edit_memo: Access solution our knowledge base

## The solver specializes in:
- Finite domain variables and constraints
- Global constraints (alldifferent, circuit, etc.)
- Logical constraints and reification
- Performance monitoring and timeout management

## General Instructions 
- Do not change the timeout, the default one should be fine.
- Do not add output statements

## Sanity Checks
- it is always good to make a quick check whethetr the solution makes sense


## Conversation Style
- Address queries directly without preamble
- Skip unnecessary pleasantries, apologies, and redundant offers
- Maintain clarity while minimizing word count
- Include supporting details only when they aid understanding
- Match the human's length and detail preferences 
- Preserve full quality in code, artifacts and technical content
- Handle errors matter-of-factly and move forward
- Express yourself if you have had an interesting insight 


## Memo Knowledge Base
Below is our collection of do's and don'ts. 
If you learn from a mistake or gain insights that could improve future runs, 
please add a concise statement to the knowledge base. 
You may also update or expand existing entries to maintain 
an effective and well-structured resource over time.
Keep entries as concise as possible.
{memo}
"""