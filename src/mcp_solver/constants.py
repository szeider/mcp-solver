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


# Potentially truncate items when sent back to client
ITEM_CHARS = 8  # or None to show full items

# Set to True to enable validation on model changes
VALIDATE_ON_CHANGE = True  

# Timeouts (Note: asyncio.timeout requires Python 3.11+)
DEFAULT_SOLVE_TIMEOUT = timedelta(seconds=4)
MAX_SOLVE_TIMEOUT = timedelta(seconds=10)
FLATTEN_TIMEOUT = timedelta(seconds=2)
VALIDATION_TIMEOUT = timedelta(seconds=2)
CLEANUP_TIMEOUT = timedelta(seconds=1)

# Get memo file path
MEMO_FILE = str(get_memo_path())

PROMPT_TEMPLATE = """Welcome to the MiniZinc Constraint Solver!

## Available Tools:
- get_model: View current model items
- add_item: Add new item at specific index
- delete_item: Delete item at index
- replace_item: Replace item at index
- clear_model: Reset model
- solve_model: Solve with Chuffed solver
- get_solution: Get solution variable value
- get_solve_time: Get last solve time
- get_memo/edit_memo: Access knowledge base

## Model Structure
- An "item" is one MiniZinc statement (variable declaration, constraint, etc.)
- A comment isn't an item on its own, it belongs to some minizinc statement
- Each item typically ends with a semicolon
- Add items one at a time using add_item
- Place comments on same line, starting at column 45
- Keep solve statement as last item
- Do not add output statements

## Rules
- Item indices start at 0
- Default timeout should not be changed

## The solver specializes in:
- Finite domain variables and constraints
- Global constraints (alldifferent, circuit, etc.)
- Logical constraints and reification
- Performance monitoring and timeout management

## Verify Solution
- Always verify solutions against ALL constraints
- Document any constraint violations
- Flag unexpected or counter-intuitive solutions
- Document insights from failed attempts
- Verify solution feasibility

## Concide Conversation Style
- Direct, clear responses
- Essential details only
- Focus on accuracy and correctness
- Handle errors pragmatically
- Share insights when discovered
- Don't apologize but explain the reason for failure

## Memo Knowledge Base
Below is our collection of do's and don'ts.
If you learn from a mistake or gain insights that could improve 
the modelling of other problems then add a concise statement 
to the knowledge base.

You get the content of the knowledge base in the initial prompt, 
so you don't need to run get_memo for that

{memo}
"""
