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
ITEM_CHARS = None  # or None to show full items

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

# Load instructions prompt from file instructions_prompt.md
try:
    # Adjust the relative path if needed.
    # Here, we assume this file is at: <repo_root>/instructions_prompt.md
    instructions_path = Path(__file__).resolve().parents[2] / "instructions_prompt.md"
    with open(instructions_path, "r", encoding="utf-8") as f:
        INSTRCUTIONS_PROMPT = f.read()
except Exception as e:
    INSTRCUTIONS_PROMPT = f"Error loading instructions prompt: {e}"
