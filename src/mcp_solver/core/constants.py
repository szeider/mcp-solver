from datetime import timedelta
from pathlib import Path


try:
    import tomllib
except ImportError:
    pass

# Potentially truncate items when sent back to client
ITEM_CHARS = None  # or None to show full items

# Set to True to enable validation on model changes
VALIDATE_ON_CHANGE = True

# Timeouts (Note: asyncio.timeout requires Python 3.11+)
MIN_SOLVE_TIMEOUT = timedelta(seconds=1)
MAX_SOLVE_TIMEOUT = timedelta(seconds=30)
VALIDATION_TIMEOUT = timedelta(seconds=2)
CLEANUP_TIMEOUT = timedelta(seconds=1)

# Get the project root directory (where the prompts folder is located)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Set the path to the prompts directory
PROMPTS_DIR = PROJECT_ROOT / "prompts"
