import os
import sys
from pathlib import Path
from datetime import timedelta

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# Potentially truncate items when sent back to client
ITEM_CHARS = None  # or None to show full items

# Set to True to enable validation on model changes
VALIDATE_ON_CHANGE = True  

# Timeouts (Note: asyncio.timeout requires Python 3.11+)
MIN_SOLVE_TIMEOUT = timedelta(seconds=1)
MAX_SOLVE_TIMEOUT = timedelta(seconds=10)
VALIDATION_TIMEOUT = timedelta(seconds=2)
CLEANUP_TIMEOUT = timedelta(seconds=1)

# Store the path to the instructions prompt file
instructions_path = Path(__file__).resolve().parents[2] / "instructions_prompt.md"
INSTRUCTIONS_PROMPT = str(instructions_path)
