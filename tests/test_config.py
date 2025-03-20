"""
Shared configuration for all test files.
"""
import os

# Configuration
MCP_CLIENT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # Use the current project directory
DEFAULT_TIMEOUT = 300  # 5 minutes default timeout

# MiniZinc configuration
MZN_PROMPT_FILE = "instructions_prompt_lite.md"

# PySAT configuration
PYSAT_PROMPT_FILE = "instructions_prompt_pysat_lite.md"

# Z3 configuration
Z3_PROMPT_FILE = "instructions_prompt_z3_lite.md"

# Get absolute paths to key directories
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROBLEMS_DIR = os.path.join(os.path.dirname(__file__), "problems")
MZN_PROBLEMS_DIR = os.path.join(PROBLEMS_DIR, "mzn")
PYSAT_PROBLEMS_DIR = os.path.join(PROBLEMS_DIR, "pysat")
Z3_PROBLEMS_DIR = os.path.join(PROBLEMS_DIR, "z3")

def get_abs_path(rel_path):
    """Convert a path relative to the root directory to an absolute path."""
    return os.path.join(ROOT_DIR, rel_path)

def get_prompt_path(prompt_file):
    """Get the absolute path to a prompt file."""
    return get_abs_path(prompt_file) 