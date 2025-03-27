"""
Shared configuration for all test files.
"""
import os
from pathlib import Path

# Configuration
MCP_CLIENT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # Use the current project directory
DEFAULT_TIMEOUT = 300  # 5 minutes default timeout

# Updated prompt file paths for the new structure
MZN_PROMPT_FILE = "prompts/mzn/instructions.md"
PYSAT_PROMPT_FILE = "prompts/pysat/instructions.md"
Z3_PROMPT_FILE = "prompts/z3/instructions.md"

# Get absolute paths to key directories
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROBLEMS_DIR = os.path.join(os.path.dirname(__file__), "problems")
MZN_PROBLEMS_DIR = os.path.join(PROBLEMS_DIR, "mzn")
PYSAT_PROBLEMS_DIR = os.path.join(PROBLEMS_DIR, "pysat")
Z3_PROBLEMS_DIR = os.path.join(PROBLEMS_DIR, "z3")
RESULTS_DIR = os.path.join(ROOT_DIR, "test_results")

def get_abs_path(rel_path):
    """Convert a path relative to the root directory to an absolute path."""
    return os.path.join(ROOT_DIR, rel_path)

def get_prompt_path(prompt_file):
    """Get the absolute path to a prompt file."""
    return get_abs_path(prompt_file)

# Helper function to load a prompt using the centralized loader
def load_prompt_for_test(mode, prompt_type="instructions"):
    """Load a prompt using the centralized prompt loader."""
    try:
        from mcp_solver.core.prompt_loader import load_prompt
        return load_prompt(mode, prompt_type)
    except ImportError:
        # If the prompt loader isn't available, fall back to file reading
        prompt_path = get_abs_path(f"prompts/{mode}/{prompt_type}.md")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        # If all else fails, try to find the old-style prompt files
        old_style_paths = {
            "mzn": {"instructions": "instructions_prompt_mzn.md"},
            "pysat": {"instructions": "instructions_prompt_pysat.md"},
            "z3": {"instructions": "instructions_prompt_z3.md"}
        }
        
        if mode in old_style_paths and prompt_type in old_style_paths[mode]:
            old_path = get_abs_path(old_style_paths[mode][prompt_type])
            with open(old_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        else:
            raise ValueError(f"Could not load prompt for {mode}/{prompt_type}: {str(e)}") 