from pathlib import Path
import logging
from typing import Literal

# Type definitions for better type checking
PromptMode = Literal["mzn", "pysat", "z3"]
PromptType = Literal["instructions", "review"]

logger = logging.getLogger(__name__)

def load_prompt(mode: PromptMode, prompt_type: PromptType) -> str:
    """
    Load a prompt file based on mode and type.
    
    Args:
        mode: The solver mode ("mzn", "pysat", or "z3")
        prompt_type: The type of prompt ("instructions" or "review")
    
    Returns:
        The content of the prompt file as a string
    
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        ValueError: If invalid mode or prompt type is provided
    """
    # Validate inputs
    if mode not in ("mzn", "pysat", "z3"):
        raise ValueError(f"Invalid mode: {mode}. Must be one of: mzn, pysat, z3")
    
    if prompt_type not in ("instructions", "review"):
        raise ValueError(f"Invalid prompt type: {prompt_type}. Must be one of: instructions, review")
    
    # Get the base prompts directory path
    base_path = Path(__file__).parent.parent.parent.parent / "prompts"
    
    if not base_path.exists():
        raise FileNotFoundError(f"Prompts directory not found at: {base_path}")
    
    # Construct the full path to the prompt file
    prompt_path = base_path / mode / f"{prompt_type}.md"
    
    logger.debug(f"Loading {prompt_type} prompt for {mode} mode from: {prompt_path}")
    
    # Read and return the prompt content
    try:
        content = prompt_path.read_text(encoding="utf-8").strip()
        logger.debug(f"Successfully loaded prompt ({len(content)} characters)")
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading prompt file {prompt_path}: {str(e)}") 