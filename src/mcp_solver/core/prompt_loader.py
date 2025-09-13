import logging
from pathlib import Path
from typing import Literal


# Type definitions for better type checking
PromptMode = Literal["mzn", "pysat", "z3", "maxsat", "asp"]
PromptType = Literal["instructions", "review"]

logger = logging.getLogger(__name__)


def get_prompt_path(mode: PromptMode, prompt_type: PromptType = "instructions") -> Path:
    """
    Get the path to a prompt file based on mode and type, without loading its content.

    Args:
        mode: The solver mode ("mzn", "pysat", "z3", "maxsat", or "asp")
        prompt_type: The type of prompt ("instructions" or "review"), defaults to "instructions"

    Returns:
        Path object pointing to the prompt file

    Raises:
        ValueError: If invalid mode or prompt type is provided
    """
    # Validate inputs
    if mode not in ("mzn", "pysat", "z3", "maxsat", "asp"):
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of: mzn, pysat, z3, maxsat, asp"
        )

    if prompt_type not in ("instructions", "review"):
        raise ValueError(
            f"Invalid prompt type: {prompt_type}. Must be one of: instructions, review"
        )

    # Get the base prompts directory path
    base_path = Path(__file__).parent.parent.parent.parent / "prompts"

    # Construct the full path to the prompt file
    prompt_path = base_path / mode / f"{prompt_type}.md"

    logger.debug(f"Prompt path for {prompt_type} in {mode} mode: {prompt_path}")
    return prompt_path


def load_prompt(mode: PromptMode, prompt_type: PromptType) -> str:
    """
    Load a prompt file based on mode and type.

    Args:
        mode: The solver mode ("mzn", "pysat", "z3", "maxsat" or "asp")
        prompt_type: The type of prompt ("instructions" or "review")

    Returns:
        The content of the prompt file as a string

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        ValueError: If invalid mode or prompt type is provided
    """
    # Get the prompt path
    prompt_path = get_prompt_path(mode, prompt_type)

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompts directory not found at: {prompt_path.parent}")

    logger.debug(f"Loading {prompt_type} prompt for {mode} mode from: {prompt_path}")

    # Read and return the prompt content
    try:
        content = prompt_path.read_text(encoding="utf-8").strip()
        logger.debug(f"Successfully loaded prompt ({len(content)} characters)")
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading prompt file {prompt_path}: {e!s}")
