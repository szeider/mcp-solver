#!/usr/bin/env python3
"""
PySAT Model Validation Module

This module provides validation functions for input to the PySAT solver components.
"""

import re
import logging
from typing import List, Tuple, Optional, Any

# Configure logger
logger = logging.getLogger(__name__)

# Constants
MAX_CODE_LENGTH = 100000
MAX_ITEMS = 100

class ModelValidationError(Exception):
    """Exception raised for model validation errors."""
    pass

def validate_index(index: Any, existing_items: Optional[List[Tuple[int, str]]] = None) -> None:
    """
    Validate a model item index.
    
    Args:
        index: The index to validate
        existing_items: Optional list of existing items to check for duplicates
        
    Raises:
        ModelValidationError: If the index is invalid
    """
    if not isinstance(index, int):
        msg = f"Index must be an integer, got {type(index).__name__}"
        logger.error(msg)
        raise ModelValidationError(msg)
    
    if index <= 0:
        msg = f"Index must be a positive integer, got {index}"
        logger.error(msg)
        raise ModelValidationError(msg)
    
    # Check if adding an item would exceed the maximum
    if existing_items is not None:
        if len(existing_items) >= MAX_ITEMS and index not in [i for i, _ in existing_items]:
            msg = f"Maximum number of code items ({MAX_ITEMS}) exceeded"
            logger.error(msg)
            raise ModelValidationError(msg)

def validate_content(content: Any) -> None:
    """
    Validate model item content.
    
    Args:
        content: The content to validate
        
    Raises:
        ModelValidationError: If the content is invalid
    """
    if not isinstance(content, str):
        msg = f"Content must be a string, got {type(content).__name__}"
        logger.error(msg)
        raise ModelValidationError(msg)
    
    if not content.strip():
        msg = "Content cannot be empty or whitespace only"
        logger.error(msg)
        raise ModelValidationError(msg)
        
    if len(content) > MAX_CODE_LENGTH:
        msg = (f"Content exceeds maximum length of {MAX_CODE_LENGTH} characters "
               f"(got {len(content)} characters)")
        logger.error(msg)
        raise ModelValidationError(msg)
    
    # Check for potentially unsafe code
    unsafe_patterns = [
        (r"__import__\s*\(", "Usage of __import__ is not allowed"),
        (r"eval\s*\(", "Usage of eval() is not allowed"),
        (r"exec\s*\(", "Usage of exec() is not allowed"),
        (r"os\.(system|popen|execl|execle|execlp|popen|spawn)", "Direct OS command execution is not allowed"),
        (r"subprocess", "Usage of subprocess module is not allowed"),
        (r"open\s*\(.+?[\"']w[\"']", "Writing to files is not allowed"),
    ]
    
    for pattern, message in unsafe_patterns:
        if re.search(pattern, content):
            msg = f"Validation failed: {message}"
            logger.error(msg)
            raise ModelValidationError(msg) 