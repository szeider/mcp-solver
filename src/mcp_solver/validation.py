#!/usr/bin/env python3
"""
Shared Validation Module for Python-based Solvers

This module provides shared validation functions for Python-based solver components
(such as PySAT and Z3) to validate input parameters and code.
"""

import re
import logging
from typing import List, Tuple, Optional, Any, Dict, Union
from datetime import timedelta

# Configure logger
logger = logging.getLogger(__name__)

# Constants
MAX_CODE_LENGTH = 100000
MAX_ITEMS = 100

class ValidationError(Exception):
    """Base exception for validation errors."""
    pass

def validate_index(index: Any, existing_items: Optional[List[Tuple[int, str]]] = None, one_based: bool = True) -> None:
    """
    Validate a model item index.
    
    Args:
        index: The index to validate
        existing_items: Optional list of existing items to check for duplicates
        one_based: Whether indices are one-based (True) or zero-based (False)
        
    Raises:
        ValidationError: If the index is invalid
    """
    if not isinstance(index, int):
        msg = f"Index must be an integer, got {type(index).__name__}"
        logger.error(msg)
        raise ValidationError(msg)
    
    min_index = 1 if one_based else 0
    if index < min_index:
        msg = f"Index must be {'a positive integer' if one_based else 'non-negative'}, got {index}"
        logger.error(msg)
        raise ValidationError(msg)
    
    # Check if adding an item would exceed the maximum
    if existing_items is not None:
        if len(existing_items) >= MAX_ITEMS:
            # Check if we're replacing an existing item
            existing_indices = [i for i, _ in existing_items]
            if index not in existing_indices:
                msg = f"Maximum number of code items ({MAX_ITEMS}) exceeded"
                logger.error(msg)
                raise ValidationError(msg)

def validate_content(content: Any) -> None:
    """
    Validate model item content.
    
    Args:
        content: The content to validate
        
    Raises:
        ValidationError: If the content is invalid
    """
    if not isinstance(content, str):
        msg = f"Content must be a string, got {type(content).__name__}"
        logger.error(msg)
        raise ValidationError(msg)
    
    if not content.strip():
        msg = "Content cannot be empty or whitespace only"
        logger.error(msg)
        raise ValidationError(msg)
        
    if len(content) > MAX_CODE_LENGTH:
        msg = (f"Content exceeds maximum length of {MAX_CODE_LENGTH} characters "
               f"(got {len(content)} characters)")
        logger.error(msg)
        raise ValidationError(msg)

def validate_python_code_safety(content: str) -> None:
    """
    Validate Python code for potentially unsafe patterns.
    
    Args:
        content: The Python code to validate
        
    Raises:
        ValidationError: If unsafe code patterns are detected
    """
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
            raise ValidationError(msg)

def validate_timeout(timeout: Any, min_timeout: timedelta, max_timeout: timedelta) -> None:
    """
    Validate a solve timeout duration.
    
    Args:
        timeout: The timeout duration to validate
        min_timeout: The minimum allowed timeout
        max_timeout: The maximum allowed timeout
        
    Raises:
        ValidationError: If the timeout is invalid
    """
    if not isinstance(timeout, timedelta):
        msg = f"Timeout must be a timedelta, got {type(timeout).__name__}"
        logger.error(msg)
        raise ValidationError(msg)
    
    if timeout < min_timeout:
        msg = f"Timeout must be at least {min_timeout.total_seconds()} seconds"
        logger.error(msg)
        raise ValidationError(msg)
        
    if timeout > max_timeout:
        msg = f"Timeout must be at most {max_timeout.total_seconds()} seconds"
        logger.error(msg)
        raise ValidationError(msg)

def get_standardized_response(success: bool, message: str, error: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Create a standardized response dictionary for validation results.
    
    Args:
        success: Whether the operation was successful
        message: A message describing the result
        error: An optional error message
        **kwargs: Additional keyword arguments to include in the response
        
    Returns:
        A dictionary with standardized response fields
    """
    response = {
        "message": message,
        "success": success
    }
    
    if error:
        response["error"] = error
    
    # Add any additional keyword arguments to the response
    response.update(kwargs)
        
    return response
