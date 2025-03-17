#!/usr/bin/env python3
"""
Test file for PySAT model input validation.

This file contains tests for validating input to the PySAT model manager,
including index and content validation.
"""

import os
import sys
import unittest
import re
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Constants for validation tests
MAX_CODE_LENGTH = 100000
MAX_ITEMS = 100

class ModelValidationError(Exception):
    """Exception raised for model validation errors."""
    pass

def validate_index(index, existing_items=None):
    """
    Validate a model item index.
    
    Args:
        index: The index to validate
        existing_items: Optional list of existing items to check for duplicates
        
    Raises:
        ModelValidationError: If the index is invalid
    """
    if not isinstance(index, int):
        raise ModelValidationError(f"Index must be an integer, got {type(index).__name__}")
    
    if index <= 0:
        raise ModelValidationError(f"Index must be a positive integer, got {index}")
    
    # Check if adding an item would exceed the maximum
    if existing_items is not None:
        if len(existing_items) >= MAX_ITEMS and index not in [i for i, _ in existing_items]:
            raise ModelValidationError(f"Maximum number of code items ({MAX_ITEMS}) exceeded")

def validate_content(content):
    """
    Validate model item content.
    
    Args:
        content: The content to validate
        
    Raises:
        ModelValidationError: If the content is invalid
    """
    if not isinstance(content, str):
        raise ModelValidationError(f"Content must be a string, got {type(content).__name__}")
    
    if not content.strip():
        raise ModelValidationError("Content cannot be empty or whitespace only")
        
    if len(content) > MAX_CODE_LENGTH:
        raise ModelValidationError(
            f"Content exceeds maximum length of {MAX_CODE_LENGTH} characters "
            f"(got {len(content)} characters)"
        )
    
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
            raise ModelValidationError(f"Validation failed: {message}")

class TestModelValidation(unittest.TestCase):
    """Test suite for model input validation."""
    
    def test_validate_index(self):
        """Test index validation."""
        # Valid indices
        validate_index(1)
        validate_index(10)
        validate_index(100)
        
        # Invalid types
        with self.assertRaises(ModelValidationError) as ctx:
            validate_index("1")
        self.assertIn("must be an integer", str(ctx.exception))
        
        with self.assertRaises(ModelValidationError) as ctx:
            validate_index(None)
        self.assertIn("must be an integer", str(ctx.exception))
        
        # Invalid values
        with self.assertRaises(ModelValidationError) as ctx:
            validate_index(0)
        self.assertIn("must be a positive integer", str(ctx.exception))
        
        with self.assertRaises(ModelValidationError) as ctx:
            validate_index(-1)
        self.assertIn("must be a positive integer", str(ctx.exception))
        
        # Maximum items check
        mock_items = [(i, "") for i in range(1, MAX_ITEMS + 1)]
        with self.assertRaises(ModelValidationError) as ctx:
            validate_index(MAX_ITEMS + 1, mock_items)
        self.assertIn(f"Maximum number of code items ({MAX_ITEMS}) exceeded", str(ctx.exception))
        
        # Valid when index already exists
        mock_items = [(i, "") for i in range(1, MAX_ITEMS + 1)]
        validate_index(1, mock_items)  # Should not raise, index already exists
    
    def test_validate_content(self):
        """Test content validation."""
        # Valid content
        validate_content("print('Hello, world!')")
        validate_content("formula = CNF()\nformula.append([1, 2, 3])")
        
        # Invalid types
        with self.assertRaises(ModelValidationError) as ctx:
            validate_content(None)
        self.assertIn("must be a string", str(ctx.exception))
        
        with self.assertRaises(ModelValidationError) as ctx:
            validate_content(123)
        self.assertIn("must be a string", str(ctx.exception))
        
        # Empty content
        with self.assertRaises(ModelValidationError) as ctx:
            validate_content("")
        self.assertIn("cannot be empty", str(ctx.exception))
        
        with self.assertRaises(ModelValidationError) as ctx:
            validate_content("   \n   ")
        self.assertIn("cannot be empty", str(ctx.exception))
        
        # Content too long
        long_content = "x" * (MAX_CODE_LENGTH + 1)
        with self.assertRaises(ModelValidationError) as ctx:
            validate_content(long_content)
        self.assertIn("exceeds maximum length", str(ctx.exception))
        
        # Unsafe patterns
        unsafe_code_samples = [
            ("__import__('os')", "Usage of __import__ is not allowed"),
            ("eval('2 + 2')", "Usage of eval() is not allowed"),
            ("exec('print(\"hello\")')", "Usage of exec() is not allowed"),
            ("os.system('ls')", "Direct OS command execution is not allowed"),
            ("subprocess.run(['ls'])", "Usage of subprocess module is not allowed"),
            ("open('file.txt', 'w')", "Writing to files is not allowed"),
        ]
        
        for code, expected_message in unsafe_code_samples:
            with self.assertRaises(ModelValidationError) as ctx:
                validate_content(code)
            self.assertIn(expected_message, str(ctx.exception))

if __name__ == "__main__":
    unittest.main() 