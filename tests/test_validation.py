#!/usr/bin/env python3
"""
Tests for the shared validation module.
"""

import pytest
from datetime import timedelta

from mcp_solver.validation import (
    validate_index, 
    validate_content, 
    validate_python_code_safety, 
    validate_timeout,
    ValidationError,
    get_standardized_response
)

class TestValidation:
    """Tests for the shared validation module."""
    
    def test_validate_index(self):
        """Test index validation."""
        # Valid indices
        validate_index(1, one_based=True)
        validate_index(100, one_based=True)
        validate_index(0, one_based=False)
        
        # Invalid indices
        with pytest.raises(ValidationError):
            validate_index("1", one_based=True)  # Not an integer
        
        with pytest.raises(ValidationError):
            validate_index(0, one_based=True)  # Too small for one-based
        
        with pytest.raises(ValidationError):
            validate_index(-1, one_based=False)  # Too small for zero-based
        
        # Test maximum items check
        existing_items = [(i, f"Item {i}") for i in range(1, 101)]  # 100 items
        
        with pytest.raises(ValidationError):
            validate_index(101, existing_items, one_based=True)  # Would exceed max
        
        # But replacing an existing item should work
        validate_index(50, existing_items, one_based=True)  # Replacing existing
    
    def test_validate_content(self):
        """Test content validation."""
        # Valid content
        validate_content("print('Hello, world!')")
        validate_content("x = 42")
        
        # Invalid content
        with pytest.raises(ValidationError):
            validate_content(42)  # Not a string
        
        with pytest.raises(ValidationError):
            validate_content("")  # Empty
        
        with pytest.raises(ValidationError):
            validate_content("   ")  # Whitespace only
        
        # Test maximum length
        long_content = "x" * 100001  # Exceeds MAX_CODE_LENGTH
        with pytest.raises(ValidationError):
            validate_content(long_content)
    
    def test_validate_python_code_safety(self):
        """Test Python code safety validation."""
        # Safe code
        validate_python_code_safety("print('Hello, world!')")
        validate_python_code_safety("x = 42")
        validate_python_code_safety("def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)")
        
        # Unsafe code
        unsafe_patterns = [
            "__import__('os')",
            "eval('2 + 2')",
            "exec('print(\"Hello\")')",
            "os.system('rm -rf /')",
            "import subprocess; subprocess.run(['ls'])",
            "open('file.txt', 'w').write('data')",
        ]
        
        for code in unsafe_patterns:
            with pytest.raises(ValidationError):
                validate_python_code_safety(code)
    
    def test_validate_timeout(self):
        """Test timeout validation."""
        min_timeout = timedelta(seconds=1)
        max_timeout = timedelta(seconds=10)
        
        # Valid timeouts
        validate_timeout(timedelta(seconds=1), min_timeout, max_timeout)
        validate_timeout(timedelta(seconds=5), min_timeout, max_timeout)
        validate_timeout(timedelta(seconds=10), min_timeout, max_timeout)
        
        # Invalid timeouts
        with pytest.raises(ValidationError):
            validate_timeout("1s", min_timeout, max_timeout)  # Not a timedelta
        
        with pytest.raises(ValidationError):
            validate_timeout(timedelta(milliseconds=500), min_timeout, max_timeout)  # Too small
        
        with pytest.raises(ValidationError):
            validate_timeout(timedelta(seconds=11), min_timeout, max_timeout)  # Too large
    
    def test_get_standardized_response(self):
        """Test standardized response formatting."""
        # Success response
        response = get_standardized_response(True, "Operation successful")
        assert response["success"] is True
        assert response["message"] == "Operation successful"
        assert "error" not in response
        
        # Error response
        response = get_standardized_response(False, "Operation failed", "Error details")
        assert response["success"] is False
        assert response["message"] == "Operation failed"
        assert response["error"] == "Error details"
        
        # Additional fields
        response = get_standardized_response(True, "Data retrieved", value=42, items=["a", "b", "c"])
        assert response["success"] is True
        assert response["message"] == "Data retrieved"
        assert response["value"] == 42
        assert response["items"] == ["a", "b", "c"] 