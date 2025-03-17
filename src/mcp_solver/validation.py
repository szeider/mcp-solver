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

def validate_python_code_safety(code: str) -> None:
    """
    Validates that Python code does not contain potentially unsafe operations.
    
    This function checks for:
    1. Dangerous imports (os, sys, subprocess, etc.)
    2. File operations (open, read, write, etc.)
    3. Network operations (socket, requests, etc.)
    4. Code execution functions (exec, eval, etc.)
    5. System command execution (os.system, subprocess.call, etc.)
    6. Basic syntax errors
    7. Common variable assignment mistakes
    
    Args:
        code: The Python code to validate
        
    Raises:
        ValidationError: If the code contains potentially unsafe operations
        SyntaxError: If the code contains syntax errors
    """
    logger = logging.getLogger(__name__)
    
    # List of dangerous imports that could allow arbitrary code execution
    dangerous_imports = [
        'os', 'subprocess', 'sys', 'builtins', 'importlib', 'runpy',
        'socket', 'requests', 'urllib', 'http', 'ftplib',
        'commands', 'popen2', 'pty', 'pipes', 'pexpect', 'asyncio.subprocess',
        'multiprocessing', 'threading', 'pickle', 'marshal', 'shelve', 'dill',
        'cryptography', 'Crypto', 'code', 'codeop', 'inspect',
        'pathlib', 'shutil', 'tempfile', 'fileinput', 'zipfile', 'tarfile',
        'ctypes', 'cffi', 'distutils'
    ]
    
    # Set of dangerous patterns that could allow code execution
    dangerous_patterns = {
        r'\bexec\s*\(': 'Use of exec() is not allowed',
        r'\beval\s*\(': 'Use of eval() is not allowed',
        r'\bcompile\s*\(': 'Use of compile() is not allowed',
        r'\bimport\s+os\b': 'Import of os module is not allowed',
        r'\bimport\s+sys\b': 'Import of sys module is not allowed',
        r'\bimport\s+subprocess\b': 'Import of subprocess module is not allowed',
        r'\bfrom\s+os\s+import': 'Import from os module is not allowed',
        r'\bfrom\s+sys\s+import': 'Import from sys module is not allowed',
        r'\bfrom\s+subprocess\s+import': 'Import from subprocess module is not allowed',
        r'\bopen\s*\(': 'File operations are not allowed',
        r'\.read\s*\(': 'File read operations are not allowed',
        r'\.write\s*\(': 'File write operations are not allowed',
        r'__import__\s*\(': 'Dynamic imports are not allowed',
        r'globals\s*\(\s*\)': 'Access to globals() is not allowed',
        r'locals\s*\(\s*\)': 'Access to locals() is not allowed',
        r'getattr\s*\(': 'Dynamic attribute access is not allowed',
        r'setattr\s*\(': 'Dynamic attribute setting is not allowed',
        r'delattr\s*\(': 'Dynamic attribute deletion is not allowed',
        r'\.system\s*\(': 'System command execution is not allowed',
        r'\.popen\s*\(': 'Process creation is not allowed',
        r'\.call\s*\(': 'Command execution is not allowed',
        r'\.run\s*\(': 'Command execution is not allowed',
        r'\.check_output\s*\(': 'Command execution is not allowed',
        r'\.check_call\s*\(': 'Command execution is not allowed',
        r'\.communicate\s*\(': 'Process communication is not allowed',
        r'\.load\s*\(': 'Loading serialized data is not allowed',
        r'\.loads\s*\(': 'Loading serialized data is not allowed',
        r'\.readline\s*\(': 'File read operations are not allowed',
        r'\.readlines\s*\(': 'File read operations are not allowed',
        r'__getattribute__': 'Low-level attribute access is not allowed',
        r'__getattr__': 'Low-level attribute access is not allowed',
        r'__setattr__': 'Low-level attribute setting is not allowed',
        r'__delattr__': 'Low-level attribute deletion is not allowed',
        r'__class__': 'Access to class internals is not allowed',
        r'__base__': 'Access to class internals is not allowed',
        r'__bases__': 'Access to class internals is not allowed',
        r'__mro__': 'Access to class resolution order is not allowed',
        r'__subclasses__': 'Access to subclasses is not allowed',
        r'__dict__': 'Access to internal dictionaries is not allowed',
        r'__globals__': 'Access to global variables is not allowed',
    }
    
    # First check for syntax errors
    try:
        import ast
        ast.parse(code)
        
        # Find potentially dangerous patterns in code
        for pattern, message in dangerous_patterns.items():
            matches = re.finditer(pattern, code)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                line_content = code.split('\n')[line_num - 1].strip()
                
                error_msg = f"{message} at line {line_num}: {line_content}"
                logger.error(error_msg)
                raise ValidationError(error_msg)
        
        # Parse imports using AST to catch them reliably
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    module_name = name.name.split('.')[0]
                    if module_name in dangerous_imports:
                        line_num = node.lineno
                        error_msg = f"Import of {module_name} module is not allowed at line {line_num}"
                        logger.error(error_msg)
                        raise ValidationError(error_msg)
            
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module.split('.')[0] if node.module else ''
                if module_name in dangerous_imports:
                    line_num = node.lineno
                    error_msg = f"Import from {module_name} module is not allowed at line {line_num}"
                    logger.error(error_msg)
                    raise ValidationError(error_msg)
        
        # Check common mistakes that might indicate syntax errors or logical issues
        lines = code.split('\n')
        for i, line in enumerate(lines):
            line_num = i + 1
            line = line.strip()
            
            # Check for common assignment errors
            if re.search(r'if\s+\w+\s*=\s*\w+', line):  # Single = in if condition
                error_msg = f"Potential mistake at line {line_num}: Assignment operator '=' used in condition instead of comparison operator '=='."
                logger.warning(error_msg)
                raise ValidationError(error_msg)
            
            # Check for missing colons
            if re.search(r'^(if|for|while|def|class|with|try|except|finally)\s+.*[^:]$', line):
                # Make sure it's not a multi-line statement or a comment
                if not line.endswith('\\') and not line.strip().startswith('#'):
                    error_msg = f"Potential syntax error at line {line_num}: Missing colon at the end of a control statement."
                    logger.warning(error_msg)
                    # Don't raise here, just warn - might be a false positive
            
            # Check for common indentation issues in the next line
            if i < len(lines) - 1:
                next_line = lines[i + 1].rstrip()
                if re.search(r'^(if|for|while|def|class|with|try|except|finally)\s+.*:$', line) and next_line and not next_line.startswith(' ') and not next_line.startswith('\t'):
                    if not next_line.strip().startswith('#') and not line.strip().endswith('\\'):
                        error_msg = f"Potential indentation error after line {line_num}: The next line should be indented."
                        logger.warning(error_msg)
                        # Don't raise here, just warn - might be a false positive
        
    except SyntaxError as e:
        line_num = e.lineno if hasattr(e, 'lineno') else '?'
        col_num = e.offset if hasattr(e, 'offset') else '?'
        error_text = e.text.strip() if hasattr(e, 'text') and e.text else 'unknown'
        
        error_msg = f"Syntax error at line {line_num}, column {col_num}: {str(e)} - '{error_text}'"
        logger.error(error_msg)
        raise ValidationError(error_msg) from e

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
