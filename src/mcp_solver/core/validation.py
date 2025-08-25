#!/usr/bin/env python3
"""
Shared Validation Module for Python-based Solvers

This module provides shared validation functions for Python-based solver components
(such as PySAT and Z3) to validate input parameters and code.
"""

import ast
import logging
import re
from datetime import timedelta
from typing import Any


# Configure logger
logger = logging.getLogger(__name__)

# Constants
MAX_CODE_LENGTH = 100000
MAX_ITEMS = 100


class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


def validate_index(
    index: Any,
    existing_items: list[tuple[int, str]] | None = None,
    one_based: bool = True,
) -> None:
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
        msg = (
            f"Content exceeds maximum length of {MAX_CODE_LENGTH} characters "
            f"(got {len(content)} characters)"
        )
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
        "os",
        "subprocess",
        "sys",
        "builtins",
        "importlib",
        "runpy",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "commands",
        "popen2",
        "pty",
        "pipes",
        "pexpect",
        "asyncio.subprocess",
        "multiprocessing",
        "threading",
        "pickle",
        "marshal",
        "shelve",
        "dill",
        "cryptography",
        "Crypto",
        "code",
        "codeop",
        "inspect",
        "pathlib",
        "shutil",
        "tempfile",
        "fileinput",
        "zipfile",
        "tarfile",
        "ctypes",
        "cffi",
        "distutils",
    ]

    # Set of dangerous patterns that could allow code execution
    dangerous_patterns = {
        r"\bexec\s*\(": "Use of exec() is not allowed",
        r"\beval\s*\(": "Use of eval() is not allowed",
        r"\bcompile\s*\(": "Use of compile() is not allowed",
        r"\bimport\s+os\b": "Import of os module is not allowed",
        r"\bimport\s+sys\b": "Import of sys module is not allowed",
        r"\bimport\s+subprocess\b": "Import of subprocess module is not allowed",
        r"\bfrom\s+os\s+import": "Import from os module is not allowed",
        r"\bfrom\s+sys\s+import": "Import from sys module is not allowed",
        r"\bfrom\s+subprocess\s+import": "Import from subprocess module is not allowed",
        r"\bopen\s*\(": "File operations are not allowed",
        r"\.read\s*\(": "File read operations are not allowed",
        r"\.write\s*\(": "File write operations are not allowed",
        r"__import__\s*\(": "Dynamic imports are not allowed",
        r"globals\s*\(\s*\)": "Access to globals() is not allowed",
        r"locals\s*\(\s*\)": "Access to locals() is not allowed",
        r"getattr\s*\(": "Dynamic attribute access is not allowed",
        r"setattr\s*\(": "Dynamic attribute setting is not allowed",
        r"delattr\s*\(": "Dynamic attribute deletion is not allowed",
        r"\.system\s*\(": "System command execution is not allowed",
        r"\.popen\s*\(": "Process creation is not allowed",
        r"\.call\s*\(": "Command execution is not allowed",
        r"\.run\s*\(": "Command execution is not allowed",
        r"\.check_output\s*\(": "Command execution is not allowed",
        r"\.check_call\s*\(": "Command execution is not allowed",
        r"\.communicate\s*\(": "Process communication is not allowed",
        r"\.load\s*\(": "Loading serialized data is not allowed",
        r"\.loads\s*\(": "Loading serialized data is not allowed",
        r"\.readline\s*\(": "File read operations are not allowed",
        r"\.readlines\s*\(": "File read operations are not allowed",
        r"__getattribute__": "Low-level attribute access is not allowed",
        r"__getattr__": "Low-level attribute access is not allowed",
        r"__setattr__": "Low-level attribute setting is not allowed",
        r"__delattr__": "Low-level attribute deletion is not allowed",
        r"__class__": "Access to class internals is not allowed",
        r"__base__": "Access to class internals is not allowed",
        r"__bases__": "Access to class internals is not allowed",
        r"__mro__": "Access to class resolution order is not allowed",
        r"__subclasses__": "Access to subclasses is not allowed",
        r"__dict__": "Access to internal dictionaries is not allowed",
        r"__globals__": "Access to global variables is not allowed",
    }

    # First check for syntax errors
    try:
        import ast

        ast.parse(code)

        # Find potentially dangerous patterns in code
        for pattern, message in dangerous_patterns.items():
            matches = re.finditer(pattern, code)
            for match in matches:
                line_num = code[: match.start()].count("\n") + 1
                line_content = code.split("\n")[line_num - 1].strip()

                error_msg = f"{message} at line {line_num}: {line_content}"
                logger.error(error_msg)
                raise ValidationError(error_msg)

        # Parse imports using AST to catch them reliably
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    module_name = name.name.split(".")[0]
                    if module_name in dangerous_imports:
                        line_num = node.lineno
                        error_msg = f"Import of {module_name} module is not allowed at line {line_num}"
                        logger.error(error_msg)
                        raise ValidationError(error_msg)

            elif isinstance(node, ast.ImportFrom):
                module_name = node.module.split(".")[0] if node.module else ""
                if module_name in dangerous_imports:
                    line_num = node.lineno
                    error_msg = f"Import from {module_name} module is not allowed at line {line_num}"
                    logger.error(error_msg)
                    raise ValidationError(error_msg)

        # Check common mistakes that might indicate syntax errors or logical issues
        lines = code.split("\n")
        for i, line in enumerate(lines):
            line_num = i + 1
            line = line.strip()

            # Check for common assignment errors
            if re.search(r"if\s+\w+\s*=\s*\w+", line):  # Single = in if condition
                error_msg = f"Potential mistake at line {line_num}: Assignment operator '=' used in condition instead of comparison operator '=='."
                logger.warning(error_msg)
                raise ValidationError(error_msg)

            # Check for missing colons
            if re.search(
                r"^(if|for|while|def|class|with|try|except|finally)\s+.*[^:]$", line
            ):
                # Make sure it's not a multi-line statement or a comment
                if not line.endswith("\\") and not line.strip().startswith("#"):
                    error_msg = f"Potential syntax error at line {line_num}: Missing colon at the end of a control statement."
                    logger.warning(error_msg)
                    # Don't raise here, just warn - might be a false positive

            # Check for common indentation issues in the next line
            if i < len(lines) - 1:
                next_line = lines[i + 1].rstrip()
                if (
                    re.search(
                        r"^(if|for|while|def|class|with|try|except|finally)\s+.*:$",
                        line,
                    )
                    and next_line
                    and not next_line.startswith(" ")
                    and not next_line.startswith("\t")
                ):
                    if not next_line.strip().startswith(
                        "#"
                    ) and not line.strip().endswith("\\"):
                        error_msg = f"Potential indentation error after line {line_num}: The next line should be indented."
                        logger.warning(error_msg)
                        # Don't raise here, just warn - might be a false positive

    except SyntaxError as e:
        line_num = e.lineno if hasattr(e, "lineno") else "?"
        col_num = e.offset if hasattr(e, "offset") else "?"
        error_text = e.text.strip() if hasattr(e, "text") and e.text else "unknown"

        error_msg = (
            f"Syntax error at line {line_num}, column {col_num}: {e!s} - '{error_text}'"
        )
        logger.error(error_msg)
        raise ValidationError(error_msg) from e


def validate_timeout(
    timeout: Any, min_timeout: timedelta, max_timeout: timedelta
) -> None:
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


def get_standardized_response(
    success: bool, message: str, error: str | None = None, **kwargs
) -> dict[str, Any]:
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
    # Ensure consistency between success flag and error presence
    if error is not None and success:
        # This is inconsistent - if there's an error, success should be False
        # Log this with a stack trace to identify the source
        logger.error(
            f"CRITICAL INCONSISTENCY DETECTED: success=True with error='{error}'",
        )

        # Always override success to be False if there's an error
        success = False  # This ensures the response is consistent

    response = {"message": message, "success": success}

    # Add error if provided
    if error:
        response["error"] = error
        # Ensure status is also set for errors
        response["status"] = "error"

    # Add any additional keyword arguments to the response
    response.update(kwargs)

    return response


class DictMisuseVisitor(ast.NodeVisitor):
    """
    AST visitor that detects dictionary misuse.

    This visitor identifies variables used as dictionaries and then
    detects when they are overwritten with non-dictionary values.
    """

    def __init__(self):
        """Initialize the visitor."""
        self.dict_vars = set()  # Variables used as dictionaries
        self.list_vars = set()  # Variables used as lists (to avoid false positives)
        self.dict_assignments = {}  # Track where dictionaries are initialized
        self.dict_misuses = []  # Detected misuses

    def visit_Subscript(self, node):
        """
        Identify variables used as subscriptable collections.

        We need to distinguish between dictionaries and lists:
        - For dictionaries: d[key] with arbitrary key
        - For lists: l[index] typically with numeric index
        """
        if isinstance(node.value, ast.Name):
            var_name = node.value.id

            # Try to determine if this is a list or dictionary access
            is_list_access = False

            # Check if index is a numeric literal (typical for lists)
            if (
                (
                    isinstance(node.slice, ast.Index)
                    and isinstance(node.slice.value, ast.Num)
                )
                or (
                    isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, int)
                )
                or (
                    isinstance(node.slice, ast.Slice)
                    and all(
                        isinstance(x, (ast.Num, ast.Constant))
                        for x in [node.slice.lower, node.slice.upper, node.slice.step]
                        if x is not None
                    )
                )
            ):
                is_list_access = True

            # If it looks like a list access and we don't already know it's a dict
            if is_list_access and var_name not in self.dict_vars:
                self.list_vars.add(var_name)
            # Otherwise assume it might be a dictionary
            elif var_name not in self.list_vars:
                self.dict_vars.add(var_name)

        self.generic_visit(node)

    def visit_Compare(self, node):
        """
        Identify variables used as dictionaries via membership tests.

        Example: key in d indicates d is used as a dictionary.
        """
        for op in node.ops:
            if isinstance(op, ast.In) or isinstance(op, ast.NotIn):
                comparator = node.comparators[0]
                if isinstance(comparator, ast.Name):
                    var_name = comparator.id
                    # Only add to dict_vars if not already known to be a list
                    if var_name not in self.list_vars:
                        self.dict_vars.add(var_name)
        self.generic_visit(node)

    def visit_Assign(self, node):
        """
        Check for dictionary misuse in assignments.

        This detects:
        1. Dictionary initialization (d = {})
        2. Dictionary misuse (d = scalar)
        3. List initialization (l = [])
        """
        # Check for dictionary initialization with {}
        if isinstance(node.value, ast.Dict) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                self.dict_vars.add(var_name)
                # Remove from list_vars if it was there
                self.list_vars.discard(var_name)
                self.dict_assignments[var_name] = getattr(node, "lineno", 0)

        # Detect dictionary initialization using dict() or defaultdict()
        elif isinstance(node.value, ast.Call) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name) and isinstance(
                node.value.func, ast.Name
            ):
                var_name = node.targets[0].id

                if node.value.func.id in ("dict", "defaultdict"):
                    self.dict_vars.add(var_name)
                    # Remove from list_vars if it was there
                    self.list_vars.discard(var_name)
                    self.dict_assignments[var_name] = getattr(node, "lineno", 0)
                # Detect list initialization with list() or []
                elif node.value.func.id == "list" or (
                    isinstance(node.value, ast.List)
                    or (isinstance(node.value, ast.ListComp))
                ):
                    self.list_vars.add(var_name)
                    # Remove from dict_vars if it was there
                    self.dict_vars.discard(var_name)

        # Detect list initialization with []
        elif isinstance(node.value, ast.List) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                self.list_vars.add(var_name)
                # Remove from dict_vars if it was there
                self.dict_vars.discard(var_name)

        # Check for dictionary misuse - overwriting a dictionary with a non-dictionary
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id

            # Only check if this is a known dictionary and not a list
            if var_name in self.dict_vars and var_name not in self.list_vars:
                # Check if we're assigning a non-dictionary value
                if not self._is_dict_value(node.value):
                    line_num = getattr(node, "lineno", "?")

                    # Avoid false positives for variable names like "solution" or "model"
                    # which are commonly used for solver results and aren't dictionary misuse
                    if var_name.lower() not in ["solution", "model", "result"]:
                        # Create user-friendly error message
                        self.dict_misuses.append(
                            {
                                "line": line_num,
                                "var_name": var_name,
                                "message": f"Error at line {line_num}: The variable '{var_name}' is being overwritten with a non-dictionary value.",
                                "suggestion": f"Use '{var_name}[key] = value' instead of '{var_name} = value' to add items to the dictionary.",
                            }
                        )

        self.generic_visit(node)

    def _is_dict_value(self, node):
        """
        Check if a value is likely to be a dictionary.

        Args:
            node: The AST node representing the value

        Returns:
            True if the value is a dictionary, False otherwise
        """
        return (
            # Dictionary literal: {}
            isinstance(node, ast.Dict)
            or
            # dict() or defaultdict() call
            (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("dict", "defaultdict")
            )
            or
            # Variable that is known to be a dictionary
            (
                isinstance(node, ast.Name)
                and node.id in self.dict_vars
                and node.id not in self.list_vars
            )
        )

    def visit_FunctionDef(self, node):
        """
        Process function definitions to catch var_mapping misuse pattern.

        This specifically looks for the var_mapping = var_count pattern
        inside functions like create_var.
        """
        function_name = node.name

        # Check if this is a create_var function or similar
        if function_name.lower().find("create_var") >= 0:
            # Get the function body
            for stmt in node.body:
                # Check for var_mapping = var_count pattern
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    if isinstance(stmt.targets[0], ast.Name):
                        target_name = stmt.targets[0].id
                        # Check if target looks like a dict variable
                        if (
                            "map" in target_name.lower()
                            or "dict" in target_name.lower()
                            or "vars" in target_name.lower()
                        ):
                            # Check if value looks like a counter
                            if isinstance(stmt.value, ast.Name):
                                value_name = stmt.value.id
                                if (
                                    "count" in value_name.lower()
                                    or "id" in value_name.lower()
                                ):
                                    # This is likely the var_mapping = var_count bug
                                    line_num = getattr(stmt, "lineno", "?")
                                    self.dict_misuses.append(
                                        {
                                            "line": line_num,
                                            "var_name": target_name,
                                            "message": f"Critical error at line {line_num}: The '{target_name}' dictionary is being overwritten with a number.",
                                            "suggestion": f"Use '{target_name}[name] = {value_name}' instead of '{target_name} = {value_name}' to add a key-value pair to the dictionary.",
                                        }
                                    )

        self.generic_visit(node)


class DictUsageVisitor(ast.NodeVisitor):
    """
    AST visitor that identifies variables used as dictionaries or lists.

    This visitor tracks variables that are used with dictionary operations
    and distinguishes between dictionaries and lists/arrays to avoid false positives.
    """

    def __init__(self):
        """Initialize the visitor with empty sets for tracking variables."""
        self.dict_vars = set()  # Variables used as dictionaries
        self.list_vars = set()  # Variables used as lists/arrays
        self.dict_assignments = {}  # Track where dictionaries are initialized

    def visit_Subscript(self, node):
        """
        Identify variables used with subscript access and distinguish between lists and dicts.

        - For lists: typically use numeric indices like l[0], l[1:3]
        - For dicts: typically use arbitrary expressions as keys
        """
        if isinstance(node.value, ast.Name):
            var_name = node.value.id

            # Try to determine if this is a list or dictionary access
            is_list_access = False

            # Check for numeric index or slice, which likely indicates a list
            if hasattr(node, "slice"):
                # Python 3.9+ direct slice
                if isinstance(node.slice, ast.Constant) and isinstance(
                    node.slice.value, int
                ):
                    is_list_access = True
                # Python 3.8 and earlier use Index wrapper
                elif isinstance(node.slice, ast.Index):
                    if isinstance(node.slice.value, ast.Num) or (
                        isinstance(node.slice.value, ast.Constant)
                        and isinstance(node.slice.value.value, int)
                    ):
                        is_list_access = True
                # Check for slice notation (list[1:5])
                elif isinstance(node.slice, ast.Slice):
                    is_list_access = True

            # If it looks like a list access and not already known to be a dict
            if is_list_access and var_name not in self.dict_vars:
                self.list_vars.add(var_name)
            # If it's not clearly a list access and not known to be a list
            elif not is_list_access and var_name not in self.list_vars:
                self.dict_vars.add(var_name)

        self.generic_visit(node)

    def visit_Compare(self, node):
        """
        Identify variables used in membership tests.

        Example: key in d could be dictionary or list/set.
        """
        for op in node.ops:
            if isinstance(op, ast.In) or isinstance(op, ast.NotIn):
                comparator = node.comparators[0]
                if isinstance(comparator, ast.Name):
                    var_name = comparator.id
                    # Only add to dict_vars if not already known to be a list
                    if var_name not in self.list_vars:
                        self.dict_vars.add(var_name)

        self.generic_visit(node)

    def visit_Assign(self, node):
        """
        Identify variables initialized as dictionaries or lists.

        Examples:
        - d = {} or d = dict() indicates a dictionary
        - l = [] or l = list() indicates a list
        """
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id

            # Dictionary literal: {}
            if isinstance(node.value, ast.Dict):
                self.dict_vars.add(var_name)
                self.list_vars.discard(var_name)  # Remove from list_vars if present
                self.dict_assignments[var_name] = getattr(node, "lineno", 0)

            # List literal: []
            elif isinstance(node.value, ast.List):
                self.list_vars.add(var_name)
                self.dict_vars.discard(var_name)  # Remove from dict_vars if present

            # List comprehension: [x for x in range(10)]
            elif isinstance(node.value, ast.ListComp):
                self.list_vars.add(var_name)
                self.dict_vars.discard(var_name)

            # Function call: dict(), list(), etc.
            elif isinstance(node.value, ast.Call) and isinstance(
                node.value.func, ast.Name
            ):
                func_name = node.value.func.id

                # Dictionary constructors
                if func_name in ("dict", "defaultdict", "OrderedDict", "Counter"):
                    self.dict_vars.add(var_name)
                    self.list_vars.discard(var_name)
                    self.dict_assignments[var_name] = getattr(node, "lineno", 0)

                # List/array constructors
                elif func_name in ("list", "array", "tuple", "set"):
                    self.list_vars.add(var_name)
                    self.dict_vars.discard(var_name)

        self.generic_visit(node)


class DictionaryMisuseValidator:
    """
    Validator for detecting dictionary misuse in Python code.

    This validator stores code fragments as they are added, then combines them
    for analysis to detect dictionary misuse errors, like overwriting a dictionary
    with a non-dictionary value.
    """

    def __init__(self):
        """Initialize a new validator with an empty list of code fragments."""
        self.code_fragments = []
        self.fragment_indices = []  # Store corresponding item indices

    def add_fragment(self, code_fragment: str, item_index: int = None):
        """
        Add a new code fragment to the collection.

        Args:
            code_fragment: The code fragment to add
            item_index: Optional index of the item (for error reporting)
        """
        self.code_fragments.append(code_fragment)
        self.fragment_indices.append(item_index)

    def get_combined_code(self) -> str:
        """
        Get all fragments combined into one string.

        Returns:
            A string with all code fragments joined by newlines
        """
        return "\n".join(self.code_fragments)

    def validate(self) -> dict[str, Any]:
        """
        Validate all collected code fragments for dictionary misuse.

        Returns:
            A dictionary with validation results, including:
            - has_errors: Whether any errors were found
            - errors: A list of error dictionaries if has_errors is True
        """
        combined_code = self.get_combined_code()

        try:
            tree = ast.parse(combined_code)

            # Find dictionary misuses using the visitor
            misuse_visitor = DictMisuseVisitor()
            misuse_visitor.visit(tree)

            # Use DictUsageVisitor to help identify arrays/lists vs dictionaries
            usage_visitor = DictUsageVisitor()
            usage_visitor.visit(tree)

            # Filter out false positives by checking for variables used as both dictionaries and lists
            # We want to remove those from the dictionary misuse list
            filtered_misuses = []
            for error in misuse_visitor.dict_misuses:
                var_name = error.get("var_name", "")

                # Skip if this variable is clearly used as a list elsewhere
                if var_name in usage_visitor.list_vars:
                    logger.debug(
                        f"Skipping false positive for list variable: {var_name}"
                    )
                    continue

                # Keep real dictionary misuses
                filtered_misuses.append(error)

            # Process any filtered misuses
            if filtered_misuses:
                logger.info(
                    f"Found {len(filtered_misuses)} dictionary misuses after filtering"
                )

                # Add item index information to errors if available
                for error in filtered_misuses:
                    line_num = error.get("line", 0)
                    if isinstance(line_num, int) and line_num > 0:
                        # Find which fragment this line belongs to
                        current_line = 0
                        for i, fragment in enumerate(self.code_fragments):
                            fragment_lines = fragment.count("\n") + 1
                            if current_line + fragment_lines >= line_num:
                                # This is the fragment containing the error
                                fragment_line = line_num - current_line
                                error["item"] = (
                                    self.fragment_indices[i]
                                    if i < len(self.fragment_indices)
                                    else "?"
                                )
                                error["fragment_line"] = fragment_line
                                break
                            current_line += fragment_lines

                return {"has_errors": True, "errors": filtered_misuses}

            return {"has_errors": False, "errors": []}

        except SyntaxError as e:
            # Handle syntax errors in the combined code
            line_num = getattr(e, "lineno", "?")
            col_num = getattr(e, "offset", "?")
            error_text = (
                getattr(e, "text", "").strip() if hasattr(e, "text") else "unknown"
            )

            logger.error(
                f"Syntax error in combined code: Line {line_num}, Col {col_num}: {e!s}"
            )

            return {
                "has_errors": True,
                "errors": [
                    {
                        "message": f"Syntax error in your code: {e!s}",
                        "line": line_num,
                        "column": col_num,
                        "text": error_text,
                        "suggestion": "Check your code syntax",
                    }
                ],
            }
