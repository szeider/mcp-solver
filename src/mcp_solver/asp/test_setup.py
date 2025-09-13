#!/usr/bin/env python3
"""
Test script for verifying the ASP mode installation of MCP-Solver.
This script checks:
  1. Required configuration files for ASP mode
  2. Clingo dependencies
  3. Basic ASP solver functionality
"""

import asyncio
import sys
from datetime import timedelta
from pathlib import Path

# Import our centralized prompt loader
from mcp_solver.core.prompt_loader import load_prompt

from .model_manager import ASPModelManager


class ASPSetupTest:
    def __init__(self):
        self.successes: list[tuple[str, str]] = []  # (test_name, details)
        self.failures: list[tuple[str, str]] = []  # (test_name, error_details)
        self.base_dir = Path(__file__).resolve().parents[3]
        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"
        self.BOLD = "\033[1m"

    def print_result(self, test_name: str, success: bool, details: str | None = None):
        """Print a test result with color and proper formatting."""
        mark = "✓" if success else "✗"
        color = self.GREEN if success else self.RED
        print(f"{color}{mark} {test_name}{self.RESET}")
        if details:
            print(f"  └─ {details}")

    def record_test(self, test_name: str, success: bool, details: str | None = None):
        """Record a test result and print it."""
        if success:
            self.successes.append((test_name, details if details else ""))
        else:
            self.failures.append((test_name, details if details else "Test failed"))
        self.print_result(test_name, success, None if success else details)

    def test_configuration_files(self):
        """Test for the presence of required configuration files."""
        print(f"\n{self.BOLD}Configuration Files:{self.RESET}")

        prompts_to_test = [("asp", "instructions"), ("asp", "review")]

        for mode, prompt_type in prompts_to_test:
            try:
                content = load_prompt(mode, prompt_type)
                self.record_test(
                    f"Prompt file: {mode}/{prompt_type}.md",
                    True,
                    f"Successfully loaded ({len(content)} characters)",
                )
            except Exception as e:
                self.record_test(
                    f"Prompt file: {mode}/{prompt_type}.md",
                    False,
                    f"Error loading prompt: {e!s}",
                )

    def test_asp_dependencies(self):
        """Test clingo installation and dependencies."""
        print(f"\n{self.BOLD}ASP (clingo) Dependencies:{self.RESET}")
        # Check clingo
        try:
            import clingo

            self.record_test(
                "clingo package", True, f"Found version {clingo.__version__}"
            )
        except ImportError as e:
            self.record_test(
                "clingo package",
                False,
                f"Error importing clingo: {e!s}\nPlease install with: pip install clingo",
            )
            return

    def test_error_handling(self):
        """Test ASP error handling functionality including validation and solution export."""
        from mcp_solver.asp import error_handling, solution

        print(f"\n{self.BOLD}ASP Error Handling:{self.RESET}")

        # Note: We now rely on Clingo for all validation, so we only test solution export here

        # Test solution export
        print(f"\n{self.BOLD}Solution Export:{self.RESET}")

        # Test successful case
        answer_sets = [["a"], ["b", "a"]]
        sol = solution.export_solution(answer_sets)
        self.record_test(
            "Solution export (valid answer sets)",
            sol.get("satisfiable") is True,
            f"Solution: {sol}",
        )

        # Test error case
        try:
            raise error_handling.ASPError(
                "Simulated ASP error", context='"line: 1", "code": "a b :- ."'
            )
        except Exception as e:
            err_sol = solution.export_solution(e)
            self.record_test(
                "Solution export (error case)",
                err_sol.get("satisfiable") is False,
                f"Error solution: {err_sol}",
            )

    def test_model_manager(self):
        """Test ASPModelManager functionality including item management and solving."""
        print(f"\n{self.BOLD}Model Manager Tests:{self.RESET}")
        mgr = ASPModelManager()
        timeout = timedelta(seconds=5)

        async def run_tests():
            # Test group 1: Model item management
            print(f"\n{self.BOLD}1. Model Item Management:{self.RESET}")

            test_cases = [
                (0, "a.", True, "Add initial valid item"),
                (1, "b :- a.", True, "Add dependent valid item"),
                (2, "a b :- .", False, "Add item with syntax error"),
                (10, "c.", False, "Add item at invalid index"),
                (
                    2,
                    "result(X) :- undefined_predicate(X).",
                    False,
                    "Add item with grounding error",
                ),
            ]

            for index, content, should_succeed, desc in test_cases:
                print(f"\nTesting: {desc}")
                result = await mgr.add_item(index, content)
                print(f"Result: {result}")
                self.record_test(
                    f"Model item management - {desc}",
                    result.get("success") == should_succeed,
                    str(result),
                )

            # Test group 2: Model solving
            print(f"\n{self.BOLD}2. Model Solving:{self.RESET}")

            solve_test_cases = [
                (["a.", "b :- a."], True, "Valid model"),
                (["a.", "b :- a.", "a b :- ."], False, "Model with syntax error"),
                (
                    ["result(X) :- undefined_predicate(X)."],
                    False,
                    "Model with grounding error",
                ),
                ([], False, "Empty model"),
            ]

            for items, should_be_satisfiable, desc in solve_test_cases:
                print(f"\nTesting: {desc}")
                mgr.code_items = items
                result = await mgr.solve_model(timeout)
                print(f"Result: {result}")
                self.record_test(
                    f"Model solving - {desc}",
                    result.get("satisfiable") == should_be_satisfiable,
                    str(result),
                )

        asyncio.run(run_tests())

    def run_all_tests(self):
        """Run all setup tests and display results."""
        print(f"{self.BOLD}=== MCP-Solver ASP Mode Setup Test ==={self.RESET}")
        print("Running test suite for ASP mode components...\n")

        test_groups = [
            ("Configuration", self.test_configuration_files),
            ("Dependencies", self.test_asp_dependencies),
            ("Error Handling", self.test_error_handling),
            ("Model Manager", self.test_model_manager),
        ]

        for group_name, test_func in test_groups:
            print(f"\n{self.BOLD}Testing {group_name}...{self.RESET}")
            print("=" * (9 + len(group_name)))
            test_func()

        print(f"\n{self.BOLD}=== Test Summary ==={self.RESET}")
        print(f"Total tests: {len(self.successes) + len(self.failures)}")
        print(f"Passed: {len(self.successes)}")
        print(f"Failed: {len(self.failures)}")

        if self.failures:
            print(f"\n{self.BOLD}Failed Tests:{self.RESET}")
            for test, details in self.failures:
                print(f"\n{self.RED}✗ {test}{self.RESET}")
                print(f"  └─ {details}")
            print(
                "\nASP mode setup incomplete. Please fix the issues above before proceeding."
            )
            sys.exit(1)
        else:
            print(
                f"\n{self.GREEN}✓ All ASP mode tests passed successfully!{self.RESET}"
            )
            print("\nSystem is ready to use MCP-Solver in ASP mode.")
            sys.exit(0)


def main():
    test = ASPSetupTest()
    test.run_all_tests()


if __name__ == "__main__":
    main()
