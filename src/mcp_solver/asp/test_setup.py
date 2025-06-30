#!/usr/bin/env python3
"""
Test script for verifying the ASP mode installation of MCP-Solver.
This script checks:
  1. Required configuration files for ASP mode
  2. Clingo dependencies
  3. Basic ASP solver functionality
"""

import sys
from pathlib import Path

# Import our centralized prompt loader
from mcp_solver.core.prompt_loader import load_prompt


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

    def test_basic_functionality(self):
        """Test basic ASP solving functionality."""
        print(f"\n{self.BOLD}Basic Functionality:{self.RESET}")

        try:
            import clingo
            ctl = clingo.Control()
            ctl.add("base", [], "a. b :- a.")
            ctl.ground([("base", [])])
            
            found_model = False
            with ctl.solve(yield_=True) as handle:
                for model in handle:
                    found_model = True
                    self.record_test("Solver execution", True, f"Found model: {model.symbols(shown=True)}")
            
            if not found_model:
                 self.record_test("Solver execution", False, "Solver ran but found no model for a simple program.")

        except Exception as e:
            self.record_test(
                "Solver initialization", False, f"Error with clingo solver: {e!s}"
            )

    def run_all_tests(self):
        """Run all setup tests and display results."""
        print(f"{self.BOLD}=== MCP-Solver ASP Mode Setup Test ==={self.RESET}")

        self.test_configuration_files()
        self.test_asp_dependencies()
        self.test_basic_functionality()

        print(f"\n{self.BOLD}=== Test Summary ==={self.RESET}")
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