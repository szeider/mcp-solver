#!/usr/bin/env python3
"""
Test script for verifying the MiniZinc installation of MCP-Solver.
This script checks:
  1. Required configuration files
  2. Core dependencies (MiniZinc with Chuffed solver)
  3. Basic solver functionality

Note: This script only tests core functionality.
Optional solvers (Z3, PySAT) have their own setup verification.
"""

import sys
from pathlib import Path

from minizinc import Instance, Model, Solver

# Import our centralized prompt loader
from mcp_solver.core.prompt_loader import load_prompt


class SetupTest:
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

    def check_file(self, file_name: str) -> bool:
        """Check if a file exists in the base directory."""
        file_path = self.base_dir / file_name
        return file_path.exists()

    def test_configuration_files(self):
        """Test for the presence of required configuration files."""
        print(f"\n{self.BOLD}Configuration Files:{self.RESET}")

        # Test prompt files using the centralized prompt loader
        prompts_to_test = [("mzn", "instructions"), ("mzn", "review")]

        for mode, prompt_type in prompts_to_test:
            try:
                # Attempt to load the prompt using the prompt loader
                content = load_prompt(mode, prompt_type)
                self.record_test(
                    f"Prompt file: {mode}/{prompt_type}.md",
                    True,
                    f"Successfully loaded ({len(content)} characters)",
                )
            except FileNotFoundError:
                self.record_test(
                    f"Prompt file: {mode}/{prompt_type}.md",
                    False,
                    "Prompt file not found",
                )
            except Exception as e:
                self.record_test(
                    f"Prompt file: {mode}/{prompt_type}.md",
                    False,
                    f"Error loading prompt: {e!s}",
                )

        # Test other required files
        other_files = [("pyproject.toml", True)]
        for file, required in other_files:
            exists = self.check_file(file)
            if required:
                self.record_test(
                    f"Configuration file: {file}",
                    exists,
                    (
                        None
                        if exists
                        else f"Required file not found at {self.base_dir / file}"
                    ),
                )

    def test_core_dependencies(self):
        """Test MiniZinc and Chuffed solver installation."""
        print(f"\n{self.BOLD}Core Dependencies:{self.RESET}")
        try:
            solver = Solver.lookup("chuffed")
            self.record_test(
                "MiniZinc Chuffed solver", True, f"Found version {solver.version}"
            )
        except Exception as e:
            self.record_test(
                "MiniZinc Chuffed solver",
                False,
                f"Chuffed solver not found: {e!s}\nPlease install MiniZinc with Chuffed solver",
            )
            return  # Skip further tests if solver not found

        try:
            from minizinc import Instance

            self.record_test("MiniZinc Python binding", True)
        except ImportError as e:
            self.record_test(
                "MiniZinc Python binding",
                False,
                f"Error importing minizinc: {e!s}\nPlease install minizinc Python package",
            )

    def test_basic_functionality(self):
        """Test basic MiniZinc functionality."""
        print(f"\n{self.BOLD}Basic Functionality:{self.RESET}")
        model_code = """
        var 0..1: x;
        constraint x = 1;
        solve satisfy;
        """

        # Test model creation
        try:
            model = Model()
            model.add_string(model_code)
            self.record_test("Model creation", True)
        except Exception as e:
            self.record_test("Model creation", False, f"Error creating model: {e!s}")
            return

        # Test solver execution
        try:
            solver = Solver.lookup("chuffed")
            instance = Instance(solver, model)
            result = instance.solve()
            self.record_test("Solver execution", True, "Successfully executed solver")
        except Exception as e:
            self.record_test("Solver execution", False, f"Error during solving: {e!s}")

    def run_all_tests(self):
        """Run all setup tests and display results."""
        print(f"{self.BOLD}=== MCP-Solver MiniZinc Setup Test ==={self.RESET}")

        self.test_configuration_files()
        self.test_core_dependencies()
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
                "\nMiniZinc setup incomplete. Please fix the issues above before proceeding."
            )
            sys.exit(1)
        else:
            print(
                f"\n{self.GREEN}✓ All MiniZinc tests passed successfully!{self.RESET}"
            )
            print("\nMiniZinc system is ready to use MCP-Solver.")
            sys.exit(0)


def main():
    test = SetupTest()
    test.run_all_tests()


if __name__ == "__main__":
    main()
