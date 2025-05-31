#!/usr/bin/env python3
"""
Test script for verifying the PySAT mode installation of MCP-Solver.
This script checks:
  1. Required configuration files for PySAT mode
  2. PySAT dependencies
  3. Basic SAT solver functionality

Note: This script only tests PySAT mode functionality.
For core MiniZinc testing, use the main test-setup script.
"""

import sys
from pathlib import Path

# Import our centralized prompt loader
from mcp_solver.core.prompt_loader import load_prompt


class PySATSetupTest:
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
        prompts_to_test = [("pysat", "instructions"), ("pysat", "review")]

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

    def test_pysat_dependencies(self):
        """Test PySAT installation and dependencies."""
        print(f"\n{self.BOLD}PySAT Dependencies:{self.RESET}")

        # Test python-sat package
        try:
            import pysat
            from pysat.formula import CNF
            from pysat.solvers import Solver

            self.record_test(
                "python-sat package", True, f"Found version {pysat.__version__}"
            )
        except ImportError as e:
            self.record_test(
                "python-sat package",
                False,
                f"Error importing pysat: {e!s}\nPlease install with: pip install python-sat",
            )
            return

        # Test available solvers
        try:
            from pysat.solvers import Glucose3

            solver = Glucose3()
            solver.delete()  # Clean up
            self.record_test(
                "SAT solver (Glucose3)", True, "Successfully initialized solver"
            )
        except Exception as e:
            self.record_test(
                "SAT solver (Glucose3)", False, f"Error initializing solver: {e!s}"
            )

    def test_basic_functionality(self):
        """Test basic SAT solving functionality."""
        print(f"\n{self.BOLD}Basic Functionality:{self.RESET}")

        # Simple SAT problem: (x1 OR x2) AND (NOT x1 OR x2)
        try:
            from pysat.formula import CNF
            from pysat.solvers import Solver

            cnf = CNF()
            cnf.append([1, 2])  # x1 OR x2
            cnf.append([-1, 2])  # NOT x1 OR x2

            self.record_test("CNF creation", True, "Successfully created CNF formula")
        except Exception as e:
            self.record_test("CNF creation", False, f"Error creating CNF: {e!s}")
            return

        # Test solver initialization and basic solving
        try:
            solver = Solver(bootstrap_with=cnf)
            is_sat = solver.solve()
            solver.delete()  # Clean up

            self.record_test(
                "Solver initialization", True, "Successfully initialized and ran solver"
            )
        except Exception as e:
            self.record_test(
                "Solver initialization", False, f"Error with solver: {e!s}"
            )

    def run_all_tests(self):
        """Run all setup tests and display results."""
        print(f"{self.BOLD}=== MCP-Solver PySAT Mode Setup Test ==={self.RESET}")

        self.test_configuration_files()
        self.test_pysat_dependencies()
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
                "\nPySAT mode setup incomplete. Please fix the issues above before proceeding."
            )
            sys.exit(1)
        else:
            print(
                f"\n{self.GREEN}✓ All PySAT mode tests passed successfully!{self.RESET}"
            )
            print("\nSystem is ready to use MCP-Solver in PySAT mode.")
            sys.exit(0)


def main():
    test = PySATSetupTest()
    test.run_all_tests()


if __name__ == "__main__":
    main()
