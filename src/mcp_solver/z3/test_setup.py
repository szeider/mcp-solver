#!/usr/bin/env python3
"""
Test script for verifying the Z3 mode installation of MCP-Solver.
This script checks:
  1. Required configuration files for Z3 mode
  2. Z3 dependencies
  3. Basic SMT solver functionality

Note: This script only tests Z3 mode functionality.
For core MiniZinc testing, use the main test-setup script.
"""

import sys
from pathlib import Path

# Import our centralized prompt loader
from mcp_solver.core.prompt_loader import load_prompt


class Z3SetupTest:
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
        prompts_to_test = [("z3", "instructions"), ("z3", "review")]

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

    def test_z3_dependencies(self):
        """Test Z3 installation and dependencies."""
        print(f"\n{self.BOLD}Z3 Dependencies:{self.RESET}")

        # Test z3-solver package
        try:
            import z3

            self.record_test(
                "z3-solver package", True, f"Found version {z3.get_version_string()}"
            )
        except ImportError as e:
            self.record_test(
                "z3-solver package",
                False,
                f"Error importing z3: {e!s}\nPlease install with: pip install z3-solver",
            )
            return

        # Test solver initialization
        try:
            solver = z3.Solver()
            self.record_test(
                "Z3 solver initialization", True, "Successfully initialized solver"
            )
        except Exception as e:
            self.record_test(
                "Z3 solver initialization",
                False,
                f"Error initializing solver: {e!s}",
            )

    def test_basic_functionality(self):
        """Test basic SMT solving functionality."""
        print(f"\n{self.BOLD}Basic Functionality:{self.RESET}")

        # Simple SMT problem: x + y = 2, x > 0, y > 0
        try:
            import z3

            x = z3.Real("x")
            y = z3.Real("y")
            solver = z3.Solver()

            solver.add(x + y == 2)
            solver.add(x > 0)
            solver.add(y > 0)

            self.record_test(
                "Constraint creation", True, "Successfully created constraints"
            )
        except Exception as e:
            self.record_test(
                "Constraint creation", False, f"Error creating constraints: {e!s}"
            )
            return

        # Test solving
        try:
            result = solver.check()
            if result == z3.sat:
                model = solver.model()
                self.record_test(
                    "Solver execution", True, "Successfully solved constraints"
                )
            else:
                self.record_test(
                    "Solver execution", False, f"Unexpected result: {result}"
                )
        except Exception as e:
            self.record_test("Solver execution", False, f"Error during solving: {e!s}")

    def run_all_tests(self):
        """Run all setup tests and display results."""
        print(f"{self.BOLD}=== MCP-Solver Z3 Mode Setup Test ==={self.RESET}")

        self.test_configuration_files()
        self.test_z3_dependencies()
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
                "\nZ3 mode setup incomplete. Please fix the issues above before proceeding."
            )
            sys.exit(1)
        else:
            print(f"\n{self.GREEN}✓ All Z3 mode tests passed successfully!{self.RESET}")
            print("\nSystem is ready to use MCP-Solver in Z3 mode.")
            sys.exit(0)


def main():
    test = Z3SetupTest()
    test.run_all_tests()


if __name__ == "__main__":
    main()
