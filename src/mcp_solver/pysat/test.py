"""
Basic PySAT test file.

This file contains simple examples for using PySAT to validate the integration
works correctly in the MCP Solver environment.
"""

import os
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Any


def simple_sat_test():
    """
    Simple SAT solving test to verify PySAT is working correctly.

    This test creates a simple SAT formula, solves it, and prints the result.
    """
    try:
        from pysat.formula import CNF
        from pysat.solvers import Glucose3

        print("PySAT import successful")

        # Create a simple SAT formula: (a OR b) AND (NOT a OR c) AND (NOT b OR NOT c)
        formula = CNF()
        formula.append([1, 2])  # a OR b
        formula.append([-1, 3])  # NOT a OR c
        formula.append([-2, -3])  # NOT b OR NOT c

        print(f"Formula: {formula.clauses}")

        # Solve the formula
        start_time = time.time()
        solver = Glucose3()
        solver.append_formula(formula)

        is_sat = solver.solve()
        model = solver.get_model() if is_sat else None
        solve_time = time.time() - start_time

        # Print results
        print(f"Is satisfiable: {is_sat}")
        if is_sat:
            print(f"Model: {model}")
        print(f"Solve time: {solve_time:.6f} seconds")

        # Always clean up
        solver.delete()
        return True
    except ImportError as e:
        print(f"PySAT import error: {e}")
        return False
    except Exception as e:
        print(f"Error in SAT test: {e}")
        return False


def repl_test():
    """
    Interactive REPL test for PySAT.

    This function provides a simple REPL for testing PySAT functionality.
    """
    try:
        from pysat.formula import CNF
        from pysat.solvers import Cadical153, Glucose3

        print("PySAT REPL - Enter clauses in DIMACS format (example: 1 2 0 for a OR b)")
        print("Enter 'quit' to exit")

        formula = CNF()
        while True:
            try:
                user_input = input("> ")
                if user_input.lower() in ("quit", "exit", "q"):
                    break

                if user_input.lower() in ("solve", "s"):
                    # Solve the current formula
                    solver = Glucose3()
                    solver.append_formula(formula)

                    is_sat = solver.solve()
                    model = solver.get_model() if is_sat else None

                    print(f"Is satisfiable: {is_sat}")
                    if is_sat:
                        print(f"Model: {model}")
                        # Convert model to a readable format (var=True/False)
                        readable = {abs(lit): lit > 0 for lit in model}
                        print(f"Readable: {readable}")

                    # Clean up the solver
                    solver.delete()
                    continue

                if user_input.lower() in ("clear", "c"):
                    formula = CNF()
                    print("Formula cleared")
                    continue

                if user_input.lower() in ("print", "p"):
                    print(f"Current formula: {formula.clauses}")
                    continue

                # Parse DIMACS format (space-separated integers with 0 delimiter)
                tokens = user_input.strip().split()
                clause = []
                for token in tokens:
                    try:
                        lit = int(token)
                        if lit == 0:  # 0 marks the end of a clause in DIMACS
                            break
                        clause.append(lit)
                    except ValueError:
                        print(f"Invalid literal: {token}")

                if clause:
                    formula.append(clause)
                    print(f"Added clause: {clause}")
                    print(f"Current formula: {formula.clauses}")
            except Exception as e:
                print(f"Error: {e}")

        return True
    except ImportError as e:
        print(f"PySAT import error: {e}")
        return False
    except Exception as e:
        print(f"Error in REPL test: {e}")
        return False


class TimeoutException(Exception):
    """Exception raised when code execution times out."""

    pass


@contextmanager
def time_limit(seconds: float):
    """
    Context manager for limiting execution time of code blocks.

    Args:
        seconds: Maximum execution time in seconds

    Raises:
        TimeoutException: If execution exceeds the time limit
    """

    def signal_handler(signum, frame):
        raise TimeoutException(f"Execution timed out after {seconds} seconds")

    # Set the timeout handler
    previous_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    finally:
        # Reset the alarm and restore previous handler
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


# Global variable to store last solution
_LAST_SOLUTION = None


def export_solution(solver=None, variables=None, objective=None) -> dict[str, Any]:
    """
    Extract and format solutions from a PySAT solver.

    Args:
        solver: PySAT solver object (optional)
        variables: Dictionary mapping variable names to values (optional)
        objective: Objective value for optimization (optional)

    Returns:
        Dictionary with solution information
    """
    global _LAST_SOLUTION

    solution = {"status": "unknown", "variables": {}, "objective": None}

    # If variables are provided directly, use them
    if variables:
        solution["variables"] = variables
        solution["status"] = "satisfied"

    # If objective is provided, add it
    if objective is not None:
        solution["objective"] = objective

    # Store the solution
    _LAST_SOLUTION = solution

    return solution


def execute_pysat_code(code_string: str, timeout: float = 4.0) -> dict[str, Any]:
    """
    Execute PySAT Python code in a secure environment with timeout handling.

    Args:
        code_string: The PySAT Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with execution results
    """
    # IMPORTANT: Properly import PySAT
    # First, remove the current directory from the path to avoid importing ourselves
    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if current_dir in sys.path:
        sys.path.remove(current_dir)
    if parent_dir in sys.path:
        sys.path.remove(parent_dir)

    # Add site-packages to the front of the path
    import site

    site_packages = site.getsitepackages()
    for p in reversed(site_packages):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Import PySAT modules
    try:
        from pysat.card import CardEnc, EncType
        from pysat.examples.rc2 import RC2
        from pysat.formula import CNF, WCNF
        from pysat.solvers import (
            Cadical103,
            Cadical153,
            Cadical195,
            Glucose3,
            Glucose4,
            Glucose42,
            Lingeling,
            MapleCM,
            Mergesat3,
            Minicard,
            Minisat22,
            MinisatGH,
        )
    except ImportError as e:
        return {
            "status": "error",
            "error": f"Failed to import PySAT: {e!s}",
            "output": [],
            "solution": None,
            "execution_time": 0,
        }

    # Reset last solution
    global _LAST_SOLUTION
    _LAST_SOLUTION = None

    # Pre-process the code to handle imports
    code_lines = code_string.split("\n")
    processed_code = []

    for line in code_lines:
        # Skip import lines but keep PySAT namespace imports
        if (
            line.strip().startswith("from pysat import")
            or line.strip().startswith("import pysat")
            or line.strip().startswith("from pysat.formula import")
            or line.strip().startswith("from pysat.solvers import")
            or line.strip().startswith("from pysat.examples import")
            or line.strip().startswith("from pysat.card import")
        ):
            continue
        else:
            processed_code.append(line)

    processed_code_string = "\n".join(processed_code)

    # Create restricted globals dict with only necessary functions/modules
    restricted_globals = {
        # Allow a limited subset of builtins
        "Exception": Exception,
        "ImportError": ImportError,
        "NameError": NameError,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        # Explicitly set open to None to force NameError
        "open": None,
    }

    # Add PySAT modules to the globals
    restricted_globals["CNF"] = CNF
    restricted_globals["WCNF"] = WCNF
    restricted_globals["Cadical103"] = Cadical103
    restricted_globals["Cadical153"] = Cadical153
    restricted_globals["Cadical195"] = Cadical195
    restricted_globals["Glucose3"] = Glucose3
    restricted_globals["Glucose4"] = Glucose4
    restricted_globals["Glucose42"] = Glucose42
    restricted_globals["Lingeling"] = Lingeling
    restricted_globals["MapleCM"] = MapleCM
    restricted_globals["MinisatGH"] = MinisatGH
    restricted_globals["Minisat22"] = Minisat22
    restricted_globals["Minicard"] = Minicard
    restricted_globals["Mergesat3"] = Mergesat3
    restricted_globals["RC2"] = RC2
    restricted_globals["CardEnc"] = CardEnc
    restricted_globals["EncType"] = EncType

    # Add our solution export function
    restricted_globals["export_solution"] = export_solution

    # Prepare result dictionary
    result = {
        "status": "unknown",
        "error": None,
        "output": [],
        "solution": None,
        "execution_time": 0,
    }

    # Capture print output
    original_stdout = sys.stdout
    from io import StringIO

    captured_output = StringIO()
    sys.stdout = captured_output

    # Execute code with timeout
    start_time = time.time()

    try:
        with time_limit(timeout):
            # Execute the code in the restricted environment
            local_vars = {}
            exec(processed_code_string, restricted_globals, local_vars)

            # Check if solution was exported via local_vars
            if "solution" in local_vars:
                result["solution"] = local_vars["solution"]
                result["status"] = "success"
            # Also check if exported via our function
            elif _LAST_SOLUTION:
                result["solution"] = _LAST_SOLUTION
                result["status"] = "success"
            else:
                result["status"] = "no_solution"
                result["error"] = (
                    "No solution was exported. Make sure to call export_solution()."
                )
    except TimeoutException as e:
        result["status"] = "timeout"
        result["error"] = str(e)
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e!s}"
        result["traceback"] = traceback.format_exc()
    finally:
        # Restore stdout and record execution time
        sys.stdout = original_stdout
        result["execution_time"] = time.time() - start_time
        result["output"] = (
            captured_output.getvalue().strip().split("\n")
            if captured_output.getvalue()
            else []
        )

    return result


def memory_management_test(iterations=50):
    """
    Test memory management with repeated solver creation and deletion.

    This test creates and deletes many solver instances to verify proper cleanup.
    We're focusing on Cadical as the preferred solver.
    """
    try:
        import gc

        from pysat.solvers import Cadical153

        print(f"Creating and deleting {iterations} solver instances...")

        # Try to measure memory if psutil is available
        try:
            import psutil

            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_tracking = True
            print(f"Initial memory usage: {initial_memory:.2f} MB")
        except ImportError:
            memory_tracking = False
            print("psutil not available, memory tracking disabled")

        # Create and delete many solvers
        for i in range(iterations):
            solver = Cadical153()

            # Add some simple clauses
            for j in range(5):
                solver.add_clause([j + 1, -(j + 2)])

            # Solve
            solver.solve()

            # Delete explicitly
            solver.delete()

            if i % 10 == 0:
                print(f"  Completed {i} iterations")
                # Force garbage collection
                gc.collect()

        if memory_tracking:
            # Final garbage collection
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"Final memory usage: {final_memory:.2f} MB")
            print(f"Memory difference: {final_memory - initial_memory:.2f} MB")

            # A large increase might indicate a memory leak
            if final_memory - initial_memory > 50:  # More than 50MB growth
                print(
                    "WARNING: Significant memory growth detected. Possible memory leak."
                )

        print("Memory management test completed successfully")
        return True
    except Exception as e:
        print(f"Error in memory management test: {e}")
        import traceback

        traceback.print_exc()
        return False


def cardinality_constraints_test():
    """
    Test cardinality constraints which are important for template implementations.

    This test validates that the 'at most k' constraint works correctly with
    different numbers of true variables.
    """
    try:
        from pysat.card import CardEnc, EncType
        from pysat.solvers import Cadical153

        print("Testing cardinality constraints...")

        # Use sequential counter encoding which supports all bounds
        encoding = EncType.seqcounter

        # Variables for our test
        variables = [1, 2, 3, 4, 5]

        # Create "at most 2" constraint
        cnf = CardEnc.atmost(variables, 2, encoding=encoding)
        print(
            f"Generated {len(cnf.clauses)} clauses for 'at most 2 of 5 variables' constraint"
        )

        # Verify the constraint works correctly
        solver = Cadical153()
        solver.append_formula(cnf)

        # Test with different numbers of true variables
        print("Validating constraint with different numbers of true variables:")

        for num_true in range(6):  # 0 to 5 true variables
            # Create assumptions to force exactly num_true variables to be true
            assumptions = []
            for i, var in enumerate(variables):
                if i < num_true:
                    assumptions.append(var)  # Force this variable to be true
                else:
                    assumptions.append(-var)  # Force this variable to be false

            is_sat = solver.solve(assumptions=assumptions)
            expected = (
                num_true <= 2
            )  # Should be satisfiable only when ≤ 2 variables are true

            print(
                f"  {num_true} true variables: {'Satisfiable' if is_sat else 'Unsatisfiable'} "
                + f"(Expected: {'Satisfiable' if expected else 'Unsatisfiable'})"
            )

            # Verify results match expectations
            if is_sat != expected:
                print(f"ERROR: Unexpected result for {num_true} true variables!")

        # Also test "exactly k" constraint
        print("\nTesting 'exactly 2' constraint:")
        cnf_exactly = CardEnc.equals(variables, 2, encoding=encoding)

        solver = Cadical153()
        solver.append_formula(cnf_exactly)

        for num_true in range(6):  # 0 to 5 true variables
            assumptions = []
            for i, var in enumerate(variables):
                if i < num_true:
                    assumptions.append(var)
                else:
                    assumptions.append(-var)

            is_sat = solver.solve(assumptions=assumptions)
            expected = (
                num_true == 2
            )  # Should be satisfiable only when exactly 2 variables are true

            print(
                f"  {num_true} true variables: {'Satisfiable' if is_sat else 'Unsatisfiable'} "
                + f"(Expected: {'Satisfiable' if expected else 'Unsatisfiable'})"
            )

        solver.delete()
        print("Cardinality constraints test completed successfully")
        return True
    except Exception as e:
        print(f"Error in cardinality constraints test: {e}")
        import traceback

        traceback.print_exc()
        return False


def maxsat_with_constraints_test():
    """
    Test MaxSAT solving with cardinality constraints.

    This test demonstrates how to combine MaxSAT (weighted clauses)
    with cardinality constraints, which will be important for the templates.
    """
    try:
        from pysat.card import CardEnc, EncType
        from pysat.examples.rc2 import RC2
        from pysat.formula import WCNF

        print("Testing MaxSAT with cardinality constraints...")

        # Create a WCNF formula
        wcnf = WCNF()

        # We'll model a simple scheduling problem:
        # - We have 5 tasks (variables 1-5)
        # - Each task has a specific value (weight)
        # - We can execute at most 3 tasks
        # - Some tasks are incompatible (hard constraints)

        # Task values (as clause weights - higher is better)
        wcnf.append([1], weight=5)  # Task 1 has value 5
        wcnf.append([2], weight=3)  # Task 2 has value 3
        wcnf.append([3], weight=8)  # Task 3 has value 8
        wcnf.append([4], weight=2)  # Task 4 has value 2
        wcnf.append([5], weight=7)  # Task 5 has value 7

        # Hard constraints (tasks that can't be done together)
        wcnf.append([-1, -2])  # Task 1 and 2 are incompatible
        wcnf.append([-3, -5])  # Task 3 and 5 are incompatible

        # Add cardinality constraint: at most 3 tasks can be executed
        # First, create the constraint as a CNF
        at_most_3 = CardEnc.atmost(
            [1, 2, 3, 4, 5], bound=3, encoding=EncType.seqcounter
        )

        # Add each clause from the CNF to the WCNF as a hard constraint
        for clause in at_most_3.clauses:
            wcnf.append(clause)

        print(
            f"Created WCNF with {len(wcnf.hard)} hard clauses and {len(wcnf.soft)} soft clauses"
        )

        # Solve with RC2 (core-guided MaxSAT solver)
        with RC2(wcnf) as rc2:
            model = rc2.compute()  # Solve
            cost = rc2.cost  # Get the cost (sum of unsatisfied weights)

            if model:
                # Extract selected tasks (positive literals in the model)
                selected_tasks = [v for v in model if v > 0 and v <= 5]
                total_value = sum([5, 3, 8, 2, 7][task - 1] for task in selected_tasks)

                # Calculate the maximum possible value (sum of all weights)
                max_possible = sum([5, 3, 8, 2, 7])

                print("MaxSAT solution found:")
                print(f"  Selected tasks: {selected_tasks}")
                print(f"  Total value: {total_value}")
                print(f"  Cost (missed value): {cost}")
                print(
                    f"  Percentage of maximum: {round(total_value / max_possible * 100, 2)}%"
                )

                # Validate constraints
                if len(selected_tasks) > 3:
                    print("ERROR: Too many tasks selected!")

                if 1 in selected_tasks and 2 in selected_tasks:
                    print("ERROR: Incompatible tasks 1 and 2 both selected!")

                if 3 in selected_tasks and 5 in selected_tasks:
                    print("ERROR: Incompatible tasks 3 and 5 both selected!")
            else:
                print("No solution found.")

        # Calculate optimal solution manually for verification
        # The optimal solution should be tasks 1, 3, 4 with value 5+8+2=15
        # or tasks 1, 4, 5 with value 5+2+7=14
        # The hard constraints prevent 3 and 5 together

        print("MaxSAT with constraints test completed successfully")
        return True
    except Exception as e:
        print(f"Error in MaxSAT with constraints test: {e}")
        import traceback

        traceback.print_exc()
        return False


def dynamic_execution_test():
    """
    Test dynamic execution of PySAT code using exec().

    This test simulates how the model manager would execute user-provided Python code.
    It uses exec() to dynamically evaluate PySAT code within a controlled environment.
    """
    print("Testing dynamic execution of PySAT code...")

    # Sample code to execute
    code_samples = [
        # Simple SAT problem similar to the first test
        """
from pysat.formula import CNF
from pysat.solvers import Glucose3

# Create a simple SAT formula
formula = CNF()
formula.append([1, 2])      # a OR b
formula.append([-1, 3])     # NOT a OR c
formula.append([-2, -3])    # NOT b OR NOT c

# Solve the formula
solver = Glucose3()
solver.append_formula(formula)

is_sat = solver.solve()
model = solver.get_model() if is_sat else None

# Export solution
export_solution(variables={
    "is_satisfiable": is_sat,
    "model": model,
    "clauses": formula.clauses
})

# Clean up
solver.delete()
        """,
        # MaxSAT example
        """
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

# Create a weighted CNF
wcnf = WCNF()
wcnf.append([-1, -2])          # Hard clause (must be satisfied)
wcnf.append([1], weight=1)     # Soft clause with weight 1
wcnf.append([2], weight=2)     # Soft clause with weight 2

# Solve MaxSAT problem
with RC2(wcnf) as rc2:
    model = rc2.compute()
    cost = rc2.cost

# Export solution
export_solution(variables={
    "maxsat_model": model,
    "maxsat_cost": cost,
    "hard_clauses": wcnf.hard,
    "soft_clauses": wcnf.soft
})
        """,
    ]

    # Execute each code sample
    for i, code in enumerate(code_samples):
        print(f"\n--- Executing code sample {i + 1} ---")

        try:
            # Execute the code
            result = execute_pysat_code(code)

            # Print results
            print(f"Status: {result['status']}")
            print(f"Execution time: {result['execution_time']:.6f} seconds")

            if result["output"]:
                print("Output:")
                for line in result["output"]:
                    print(f"  {line}")

            if result["solution"]:
                print("Solution:")
                for key, value in result["solution"].items():
                    print(f"  {key}: {value}")

            if result["error"]:
                print(f"Error: {result['error']}")

        except Exception as e:
            print(f"Error in execution: {e}")
            import traceback

            traceback.print_exc()

    return True


if __name__ == "__main__":
    print("Running basic PySAT tests...")

    # Run the simple SAT test
    print("\n=== Simple SAT Test ===")
    simple_sat_test()

    # Run the memory management test
    print("\n=== Memory Management Test ===")
    memory_management_test()

    # Run the cardinality constraints test
    print("\n=== Cardinality Constraints Test ===")
    cardinality_constraints_test()

    # Run the maxsat with constraints test
    print("\n=== MaxSAT with Constraints Test ===")
    maxsat_with_constraints_test()

    # Run the dynamic execution test
    print("\n=== Dynamic Execution Test ===")
    dynamic_execution_test()

    # Run the REPL test if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--repl":
        print("\n=== REPL Test ===")
        repl_test()
