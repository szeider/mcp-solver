#!/usr/bin/env python3
"""
Test script for verifying the installation of MCP-Solver and
MiniZinc Python binding with the Chuffed solver.
This script performs:
  1. A check for the existence of the instructions prompt file (instructions_prompt.md)
  2. Running a simple MiniZinc model using the Chuffed solver.
  3. Printing detailed messages at each step.
"""

import os
import sys
from pathlib import Path
from minizinc import Model, Instance, Solver, MiniZincError

def check_file(file_name: str, base_dir: Path) -> bool:
    file_path = base_dir / file_name
    if file_path.exists():
        print(f"[OK] Found '{file_name}' at: {file_path.resolve()}")
        return True
    else:
        print(f"[ERROR] '{file_name}' NOT found at: {file_path.resolve()}")
        return False

def run_minizinc_test():
    # Define a simple MiniZinc model as a string.
    # This model declares a variable x (0..1) and forces it to be 1.
    model_code = """
    var 0..1: x;
    constraint x = 1;
    solve satisfy;
    """
    print("\n--- MiniZinc Model Code ---")
    print(model_code)
    
    # Create a MiniZinc model and add the string.
    model = Model()
    try:
        model.add_string(model_code)
        print("MiniZinc model created and code added successfully.")
    except MiniZincError as e:
        print("Error adding MiniZinc code to the model:")
        print(e)
        sys.exit(1)
    
    # Look up the Chuffed solver (our default).
    try:
        solver = Solver.lookup("chuffed")
        print(f"Chuffed solver found: Name='{solver.name}', Version='{solver.version}', ID='{solver.id}'")
    except Exception as e:
        print("Error looking up the Chuffed solver:")
        print(e)
        sys.exit(1)
    
    # Create an instance with the model and the Chuffed solver.
    instance = Instance(solver, model)
    print("MiniZinc instance created using the Chuffed solver.")
    
    # Solve the instance and output the result.
    try:
        print("Starting to solve the MiniZinc instance...")
        result = instance.solve()
        print("Solving completed.")
    except MiniZincError as e:
        print("Error during MiniZinc solving process:")
        print(e)
        sys.exit(1)
    
    # Print detailed result information.
    print("\n--- MiniZinc Solve Result ---")
    print("Status    :", result.status)
    print("Solution  :", result.solution)
    print("Statistics:", result.statistics)

def main():
    print("=== Starting Installation Test for MCP-Solver ===")
    
    # Set the base directory to the project root.
    # Since this file is located at /project_root/src/mcp_solver/test_setup.py,
    # the project root is two levels up.
    base_dir = Path(__file__).resolve().parents[2]
    print("Base directory is set to:", base_dir)
    
    # Check for the required instructions prompt file.
    if not check_file("instructions_prompt.md", base_dir):
        print("Required configuration file 'instructions_prompt.md' is missing. Aborting test.")
        sys.exit(1)
    else:
        print("Required configuration file found.")
    
    # Run MiniZinc binding test using the Chuffed solver.
    run_minizinc_test()
    
    print("\n=== Installation Test Completed Successfully ===")

if __name__ == "__main__":
    main()
