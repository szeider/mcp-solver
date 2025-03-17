"""
Basic test for PySAT integration.

This script tests the basic functionality of the PySAT integration.
"""

import sys
import os
import time

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath('src'))

from mcp_solver.pysat.environment import execute_pysat_code
from mcp_solver.pysat.model_manager import PySATModelManager

def test_simple_sat():
    """Test a simple SAT problem."""
    code = """
# Import PySAT components
from pysat.formula import CNF
from pysat.solvers import Cadical153

# Create a CNF formula
formula = CNF()

# Add clauses: (a OR b) AND (NOT a OR c) AND (NOT b OR NOT c)
formula.append([1, 2])       # Clause 1: a OR b
formula.append([-1, 3])      # Clause 2: NOT a OR c
formula.append([-2, -3])     # Clause 3: NOT b OR NOT c

# Create solver and add the formula
solver = Cadical153()
solver.append_formula(formula)

# Solve the formula using direct conditional check
if solver.solve():
    model = solver.get_model()
    
    # Create a mapping of variable names to IDs
    variables = {
        "a": 1,
        "b": 2,
        "c": 3
    }
    
    # Print results
    print(f"Is satisfiable: True")
    print(f"Model: {model}")
    
    # Export the solution
    export_solution({
        "satisfiable": True,
        "model": model,
        "assignment": {
            "a": 1 in model,
            "b": 2 in model,
            "c": 3 in model
        }
    })
else:
    print(f"Is satisfiable: False")
    # Export the solution
    export_solution({
        "satisfiable": False
    })

# Free solver memory
solver.delete()
"""
    
    result = execute_pysat_code(code)
    
    print("=== Simple SAT Test ===")
    print(f"Success: {result['success']}")
    print(f"Output: {result['output']}")
    if result['error']:
        print(f"Error: {result['error']}")
    if result['solution']:
        print(f"Solution: {result['solution']}")
    print()
    
    return result['success']

def test_maxsat():
    """Test a MaxSAT problem."""
    code = """
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

# Create a WCNF formula
wcnf = WCNF()

# Add hard constraints (must be satisfied)
wcnf.append([1, 2])  # a OR b

# Add soft constraints with weights
wcnf.append([-1], weight=10)  # NOT a (weight 10)
wcnf.append([-2], weight=5)   # NOT b (weight 5)

# Solve with RC2
with RC2(wcnf) as rc2:
    model = rc2.compute()
    cost = rc2.cost

# Export solution with variables and objective
variables = {"a": 1, "b": 2}
export_solution(rc2, variables, cost)
"""
    
    result = execute_pysat_code(code)
    
    print("=== MaxSAT Test ===")
    print(f"Success: {result['success']}")
    print(f"Output: {result['output']}")
    if result['error']:
        print(f"Error: {result['error']}")
    if result['solution']:
        print(f"Solution: {result['solution']}")
    print()
    
    return result['success']

def test_cardinality_constraints():
    """Test cardinality constraints."""
    code = """
from pysat.formula import CNF
from pysat.solvers import Cadical153
from pysat.card import CardEnc, EncType

# Create variables (1 to 5)
variables = [1, 2, 3, 4, 5]

# At most 2 variables can be True
atmost2 = CardEnc.atmost(variables, 2, encoding=EncType.seqcounter)

# Create the formula
formula = CNF()
for clause in atmost2.clauses:
    formula.append(clause)

# Create solver and add the formula
solver = Cadical153()
solver.append_formula(formula)

# Solve using direct conditional check
if solver.solve():
    model = solver.get_model()
    
    # Count how many variables are True
    true_count = sum(1 for var in variables if var in model)
    
    # Print results
    print(f"Is satisfiable: True")
    print(f"Model: {model}")
    print(f"True count: {true_count}")
    
    # Create variable mapping
    var_names = {f"var_{i}": i for i in variables}
    
    # Export the solution
    export_solution({
        "satisfiable": True,
        "model": model,
        "assignment": {f"var_{i}": i in model for i in variables}
    })
else:
    print(f"Is satisfiable: False")
    export_solution({
        "satisfiable": False
    })

# Free solver memory
solver.delete()
"""
    
    result = execute_pysat_code(code)
    
    print("=== Cardinality Constraints Test ===")
    print(f"Success: {result['success']}")
    print(f"Output: {result['output']}")
    if result['error']:
        print(f"Error: {result['error']}")
    if result['solution']:
        print(f"Solution: {result['solution']}")
    print()
    
    return result['success']

def test_model_manager():
    """Test the PySAT model manager."""
    print("=== Model Manager Test ===")
    
    try:
        # Create model manager
        manager = PySATModelManager(lite_mode=True)
        
        # Add the code
        async def run_test():
            try:
                # Clear model
                await manager.clear_model()
                
                # Add items
                await manager.add_item(1, """
# Import PySAT components
from pysat.formula import CNF
from pysat.solvers import Cadical153

# Create a CNF formula
formula = CNF()
""")
                
                await manager.add_item(2, """
# Add clauses: (a OR b) AND (NOT a OR c) AND (NOT b OR NOT c)
formula.append([1, 2])       # Clause 1: a OR b
formula.append([-1, 3])      # Clause 2: NOT a OR c
formula.append([-2, -3])     # Clause 3: NOT b OR NOT c
""")
                
                await manager.add_item(3, """
# Create solver and add the formula
solver = Cadical153()
solver.append_formula(formula)

# Solve the formula using direct conditional check
if solver.solve():
    model = solver.get_model()
    
    # Create a mapping of variable names to IDs
    variables = {
        "a": 1,
        "b": 2,
        "c": 3
    }
    
    # Export the solution
    export_solution({
        "satisfiable": True,
        "model": model,
        "assignment": {
            "a": 1 in model,
            "b": 2 in model,
            "c": 3 in model
        }
    })
else:
    export_solution({
        "satisfiable": False
    })

# Free solver memory
solver.delete()
""")
                
                # Get model
                model_items = manager.get_model()
                print(f"Model has {len(model_items)} items")
                
                # Solve model with timeout
                from datetime import timedelta
                solve_result = await manager.solve_model(timeout=timedelta(seconds=5))
                print(f"Solve result: {solve_result}")
                
                # Get solution
                solution = manager.get_solution()
                print(f"Solution: {solution}")
                
                # Get variable value
                var_value = manager.get_variable_value("a")
                print(f"Value of 'a': {var_value}")
                
                # Get solve time
                solve_time = manager.get_solve_time()
                print(f"Solve time: {solve_time}")
                
                # Test replace item
                await manager.replace_item(2, """
# Add different clauses
formula.append([1, 2])       # Clause 1: a OR b
formula.append([-1, -2])     # Different clause
""")
                
                # Solve again
                solve_result = await manager.solve_model(timeout=timedelta(seconds=5))
                print(f"Solve result after replacement: {solve_result}")
                
                # Delete item
                await manager.delete_item(1)
                model_items = manager.get_model()
                print(f"Model has {len(model_items)} items after deletion")
                
                return True
            except Exception as e:
                import traceback
                print(f"Error in run_test: {e}")
                print(traceback.format_exc())
                return False
        
        import asyncio
        success = asyncio.run(run_test())
        print(f"Model manager test success: {success}")  # Debug output
        return success
        
    except Exception as e:
        import traceback
        print(f"Error in test_model_manager: {e}")
        print(traceback.format_exc())
        return False

def run_all_tests():
    """Run all tests."""
    
    tests = [
        ("Simple SAT", test_simple_sat),
        ("MaxSAT", test_maxsat),
        ("Cardinality Constraints", test_cardinality_constraints),
        ("Model Manager", test_model_manager)
    ]
    
    results = {}
    
    print("Running PySAT integration tests...\n")
    
    for name, test_func in tests:
        print(f"Running test: {name}")
        try:
            start_time = time.time()
            success = test_func()
            end_time = time.time()
            results[name] = {
                "success": success,
                "time": end_time - start_time
            }
        except Exception as e:
            import traceback
            print(f"Error in test {name}: {e}")
            print(traceback.format_exc())
            results[name] = {
                "success": False,
                "error": str(e)
            }
    
    # Print summary
    print("\n=== Test Summary ===")
    all_success = True
    for name, result in results.items():
        success = result.get("success", False)
        all_success = all_success and success
        status = "✅ Passed" if success else "❌ Failed"
        time_str = f" ({result.get('time', 0):.3f}s)" if "time" in result else ""
        print(f"{name}: {status}{time_str}")
        if not success and "error" in result:
            print(f"  Error: {result['error']}")
    
    return all_success

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 