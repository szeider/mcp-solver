"""
Integration tests for MCP Solver.
Tests features that work across different solvers.
"""
import pytest
import asyncio
from datetime import timedelta
import os
import sys

# Import fixtures
from conftest import mzn_manager, pysat_manager, z3_manager

def test_all_solvers_initialization():
    """Test that all solvers can be initialized"""
    # All three managers should be initialized already by the fixtures
    assert mzn_manager.initialized == True
    assert pysat_manager.initialized == True
    assert z3_manager.initialized == True

def test_all_solvers_clear_model():
    """Test that all solvers can clear their models"""
    # Clear all models
    asyncio.run(mzn_manager.clear_model())
    asyncio.run(pysat_manager.clear_model())
    asyncio.run(z3_manager.clear_model())
    
    # No exceptions means success
    assert True

def test_common_interface_consistency():
    """Test that all solvers have the same interface methods"""
    # Get all methods from each manager
    mzn_methods = set(dir(mzn_manager))
    pysat_methods = set(dir(pysat_manager))
    z3_methods = set(dir(z3_manager))
    
    # Define the common interface methods expected in all solvers
    common_methods = {
        "clear_model",
        "add_item",
        "delete_item",
        "replace_item",
        "solve_model",
        "get_solution",
        "get_solve_time"
    }
    
    # All common methods should be in each manager
    for method in common_methods:
        assert method in mzn_methods, f"Method {method} not found in MiniZinc manager"
        assert method in pysat_methods, f"Method {method} not found in PySAT manager"
        assert method in z3_methods, f"Method {method} not found in Z3 manager"

@pytest.mark.parametrize("manager,code,expected", [
    (
        "mzn_manager", 
        ["var 1..10: x;", "constraint x > 5;", "solve satisfy;"],
        True
    ),
    (
        "pysat_manager", 
        [
            """
from pysat.formula import CNF
from pysat.solvers import Glucose3

def export_solution(sol_dict):
    global solution
    solution = sol_dict

formula = CNF()
formula.append([1])
            """,
            """
solver = Glucose3()
solver.append_formula(formula)
is_sat = solver.solve()
solution = {"satisfiable": is_sat}
if is_sat:
    solution["model"] = solver.get_model()
export_solution(solution)
solver.delete()
            """
        ],
        True
    ),
    (
        "z3_manager", 
        [
            """
from z3 import *

def export_solution(sol_dict):
    global solution
    solution = sol_dict

x = Int('x')
solver = Solver()
solver.add(x > 5)
solver.add(x < 10)
            """,
            """
if solver.check() == sat:
    model = solver.model()
    solution = {
        "satisfiable": True,
        "x": model[x].as_long()
    }
else:
    solution = {
        "satisfiable": False
    }
export_solution(solution)
            """
        ],
        True
    )
])
def test_solve_on_all_solvers(manager, code, expected, request):
    """Test that all solvers can solve models"""
    # Get the manager fixture
    mgr = request.getfixturevalue(manager)
    
    # Clear the model
    asyncio.run(mgr.clear_model())
    
    # Add code items
    for i, item in enumerate(code, 1):
        asyncio.run(mgr.add_item(i, item))
    
    # Solve the model
    result = asyncio.run(mgr.solve_model(timedelta(seconds=5)))
    
    # Check the result
    if manager == "mzn_manager":
        assert result["status"] == "SAT"
    else:
        assert result["success"] == expected 