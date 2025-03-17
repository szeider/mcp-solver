"""
Unit tests for PySAT model manager.
"""
import pytest
import asyncio
from datetime import timedelta
import os
import sys

# Import fixtures
from conftest import pysat_manager

def test_clear_model(pysat_manager):
    """Test clearing the model"""
    asyncio.run(pysat_manager.clear_model())
    assert True  # Model cleared with no exception

def test_add_item(pysat_manager):
    """Test adding an item to the model"""
    asyncio.run(pysat_manager.clear_model())
    result = asyncio.run(pysat_manager.add_item(1, "from pysat.formula import CNF\ncnf = CNF()"))
    assert result["success"] == True
    
def test_delete_item(pysat_manager):
    """Test deleting an item from the model"""
    asyncio.run(pysat_manager.clear_model())
    asyncio.run(pysat_manager.add_item(1, "from pysat.formula import CNF"))
    asyncio.run(pysat_manager.add_item(2, "cnf = CNF()"))
    result = asyncio.run(pysat_manager.delete_item(1))
    assert result["success"] == True
    
def test_replace_item(pysat_manager):
    """Test replacing an item in the model"""
    asyncio.run(pysat_manager.clear_model())
    asyncio.run(pysat_manager.add_item(1, "from pysat.formula import CNF"))
    result = asyncio.run(pysat_manager.replace_item(1, "from pysat.formula import CNF, WCNF"))
    assert result["success"] == True

def test_solve_simple_model(pysat_manager):
    """Test solving a simple model"""
    asyncio.run(pysat_manager.clear_model())
    code = """
from pysat.formula import CNF
from pysat.solvers import Glucose3

# Create a CNF formula with a single variable
formula = CNF()
formula.append([1])  # Variable 1 must be True

# Solve with Glucose3
solver = Glucose3()
solver.append_formula(formula)
is_sat = solver.solve()

# Export solution - always required
solution = {}
if is_sat:
    solution["model"] = solver.get_model()
    solution["satisfiable"] = True
else:
    solution["satisfiable"] = False

def export_solution(sol_dict):
    global solution
    solution = sol_dict

export_solution(solution)
solver.delete()
"""
    asyncio.run(pysat_manager.add_item(1, code))
    result = asyncio.run(pysat_manager.solve_model(timedelta(seconds=2)))
    assert result["success"] == True
    assert result.get("solution", {}).get("satisfiable") == True

def test_get_model(pysat_manager):
    """Test getting the model content"""
    asyncio.run(pysat_manager.clear_model())
    code = "# This is a test model\nfrom pysat.formula import CNF"
    asyncio.run(pysat_manager.add_item(1, code))
    model = pysat_manager.get_model()
    assert isinstance(model, dict)
    assert "items" in model
    assert len(model["items"]) == 1 