"""
Unit tests for Z3 model manager.
"""
import pytest
import asyncio
from datetime import timedelta
import os
import sys

# Import fixtures
from conftest import z3_manager

def test_clear_model(z3_manager):
    """Test clearing the model"""
    asyncio.run(z3_manager.clear_model())
    assert True  # Model cleared with no exception

def test_add_item(z3_manager):
    """Test adding an item to the model"""
    asyncio.run(z3_manager.clear_model())
    result = asyncio.run(z3_manager.add_item(1, "from z3 import *\nx = Int('x')"))
    assert result["success"] == True
    
def test_delete_item(z3_manager):
    """Test deleting an item from the model"""
    asyncio.run(z3_manager.clear_model())
    asyncio.run(z3_manager.add_item(1, "from z3 import *"))
    asyncio.run(z3_manager.add_item(2, "x = Int('x')"))
    result = asyncio.run(z3_manager.delete_item(1))
    assert result["success"] == True
    
def test_replace_item(z3_manager):
    """Test replacing an item in the model"""
    asyncio.run(z3_manager.clear_model())
    asyncio.run(z3_manager.add_item(1, "from z3 import *"))
    result = asyncio.run(z3_manager.replace_item(1, "from z3 import *, sat"))
    assert result["success"] == True

def test_solve_simple_model(z3_manager):
    """Test solving a simple model"""
    asyncio.run(z3_manager.clear_model())
    code = """
from z3 import *

# Create solver
solver = Solver()

# Define variables
x = Int('x')

# Add constraints
solver.add(x > 5)
solver.add(x < 10)

# Define result variable
solution = {}

# Solve
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

# Export solution
def export_solution(sol_dict):
    global solution
    solution = sol_dict

export_solution(solution)
"""
    asyncio.run(z3_manager.add_item(1, code))
    result = asyncio.run(z3_manager.solve_model(timedelta(seconds=2)))
    assert result["success"] == True
    assert result.get("solution", {}).get("satisfiable") == True
    assert 5 < result.get("solution", {}).get("x", 0) < 10

def test_get_model(z3_manager):
    """Test getting the model content"""
    asyncio.run(z3_manager.clear_model())
    code = "# This is a test model\nfrom z3 import *"
    asyncio.run(z3_manager.add_item(1, code))
    model = z3_manager.get_model()
    assert isinstance(model, dict)
    assert "items" in model
    assert len(model["items"]) == 1 