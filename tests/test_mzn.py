"""
Unit tests for MiniZinc model manager.
"""
import pytest
import asyncio
from datetime import timedelta
import os
import sys

# Import fixtures
from conftest import mzn_manager

def test_clear_model(mzn_manager):
    """Test clearing the model"""
    asyncio.run(mzn_manager.clear_model())
    assert True  # Model cleared with no exception

def test_add_item(mzn_manager):
    """Test adding an item to the model"""
    asyncio.run(mzn_manager.clear_model())
    result = asyncio.run(mzn_manager.add_item(1, "var int: x;"))
    assert result["success"] == True
    assert "model" in result
    assert "x" in result["model"]

def test_delete_item(mzn_manager):
    """Test deleting an item from the model"""
    asyncio.run(mzn_manager.clear_model())
    asyncio.run(mzn_manager.add_item(1, "var int: x;"))
    asyncio.run(mzn_manager.add_item(2, "var int: y;"))
    result = asyncio.run(mzn_manager.delete_item(1))
    assert result["success"] == True
    assert "model" in result
    assert "x" not in result["model"]
    assert "y" in result["model"]

def test_replace_item(mzn_manager):
    """Test replacing an item in the model"""
    asyncio.run(mzn_manager.clear_model())
    asyncio.run(mzn_manager.add_item(1, "var int: x;"))
    result = asyncio.run(mzn_manager.replace_item(1, "var int: z;"))
    assert result["success"] == True
    assert "model" in result
    assert "z" in result["model"]
    assert "x" not in result["model"]

def test_solve_simple_model(mzn_manager):
    """Test solving a simple model"""
    asyncio.run(mzn_manager.clear_model())
    asyncio.run(mzn_manager.add_item(1, "var 1..10: x;"))
    asyncio.run(mzn_manager.add_item(2, "constraint x > 5;"))
    asyncio.run(mzn_manager.add_item(3, "solve satisfy;"))
    result = asyncio.run(mzn_manager.solve_model(timedelta(seconds=2)))
    assert result["status"] == "SAT"
    assert "solution" in result
    assert int(result["solution"]["x"]) > 5

def test_get_solution(mzn_manager):
    """Test getting a solution"""
    asyncio.run(mzn_manager.clear_model())
    asyncio.run(mzn_manager.add_item(1, "var 1..10: x;"))
    asyncio.run(mzn_manager.add_item(2, "constraint x > 5;"))
    asyncio.run(mzn_manager.add_item(3, "solve satisfy;"))
    asyncio.run(mzn_manager.solve_model(timedelta(seconds=2)))
    result = mzn_manager.get_solution()
    assert "solution" in result
    assert int(result["solution"]["x"]) > 5

def test_get_variable_value(mzn_manager):
    """Test getting a variable value"""
    asyncio.run(mzn_manager.clear_model())
    asyncio.run(mzn_manager.add_item(1, "var 1..10: x;"))
    asyncio.run(mzn_manager.add_item(2, "constraint x > 5;"))
    asyncio.run(mzn_manager.add_item(3, "solve satisfy;"))
    asyncio.run(mzn_manager.solve_model(timedelta(seconds=2)))
    result = mzn_manager.get_variable_value("x")
    assert "value" in result
    assert int(result["value"]) > 5

def test_get_solve_time(mzn_manager):
    """Test getting solve time"""
    asyncio.run(mzn_manager.clear_model())
    asyncio.run(mzn_manager.add_item(1, "var 1..10: x;"))
    asyncio.run(mzn_manager.add_item(2, "constraint x > 5;"))
    asyncio.run(mzn_manager.add_item(3, "solve satisfy;"))
    asyncio.run(mzn_manager.solve_model(timedelta(seconds=2)))
    result = mzn_manager.get_solve_time()
    assert "solve_time" in result
    assert result["solve_time"] >= 0 