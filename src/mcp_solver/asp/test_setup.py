# test_setup.py
# Test setup for ASP integration 

import asyncio
from datetime import timedelta
from .model_manager import ASPModelManager
from .templates.basic_templates import facts, rule

def test_simple_asp():
    """Test solving a simple ASP program with one answer set."""
    mgr = ASPModelManager()
    
    fact_list = ['a', 'b']
    asp_facts = facts(fact_list)
    asp_rule = rule('c', ['a', 'b'])

    mgr.code_items = [
        asp_facts,
        asp_rule,
    ]

    result = asyncio.run(mgr.solve_model(timedelta(seconds=5)))
    print("Test simple ASP:", result)
    assert result["success"]
    assert len(result["answer_sets"]) >= 1
    print("Answer sets:", result["answer_sets"])
    sol = mgr.get_solution()
    print("Get solution:", sol)
    assert sol["success"]
    var_val = mgr.get_variable_value("c")
    print("Get variable value for 'c':", var_val)
    assert var_val["success"]
    print("Solve time:", mgr.get_solve_time())

def test_error_handling():
    """Test error handling for invalid ASP code."""
    mgr = ASPModelManager()
    mgr.code_items = ["this is not valid ASP code"]
    result = asyncio.run(mgr.solve_model(timedelta(seconds=2)))
    assert not result["success"]
    print("Test error handling:", result)

def test_empty_model():
    """Test solving with an empty model."""
    mgr = ASPModelManager()
    result = asyncio.run(mgr.solve_model(timedelta(seconds=2)))
    assert not result["success"]
    print("Test empty model:", result)

def main():
    print("Running ASPModelManager tests...")
    test_simple_asp()
    test_error_handling()
    test_empty_model()
    print("All ASPModelManager tests completed.") 

if __name__ == "__main__":
    main()