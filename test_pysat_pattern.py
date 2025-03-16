"""
Simplified test for direct conditional pattern in PySAT solver.
"""

import sys
import os
import re

# Test regex patterns for both direct conditional and variable assignment
direct_pattern = re.compile(r'if\s+\w+\.solve\(\)')
var_pattern = re.compile(r'(\w+)\s*=\s*\w+\.solve\(\)')

def test_regex_patterns():
    """Test that our regex patterns work correctly."""
    # Test direct conditional pattern
    direct_matches = direct_pattern.findall("""
    if solver.solve():
        print("Satisfiable")
    """)
    assert len(direct_matches) == 1, "Should find one direct conditional pattern"
    
    # Test variable assignment pattern
    var_matches = var_pattern.findall("""
    is_sat = solver.solve()
    if is_sat:
        print("Satisfiable")
    """)
    assert len(var_matches) == 1, "Should find one variable assignment pattern"
    assert var_matches[0] == "is_sat", "Should extract the correct variable name"
    
    # Test with different variable name
    var_matches2 = var_pattern.findall("""
    satisfiable = solver.solve()
    if satisfiable:
        print("Satisfiable")
    """)
    assert len(var_matches2) == 1, "Should find one variable assignment pattern"
    assert var_matches2[0] == "satisfiable", "Should extract the correct variable name"

def print_results():
    """Print test results in a readable format."""
    # Test direct conditional pattern
    direct_line = "if solver.solve():"
    direct_match = direct_pattern.search(direct_line)
    print(f"Direct conditional pattern:")
    print(f"  Pattern: {direct_pattern.pattern}")
    print(f"  Line: '{direct_line}'")
    print(f"  Match: {direct_match is not None}")
    print()
    
    # Test variable assignment pattern
    var_line1 = "is_sat = solver.solve()"
    var_line2 = "satisfiable = solver.solve()"
    var_match1 = var_pattern.search(var_line1)
    var_match2 = var_pattern.search(var_line2)
    
    print(f"Variable assignment pattern:")
    print(f"  Pattern: {var_pattern.pattern}")
    print(f"  Line 1: '{var_line1}'")
    print(f"  Match 1: {var_match1 is not None}")
    if var_match1:
        print(f"  Extracted variable: '{var_match1.group(1)}'")
    
    print(f"  Line 2: '{var_line2}'")
    print(f"  Match 2: {var_match2 is not None}")
    if var_match2:
        print(f"  Extracted variable: '{var_match2.group(1)}'")

if __name__ == "__main__":
    print("Testing regex patterns for PySAT code transformation\n")
    try:
        test_regex_patterns()
        print("✅ All tests passed!")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
    
    print("\nDetailed results:")
    print_results() 