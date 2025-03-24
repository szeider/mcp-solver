# Nested Scope Test

Create a Z3 model that defines variables in nested function scopes to verify that scope management works correctly.

## Problem:

Write a Z3 model that:
1. Creates an outer function that defines some variables
2. Creates an inner function that adds constraints to those variables
3. Uses those functions to build a model
4. Exports the solution

This tests if variables defined across multiple nested function scopes are properly accessible during solution extraction.

IMPORTANT: Make sure to call export_solution(solver=solver, variables=variables) to properly export the solution. 