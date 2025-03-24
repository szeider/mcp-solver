# Variable Scope Test

Create a Z3 model that defines variables in a function and successfully exports them.

## Problem:

Write a Z3 model that follows these steps:
1. Define a function that creates variables and constraints
2. Call the function to build your model 
3. Export the solution using export_solution (this is crucial)

Your function should create at least two variables with constraints.
This tests if variables defined in a function scope are accessible during solution extraction.

IMPORTANT: Make sure to uncomment and use export_solution(solver=solver, variables=variables) to properly export the solution. 