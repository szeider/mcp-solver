# MaxSAT Example Problem

This example demonstrates how to use MaxSAT for optimization problems following the required standard pattern.

## Problem Description

We have two variables x₁ and x₂ with:

1. Hard constraints:
   - At least one variable must be true (x₁ OR x₂)
   - At most one variable can be true (NOT x₁ OR NOT x₂)

2. Soft constraints:
   - Prefer x₁ to be true (weight: 1)
   - Prefer x₂ to be true (weight: 2)

Since both variables can't be true simultaneously due to the hard constraints, and the soft constraint for x₂ has a higher weight (2 > 1), the optimal solution should set x₂ to true and x₁ to false, with a cost of 1 (for violating the soft constraint on x₁).

## Task

Implement this MaxSAT problem following the required standard pattern:

1. Create a WCNF formula with the hard and soft constraints
2. Create a standard solver for MCP validation 
3. Use RC2 for MaxSAT optimization
4. Use export_maxsat_solution to handle the MaxSAT result
5. Follow with solver.solve() and export_solution to complete the MCP requirements