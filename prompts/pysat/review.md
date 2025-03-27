# Solution Correctness Verification

## Task

You are given a problem description, a PySAT encoding, and a solution. Verify the correctness of the solution.

## Evaluation Criteria

- **For satisfiable solutions**: Verify that all constraints in the problem description are satisfied. Answer *correct* if satisfied, otherwise *incorrect*.
- **For unsatisfiable solutions**: Verify that all clauses produced by teh encoding are actually required by the problem statement or are valid symmetry breaking constraints. Answer *correct* if valid, otherwise *incorrect*.
- **For no solution/timeout/unverifiable cases**: Answer *unknown*.

## Output Format

1. Verdict: *correct*, *incorrect*, or *unknown*
2. Confidence: Integer from 0 (not confident) to 10 (very confident)
3. Brief justification of your assessment

## Data

### Problem Statement

$PROBLEM

### PySAT Encoding

$MODEL

### Solution

$SOLUTION

