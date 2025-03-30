# Solution Correctness Verification

## Task

You are given a problem description, a MiniZinc model, and a solution. Verify the correctness of the solution.

## Evaluation Criteria

- **For satisfiable solutions**: Verify that all constraints in the problem description are satisfied. Answer *correct* if satisfied, otherwise *incorrect*.

  *Check constraint by constraint.*

- **For unsatisfiable solutions**: Verify that all constraints in the MiniZinc model are actually required by the problem statement or are valid symmetry breaking constraints. Answer *correct* if valid, otherwise *incorrect*.

  **check constraint by constraint**

- **For no solution/timeout/unverifiable cases**: Answer *unknown*.

## Output Format

IMPORTANT: Your answer MUST follow this exact JSON format:

```json
{
  "correctness": "correct",  // Must be exactly one of: "correct", "incorrect", or "unknown"
  "confidence": 10,          // Integer from 0 (not confident) to 10 (very confident)
  "explanation": "Your detailed justification of the assessment here, list all constraints."
}
```

DO NOT include anything else before or after the JSON object. Format your entire answer as a valid JSON object as shown above.

## Data

### Problem Statement

$PROBLEM

### MiniZinc Model

$MODEL

### Solution

$SOLUTION

