# Solution Correctness Verification

## Task

You are given a problem description, a Python Z3 encoding, and a solution. Verify the correctness of the solution.

## Evaluation Criteria

- **For satisfiable solutions**: Verify that all constraints in the problem description are satisfied. Answer *correct* if satisfied, otherwise *incorrect*.
- **For unsatisfiable solutions**: Verify that all clauses produced by the encoding are actually required by the problem statement or are valid symmetry breaking constraints. Answer *correct* if valid, otherwise *incorrect*.
- **For no solution/timeout/unverifiable cases**: Answer *unknown*.

## Output Format

IMPORTANT: Your answer MUST follow this exact JSON format:

```json
{
  "explanation": "Your detailed justification of the assessment here, list all constraints. END WITH A CLEAR CONCLUSION STATEMENT.",
  "correctness": "correct"
}
```

The "correctness" field MUST be exactly one of: "correct", "incorrect", or "unknown".

IMPORTANT: Follow this exact process:
1. First, analyze each constraint in detail in the "explanation" field
2. End your explanation with a clear conclusion: "The solution is correct." or "The solution is incorrect."
3. Only AFTER completing your explanation, set the "correctness" field to match your conclusion

IMPORTANT: Before finalizing your response, always check that:
1. Your explanation ends with a clear conclusion statement
2. The "correctness" field matches your conclusion exactly
3. If your explanation concludes "The solution is correct", then set "correctness" to "correct"
4. If your explanation concludes "The solution is incorrect", then set "correctness" to "incorrect"

DO NOT include anything else before or after the JSON object. Format your entire answer as a valid JSON object.

## Data

### Problem Statement

$PROBLEM

### Z3 Code

$MODEL

### Solution

$SOLUTION

