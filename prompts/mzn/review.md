# Solution Correctness Verification

## Task

You are given a problem description, a MiniZinc model, and a solution. Verify the correctness of the solution.

## Verification Process

Follow these steps carefully:

1. Create a structured table of the solution values:
   - For each decision variable, record its value
   - For array variables, list each index and its corresponding value
   - Present this table at the beginning of your explanation

2. For each constraint:
   - State the constraint precisely
   - Extract the relevant variable values from your table
   - Show your evaluation process step-by-step
   - Conclude whether the constraint is satisfied or violated

3. If you identify a violation:
   - Double-check by re-extracting the exact values from the solution
   - Explicitly show how these values violate the constraint
   - Include the relevant indices and their values to avoid indexing errors

4. Before finalizing your verdict:
   - Re-verify any reported violations
   - Ensure you haven't misread or misinterpreted array indices or values

## Evaluation Criteria

- **For satisfiable solutions**: Verify that all constraints in the problem description are satisfied. Answer *correct* if satisfied, otherwise *incorrect*. You do not need to verify optimality - only check if the solution satisfies all constraints.

  Present your constraint verification in a structured format:
  - Constraint: [State the constraint]
  - Values: [List relevant variable values]
  - Evaluation: [Show calculation or reasoning]
  - Result: [Satisfied/Violated]

- **For unsatisfiable solutions**: Verify that all constraints in the MiniZinc model are actually required by the problem statement or are valid symmetry breaking constraints. Answer *correct* if valid, otherwise *incorrect*.

  Check constraint by constraint using the same structured format.

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

### MiniZinc Model

$MODEL

### Solution

$SOLUTION
