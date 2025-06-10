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

- **For satisfiable solutions**: Verify that all constraints in the problem description are satisfied. Answer *correct* if satisfied, otherwise *incorrect*. You do not need to verify optimality, only check if the solution satisfies all hard constraints.

  Present your constraint verification in a structured format:
  - Constraint: [State the constraint]
  - Values: [List relevant variable values]
  - Evaluation: [Show calculation or reasoning]
  - Result: [Satisfied/Violated]

- **For unsatisfiable solutions**: Verify that all constraints in the MiniZinc model are actually required by the problem statement or are valid symmetry breaking constraints. Answer *correct* if valid, otherwise *incorrect*.

  Check constraint by constraint using the same structured format.

  **IMPORTANT**: You do NOT need to explain WHY the instance is unsatisfiable. Trust the solver's determination. Your task is only to verify that each constraint in the model is grounded in the problem description.

  Note that "unsatisfiable" is a perfectly fine result. So if all constraints added to the model are valid representations of the problem requirements, then your verdict should be *correct*.

- **For no solution/timeout/unverifiable cases**: Answer *unknown*.

## Output Format

After your detailed analysis, provide your verdict using simple XML tags.

IMPORTANT: Your answer MUST follow this structure:
1. First provide a detailed explanation of your reasoning
2. Analyze each constraint in detail
3. End with a clear conclusion statement: "The solution is correct." or "The solution is incorrect."
4. Finally, add exactly ONE of these verdict tags on a new line:
   <verdict>correct</verdict>
   <verdict>incorrect</verdict>
   <verdict>unknown</verdict>

For example:
```
[Your detailed analysis here]

After checking all constraints, I can confirm that each one is satisfied by the provided solution values.

The solution is correct.

<verdict>correct</verdict>
```

The verdict must be EXACTLY one of: "correct", "incorrect", or "unknown" - nothing else.

IMPORTANT: Before finalizing your response, always check that:
1. Your explanation ends with a clear conclusion statement
2. The verdict tag matches your conclusion exactly 
3. If your explanation concludes "The solution is correct", then use <verdict>correct</verdict>
4. If your explanation concludes "The solution is incorrect", then use <verdict>incorrect</verdict>
5. If you cannot determine correctness or establish incorrectness, use <verdict>unknown</verdict>

## Data

### Problem Statement

$PROBLEM

### MiniZinc Model

$MODEL

### Solution

$SOLUTION
