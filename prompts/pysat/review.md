# Solution Correctness Verification

## Task

You are given a problem description, a PySAT encoding, and a solution. Verify the correctness of the solution.

## Evaluation Criteria

- **For satisfiable solutions**: Verify that all constraints in the problem description are satisfied. Answer *correct* if satisfied, otherwise *incorrect*. You do not need to verify optimality, only check if the solution satisfies all hard constraints.
- **For unsatisfiable solutions**: Verify that all clauses produced by the encoding are actually required by the problem statement or are valid symmetry-breaking constraints. Answer *correct* if valid, otherwise *incorrect*.

  **IMPORTANT**: You do NOT need to explain WHY the instance is unsatisfiable. Trust the solver's determination. Your task is only to verify that each constraint in the model is grounded in the problem description.

  Note that "unsatisfiable" is a perfectly fine result. So if all constraints added to the model are valid representations of the problem requirements, then your verdict should be *correct*.

- **For no solution/timeout/unverifiable cases**: Answer *unknown*.

## Output Format

After your detailed analysis, provide your verdict using simple XML tags.

IMPORTANT: Your answer MUST follow this structure:
1. First provide a detailed explanation of your reasoning
2. Analyze each clause in detail
3. End with a clear conclusion statement: "The solution is correct." or "The solution is incorrect."
4. Finally, add exactly ONE of these verdict tags on a new line:
   <verdict>correct</verdict>
   <verdict>incorrect</verdict>
   <verdict>unknown</verdict>

For example:
```
[Your detailed analysis here]

After checking all clauses, I can confirm that each one is satisfied by the provided solution values.

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

### PySAT Encoding

$MODEL

### Solution

$SOLUTION

