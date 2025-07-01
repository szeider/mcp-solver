# ASP (clingo) Solution Review Template

Your task is to review an Answer Set Programming (ASP) solution for structural correctness, adherence to the problem statement, and best practices.

## Problem Statement

${PROBLEM}

## Model Implementation

${MODEL}

## Solution Results

${SOLUTION}

## Review Guidelines

⚠️ **IMPORTANT**: Do NOT try to verify that the logic of every rule is mathematically correct. However, DO verify that the solution satisfies the hard constraints and requirements stated in the problem.

Focus on these checks:

1. **Structural Correctness**
   - Are all facts, rules, and constraints properly terminated with periods?
   - Are variables properly grounded (no unsafe variables)?
   - Are integrity constraints (`:- ... .`) used for hard requirements?
   - Are optimization statements (`#minimize`, `#maximize`) used only if required?
   - Is the model divided into clear sections (facts, domains, rules, constraints, optimization)?

2. **Hard Constraint Satisfaction**
   - Based on the solution, check if all integrity constraints are satisfied.
   - For example: If the problem says "no two adjacent nodes have the same color", verify this in the solution.
   - Focus on constraints you can easily verify from the problem statement and solution.
   - If unsure about complex encodings, skip them.

3. **Solution Format & Completeness**
   - Does the solution include all required outputs from the problem statement?
   - Are all relevant variables/atoms present in the solution?
   - Does the solution format match what the problem asks for?

4. **For Unsatisfiable Solutions**
   - **IMPORTANT**: You do NOT need to explain WHY the problem is unsatisfiable.
   - Simply verify that all integrity constraints in the model are justified by the problem statement.
   - Note that "unsatisfiable" is a perfectly valid result—if all constraints are justified by the problem, mark it as correct.
   - Trust the solver's determination of unsatisfiability.

5. **Basic Sanity Checks**
   - Are all atoms/variables in the solution properly grounded?
   - Are there any obvious contradictions or missing assignments?

6. **Optimality Verification**
   - If the solution claims to be optimal (e.g., with a cost value), check if this makes sense given the problem and constraints.
   - For optimization problems, verify that the reported cost matches the solution's assignments.

7. **DO NOT Check**
   - ❌ Whether the logic of each rule is mathematically correct (trust the encoding for non-obvious cases)
   - ❌ Whether the solution is globally optimal (trust the solver)
   - ❌ Complex Boolean formula transformations
   - ❌ Whether soft constraint polarities match your interpretation

## Output Format

After your detailed analysis, provide your verdict using simple XML tags.

**Your answer MUST follow this structure:**
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

The verdict must be EXACTLY one of: "correct", "incorrect", or "unknown" nothing else.

**Before finalizing your response, always check that:**
1. Your explanation ends with a clear conclusion statement
2. The verdict tag matches your conclusion exactly
3. If your explanation concludes "The solution is correct", then use <verdict>correct</verdict>
4. If your explanation concludes "The solution is incorrect", then use <verdict>incorrect</verdict>
5. If you cannot determine correctness or establish incorrectness, use <verdict>unknown</verdict>

## Data

### Problem Statement

$PROBLEM

### ASP Model

$MODEL

### Solution

$SOLUTION
