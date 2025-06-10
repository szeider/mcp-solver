# MaxSAT Solution Review Template

Your task is to review a MaxSAT solution for basic correctness and proper formatting.

## Problem Statement

${PROBLEM}

## Model Implementation

${MODEL}

## Solution Results

${SOLUTION}

## Review Guidelines

⚠️ **IMPORTANT**: Do NOT try to verify that constraint encodings are correct. However, DO verify that the solution satisfies the hard constraints stated in the problem.

Focus on these checks:

1. **Structural Correctness**
   - Is WCNF (not CNF) used for the MaxSAT problem?
   - Are hard constraints added WITHOUT weights? (just `wcnf.append([literals])`)
   - Are soft constraints added WITH weights? (`wcnf.append([literals], weight=value)`)
   - Is the RC2 solver used and called correctly?
   - Is export_solution() called with results?

2. **Hard Constraint Satisfaction**
   - Based on the solution values, check if obvious hard constraints are satisfied
   - For example: If problem says "at most 2 items per slot", verify no slot has 3+ items
   - For example: If problem says "A requires B", verify that if A is selected, B is also selected
   - Focus on constraints you can easily verify from the problem statement
   - If you're unsure about complex constraint encodings, skip them

3. **Solution Format & Completeness**
   - Does the solution include `"satisfiable": True/False`?
   - If satisfiable, does it include the cost from solver.cost?
   - Are the required outputs from the problem statement included?
   - Do the variable counts match the problem (e.g., if 5 items mentioned, are 5 items in solution)?
   - Does the solution format match what the problem asks for?

4. **For Unsatisfiable Solutions**
   - **IMPORTANT**: You do NOT need to explain WHY the problem is unsatisfiable
   - Simply verify that all hard constraints in the model are grounded in the problem description
   - Check that each hard constraint represents a requirement from the problem statement
   - Note that "unsatisfiable" is a perfectly valid result - if all constraints are justified by the problem, mark it as correct
   - Trust the solver's determination of unsatisfiability

5. **Basic Sanity Checks**
   - Are positive integer variables used (1, 2, 3...)?
   - Is the cost a non-negative number for satisfiable solutions?
   - Are the reported values consistent (e.g., if cost=10, which soft constraints were violated?)

6. **Optimality Verification**
   - If solver reports "optimal" with cost > 0, verify if this makes sense:
     - Does the solution satisfy all explicit requirements in the problem?
     - For "select at least k items" problems, check if k items were actually selected
     - Note: "Solution optimal with cost X" doesn't mean perfect - it means best possible given the encoding

7. **DO NOT Check**
   - ❌ Whether the constraint encoding logic is correct (e.g., don't verify if `[-a, -b, c]` correctly encodes "if a and b then c")
   - ❌ Whether the solution is optimal (trust the solver)
   - ❌ Whether there could be a "better" solution
   - ❌ Complex Boolean formula transformations
   - ❌ Whether soft constraint polarities match your interpretation

## Final Verdict

Based on your review, provide a final verdict with one of the following:

<verdict>correct</verdict> - If the solution follows proper MaxSAT structure and formats results correctly
<verdict>incorrect</verdict> - If there are structural issues (wrong solver, missing weights, bad format, etc.)
<verdict>unknown</verdict> - If you cannot determine correctness based on the information provided

**Remember**: 
- If the solver found a solution (satisfiable=True), the constraint logic is almost certainly correct
- Focus on structure and format, not mathematical correctness
- A solution can be "correct" even if you don't understand the constraint encodings