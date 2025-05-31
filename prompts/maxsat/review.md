# MaxSAT Solution Review Template

Your task is to review a MaxSAT solution for correctness, optimality, and quality.

## Problem Statement

${PROBLEM}

## Model Implementation

${MODEL}

## Solution Results

${SOLUTION}

## Review Guidelines

Carefully review the solution and assess its correctness based on the following criteria:

1. **Encoding Correctness**
   - Is the problem correctly formulated as a MaxSAT problem?
   - Are hard constraints properly encoded (no weights)?
   - Are soft constraints properly encoded with appropriate weights?
   - Is the WCNF formula used correctly?
   - Is the RC2 solver used for optimization?

2. **Solution Correctness**
   - Are all hard constraints satisfied in the solution?
   - Is the solution cost (sum of weights of violated soft constraints) correctly calculated?
   - Is the objective value reasonable for the problem?
   - Are the variable assignments consistent with the problem requirements?

3. **Implementation Quality**
   - Is the code well-structured and readable?
   - Are variables named meaningfully?
   - Is the export_maxsat_solution function used correctly?
   - Are there any potential issues or bugs in the implementation?

4. **Optimality**
   - Is the solution optimal (minimizing cost/maximizing objective)?
   - Could there be a better solution that the solver missed?
   - Are there any violated soft constraints that could be satisfied without violating hard constraints?

## Final Verdict

Based on your review, provide a final verdict with one of the following:

<verdict>correct</verdict> - If the solution is correct, optimal, and well-implemented
<verdict>incorrect</verdict> - If there are significant issues with the solution
<verdict>unknown</verdict> - If you cannot determine correctness based on the information provided

Your review should be detailed, constructive, and focus on the MaxSAT aspects of the solution. Provide specific examples and suggestions for improvement where applicable.