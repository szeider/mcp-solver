# Review Checklist for MaxSAT Optimization Models

When working with MaxSAT optimization models, use this review checklist to ensure correctness, efficiency, and interpretability.

## Solution Verification

- [ ] All hard constraints are correctly encoded and satisfied in the solution
- [ ] Soft constraints are correctly weighted according to their importance
- [ ] Solution cost matches the sum of weights of violated soft constraints
- [ ] Optimization objective is correctly calculated
- [ ] Variable assignments make sense in the problem context

## Encoding Correctness

- [ ] Using WCNF (not CNF) for MaxSAT optimization problems
- [ ] Hard constraints have no weight specified
- [ ] Soft constraints have appropriate positive weights
- [ ] RC2 solver is used for MaxSAT optimization
- [ ] Variables are correctly mapped to meaningful names
- [ ] All problem constraints are properly encoded as clauses

## Solution Export

- [ ] Using `export_maxsat_solution()` to return results
- [ ] Solution includes satisfiability status ("satisfiable"/"unsatisfiable")
- [ ] Solution includes optimization information (cost/objective)
- [ ] Problem-specific dictionaries are included (e.g., "selected_items")
- [ ] Variable assignments are correctly extracted from the model

## Performance Considerations

- [ ] Number of variables and clauses is reasonable for the problem size
- [ ] Constraints are simplified where possible
- [ ] Solver timeout is appropriate for the problem complexity
- [ ] Consider alternative encodings for large cardinality constraints

## Common Errors to Check

- [ ] All variables are properly declared and used
- [ ] No dictionary assignment errors (use `dict[key] = value` pattern)
- [ ] No misplaced hard/soft constraints
- [ ] No weight scale inconsistencies (weights should be comparable)
- [ ] All dependencies between variables are correctly encoded
- [ ] No variables with ID 0 (must start from 1)

## Optimization Analysis

- [ ] The objective function accurately represents the optimization goal
- [ ] Weights reflect the relative importance of different constraints
- [ ] Solution optimality can be verified independently
- [ ] Trade-offs between conflicting objectives are appropriate
- [ ] Corner cases are handled correctly

## Code Structure and Readability

- [ ] Variable naming is clear and consistent
- [ ] Code is well-structured and modular
- [ ] Comments explain the encoding strategy
- [ ] Problem formulation is clear and matches requirements
- [ ] Solution interpretation is provided

For more details on MaxSAT optimization techniques, see the PySAT documentation.