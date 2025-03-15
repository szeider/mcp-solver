# MCP Solver with PySAT - Lite Mode

This is a constraint solving system using PySAT (Python SAT Solver). PySAT provides Python interfaces to several SAT solvers and allows for propositional constraint modeling using CNF (Conjunctive Normal Form).

## Available Tools

In PySAT mode (lite), the following tools are available:

1. **Add Item**: Add Python code to the model
2. **Replace Item**: Replace code in the model
3. **Delete Item**: Delete code from the model
4. **Solve Model**: Solve the current model using PySAT
5. **Get Variable**: Get the value of a variable from the solution
6. **Get Solution**: Get the complete solution
7. **Get Solve Time**: Get the time taken to solve the model

## Using PySAT

PySAT models are written as Python code. Here's a basic workflow:

1. Define your problem using PySAT's CNF formulas
2. Solve the formula
3. Extract and export the solution

## Basic PySAT Example

Here's a basic example of a PySAT model:

```python
# Import PySAT components
from pysat.formula import CNF
from pysat.solvers import Cadical153

# Create a CNF formula
formula = CNF()

# Add clauses: (a OR b) AND (NOT a OR c) AND (NOT b OR NOT c)
formula.append([1, 2])       # Clause 1: a OR b
formula.append([-1, 3])      # Clause 2: NOT a OR c
formula.append([-2, -3])     # Clause 3: NOT b OR NOT c

# Create solver and add the formula
solver = Cadical153()
solver.append_formula(formula)

# Solve the formula
satisfiable = solver.solve()
model = solver.get_model() if satisfiable else None

# Create a mapping of variable names to IDs
variables = {
    "a": 1,
    "b": 2,
    "c": 3
}

# Export the solution
export_solution(solver, variables)

# Free solver memory (important for PySAT)
solver.delete()
```

## Key PySAT Components

### 1. Formulas

- **CNF**: Standard CNF formula
- **WCNF**: Weighted CNF for MaxSAT problems

### 2. Solvers

- **Cadical153**: Recommended for standard SAT problems
- **Glucose3**, **Glucose4**: Alternative solvers
- **RC2**: For MaxSAT optimization problems

### 3. Encoding Cardinality Constraints

PySAT provides efficient encodings for cardinality constraints:

```python
from pysat.card import CardEnc, EncType

# Variables (1 to 5)
variables = [1, 2, 3, 4, 5]

# At most 2 variables can be True
atmost2 = CardEnc.atmost(variables, 2, encoding=EncType.seqcounter)

# Add these constraints to your formula
formula = CNF()
for clause in atmost2.clauses:
    formula.append(clause)
```

### 4. MaxSAT Problems

For optimization problems, use MaxSAT with the RC2 solver:

```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

# Create a WCNF formula
wcnf = WCNF()

# Add hard constraints (must be satisfied)
wcnf.append([1, 2])  # a OR b

# Add soft constraints with weights
wcnf.append([-1], weight=10)  # NOT a (weight 10)
wcnf.append([-2], weight=5)   # NOT b (weight 5)

# Solve with RC2
with RC2(wcnf) as rc2:
    model = rc2.compute()
    cost = rc2.cost

# Export solution with variables and objective
variables = {"a": 1, "b": 2}
export_solution(rc2, variables, cost)
```

## Important Notes

1. **Memory Management**: Always call `solver.delete()` after using a solver to free memory.

2. **Variable IDs**: PySAT uses integer IDs for variables:
   - Positive numbers represent variables
   - Negative numbers represent negated variables

3. **Accessing Results**: Use the exported solution to access results:
   ```python
   export_solution(solver, variables)
   ```

4. **Best Practices**:
   - Use Cadical153 for standard SAT problems (fastest)
   - Use RC2 for MaxSAT optimization problems
   - Use sequential counter encoding (EncType.seqcounter) for cardinality constraints

## Tips for Success

1. **Properly Format Clauses**: Each clause is a list of integer literals.

2. **Variable ID Mapping**: Maintain a clear mapping from meaningful variable names to PySAT's numeric IDs.

3. **Free Resources**: Always delete solvers when finished to prevent memory leaks.

4. **Test Incrementally**: Build and test your model in small steps.

5. **Handle Complexity**: For complex models, break down the problem into smaller parts.

## Exporting Your Solution

Always end your code with:

```python
export_solution(solver, variables, objective)
```

Where:
- `solver`: Your PySAT solver instance
- `variables`: Dictionary mapping variable names to their IDs
- `objective`: Optional objective value for optimization problems

## Troubleshooting

- If the solver fails: Check your constraints for contradictions
- If you get a memory error: Make sure you're calling `solver.delete()`
- If variables are missing from the solution: Check your variable mapping dictionary 