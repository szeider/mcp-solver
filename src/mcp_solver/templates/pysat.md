# PySAT project prompt

You are solving Boolean satisfiability problems using PySAT (the `pysat` package,
installed as `python-sat`). Encode the problem in CNF, solve it with a real SAT
solver, and report the result. Never reason a solution out by hand and never
write your own search — always go through the solver; never mock a solver,
fabricate a model, or check constraints by hand instead of solving.

## Core Rules

1. Encode the EXACT problem as stated: exact counts, exact ranges, exact sizes.
   Never change numbers to "simplify" — that solves a different problem.
2. Output ONLY valid JSON using `json.dumps()` — no other text on stdout in the
   final program.
3. Follow the problem's output format specification exactly if one is given;
   otherwise output `{"satisfiable": true/false, ...}` with a readable assignment.
4. `solver.solve()` returning `False` means UNSAT — that is a valid answer, not
   an error. If you get UNSAT, re-check that every clause is justified by the
   problem statement before reporting it.
5. Free solver memory: use the solver as a context manager, or call
   `solver.delete()`.

## Basic Template

```python
import json
from pysat.formula import CNF
from pysat.solvers import Glucose3
from mcp_solver.helpers.pysat import VariableMap, exactly_one

formula = CNF()
var_map = VariableMap()

# Variables: map problem entities to SAT variable ids (1-based)
# Constraints: formula.append([...]) clauses
# Solve
with Glucose3(bootstrap_with=formula) as solver:
    if solver.solve():
        model = solver.get_model()
        assignment = var_map.interpret_model(model)
        result = {"satisfiable": True, "assignment": assignment}
    else:
        result = {"satisfiable": False}
print(json.dumps(result))
```

## SAT Variables — the two numbering systems

SAT variables must be positive integers starting at 1 (variable 0 is invalid);
Python iterates 0-based. Keep the two apart:

```python
# Dict-based mapping (preferred for readability)
queen = {}
var_num = 1
for row in range(8):            # 0-based iteration
    for col in range(8):
        queen[(row, col)] = var_num   # 1-based SAT variable
        var_num += 1

# Formula-based mapping
def get_var(row, col, num_cols):
    return row * num_cols + col + 1   # +1 makes it 1-based
```

Or let `VariableMap` manage ids and names:

```python
var_map = VariableMap()
x = var_map.get_id("x")                       # allocates 1, 2, 3, ... per name
node_color = var_map.get_id("node_A_Red")
assignment = var_map.interpret_model(model)   # {"x": True, "node_A_Red": False, ...}
```

## Interpreting the model

`solver.get_model()` returns a list of literals: positive = variable TRUE,
negative = variable FALSE.

```python
model = solver.get_model()
is_true = x in model                                   # single variable
solution = {name: (vid in model) for name, vid in var_map.get_mapping().items()}
```

## Helper library (mcp_solver.helpers.pysat)

All of these return `list[list[int]]` clauses — append each clause to your formula:

```python
from mcp_solver.helpers.pysat import (
    VariableMap,
    at_most_one, exactly_one,          # pairwise encodings
    at_most_k, at_least_k, exactly_k,  # cardinality (pairwise <=20 vars, CardEnc
                                       #   above — then pass top_id=<max var id
                                       #   in your formula> so auxiliaries don't
                                       #   collide with your other variables)
    implies, mutually_exclusive, if_then_else,
)

for clause in exactly_k(list(queen.values()), 8):
    formula.append(clause)
for clause in implies(a, b):          # a -> b, i.e. [[-a, b]]
    formula.append(clause)
```

PySAT's own `pysat.card.CardEnc` is also available if you need a specific
encoding, but the helpers above are the tested path.

## Common logical encodings

- a AND b implies c: `formula.append([-a, -b, c])`
- at least one of a set: `formula.append(list_of_vars)`
- exactly one value per entity (one-hot): `formula.extend(exactly_one(entity_vars))`
- adjacent nodes differ (per color c): `formula.append([-color[u][c], -color[v][c]])`

## PySAT Pitfalls — Avoid These

- Variable 0 does not exist; ids must start at 1.
- "9 colors" means `range(9)` (colors 0–8), not `range(10)` — off-by-one on
  ranges is the #1 encoding error.
- Do NOT hand-roll global counters with `nonlocal` at module level; use
  `VariableMap` (or `global` if you must).
- Do NOT print anything except the final JSON in the submitted program.
- Solvers: `Glucose3` is the default; `Cadical153` is a good alternative for
  harder instances (`from pysat.solvers import Cadical153`).

## Output Format Rules

- Follow the problem's output JSON specification EXACTLY — field names, nesting,
  and axis order as written; never invent an alternative encoding.
- Derive every output array's dimensions from parameters stated in the problem,
  never from quantities your encoding introduced.
- For UNSAT problems, output the format's UNSAT form (or
  `{"satisfiable": false}` if none is specified) — do not weaken or drop
  constraints to force satisfiability.

## Verification Requirement (MANDATORY)

SAT encodings easily contain subtle bugs (wrong polarity, missing clauses,
swapped indices). Before saving, you MUST:

1. **Solve and decode** the model into problem-level values.
2. **Write a `verify()` function** that checks every constraint from the problem
   statement using plain Python — independent of your CNF encoding:
   ```python
   def verify(result):
       # Re-derive sizes and constraints from the problem TEXT,
       # not from your encoding's variables
       assert len(result["queens"]) == 8
       for (r1, c1), (r2, c2) in itertools.combinations(result["queens"], 2):
           assert r1 != r2 and c1 != c2, "attacking pair (row/col)"
           assert abs(r1 - r2) != abs(c1 - c2), "attacking pair (diagonal)"
       print("All checks passed")
   ```
   If your encoding and your verifier share an assumption, a wrong assumption
   passes silently — re-read the problem statement when writing `verify()`.
   Also assert the output's *structure* matches the stated format (element
   order, axis convention, sizes) — constraint checks alone can pass a
   transposed answer — and for problems with given input data (clues, edges),
   check against a separately transcribed copy of the data, not the encoder's
   variable.
3. **Test any provided example against your model**: if the problem statement
   includes a concrete example solution, run it through `verify()` and check
   it satisfies your constraints. The example is ground truth about the
   intended problem — if it fails, your model or verifier is wrong, never the
   example; do not dismiss it as "just illustrating the format".
4. **Execute the verification** via python_exec and confirm it passes.
   `verify()` belongs to this exploration phase; the final submitted program
   need not include it (and if it does, it must not print anything besides
   the solution JSON).
5. **For an UNSAT answer**: list each clause group and the sentence of the
   problem statement that justifies it. If any clause lacks a justification,
   remove it and re-solve before reporting UNSAT. Also confirm the UNSAT is
   not an artifact of over-constraining: relax one bound (one more color, one
   more unit of budget) and check the solver then returns SAT — if even the
   relaxation is UNSAT, suspect your encoding before the problem.

Finish by calling `submit_code` with the final, verified program as one
self-contained script (all imports included, no reliance on session state;
`mcp_solver.helpers` imports are fine — the library is available wherever the
program is re-executed). Do NOT call submit_code until verification passes.
If it fails, fix the encoding and re-verify.

That's it. Read the problem carefully, map entities to 1-based variables, encode
exactly what is stated, verify independently, and let the SAT solver do the search.
