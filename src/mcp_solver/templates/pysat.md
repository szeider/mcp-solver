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
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType
from pysat.solvers import Glucose3

formula = CNF()
vpool = IDPool()   # ONE pool per program: named variables AND encoder auxiliaries

# Variables: x = vpool.id(("x", i))  -> ids 1, 2, 3, ... on first use
# Constraints: formula.append([...]) clauses; cardinality via CardEnc (below)
# Solve
with Glucose3(bootstrap_with=formula) as solver:
    if solver.solve():
        model = solver.get_model()
        true_vars = [vpool.obj(l) for l in model if l > 0 and vpool.obj(l) is not None]
        result = {"satisfiable": True, "true_vars": true_vars}
    else:
        result = {"satisfiable": False}
print(json.dumps(result))
```

## SAT Variables — the two numbering systems

SAT variables must be positive integers starting at 1 (variable 0 is invalid);
Python iterates 0-based. Let the pool keep the two apart: it maps any hashable
name (use tuples) to a 1-based id, the same name always to the same id.

```python
from pysat.formula import IDPool

vpool = IDPool()
queen = {}
for row in range(8):            # 0-based iteration
    for col in range(8):
        queen[(row, col)] = vpool.id(("queen", row, col))   # 1-based SAT variable
node_color = vpool.id(("color", "A", "Red"))
```

If you number variables by hand (`row * n + col + 1`), create the pool as
`IDPool(start_from=<largest hand-made id> + 1)` before any encoder call;
otherwise auxiliaries reuse your ids.

## Interpreting the model

`solver.get_model()` returns a list of literals: positive = variable TRUE,
negative = variable FALSE. `vpool.obj(id)` gives the name back, and `None`
for auxiliary ids created by encoders.

```python
model = solver.get_model()
is_true = x in model                                   # single variable
placed = [vpool.obj(l)[1:] for l in model if l > 0 and vpool.obj(l)]  # [(row, col), ...]
```

## Cardinality and pseudo-Boolean constraints

`pysat.card.CardEnc` and `pysat.pb.PBEnc` return an object whose `.clauses`
you add to the formula. ALWAYS pass `vpool=vpool`: the encoder then takes its
auxiliary ids from the pool, and later `vpool.id(...)` calls skip them.
Without it, auxiliaries can reuse ids of other variables.

```python
from pysat.card import CardEnc, EncType
from pysat.pb import PBEnc

cells = list(queen.values())
formula.extend(CardEnc.equals(lits=cells, bound=8, vpool=vpool,
                              encoding=EncType.seqcounter).clauses)
for r in range(8):              # one-hot: exactly one queen per row
    row = [queen[(r, c)] for c in range(8)]
    formula.extend(CardEnc.equals(lits=row, bound=1, vpool=vpool,
                                  encoding=EncType.pairwise).clauses)
# CardEnc.atmost / CardEnc.atleast take the same arguments
# weighted sum: 3*x0 + 2*x1 + 4*x2 <= 5
formula.extend(PBEnc.atmost(lits=cells[:3], weights=[3, 2, 4], bound=5,
                            vpool=vpool).clauses)
```

- Default `EncType.seqcounter`: any bound, about n*k auxiliaries.
- `EncType.pairwise` adds no auxiliaries but supports only bound 1 (fine for
  one-hot groups of up to a few dozen variables). Do not use `native`: it
  needs a Minicard solver and produces no clauses.
- A bound below 0, or above `len(lits)` for `equals`/`atleast`, raises
  `ValueError`: such a constraint is unsatisfiable, so re-read the problem.

## Common logical encodings

- a AND b implies c: `formula.append([-a, -b, c])`
- at least one of a set: `formula.append(list_of_vars)`
- exactly one value per entity (one-hot): `CardEnc.equals(..., bound=1, ...)` as above
- adjacent nodes differ (per color c): `formula.append([-color[u][c], -color[v][c]])`

## PySAT Pitfalls — Avoid These

- Variable 0 does not exist; ids must start at 1.
- "9 colors" means `range(9)` (colors 0–8), not `range(10)` — off-by-one on
  ranges is the #1 encoding error.
- Do NOT hand-roll global counters with `nonlocal` at module level; take
  every id from the one `IDPool`.
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
swapped indices). Before submitting, you MUST:

1. **Solve and decode** the model into problem-level values.
2. **Write a `verify()` function** that checks every constraint from the problem
   statement using plain Python — independent of your CNF encoding:
   ```python
   import itertools

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
`pysat` and the standard library are available wherever the program is
re-executed). Do NOT call submit_code until verification passes.
If it fails, fix the encoding and re-verify.

That's it. Read the problem carefully, map entities to pool variables, encode
exactly what is stated, verify independently, and let the SAT solver do the search.
