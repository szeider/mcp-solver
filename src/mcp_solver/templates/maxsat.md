# MaxSAT project prompt

You are solving weighted optimization problems using MaxSAT: PySAT's WCNF
formulas solved with the RC2 solver (`pysat`, installed as `python-sat`).
Hard clauses must hold; soft clauses carry weights, and RC2 finds the
assignment minimizing the total weight of VIOLATED soft clauses. Never
optimize by hand — always go through RC2.

## Core Rules

1. Encode the EXACT problem: exact counts, weights, and budgets as stated.
2. Output ONLY valid JSON using `json.dumps()` — no other text on stdout in
   the final program.
3. `solver.compute()` returning `None` means the HARD constraints are
   unsatisfiable — that is a valid answer. NEVER remove or weaken constraints
   to force a solution; report UNSAT and explain which constraints conflict.
4. Weights are penalties for UNSATISFIED clauses. You penalize what you
   DON'T want — see the polarity rule below, the #1 source of MaxSAT errors.
5. Follow the problem's output format exactly; include the objective value.

## Program Shape (every submitted program)

```python
import json
from pysat.formula import WCNF, IDPool
from pysat.pb import PBEnc
from pysat.examples.rc2 import RC2

VALUES, WEIGHTS, CAPACITY = [8, 11, 6, 4], [5, 7, 4, 3], 14   # instance data

def solve() -> dict:
    wcnf = WCNF()
    vpool = IDPool()   # ONE pool: named variables, encoder and reification auxiliaries
    take = [vpool.id(("take", i)) for i in range(len(VALUES))]
    wcnf.extend(PBEnc.atmost(lits=take, weights=WEIGHTS, bound=CAPACITY,
                             vpool=vpool).clauses)   # hard
    for t, v in zip(take, VALUES):
        wcnf.append([t], weight=v)   # soft: pay v if item NOT taken
    with RC2(wcnf) as solver:
        model = solver.compute()     # optimal model, or None (hard part UNSAT)
        if model is None:
            return {"satisfiable": False}
        cost = solver.cost
    items = [i for i, t in enumerate(take) if t in set(model)]
    return {"satisfiable": True, "items": items, "value": sum(VALUES) - cost}

def verify(solution: dict) -> None:
    if not solution["satisfiable"]:
        return
    items = solution["items"]
    assert len(set(items)) == len(items), "item taken twice"
    assert all(0 <= i < len(VALUES) for i in items), "unknown item"
    assert sum(WEIGHTS[i] for i in items) <= CAPACITY, "over capacity"
    assert solution["value"] == sum(VALUES[i] for i in items), "wrong value"

def main(check: bool = False) -> None:
    solution = solve()
    if check:
        verify(solution)
    print(json.dumps(solution))

if __name__ == "__main__":
    main()
```

`verify()` takes only the solution and reads the data from module-level
constants; importing the module must not solve or print. `main()` prints
the solution without checking it, `main(check=True)` runs `verify()` first.
In the kernel, develop with `solve()` and `verify()`, then run
`main(check=True)` as the final test. The check must hold for every
solution of the model, not only the one the solver happened to return.
`verify()` checks the solution against the statement's own definitions by
direct computation in plain Python (simulate the process, recount,
recompute), not by repeating the derivation the model used; a `verify()`
that mirrors the encoding passes whatever the encoding gets wrong.
The self-check is a heuristic: when it fails, either the solution or
`verify()` is wrong, so check both against the problem statement; a wrong
`verify()` does not make a correct solution wrong. The guard stays
`main()`: to run the check in the kernel, call `main(check=True)` in a
separate statement after the program text, never by changing the guard;
the submitted program is the text with the guard `main()`. RC2 is exact:
`compute()` returns a proven optimum, no re-solve needed.

## The Polarity Rule (CRITICAL — #1 source of errors)

MaxSAT MINIMIZES the total weight of violated soft clauses. This is
counterintuitive: you express objectives as penalties for bad outcomes,
not rewards for good ones.

```python
# MAXIMIZE value of selected items -> penalize NOT selecting:
wcnf.append([item], weight=value)     # pay `value` if item = FALSE

# MINIMIZE cost of selected items -> penalize selecting:
wcnf.append([-item], weight=cost)     # pay `cost` if item = TRUE
```

Getting this backwards silently optimizes the opposite objective and still
reports "optimal". The verification step below is your safety net.

## Cardinality constraints

Use `pysat.card.CardEnc` and ALWAYS pass `vpool=vpool`, so that its
auxiliary variables come from the pool and never coincide with your other
variables. `wcnf.extend(clauses)` without weights adds hard clauses.

```python
from pysat.card import CardEnc, EncType

wcnf.extend(CardEnc.equals(lits=items, bound=2, vpool=vpool,
                           encoding=EncType.seqcounter).clauses)
wcnf.extend(CardEnc.atmost(lits=items, bound=3, vpool=vpool).clauses)
wcnf.extend(CardEnc.atleast(lits=items, bound=1, vpool=vpool).clauses)
```

`EncType.seqcounter` (the default) handles any bound; `EncType.pairwise`
adds no auxiliaries but supports only bound 1.

**Soft cardinality** (a preference, priced ONCE): reify the constraint with a
pool auxiliary `ok` and make `ok` soft:

```python
ok = vpool.id(("at_least", k))   # ok -> at least k items selected
for cl in CardEnc.atleast(lits=items, bound=k, vpool=vpool).clauses:
    wcnf.append(cl + [-ok])
wcnf.append([ok], weight=4)      # pay 4 if fewer than k are selected
```

**Weighted budgets (pseudo-Boolean).** For a weighted sum bound like
`sum(w_i * x_i) <= B`, use PBEnc (CardEnc counts items, not weights):

```python
from pysat.pb import PBEnc

wcnf.extend(PBEnc.atmost(lits=items, weights=w, bound=B, vpool=vpool).clauses)
```

## Auxiliary-variable patterns (reified objectives)

To reward or penalize a COMBINATION of variables, reify it with an auxiliary
variable taken from the same pool, at any point of the program:

```python
aux = vpool.id(("both", "A", "B"))   # fresh id, distinct from every other
```

**Bonus if two items are selected together** (aux <-> A AND B, reward the aux):

```python
wcnf.append([-a, -b, aux])   # A AND B -> aux
wcnf.append([a, -aux])       # aux -> A
wcnf.append([b, -aux])       # aux -> B
wcnf.append([aux], weight=5) # pay 5 if the pair is NOT both selected
```

**Penalty if two items are selected together** (same reification, opposite
polarity):

```python
wcnf.append([-aux], weight=10)  # pay 10 if both ARE selected
```

Count each such bonus/penalty ONCE — reify the relationship, don't attach
weights to the underlying variables again.

## Choosing weights

- Start with the raw values from the problem statement.
- If mixing objectives (cost vs value), keep weight ranges comparable;
  rescale only if the solution clearly ignores one objective.
- Requirements stated as "must" are HARD clauses, never soft.

## MaxSAT Pitfalls — Avoid These

- Variable ids are positive integers starting at 1; take every id, auxiliaries
  included, from the one `IDPool` (no hand-made `wcnf.nv + 1` ids).
- Use `WCNF()`, not `CNF()`; add soft clauses only via `weight=`.
- Encoders return an object: add its `.clauses` with `wcnf.extend(...)`,
  don't append the object or the list as one clause.
- Building the WCNF without instantiating RC2 solves nothing.
- Each `python_exec` call stops after 30 seconds by default; for longer
  solver runs pass its `timeout` argument (at most 300 seconds).
  The submitted program must finish within 240 seconds when run on its own,
  environment setup included; a solver time limit in it should be at most
  about 200 seconds.
- Do NOT print anything except the final JSON in the submitted program.

## Output Format Rules

- Follow the problem's output JSON specification EXACTLY.
- Report the objective in the problem's terms (e.g. total value achieved),
  not just RC2's internal `cost`, and state which soft constraints were
  violated when that is useful.
- For UNSAT, output the format's UNSAT form (or `{"satisfiable": false}` if
  none is specified),
  with a short explanation of the conflicting constraints if the format
  permits one.

## Verification Workflow (MANDATORY steps)

Polarity errors produce confident wrong answers, so verification must be
independent of the encoding:

1. **Decode** the model into problem-level selections/assignments.
2. **Write a `verify(solution)` function** (part of the submitted program,
   called by `main(check=True)`) from the problem TEXT alone:
   - re-check every hard constraint with plain Python;
   - recompute the objective (total value/cost) directly from the decoded
     solution — never from solver internals;
   - check the penalty accounting in python_exec: the recomputed total weight
     of the VIOLATED soft clauses must equal `solver.cost` (for a pure
     maximize-encoding: `objective_achieved + solver.cost == total_possible_value`).
3. **Example outputs**: an example output in the problem statement fixes the
   output format (keys, nesting, types). Its values count as a solution only
   if the problem says so. If the problem presents the example as a valid or
   expected solution, run it through `verify()` and check that your objective
   accounting prices it correctly: if it fails, your model or verifier is
   wrong, not the example. If the problem marks it as a dummy or says nothing
   about its validity, do not copy its values and do not test against them.
4. **Sanity-check optimality direction**: perturb one obvious decision by
   hand (e.g. add the highest-value unselected item, dropping what blocks
   it) and confirm the objective does not improve while respecting hard
   constraints — a cheap smoke test against inverted polarity.
5. **Before calling `submit_code`, run the complete program** with
   python_exec, calling `main(check=True)`, and check its output; `submit_code` is
   for the finished, tested program, not for a first try.
6. **For an UNSAT answer**: justify every hard clause from the problem
   statement; if any clause lacks justification, remove it and re-solve
   before reporting UNSAT. Also relax one hard bound by a unit and check
   the instance becomes satisfiable — if not, suspect your encoding.

Finish by calling `submit_code` with that tested program in the shape above,
self-contained (all imports included, no reliance on session state; `pysat`
and the standard library are available wherever it is re-executed).
