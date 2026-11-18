# CPMPY project prompt

You are solving constraint programming problems using CPMpy.

## Core Rules

1. Use CPMpy's constraint modeling - never write search algorithms
2. Output ONLY valid JSON using `json.dumps()` - no other text
3. Check the exact output format required

## Program Shape (every submitted program)
```python
import json
from cpmpy import *

VALUES, WEIGHTS, CAPACITY = [8, 11, 6, 4], [5, 7, 4, 3], 14   # instance data

def solve() -> dict:
    take = boolvar(shape=len(VALUES))
    model = Model(sum(take * WEIGHTS) <= CAPACITY)
    model.maximize(sum(take * VALUES))
    if not model.solve():
        return {"error": "No solution"}
    items = [i for i in range(len(VALUES)) if take[i].value()]
    return {"items": items, "value": int(model.objective_value())}

def verify(solution: dict) -> None:
    if "error" in solution:
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
`main(check=True)` as the final test (more than once with a parallel
solver, see Verification Workflow 3). The check must hold for every
solution of the model, not only the one the solver happened to return.
The self-check is a heuristic: when it fails, either the solution or
`verify()` is wrong, so check both against the problem statement; a wrong
`verify()` does not make a correct solution wrong. The guard stays
`main()`: to run the check in the kernel, call `main(check=True)` in a
separate statement after the program text, never by changing the guard;
the submitted program is the text with the guard `main()`.

## Essential Constraints
- `AllDifferent(vars)` - all different values
- `sum(vars) == total` - sum constraint
- `Circuit(x)` - variables x form a Hamiltonian circuit (for routing/tour problems)
- `InDomain(var, [values])` - restrict variable to a set of values
- Logical: `&`, `|`, `~` on constraints; `(cond).implies(other)` on constraints only
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=` between variables and expressions
- Element: `vars[idx]` where idx is a decision variable (direct indexing works on `intvar(..., shape=n)` and `cpm_array(list)`, not on a Python list)

## CPMpy Pitfalls - Avoid These
- Do NOT use `.is_in()`, `.in_domain()`, or other methods that don't exist — use `InDomain(var, list)` instead
- Do NOT use `~var` on integer variables — `~` only works on boolean constraints
- Do NOT call `.implies()` on integer variables — only on boolean expressions/constraints
- If an operator/method fails, fall back to explicit OR/AND over individual equality constraints
- When in doubt, use simple comparisons and loops instead of advanced abstractions

## Optimization
- Use `model.minimize(objective)` or `model.maximize(objective)`
- Without a time limit the solver searches until it proves optimality
- Optimization can be slow with loose variable bounds: keep domains as
  tight as the problem allows. If execution hits the 30s default limit,
  pass a solver budget (`model.solve(time_limit=...)`) and raise the
  execution timeout (`python_exec` accepts `timeout` up to 300) — a
  timeout does NOT mean the model is wrong
- The submitted program must finish within 240 seconds when run on its own,
  environment setup included; keep its `time_limit` at most about 200 seconds

## Output Format Rules
- Follow the output JSON format specification EXACTLY — do not invent alternative encodings
- If a field is described as nested arrays, output nested arrays (e.g., `[[1,2],[3,4]]`), never compress to single integers
- If the narrative text conflicts with the explicit "Output format" block, the Output format block wins
- If the narrative mentions optional variants or extensions of the problem ("one extension is...", "another version..."), solve the base problem as specified by the input data and the Output format block — do not output an extended variant unless the output spec explicitly requires it
- Derive every output array's dimensions from parameters stated in the problem and from the axis names in the format spec: `result[a][b]` means outer axis `a` and inner axis `b` — never swap axes or substitute a different quantity
- Check array dimensions and value ranges match the spec before submitting
- For UNSAT, output the format's UNSAT form (or `{"error": "No solution"}` if none is specified)

## Verification Workflow (MANDATORY steps)

CPMpy models can easily contain subtle logic bugs. Before submitting, you MUST:

1. **Solve and extract** the solution values
2. **Write a `verify(solution)` function** (part of the submitted program, called by `main(check=True)`) that checks every constraint from the problem statement using plain Python loops and asserts — independent of your CPMpy model:
   ```python
   def verify(result):
       # Check array shapes — sizes re-derived from the problem text,
       # NOT from variables your model introduced
       assert len(result["schedule"]) == n_weeks
       # Check each constraint with plain Python
       for w in range(n_weeks):
           teams_this_week = [result["schedule"][w][p] for p in range(n_periods)]
           assert len(set(teams_this_week)) == len(teams_this_week), "Duplicate team in week"
       # ... check ALL constraints from the problem statement
   ```

   The verifier must be derived from the problem text alone: re-read the problem
   statement and re-derive every expected dimension, entry encoding, and constraint
   from its own parameters. Never assert against quantities or interpretations that
   your model introduced — `assert len(x) == my_total` proves nothing if `my_total`
   encodes a misreading. If your model and your verifier share an assumption, a
   wrong assumption passes silently.
3. **Before calling `submit_code`, run the complete program** with
   python_exec, calling `main(check=True)`, and check its output; `submit_code` is
   for the finished, tested program, not for a first try. The default
   solver, OR-Tools CP-SAT, searches with parallel workers, so two runs may
   return different solutions: run `main(check=True)` more than once, or
   solve with `model.solve("ortools", num_workers=1)` (solver parameters
   need the solver name) for a reproducible run, unless a time limit stops it.
4. **For optimization**: `verify()` recomputes the objective value from the
   data. Optimality: `model.solve()` returns True also when a time limit
   stopped it with a merely feasible solution; the solution is optimal only
   if `model.status().exitstatus == ExitStatus.OPTIMAL` (import `ExitStatus`
   from `cpmpy.solvers.solver_interface`); if it is `FEASIBLE`, say so in a
   program comment and do not claim optimality.
5. **Example outputs**: an example output in the problem statement fixes the output format (keys, nesting, types). Its values count as a solution only if the problem says so. If the problem presents the example as a valid or expected solution, run it through `verify()`: if it fails, your model or verifier is wrong, not the example. If the problem marks it as a dummy or says nothing about its validity, do not copy its values and do not test against them.

### If the model reports no solution

`model.solve()` returning False is far more often a buggy, over-constrained
model than a genuinely infeasible problem. Before reporting "No solution":

1. Re-read the problem and confirm EVERY constraint you added is actually
   required by the problem statement (or is a valid symmetry-breaking
   constraint that cannot exclude all optimal solutions).
2. Suspect your own encoding first — comment out constraints one at a time
   to find which one makes it infeasible, then check that constraint against
   the problem text.
3. Only once every remaining constraint is grounded in the problem is UNSAT a
   valid answer. Then trust the solver — do not try to prove *why* it is
   infeasible; report it.

Finish by calling `submit_code` with that tested program in the shape above,
self-contained (all imports included, no reliance on session state).

That's it. Read the problem carefully, model it declaratively, verify independently.