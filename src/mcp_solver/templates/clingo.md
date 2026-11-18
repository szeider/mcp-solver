# Answer Set Programming (ASP) Task - v10

## Section 1: Mission Briefing & Core Requirements

### 1.1 Primary Goal
Produce a solution using Answer Set Programming (ASP) that correctly models and solves the problem using the clingo Python API.

### 1.2 Non-Negotiable Requirements
1. **Tool**: You MUST use the clingo Python API for modeling and solving
2. **Execution Time**: Each `python_exec` call stops after 30 seconds by default; for longer solver runs pass its `timeout` argument (at most 300 seconds). The submitted program must finish within 240 seconds when run on its own, environment setup included; a solver time limit in it should be at most about 200 seconds.
   - **Stated bounds**: If the problem states a bound or an expected optimal value (e.g., "expected minimum cost: 50"), add it as a hard constraint and keep `#minimize`/`#maximize`: the constraint prunes the search, the objective still proves the optimum.
   - **Example**: `:- total_cost(Cost), Cost > 50.` next to `#minimize { Cost : total_cost(Cost) }`. If the optimization does not finish within the time limit, solve without the objective and submit the solution that meets the stated value, with a program comment that optimality rests on the problem statement, not on a proof.
3. **Output Format**: Print the final solution as a single JSON object to stdout

### 1.3 Output Specifications
- Generate ONLY executable Python code using clingo
- Use clear, meaningful predicate names (e.g., `assigned(task1, worker2)` not `p(1,2)`)
- Extract solutions from answer sets and format as JSON
- For UNSAT, output the format's UNSAT form (or `{"error": "No solution exists"}` if none is specified)

### 1.4 Final Checklist
Before completing your solution, verify:
☐ 1. **Uses clingo API** - Solution uses `clingo.Control()` and ASP rules
☐ 2. **Model solves correctly** - At least one answer set is found (or UNSAT handled)
☐ 3. **Optimization applied** - #minimize/#maximize for the objective; a bound stated in the problem is added as a hard constraint as well (see the requirement above)
☐ 4. **Solution extracted** - Answer set atoms properly extracted and formatted
☐ 4b. **No Model escapes on_model** - All solution data extracted to plain Python objects inside the callback (touching a Model after solve() returns segfaults)
☐ 5. **Output format correct** - JSON output with appropriate structure
☐ 6. **Predicates meaningful** - Clear names that reflect the problem domain
☐ 7. **Constraints verified independently** - All problem constraints are encoded AND re-checked by a plain-Python verify(solution) in the submitted program, called before printing, whose expectations are re-derived from the problem text, not from your encoding's own interpretation
☐ 8. **Program tested, then submitted** - Before calling `submit_code`, run the complete program (solve, verify, print) with python_exec and check its output; `submit_code` is for the finished, tested program (self-contained, shape of 3.1), not for a first try

## Section 2: Critical Rules of Engagement

### 2.1 MANDATORY RULE: Core ASP Syntax

**FUNDAMENTAL RULE**: You must follow these basic syntax rules. Errors here are the most common cause of failures.

#### A. Constants vs. Variables
This is the single most important distinction in ASP syntax.

* **Constants (Symbols)**: Must start with a **lowercase letter**, be a **number**, or be enclosed in **double quotes `""`**.
  * `item(apple).` (lowercase)
  * `cost(100).` (number)
  * `amino_acid(1, "H").` (quoted string - **use this for input data that is uppercase**)

* **Variables**: Must start with an **uppercase letter** or an **underscore `_`**.
  * `has_color(Item, Color) :- ...`
  * `cost(X, C) :- ...`

**Common Mistake and Correction:**

| Incorrect (produces "unsafe variable" error) | Correct |
|:---|:---|
| `amino_acid(1, H).` | `amino_acid(1, "H").` **(Best Practice)** or `amino_acid(1, h).` |
| `type(engine, V8).` | `type(engine, "V8").` or `type(engine, v8).` |
| `player(P1).` | `player(p1).` or `player("P1").` |

#### B. The Closed World Principle: Never Hallucinate Data

**FUNDAMENTAL RULE**: Your ASP program operates under a **Closed World Assumption**. This means anything not explicitly stated as a fact or derivable from a rule is considered **false**. You MUST NOT invent or assume data that isn't provided.

All data must come from two places only:
1.  **Facts generated from the problem description.**
2.  **New facts derived by your rules from existing facts.**

**Common Mistake and Correction:**

| Incorrect (Hallucinates data) | Correct |
|:---|:---|
| **Problem:** "The available parts are a `cpu` and a `gpu`." <br/><br/> `part(cpu; gpu; ram).` <br/><br/> **Error**: The program invented `part(ram)`, which was not in the problem description. | **Problem:** "The available parts are a `cpu` and a `gpu`." <br/><br/> `part(cpu). part(gpu).` <br/><br/> **Explanation**: Only facts directly corresponding to the input are created. |
| **Problem:** "Schedule tasks for workers `w1`, `w2`." <br/><br/> `worker(W) :- assigned(_, W).` <br/><br/> **Error**: This only defines workers *if* they get a task. If `w2` is a valid worker but gets no tasks, `worker(w2)` will not be true. | **Problem:** "Schedule tasks for workers `w1`, `w2`." <br/><br/> `worker(w1). worker(w2).` <br/><br/> **Explanation**: The set of all possible workers is explicitly defined as facts from the problem input. |
| **Problem:** "Costs are: R1→C1: 10, R1→C2: 12" <br/><br/> `cost("R1","C1",10). cost("R1","C2",12). cost("R2","C2",12).` <br/><br/> **Error**: Invented cost for R2→C2 not in problem. | **Problem:** "Costs are: R1→C1: 10, R1→C2: 12" <br/><br/> `cost("R1","C1",10). cost("R1","C2",12).` <br/><br/> **Explanation**: Only the explicitly stated costs are created as facts. |

#### Hierarchy of Truth: Data Tables vs. Example Outputs

**CRITICAL RULE**: When a problem specification contains both data tables/lists AND example outputs, the **data tables are the single, immutable source of truth**. An example output in the problem statement fixes the output format (keys, nesting, types). Its values count as a solution only if the problem says so. If the problem presents the example as a valid or expected solution, run it through verify(): if it fails, your model or verifier is wrong, not the example. If the problem marks it as a dummy or says nothing about its validity, do not copy its values and do not test against them.

**NEVER modify, invent, or transfer data to match an example output.** Your model must solve the problem defined by the data tables, even if the example's values appear inconsistent.

**Why This Matters:**
- Example values not stated to be valid may be dummies, typos or from another instance
- Modifying data to match examples violates the Closed World Principle

**Common Mistake and Correction:**

| Incorrect (Modifies data to match example) | Correct (Uses only primary data) |
|:---|:---|
| **Data Table:** "Connection costs: A→B: 5, A→C: 10" <br/><br/> **Example Output:** Shows B connected to C <br/><br/> `cost("A","B",5). cost("A","C",10). cost("B","C",8).` <br/><br/> **Error**: Invented cost(B,C,8) to match example structure | **Data Table:** "Connection costs: A→B: 5, A→C: 10" <br/><br/> **Example Output:** Shows B connected to C <br/><br/> `cost("A","B",5). cost("A","C",10).` <br/><br/> **Explanation**: Use ONLY the provided costs. The solver finds the optimal solution using the true data. |
| **Data:** "Capacities: Server1: 100GB, Server2: 200GB" <br/><br/> **Example:** Shows Server3 in use <br/><br/> `capacity("Server1",100). capacity("Server2",200). capacity("Server3",150).` <br/><br/> **Error**: Added Server3 because example showed it | **Data:** "Capacities: Server1: 100GB, Server2: 200GB" <br/><br/> **Example:** Shows Server3 in use <br/><br/> `capacity("Server1",100). capacity("Server2",200).` <br/><br/> **Explanation**: Example is from a different instance. Use only the provided servers. |

**Priority Order:**
1. **Primary data tables/lists** (highest priority)
2. **Explicit problem constraints**
3. **Example output structure** (format guidance only)
4. **Example output values** (a solution only if the problem says so)

#### C. Structure of Rules and Facts
* Every statement (fact or rule) **MUST** end with a period (`.`).
* In a rule body, a comma (`,`) means **AND**.
* Comments start with a percent sign (`%`).

```asp
% This is a FACT. It ends with a period.
city("paris").

% This is a RULE. Note the period at the end.
% "is_in_france(X) is true IF city(X) is true AND X is "paris"."
is_in_france(X) :- city(X), X == "paris".
```

### 2.2 MANDATORY RULE: Variable Safety

**FUNDAMENTAL RULE**: Every variable appearing in a rule MUST be grounded by appearing in at least one positive, non-aggregate, non-arithmetic literal in the rule's body.

**Why this matters**: ASP needs to know which values to check. Unsafe variables have no defined domain.

**Examples of UNSAFE vs SAFE rules**:
```asp
% UNSAFE - Variable S has no positive grounding
:- item(I), not on_shelf(I, S).
% Error: 'S' is unsafe - solver doesn't know which shelves to check

% SAFE - Variable S is grounded by shelf(S)
:- item(I), shelf(S), not on_shelf(I, S).
% Now solver checks all combinations of items and shelves

% UNSAFE - Variables X,Y only appear in negation
:- robot(R), not at(R,X,Y).
% Error: 'X' and 'Y' are unsafe

% SAFE - Variables X,Y grounded by location/2
:- robot(R), location(X,Y), not at(R,X,Y).
% Now solver knows to check all valid locations
```

#### Variable Safety with Negation as Failure (`not`)

A critical application of this rule is when using negation (`not`). Any variable used inside a `not` literal **must** be grounded by a positive literal elsewhere in the same rule body. The solver cannot find solutions by iterating through infinite possibilities of what is *not* true.

**Example: Finding unassigned tasks**
Imagine you have `task(1..10)` and `assigned(T, W)` facts for tasks that are already assigned. You want to write a rule to find unassigned tasks.

| Incorrect (produces "unsafe variable" error) | Correct |
|:---|:---|
| `unassigned(T) :- not assigned(T, W).` <br/><br/> **Error:** The variable `W` is unsafe. The solver doesn't know which set of workers (`W`) to check for non-assignment. | `unassigned(T) :- task(T), not assigned(T, _).` <br/><br/> **Explanation:** `task(T)` grounds the `T` variable. The underscore `_` in `not assigned(T, _)` checks for the existence of *any* assignment for that specific task `T`, correctly identifying tasks with no assignments. |

#### Choice Rule Variable Safety

**CRITICAL RULE**: Variables in choice rules must be grounded either:
1. **In the choice domain** (after the `:` inside `{...}`), OR
2. **In the rule body** (after the `:-`)

Variables appearing in the choice head but nowhere else are UNSAFE.

**General Pattern:**
```asp
% UNSAFE - Variables in choice head but not grounded
{chosen(X, Var1, Var2)} :- domain(X).
% Error: 'Var1' and 'Var2' have no defined domain

% SAFE - Option 1: Ground in choice domain
{chosen(X, Var1, Var2) : property1(Var1), property2(Var2)} :- domain(X).

% SAFE - Option 2: Ground in rule body
{chosen(X, Var1, Var2)} :- domain(X), property1(Var1), property2(Var2).
```

**Example: Assignment**
```asp
% UNSAFE
{assign(Task, Worker, Priority)} :- task(Task).

% SAFE - Ground in choice domain
{assign(Task, Worker, Priority) : worker(Worker), priority(Priority)} :- task(Task).
```

**Example: Pairing**
```asp
% UNSAFE
{pair(Group, P1, P2)} :- group(Group).

% SAFE
{pair(Group, P1, P2) : member(P1), member(P2)} :- group(Group), P1 != P2.
```

### 2.3 MANDATORY RULE: Aggregate Placement

**Aggregates (#count, #sum, #min, #max) can ONLY be used:**
- In the BODY of a rule (for conditions)
- In #minimize/#maximize statements
- NEVER as an argument of a head atom, as in `total(#sum{...})`

**Why**: Rule heads generate new facts, while aggregates compute over existing facts.

```asp
% CORRECT - Aggregate in constraint body
:- #count { X : selected(X) } > 5.

% CORRECT - Aggregate in optimization
#minimize { Cost,X : selected(X), cost(X,Cost) }.

% INCORRECT - Cannot use aggregate in rule head
% total(#sum{C,X : cost(X,C)}) :- ...  % SYNTAX ERROR!

% CORRECT - Use auxiliary predicates instead
total(T) :- T = #sum{C,X : selected(X), cost(X,C)}.
```

### 2.4 MANDATORY RULE: State Exclusivity (Fluents)

**FUNDAMENTAL PRINCIPLE**: A property that changes over time (a "fluent") can only have ONE value at any given time. You MUST enforce mutual exclusivity.

**Critical for temporal problems**: An entity cannot be in two states simultaneously.

```asp
% INCORRECT - Allows item to be in two places
item_at(I,X,Y,T+1) :- item_at(I,X,Y,T), not pickup(_,I,T).
carrying(R,I,T+1) :- pickup(R,I,T).
% BUG: Item can be both at location AND carried!

% CORRECT - Enforce mutual exclusivity
% Item cannot be in two different places
:- item_at(I,X1,Y1,T), item_at(I,X2,Y2,T), (X1,Y1) != (X2,Y2).

% Item cannot be both on grid AND carried
:- item_at(I,X,Y,T), carrying(_,I,T).
```

### 2.5 Required 4-Step Workflow

#### Step 1: Analyze & Model
- Identify objects/entities, relationships, constraints, optimization criteria
- Design appropriate predicates
- **Parse input data and generate ASP facts from problem description**
- **Identify fluents (changing properties) and enforce exclusivity**

#### Step 2: Implement with Clingo
- Create Control object
- Add ASP rules (facts, choice rules, constraints, optimization)
- Ground the program
- **Follow the Three-Step Action Pattern for temporal problems**

#### Step 3: Solve & Extract Solutions
- Configure solver parameters
- Extract atoms from answer sets
- Handle multiple solutions or optimization

#### Step 4: Format & Verify Output
- Convert to required JSON format
- Verify solution satisfies constraints
- Handle UNSAT cases gracefully. A justified UNSAT is a correct final
  answer — verify only that every integrity constraint is warranted by the
  problem text; do not try to prove *why* it is unsatisfiable. Optimality:
  trust the solver when the solve finished (`exhausted`); if it was stopped
  early, say so in a program comment and do not claim optimality.

### 2.6 How to Express Constraints - The Eliminative Mindset

**CRITICAL**: ASP constraints work by ELIMINATION, not assertion. You forbid what's invalid, not state what's valid.

**Common Constraint Patterns:**

```asp
% To enforce X >= Y, forbid X < Y
% WRONG: :- capacity >= 32.  % This would forbid ALL solutions!
% CORRECT: Forbid capacity being less than 32
:- selected_ram(R), ram(R,_,Capacity,_,_), Capacity < 32.

% To enforce X > Y, forbid X <= Y
:- deadline(D), current_day(C), D <= C.  % Deadline must be after today

% To enforce X != "bad_value", forbid X = "bad_value"  
:- status(X, "invalid").  % No entity can have invalid status

% To enforce at least one selection, forbid having none
:- not selected(_).  % At least one must be selected

% To enforce mutual exclusion, forbid both being true
:- has_property(X, hot), has_property(X, cold).  % Can't be both
```

**Remember**: Think "what makes a solution invalid?" then write `:- invalid_condition.`

### 2.7 Hard vs Weak Constraints

**Use the right tool for the job:**

**Integrity Constraints (`:-`)**: Define what is IMPOSSIBLE
```asp
% These eliminate invalid answer sets entirely
:- overlapping_tasks(T1,T2).  % No overlaps allowed
:- budget_exceeded.            % Hard budget limit
```

**Weak Constraints (`:~`)**: Define what is UNDESIRABLE  
```asp
% These guide optimization without eliminating solutions
:~ delayed(Task). [1@2,Task]   % Prefer on-time, priority 2
:~ cost(C). [C@1]              % Minimize cost, priority 1
```

### 2.8 Multi-Objective Optimization

When using multiple `#minimize` or `#maximize` statements, clingo sums them into one objective unless each weight has a priority `@P`; with priorities it optimizes level-by-level, highest `P` first:

```asp
% Optimized in this order:
#maximize { Score@4 : ai_performance(Score) }.    % First priority
#maximize { Score@3 : gaming_score(Score) }.      % Second priority  
#maximize { Score@2 : user_preference(Score) }.   % Third priority
#minimize { Cost@1 : total_cost(Cost) }.          % Fourth priority
```

This allows sophisticated trade-off balancing without complex weight tuning.

## Section 3: Clingo API & Implementation Guide

### 3.1 Program Shape (every submitted program)

```python
import json
import clingo

TASKS, WORKERS = ["t1", "t2"], ["w1", "w2"]   # instance data
COST = {("t1", "w1"): 4, ("t1", "w2"): 2, ("t2", "w1"): 3, ("t2", "w2"): 5}

PROGRAM = """
{ assigned(T,W) : worker(W) } = 1 :- task(T).
:- assigned(T1,W), assigned(T2,W), T1 < T2.   % one task per worker
#minimize { C,T : assigned(T,W), cost(T,W,C) }.
"""

def solve() -> dict:
    facts = [f"task({t})." for t in TASKS] + [f"worker({w})." for w in WORKERS]
    facts += [f"cost({t},{w},{c})." for (t, w), c in COST.items()]
    ctl = clingo.Control()
    ctl.add("base", [], " ".join(facts) + PROGRAM)
    ctl.ground([("base", [])])
    best = {}
    def on_model(m):   # extract plain data here (see 3.6)
        best["assigned"] = {str(a.arguments[0]): str(a.arguments[1])
                            for a in m.symbols(atoms=True) if a.match("assigned", 2)}
        best["cost"] = m.cost[0]
    if not ctl.solve(on_model=on_model).satisfiable:
        return {"error": "No solution exists"}
    return best

def verify(solution: dict) -> None:
    if "error" in solution:
        return
    a = solution["assigned"]
    assert sorted(a) == sorted(TASKS), "every task exactly once"
    assert all(w in WORKERS for w in a.values()), "unknown worker"
    assert len(set(a.values())) == len(a), "a worker with two tasks"
    assert solution["cost"] == sum(COST[t, w] for t, w in a.items()), "wrong cost"

if __name__ == "__main__":
    solution = solve()
    verify(solution)
    print(json.dumps(solution))
```

`verify()` takes only the solution and reads the data from module-level
constants; importing the module must not solve or print.

### 3.2 CRITICAL: Finding One vs. All Solutions (Solver Configuration)

A frequent and critical error is misconfiguring the solver to search for all possible solutions when only one is required. This is a leading cause of timeouts on moderately complex problems. The number of models to find is controlled by the first argument to `clingo.Control()`.

> **Fundamental Rule:** If the problem asks for *a* feasible solution, you MUST limit the solver to find only one.

| Configuration | Behavior | Use Case | Consequence of Misuse |
| :--- | :--- | :--- | :--- |
| `ctl = clingo.Control()` | **Stops after the first solution (model) is found; with `#minimize`/`#maximize` it continues until the optimum (the last model is proven optimal only if the solve result's `exhausted` is True; a solve stopped early has `interrupted` True).** | **This should be your default for most tasks.** Use when the problem asks for *any* valid solution. | (Safe) |
| `ctl = clingo.Control(["1"])` | **Stops after the first solution, also with `#minimize`/`#maximize` ("optimality of last model not guaranteed").** | Satisfaction problems only. | **Non-optimal answers** for optimization problems. |
| `ctl = clingo.Control(["0"])` | **Finds all possible solutions.** | Use **only** when the problem explicitly requires enumerating every single valid combination. | **Leads to timeouts.** The solver will continue working long after finding a valid solution, needlessly exploring the entire search space. |


### 3.3 The Three-Step Action Modeling Pattern (CRITICAL for temporal problems)

For EVERY action in temporal/planning problems, you MUST model three aspects:

#### 1. Choice Rule (Generation)
Define when an action CAN possibly occur:
```asp
% Robot can try to pickup any item at any valid time
{ pickup(R,I,T) : item(I) } 1 :- robot(R), time(T), T < max_time.
```

#### 2. Precondition Constraints (Validation)
Define when an action is INVALID (prune invalid choices):
```asp
% Cannot pickup if robot and item at different locations
:- pickup(R,I,T), robot_at(R,RX,RY,T), item_at(I,IX,IY,T), (RX,RY) != (IX,IY).

% Cannot pickup if already carrying something
:- pickup(R,I,T), carrying(R,_,T).
```

#### 3. Effect Rules (State Change)
Define the NEW STATE resulting from a valid action:
```asp
% Pickup causes carrying at next timestep
carrying(R,I,T+1) :- pickup(R,I,T).

% Item no longer has location after pickup
% (Enforced by mutual exclusivity constraint)
```

### 3.4 Passing Data to ASP

```python
def generate_facts(problem_data):
    facts = []
    for item in problem_data["items"]:
        facts.append(f'item("{item}").')
    return " ".join(facts)
```

### 3.5 Extracting Solutions

```python
def extract_solution(model):
    solution = {}
    for atom in model.symbols(atoms=True):
        if atom.match("assigned", 2):
            task = str(atom.arguments[0])
            worker = str(atom.arguments[1])
            if worker not in solution:
                solution[worker] = []
            solution[worker].append(task)
    return solution
```

### 3.6 CRITICAL: The Model Object Is Only Valid Inside on_model (Segfault Risk)

The `clingo.Model` object passed to your `on_model` callback is only valid DURING
that callback. After `ctl.solve()` returns, clingo frees the underlying model —
touching it later is a use-after-free that crashes the Python process with a
segmentation fault (exit code 139) and NO Python traceback.

Extract everything you need into plain Python objects (ints, strings, lists, dicts)
INSIDE the callback. Never store the Model object itself for later use.

```python
# WRONG - segfaults after solve() returns:
saved = None
def on_model(m):
    global saved
    saved = m                        # Model outlives the callback
ctl.solve(on_model=on_model)
atoms = saved.symbols(atoms=True)    # use-after-free -> SIGSEGV, no traceback

# CORRECT - extract inside the callback:
result = []
def on_model(m):
    for a in m.symbols(atoms=True):
        if a.match("assigned", 2):
            result.append((str(a.arguments[0]), str(a.arguments[1])))
ctl.solve(on_model=on_model)
# work with the plain tuples in result
```

This also applies to verification: snapshot the atoms into plain Python inside
`on_model`, then run your checks on the snapshot after solving.

## Section 4: Problem-Solving Pattern Library

### 4.1 Modeling State and Change (NEW - CRITICAL)

#### A. Define Exclusive States (Fluents)
A fluent is a property that changes over time. It can only have ONE value at any time.

**Examples of fluents:**
- Location of an object
- Whether a resource is available
- State of a process (idle, running, complete)

**Enforce exclusivity:**
```asp
% Object cannot be in two places
:- at(Obj,Loc1,T), at(Obj,Loc2,T), Loc1 != Loc2.

% Resource cannot be both free and occupied
:- free(R,T), occupied(R,T).

% Process cannot be in multiple states
:- state(P,S1,T), state(P,S2,T), S1 != S2.
```

#### B. Frame Axioms: Persistence vs Change
Frame axioms define what persists. They must NOT fire when an action changes that state.

**Pattern:**
```asp
% State persists IF no action changes it
fluent(Args,T+1) :- fluent(Args,T), time(T+1), 
    not action_that_changes_fluent(Args,T).
```

**Example:**
```asp
% Item location persists if not picked up
item_at(I,X,Y,T+1) :- item_at(I,X,Y,T), time(T+1), 
    not pickup(_,I,T).

% Carrying persists if not put down
carrying(R,I,T+1) :- carrying(R,I,T), time(T+1), 
    not putdown(R,I,T).
```

Debugging temporal models: see Section 5.6.

### 4.2 Assignment Problems

**Pattern: Worker-Task Assignment**
```asp
% Each task assigned to exactly one worker
1 { assigned(T,W) : worker(W) } 1 :- task(T).

% Worker capacity constraint
:- worker(W), capacity(W,C), #count { T : assigned(T,W) } > C.
```

### 4.3 Graph Problems

**Pattern: Graph Coloring**
```asp
% Each node gets exactly one color
1 { colored(N,C) : color(C) } 1 :- node(N).

% Adjacent nodes different colors
:- colored(N1,C), colored(N2,C), edge(N1,N2).
```

### 4.4 Scheduling Problems

**CRITICAL: Always Define a Finite Time Horizon**

For scheduling, planning, and any temporal problem, you MUST define a finite time domain. Without this, a recursive rule such as `at(T+1) :- at(T).` does not finish grounding, a time variable without a domain is unsafe, and an aggregate over an empty time domain evaluates to `#inf`.

```asp
% REQUIRED: Define finite time domain first
time(0..max_time).  % Use the problem's latest deadline or a reasonable upper bound
```

**Pattern: Job Scheduling**
```asp
% STEP 1: Define time domain (CRITICAL!)
time(0..20).  % Set based on problem constraints

% STEP 2: Each job starts at exactly one time
1 { start(J,T) : time(T) } 1 :- job(J).

% No time overflow
:- start(J,T), duration(J,D), T+D-1 > 20.

% No overlap
:- start(J1,T1), start(J2,T2), J1 != J2,
   duration(J1,D1), T1 <= T2, T2 < T1+D1.
```

**Why This Matters:**
- Without `time(0..N)`, variables like `T` in `#minimize { T : makespan(T) }` have no bound
- If `time/1` is empty, `#max` over it is `#inf`; clingo then drops the `#minimize` tuple (`info: tuple ignored: #inf@0`) and the program is usually UNSAT
- Always use the latest deadline or a reasonable upper bound as your time horizon

### 4.5 Temporal Reasoning with Frame Axioms

**Critical for**: Robot planning, logistics, action sequences

**Complete Pattern for Temporal Problems:**

```asp
% 1. ENTITIES AND TIME
#const max_time = 10.
robot(r1). item(i1). time(0..max_time).
location(1..5, 1..5).
delta(up,0,1). delta(down,0,-1). delta(left,-1,0). delta(right,1,0).

% 2. INITIAL STATE
robot_at(r1,1,1,0).
item_at(i1,2,2,0).

% 3. ACTION GENERATION (Choice Rules)
{ move(R,Dir,T) : delta(Dir,_,_) } 1 :- robot(R), time(T), T < max_time.
{ pickup(R,I,T) : item(I) } 1 :- robot(R), time(T), T < max_time.
{ putdown(R,I,T) : item(I) } 1 :- robot(R), time(T), T < max_time.

% 4. MUTUAL EXCLUSION (State Constraints)
% Robot cannot be in two places
:- robot_at(R,X1,Y1,T), robot_at(R,X2,Y2,T), (X1,Y1) != (X2,Y2).

% Item cannot be both at location and carried
:- item_at(I,X,Y,T), carrying(_,I,T).

% 5. ACTION PRECONDITIONS
% Cannot move outside grid
:- move(R,D,T), delta(D,DX,DY), robot_at(R,X,Y,T), not location(X+DX,Y+DY).

% Cannot pickup from different location
:- pickup(R,I,T), robot_at(R,RX,RY,T), item_at(I,IX,IY,T), (RX,RY) != (IX,IY).

% Cannot put down an item that is not carried
:- putdown(R,I,T), not carrying(R,I,T).

% 6. ACTION EFFECTS
% Movement changes robot position
robot_at(R,X+DX,Y+DY,T+1) :- move(R,D,T), delta(D,DX,DY), robot_at(R,X,Y,T).

% Pickup creates carrying relationship
carrying(R,I,T+1) :- pickup(R,I,T).

% Putdown creates item location
item_at(I,X,Y,T+1) :- putdown(R,I,T), robot_at(R,X,Y,T).

% 7. FRAME AXIOMS (Persistence)
% Robot position persists if no movement
robot_at(R,X,Y,T+1) :- robot_at(R,X,Y,T), time(T+1), not move(R,_,T).

% Item location persists if not picked up
item_at(I,X,Y,T+1) :- item_at(I,X,Y,T), time(T+1),
    not pickup(_,I,T).

% Carrying persists if not put down
carrying(R,I,T+1) :- carrying(R,I,T), time(T+1),
    not putdown(R,I,T).
```

### 4.6 Combinatorial Puzzles

**Pattern: N-Queens**
```asp
#const n=8.
row(1..n). col(1..n).
1 { queen(R,C) : col(C) } 1 :- row(R).
:- queen(R1,C), queen(R2,C), R1 != R2.
:- queen(R1,C1), queen(R2,C2), R1 != R2, |R1-R2| == |C1-C2|.
```

### 4.7 Enforcing Dependent Properties (A implies B)

A common task is to enforce that if one property `P` exists, a corresponding property `Q` must also exist. A frequent mistake is to only forbid the case `P and not Q` using complex joins, which can be brittle. A more robust pattern is to state the implication directly as a constraint.

**Pattern: If a keyed path exists, a keyless return MUST exist.**

Let's say for a connection `conn(R1, R2, Key)`, we must ensure `conn(R2, R1, "null")` exists.

**Incorrect Attempt (Brittle):**
This attempts to join all possibilities and forbid the bad ones. It fails if multiple return paths exist or if the solver simply avoids creating a bidirectional link.
```asp
% Fails to enforce generation; only prunes specific bad combinations.
:- conn(R1, R2, Req1), conn(R2, R1, Req2), key(Req1), Req2 != "null".
```

**Correct Pattern (Robust Logic):**
This pattern forbids the state where the premise (`keyed path exists`) is true but the conclusion (`keyless return path exists`) is false. This forces the solver to generate the conclusion to satisfy the constraint.

```asp
% Define the premise: a keyed path exists from R1 to R2.
keyed_path(R1, R2) :- conn(R1, R2, Req), key(Req).

% Enforce the conclusion: if a keyed path exists, a keyless return MUST exist.
% This translates to: "It is an error if a keyed path from R1 to R2 exists
% AND a keyless path from R2 to R1 does NOT exist."
:- keyed_path(R1, R2), not conn(R2, R1, "null").
```
This approach correctly models the logical implication `keyed_path(R1, R2) -> conn(R2, R1, "null")` and is the standard way to express such dependencies in ASP.

### 4.8 Default Logic with Negation as Failure

**Pattern: Default Property with Exceptions**

This pattern implements defeasible reasoning where properties hold by default unless exceptions apply:

```asp
% General Pattern:
% A property holds by default...
has_property(X) :- entity(X), not lacks_property(X).

% ...unless a specific exception applies
lacks_property(X) :- exception_condition_1(X).
lacks_property(X) :- exception_condition_2(X).

% Example: Birds fly unless they're penguins or injured
flies(B) :- bird(B), not cannot_fly(B).
cannot_fly(B) :- penguin(B).
cannot_fly(B) :- injured(B).

% Example: Statements are true unless proven false
believed(Statement) :- statement(Statement), not disproven(Statement).
disproven(Statement) :- contradicts(Statement, Fact), proven(Fact).
```

**Key Insight**: The `not` operator implements Closed World Assumption - what cannot be proven true is assumed false.

### 4.9 Sequential Planning Problems (Tower of Hanoi, Blocks World)

**Critical for**: Multi-step sequential planning where exactly ONE action occurs per timestep.

**ANTI-PATTERN: Multiple Simultaneous Actions + Global Count**

This is the most common mistake in planning problems:

```asp
% WRONG: Allows multiple moves per timestep (one per disk!)
{ move(D, P1, P2, T) : peg(P1), peg(P2), P1 != P2 } 1 :- disk(D), time(T), T < max_time.

% Then uses slow global constraint to force total count
:- #count { D, P1, P2, T : move(D, P1, P2, T) } != 19.
```

**Why this is catastrophically slow:**
1. Generates moves for ALL disks from ANY peg to ANY peg (even illegal ones)
2. Allows up to N simultaneous moves per timestep (one per disk)
3. Uses expensive #count over entire plan to enforce sequence length
4. Creates massive search space that times out

**CORRECT PATTERN: Sequential Actions with Validity Checking**

The key principles:
1. **Define what CAN be moved** (e.g., top disk on a peg)
2. **Choose exactly ONE valid action per timestep**
3. **No global count needed** - sequence length emerges from timesteps

**Complete Tower of Hanoi Example:**

```asp
#const max_time = 19.
disk(1..4). peg(a;b;c;d).
time(0..max_time-1).  % Action timesteps (0..18 for 19 moves)

% INITIAL STATE
on_peg(1,a,0). on_peg(2,a,0). on_peg(3,a,0). on_peg(4,a,0).

% --- 1. DEFINE WHAT CAN BE MOVED ---
% A disk is "top" if no smaller disk is on the same peg
top(D, P, T) :- on_peg(D, P, T), not covered(D, P, T).
covered(D, P, T) :- on_peg(D, P, T), on_peg(D2, P, T), D2 < D.

% --- 2. CHOOSE EXACTLY ONE VALID MOVE PER TIMESTEP ---
% At each time T, move exactly one top disk to a different peg
1 { move(D, P1, P2, T) : top(D, P1, T), peg(P2), P1 != P2 } 1 :- time(T).

% --- 3. PRECONDITIONS ---
% Cannot place larger disk on smaller disk
:- move(D, _, P2, T), top(D2, P2, T), D > D2.

% --- 4. EFFECTS ---
% Moved disk is on new peg at T+1
on_peg(D, P2, T+1) :- move(D, _, P2, T).

% --- 5. FRAME AXIOMS ---
% Disks not moved stay on their peg
on_peg(D, P, T+1) :- on_peg(D, P, T), T < max_time, not move(D, P, _, T).

% --- 6. STATE EXCLUSIVITY ---
% Disk cannot be on two pegs
:- on_peg(D, P1, T), on_peg(D, P2, T), P1 != P2.

% --- 7. DOMAIN-SPECIFIC CONSTRAINTS ---
% Pilgrim's journey: every disk must visit both B and C
visited(D, P) :- move(D, _, P, _).
:- disk(D), not visited(D, b).
:- disk(D), not visited(D, c).

% --- 8. GOAL ---
% All disks on peg D at final timestep
:- disk(D), not on_peg(D, d, max_time).
```

**Why this is fast:**
- Only generates valid moves (from top disks only)
- Exactly one move per timestep (correctly models sequential plan)
- No expensive global #count constraint
- Search space is minimal and focused

**Key Pattern Elements:**

1. **Define "Valid"**: Use auxiliary predicate (`top/3`) to identify what can be acted upon
2. **One Action Per Time**: `1 { action(...) : valid(...) } 1 :- time(T)`
3. **Constrain at Source**: Only generate moves from valid sources in choice rule
4. **No Global Counts**: Let sequence length emerge from timesteps naturally

**Blocks World Pattern:**

```asp
% Define what can be moved
clear(B,T) :- block(B), time(T), not on(_, B, T).
on_table_space(T) :- time(T), #count { B : on_table(B,T) } < max_table_slots.

% Exactly one move per timestep
1 { move_to_block(B1, B2, T) : clear(B1,T), clear(B2,T), B1 != B2 ;
    move_to_table(B, T) : clear(B,T), on_table_space(T) } 1 :- time(T).
```

**General Takeaway:** For sequential planning, always:
- Define valid actions via auxiliary predicates
- Use `1 { ... } 1 :- time(T)` for exactly one action per timestep
- Avoid generate-and-filter patterns in choice rules
- Eliminate global #count constraints over entire plans

### 4.10 "Reachable Only After" / Ordered Reachability Constraints

When a problem requires that some target element become reachable only after all
other elements (or only after specific ones) under an evolving reachability process,
do NOT model it as an existential ordering — guessing a permutation with the target
pinned last only proves that SOME order exists, which is weaker than what such
problems ask. The intended reading is almost always about the actual reachability
process: everything reachable gets reached as soon as it is reachable.

Model the process directly with a time-indexed fixpoint and constrain first-reach
times:

```asp
step(0..max_step).
reached(S, 0) :- start(S).
reached(Y, T+1) :- edge(X, Y), reached(X, T), step(T+1), unlocked(X, Y, T).
reached(X, T+1) :- reached(X, T), step(T+1).                    % persistence
first_reached(X, 0) :- reached(X, 0).
first_reached(X, T) :- reached(X, T), not reached(X, T-1), T > 0, step(T).

% target must not become reachable before (or together with) any other element:
:- first_reached(target, Tg), first_reached(X, Tx), X != target, Tx >= Tg.
```

Adapt `unlocked/3` to whatever gates progression (collected items, satisfied
prerequisites, ...). Verify with a plain-Python simulation of the same process:
repeatedly add everything currently reachable, one step at a time, and check the
required ordering of first-reach steps against the problem text — do not reuse
your encoding's ordering predicates in the verifier.

### 4.11 Runs in Line Puzzles (Nonograms)

Two errors recur. (1) A gap between every pair of consecutive runs: only runs of the *same* color need one; runs of different colors may touch, and a colored clue can fill its line exactly (3+2+3 in 8 cells), so a gap rule for all runs is UNSAT. (2) Optional runs (`{...} 1`) or free cells outside the runs: unless the problem says otherwise, the clue lists all runs, so place each run exactly once and make every other cell background.

```asp
% Lines L = r(Row) or c(Col); color 0 = background.
% clue(L,I,K,N): the I-th run (I = 1,2,...) of line L has color K, length N.
1 { cell(R,C,K) : color(K) } 1 :- row(R), col(C).
slot(r(R),C) :- row(R), col(C).   slot(c(C),R) :- row(R), col(C).
at(r(R),C,K) :- cell(R,C,K).      at(c(C),R,K) :- cell(R,C,K).
1 { start(L,I,S) : slot(L,S) } 1 :- clue(L,I,_,_).
:- start(L,I,S), clue(L,I,_,N), not slot(L,S+N-1).
in(L,P,K) :- start(L,I,S), clue(L,I,K,N), slot(L,P), S <= P, P < S+N.
% order; a gap only between runs of the same color
:- start(L,I,S), start(L,I+1,T), clue(L,I,_,N), T < S+N.
:- start(L,I,S), start(L,I+1,T), clue(L,I,K,N), clue(L,I+1,K,_), T = S+N.
% run cells have the run's color; every other cell is background
:- in(L,P,K), not at(L,P,K).
:- at(L,P,K), K != 0, not in(L,P,K).
```

## Section 5: Debugging & Advanced Techniques

### 5.1 Efficient Generation: Constrain Choices Early (Anti-Pattern: Generate-and-Filter)

A common performance pitfall is to generate a massive number of candidate solutions in a choice rule and then use separate integrity constraints to filter out the vast majority of invalid ones. This forces the grounder and solver to consider many useless options.

**The Principle: Constrain at the Source.** Build validity checks directly into the generating choice rule itself, so the solver only ever considers valid candidates.

**Anti-Pattern: Generate, then Filter**
This approach generates all combinations and then prunes them.

```asp
% DOMAIN: 100 items, 50 containers. compatible(I,C) defines valid pairs.
item(1..100). container(1..50).
compatible(1,1). compatible(2,1). %... many other compatibility facts

% ANTI-PATTERN: Generate all 100*50=5000 possible assignments
{ placed(I,C) : container(C) } 1 :- item(I).

% Then, filter out the invalid ones
:- placed(I,C), not compatible(I,C).
```
**Why it's slow:** The solver must first consider 5,000 potential `placed/2` atoms before eliminating most of them.

**Good Pattern: Constrain During Generation**
This approach only generates combinations that are already known to be valid.

```asp
% DOMAIN: Same as above.
item(1..100). container(1..50).
compatible(1,1). compatible(2,1). %...

% GOOD PATTERN: Only generate assignments (I,C) where compatible(I,C) is true.
{ placed(I,C) : container(C), compatible(I,C) } 1 :- item(I).

% No filter rule is needed.
```
**Why it's fast:** The solver's search space is restricted from the very beginning to only valid pairs. If there are only 200 compatible pairs, it considers 200 options, not 5,000.

**General Takeaway:** Push filtering conditions from the body of separate constraints (`:- ...`) into the domain of the choice rule (`{... : conditions}`).

### 5.2 CRITICAL: Canonical Counting for Symmetric Relations

When counting relationships between pairs of entities (connections, edges, paths), you must ensure each pair is counted **exactly once**. A symmetric definition can lead to double-counting, making cardinality constraints impossible to satisfy.

**The Problem: Double-Counting**

A common mistake is to define a relationship symmetrically, which causes each instance to be counted multiple times:

```asp
% INCORRECT: Double-counts one-way paths
% If R3 -> R4 is one-way, this creates BOTH oneway("R3","R4") AND oneway("R4","R3")
oneway(R1, R2) :- conn(R1, R2, _), not conn(R2, R1, _), R2 != "Goal".
oneway(R1, R2) :- conn(R2, R1, _), not conn(R1, R2, _), R2 != "Goal".

% This constraint expects count=1, but gets count=2 for a single path!
:- #count { R1, R2 : oneway(R1, R2) } != 1.
```

**The Solution: Canonical Ordering**

Establish a canonical representation for each pair by enforcing an ordering constraint (typically using `<`). This ensures each pair is represented exactly once:

```asp
% CORRECT: Canonical counting with ordering
% A one-way path between R1 and R2 (where R1 < R2) creates exactly ONE atom
oneway_pair(R1, R2) :- R1 < R2, conn(R1, R2, _), not conn(R2, R1, _).
oneway_pair(R1, R2) :- R1 < R2, conn(R2, R1, _), not conn(R1, R2, _).

% Now the constraint works correctly
:- #count { R1, R2 : oneway_pair(R1, R2) } != 1.
```

**Pattern Examples:**

```asp
% Bidirectional connections (canonical form)
bidirectional(R1, R2) :- R1 < R2, conn(R1, R2, _), conn(R2, R1, _).

% Counting edges in undirected graphs
edge_count(N) :- N = #count { V1, V2 : V1 < V2, edge(V1, V2) }.

% Finding isolated pairs
isolated_pair(A, B) :- A < B, entity(A), entity(B),
    not connected(A, B), not connected(B, A).
```

**When to Use This Pattern:**

- Counting bidirectional or one-way connections
- Measuring relationships in undirected graphs
- Any symmetric or pair-based relationship requiring accurate counting
- Whenever you need to distinguish between "relationship exists" and "count of relationships"

### 5.3 CRITICAL: Controlling Cardinality with Choice Rule Bounds

While Section 5.1 focuses on filtering the *domain* of a choice rule, it is equally critical to control the *number* of atoms chosen from that domain. An unbounded choice rule `{...}` allows the solver to select *any number* of atoms, which is often not the desired behavior and can lead to logical errors and massive search spaces.

> **Fundamental Rule:** Always specify bounds `L { ... } U` on your choice rules unless you explicitly intend to allow an arbitrary number of choices.

#### Anti-Pattern: Unbounded Choice for Relationships

This mistake was observed in a Metroidvania map generation task where the goal was to create at most one connection between any two rooms.

**The Code:**
```asp
% For each room pair, choose a requirement for a connection
{ conn(R1, R2, Req) : requirement(Req) } :- room_pair(R1, R2).
```

**The Problem:**
This rule allows the solver to generate **multiple** connections between the same two rooms, e.g., `conn(r1,r2,"null")` and `conn(r1,r2,"RedKey")` simultaneously. This is logically incorrect for the problem and creates a combinatorial explosion in the search space, often leading to `grounding stopped` errors or timeouts.

#### Correct Pattern: Bounded Choice for Unique Relationships

By adding bounds, you enforce the intended logic directly and efficiently.

**The Code:**
```asp
% For each room pair, choose AT MOST ONE requirement for a connection
0 { conn(R1, R2, Req) : requirement(Req) } 1 :- room_pair(R1, R2).
```

**The Solution:**
This rule correctly states: "For any pair of rooms (R1, R2), choose **zero or one** requirement for a connection." This enforces the intended unique relationship, drastically reduces the search space, and prevents logical errors.

#### General Patterns for Cardinality Control

| Goal | ASP Pattern | Explanation |
| :--- | :--- | :--- |
| **Assign a unique property** | `1 { has_property(X, P) : property(P) } 1 :- entity(X).` | Each entity `X` must have **exactly one** property `P`.|
| **Model an optional, unique connection** | `0 { connected(X, Y, Type) : conn_type(Type) } 1 :- pair(X, Y).` | There is **at most one** connection between `X` and `Y`.|
| **Select a fixed number of items** | `k { selected(I) : item(I) } k :- ...` | **Exactly `k`** items must be selected from the domain. |
| **Select a minimum number of items** | `k { selected(I) : item(I) } :- ...` | **At least `k`** items must be selected. |

### 5.4 Avoiding Grounding Explosion: Decompose Complex Arithmetic (Anti-Pattern: Monolithic Arithmetic)

Writing a single, large arithmetic constraint that involves many independent variables can cause a "grounding explosion." The grounder must instantiate the rule for every possible combination of values for those variables.

**The Principle: Aggregate, Don't Enumerate.** Use auxiliary predicates and aggregates (`#sum`, `#count`, etc.) to compute intermediate results. Then, write simpler constraints over those results.

**Anti-Pattern: Monolithic Arithmetic Constraint**
This approach forces the grounder to iterate through all value combinations.

```asp
% DOMAIN: Assign a value from 1..10 to four variables a,b,c,d.
var(a;b;c;d). value(1..10).
1 { assign(V, N) : value(N) } 1 :- var(V).

% ANTI-PATTERN: Compare sums directly.
% This requires grounding for every combination of N1,N2,N3,N4 from 1..10.
% That's 10 * 10 * 10 * 10 = 10,000 ground rules.
:- assign(a, N1), assign(b, N2), assign(c, N3), assign(d, N4),
   N1 + N2 != N3 + N4.
```
**Why it's slow:** The grounder creates a massive number of rules that are computationally expensive to process, even if the logic is simple.

**Good Pattern: Decompose with Aggregates**
This approach computes the sums first, then compares the two results.

```asp
% DOMAIN: Same as above.
var(a;b;c;d). value(1..10).
1 { assign(V, N) : value(N) } 1 :- var(V).

% GOOD PATTERN: Use #sum to calculate intermediate values.
sum_ab(S) :- S = #sum { N,V : assign(V,N), V=(a;b) }.
sum_cd(S) :- S = #sum { N,V : assign(V,N), V=(c;d) }.

% Now, constrain the two resulting sums. This generates very few ground rules.
:- sum_ab(S1), sum_cd(S2), S1 != S2.
```
**Why it's fast:** The `#sum` aggregate is highly optimized. The solver computes each sum just once. The final constraint only compares the two resulting sum values, avoiding the combinatorial explosion.

**General Takeaway:** Instead of writing complex arithmetic expressions with many variables in a single rule body, use aggregates to compute sub-expressions and constrain their results.

### 5.5 Recursive Aggregates (#sum over a derived relation)

When a threshold depends on a relation you are still deriving, aggregate over
that relation with a helper predicate to keep the recursion through `#sum` readable
(classic example: transitive company control via ownership percentages):

```asp
controls(X,X) :- company(X).
contrib(A,B,A,P) :- owns(A,B,P).
contrib(A,B,C,P) :- controls(A,C), owns(C,B,P), A != C.
sum(A,B,S)      :- company(A), company(B), S = #sum { P,C : contrib(A,B,C,P) }.
controls(A,B)   :- sum(A,B,S), S > 50, A != B.
```

Use a helper (`contrib/4`) so the aggregate ranges over ground tuples; do not
inline a `#sum` that recurses on the predicate it defines.

### 5.6 Debugging Temporal Models (CRITICAL)

**Common issues and solutions:**

1. **Invalid state combinations**
   - Check: Are mutually exclusive states enforced?
   - Fix: Add exclusivity constraints

2. **Actions happening without preconditions**
   - Check: Are all preconditions validated?
   - Fix: Add precondition constraints

3. **States not changing after actions**
   - Check: Are effect rules defined?
   - Fix: Add effect rules for each action

4. **UNSAT when solution should exist**
   - Debug: Comment out constraints one by one
   - Test: Create minimal test case
   - Isolate: Use "teleportation" test (bypass actions, technique 2 below)

**Debugging techniques:**

1. **Test state exclusivity first**
   ```asp
   % Add test constraints to verify exclusivity
   :- item_at(test_item,_,_,5), carrying(_,test_item,5).
   ```

2. **Isolate action logic**
   ```asp
   % Test if goals are achievable without actions
   item_at(I,X,Y,max_time) :- item_destination(I,X,Y).
   ```

3. **Trace single actions**
   ```asp
   % Force specific action to test effects
   move(r1,up,0).
   ```

4. **Use clingo's debugging features**
   ```python
   # Show ground atoms
   ctl.ground([("base", [])])
   for atom in ctl.symbolic_atoms:
       if "item_at" in str(atom.symbol):
           print(atom.symbol)
   ```

### 5.7 Common Errors and Solutions

1. **"Unsafe variable" errors**:
   - Add positive predicate defining variable's domain

2. **"Syntax error, unexpected #sum"**:
   - Move aggregate to body or optimization

3. **No answer sets (UNSAT)**:
   - Check mutual exclusivity constraints
   - Verify action preconditions aren't too restrictive
   - Test minimal cases

4. **Invalid plans/solutions**:
   - Check state exclusivity enforcement
   - Verify frame axioms and action effects

### 5.8 Best Practices

1. **Start with state model**: Define fluents and exclusivity first
2. **Test incrementally**: Add one action type at a time
3. **Use meaningful names**: `robot_at(r1,3,4,5)` not `p(1,3,4,5)`
4. **Document assumptions**: Comment complex constraints
5. **Verify exclusivity**: Always enforce mutual exclusion for fluents