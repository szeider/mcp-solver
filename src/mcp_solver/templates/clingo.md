# Answer Set Programming (ASP) Task - v10

## Section 1: Mission Briefing & Core Requirements

### Primary Goal
Produce a solution using Answer Set Programming (ASP) that correctly models and solves the problem using the clingo Python API.

### Non-Negotiable Requirements
1. **Tool**: You MUST use the clingo Python API for modeling and solving
2. **Execution Time**: Your script must complete within 10 seconds
   - **Optimization vs Bounds**: If the problem provides an expected optimal value or bound (e.g., "expected minimum cost: 50"), use that value as a CONSTRAINT rather than using `#minimize`/`#maximize`. Finding the absolute optimum can be extremely expensive. Instead, find any solution that satisfies the given bound.
   - **Example**: Instead of `#minimize { Cost : total_cost(Cost) }`, use `:- total_cost(Cost), Cost > 50.`
3. **Output Format**: Print the final solution as a single JSON object to stdout
4. **Task Planning**: Structure your approach with clear steps:
   - Mentally break down the problem into 5-8 logical tasks
   - Follow the structured workflow defined below
   - Use code comments to track your progress through tasks
5. **Error Documentation**: Create `feedback.md` for environment/specification issues

### Output Specifications
- Generate ONLY executable Python code using clingo
- Use clear, meaningful predicate names (e.g., `assigned(task1, worker2)` not `p(1,2)`)
- Extract solutions from answer sets and format as JSON
- Handle UNSAT cases gracefully with error JSON: `{"error": "No solution exists", "reason": "..."}`

### Final Checklist
Before completing your solution, verify:
☐ 1. **Uses clingo API** - Solution uses `clingo.Control()` and ASP rules
☐ 2. **Model solves correctly** - At least one answer set is found (or UNSAT handled)
☐ 3. **Optimization applied** - If problem asks for optimal, use #minimize/#maximize
☐ 4. **Solution extracted** - Answer set atoms properly extracted and formatted
☐ 4b. **No Model escapes on_model** - All solution data extracted to plain Python objects inside the callback (touching a Model after solve() returns segfaults)
☐ 5. **Output format correct** - JSON output with appropriate structure
☐ 6. **Predicates meaningful** - Clear names that reflect the problem domain
☐ 7. **Constraints verified independently** - All problem constraints are encoded AND re-checked by a plain-Python verify() whose expectations are re-derived from the problem text, not from your encoding's own interpretation
☐ 8. **Program submitted** - Call `submit_code` with the complete, verified,
self-contained program (all imports included, no reliance on session state)

## Section 2: Critical Rules of Engagement

### MANDATORY RULE: Core ASP Syntax

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
|:----------------------------------------------|:--------|
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
|:----------------------------------------------|:--------|
| **Problem:** "The available parts are a `cpu` and a `gpu`." <br/><br/> `part(cpu; gpu; ram).` <br/><br/> **Error**: The program invented `part(ram)`, which was not in the problem description. | **Problem:** "The available parts are a `cpu` and a `gpu`." <br/><br/> `part(cpu). part(gpu).` <br/><br/> **Explanation**: Only facts directly corresponding to the input are created. |
| **Problem:** "Schedule tasks for workers `w1`, `w2`." <br/><br/> `worker(W) :- assigned(_, W).` <br/><br/> **Error**: This only defines workers *if* they get a task. If `w2` is a valid worker but gets no tasks, `worker(w2)` will not be true. | **Problem:** "Schedule tasks for workers `w1`, `w2`." <br/><br/> `worker(w1). worker(w2).` <br/><br/> **Explanation**: The set of all possible workers is explicitly defined as facts from the problem input. |
| **Problem:** "Costs are: R1→C1: 10, R1→C2: 12" <br/><br/> `cost("R1","C1",10). cost("R1","C2",12). cost("R2","C2",12).` <br/><br/> **Error**: Invented cost for R2→C2 not in problem. | **Problem:** "Costs are: R1→C1: 10, R1→C2: 12" <br/><br/> `cost("R1","C1",10). cost("R1","C2",12).` <br/><br/> **Explanation**: Only the explicitly stated costs are created as facts. |

#### Hierarchy of Truth: Data Tables vs. Example Outputs

**CRITICAL RULE**: When a problem specification contains both data tables/lists AND example outputs, the **data tables are the single, immutable source of truth**. Example outputs demonstrate the required **format** only - their **values** are illustrative.

**NEVER modify, invent, or transfer data to match an example output.** Your model must solve the problem defined by the data tables, even if the example's values appear inconsistent.

**Why This Matters:**
- Example outputs may contain typos or be slightly out of sync with primary data
- Example values are often from a different instance or simplified scenario
- Modifying data to match examples violates the Closed World Principle

**Common Mistake and Correction:**

| Incorrect (Modifies data to match example) | Correct (Uses only primary data) |
|:----------------------------------------------|:--------|
| **Data Table:** "Connection costs: A→B: 5, A→C: 10" <br/><br/> **Example Output:** Shows B connected to C <br/><br/> `cost("A","B",5). cost("A","C",10). cost("B","C",8).` <br/><br/> **Error**: Invented cost(B,C,8) to match example structure | **Data Table:** "Connection costs: A→B: 5, A→C: 10" <br/><br/> **Example Output:** Shows B connected to C <br/><br/> `cost("A","B",5). cost("A","C",10).` <br/><br/> **Explanation**: Use ONLY the provided costs. The solver finds the optimal solution using the true data. |
| **Data:** "Capacities: Server1: 100GB, Server2: 200GB" <br/><br/> **Example:** Shows Server3 in use <br/><br/> `capacity("Server1",100). capacity("Server2",200). capacity("Server3",150).` <br/><br/> **Error**: Added Server3 because example showed it | **Data:** "Capacities: Server1: 100GB, Server2: 200GB" <br/><br/> **Example:** Shows Server3 in use <br/><br/> `capacity("Server1",100). capacity("Server2",200).` <br/><br/> **Explanation**: Example is from a different instance. Use only the provided servers. |

**Priority Order:**
1. **Primary data tables/lists** (highest priority)
2. **Explicit problem constraints**
3. **Example output structure** (format guidance only)
4. **Example output values** (ignore if inconsistent)

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

### MANDATORY RULE: Variable Safety

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
|:----------------------------------------------|:--------|
| `unassigned(T) :- not assigned(T, W).` <br/><br/> **Error:** The variable `W` is unsafe. The solver doesn't know which set of workers (`W`) to check for non-assignment. | `unassigned(T) :- task(T), not assigned(T, _).` <br/><br/> **Explanation:** `task(T)` grounds the `T` variable. The underscore `_` in `not assigned(T, _)` checks for the existence of *any* assignment for that specific task `T`, correctly identifying tasks with no assignments. |

**Example: Circuit diagnosis - ordering matters!**
When a variable appears in multiple places, it must be grounded before being used in negation.

| Incorrect (unsafe variable error) | Correct |
|:----------------------------------------------|:--------|
| `wire_value(T, W, V) :- not faulty(G, _), gate_output(G, W), correct_output(T, G, V).` <br/><br/> **Error:** `G` is unsafe - it first appears in `not faulty(G, _)`. | `wire_value(T, W, V) :- gate_output(G, W), not faulty(G, _), correct_output(T, G, V).` <br/><br/> **Explanation:** `gate_output(G, W)` grounds `G` first. Now `not faulty(G, _)` checks if *that specific gate* is healthy. |

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

### MANDATORY RULE: Aggregate Placement

**Aggregates (#count, #sum, #min, #max) can ONLY be used:**
- In the BODY of a rule (for conditions)
- In #minimize/#maximize statements
- NEVER in the HEAD of a regular rule

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

### MANDATORY RULE: State Exclusivity (Fluents)

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

### Required 5-Step Workflow

#### Step 0: Plan Your Approach (MANDATORY - DO THIS FIRST!)

**Structure your solution with these logical tasks:**
```python
# Task Plan:
# 1. Analyze problem and identify constraints
# 2. Design ASP predicates and model structure  
# 3. Implement facts and choice rules
# 4. Add constraints and optimization
# 5. Solve and extract solution
# 6. Format output as JSON
# 7. Test and verify solution
```

Use comments in your code to indicate which task you're working on:
```python
# Task 1: Analyze problem and identify constraints
# [Your analysis code here]

# Task 2: Design ASP predicates and model structure
# [Your design code here]
```

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
- Handle UNSAT cases gracefully

### When to Create feedback.md

Create `feedback.md` ONLY for:
- Missing packages or imports
- Inconsistencies or ambiguities in problem description
- Conflicting constraints that make problem unsolvable
- Environment setup problems
- API compatibility issues

NOT for your own logic errors or debugging.

### 2.5 How to Express Constraints - The Eliminative Mindset

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

### 2.6 Hard vs Weak Constraints

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
:~ delayed(Task). [1@2]        % Prefer on-time, priority 2
:~ cost(C). [C@1]              % Minimize cost, priority 1
```

### 2.7 Multi-Objective Optimization

When using multiple `#minimize` or `#maximize` statements, clingo optimizes level-by-level in order:

```asp
% Optimized in this order:
#maximize { Score*3 : ai_performance(Score) }.    % First priority
#maximize { Score*2 : gaming_score(Score) }.      % Second priority  
#maximize { Score : user_preference(Score) }.     % Third priority
#minimize { Cost : total_cost(Cost) }.            % Fourth priority
```

This allows sophisticated trade-off balancing without complex weight tuning.

## Section 3: Clingo API & Implementation Guide

### Basic API Flow

```python
import clingo
import json

# Create control object
ctl = clingo.Control()

# Add ASP program
program = """
% Facts from input
% ... generated facts ...

% Choice rules
{ choice(X) } :- domain(X).

% Constraints
:- invalid_condition.

% Optimization
#minimize { Cost,X : selected(X), cost(X,Cost) }.
"""
ctl.add("base", [], program)

# Ground the program
ctl.ground([("base", [])])

# Solve and extract
solution = None
def on_model(m):
    global solution
    atoms = m.symbols(atoms=True)
    # Process atoms into solution
    
result = ctl.solve(on_model=on_model)

# Output
if result.satisfiable:
    print(json.dumps(solution))
else:
    print(json.dumps({"error": "No solution exists"}))
```

### CRITICAL: Finding One vs. All Solutions (Solver Configuration)

A frequent and critical error is misconfiguring the solver to search for all possible solutions when only one is required. This is a leading cause of timeouts on moderately complex problems. The number of models to find is controlled by the first argument to `clingo.Control()`.

> **Fundamental Rule:** If the problem asks for *a* feasible solution, you MUST limit the solver to find only one.

| Configuration                  | Behavior                                                | Use Case                                                                              | Consequence of Misuse                                                                                             |
| :----------------------------- | :------------------------------------------------------ | :------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------- |
| `ctl = clingo.Control(["1"])`   | **Stops after the first solution (model) is found.**    | **This should be your default for most tasks.** Use when the problem asks for *any* valid solution. | (Safe)                                                                                                            |
| `ctl = clingo.Control()` or `ctl = clingo.Control(["0"])` | **Finds all possible solutions.** | Use **only** when the problem explicitly requires enumerating every single valid combination. | **Leads to timeouts.** The solver will continue working long after finding a valid solution, needlessly exploring the entire search space. |

**Example from a Failed Task (Bin Packing):**
- **The Goal:** Find one valid way to pack bins.
- **The Mistake:** `ctl = clingo.Control(["0"])` was used. The solver found a valid packing quickly but then continued searching for all other possible valid packings, eventually timing out.
- **The Fix:** Changing the call to `ctl = clingo.Control(["1"])` allows the program to terminate immediately after the first valid model is found, solving the problem well within the time limit.

### The Three-Step Action Modeling Pattern (CRITICAL for temporal problems)

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

### Passing Data to ASP

```python
def generate_facts(problem_data):
    facts = []
    for item in problem_data["items"]:
        facts.append(f'item("{item}").')
    return " ".join(facts)
```

### Extracting Solutions

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

### CRITICAL: The Model Object Is Only Valid Inside on_model (Segfault Risk)

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

#### C. Debugging Temporal Models

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
   - Isolate: Use "teleportation" test (bypass actions)

### 4.2 Assignment Problems

**Pattern: Worker-Task Assignment**
```asp
% Each task assigned to exactly one worker
1 { assigned(T,W) : worker(W) } 1 :- task(T).

% Worker capacity constraint
:- worker(W), #count { T : assigned(T,W) } > capacity(W).
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

For scheduling, planning, and any temporal problem, you MUST define a finite time domain. Without this, the solver may attempt to explore an infinite search space, leading to grounding errors (`#inf@0`) or extreme slowness.

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
- The grounder will produce `#inf@0` errors or create enormous search spaces
- Always use the latest deadline or a reasonable upper bound as your time horizon

### 4.5 Temporal Reasoning with Frame Axioms

**Critical for**: Robot planning, logistics, action sequences

**Complete Pattern for Temporal Problems:**

```asp
% 1. ENTITIES AND TIME
robot(r1). item(i1). time(0..max_time).
location(1..5, 1..5).

% 2. INITIAL STATE
robot_at(r1,1,1,0).
item_at(i1,2,2,0).

% 3. ACTION GENERATION (Choice Rules)
{ move(R,Dir,T) : direction(Dir) } 1 :- robot(R), time(T), T < max_time.
{ pickup(R,I,T) : item(I) } 1 :- robot(R), time(T), T < max_time.

% 4. MUTUAL EXCLUSION (State Constraints)
% Robot cannot be in two places
:- robot_at(R,X1,Y1,T), robot_at(R,X2,Y2,T), (X1,Y1) != (X2,Y2).

% Item cannot be both at location and carried
:- item_at(I,X,Y,T), carrying(_,I,T).

% 5. ACTION PRECONDITIONS
% Cannot move outside grid
:- move(R,up,T), robot_at(R,X,Y,T), Y >= max_y.

% Cannot pickup from different location
:- pickup(R,I,T), robot_at(R,RX,RY,T), item_at(I,IX,IY,T), (RX,RY) != (IX,IY).

% 6. ACTION EFFECTS
% Movement changes robot position
robot_at(R,X,Y+1,T+1) :- move(R,up,T), robot_at(R,X,Y,T).

% Pickup creates carrying relationship
carrying(R,I,T+1) :- pickup(R,I,T).

% Putdown creates item location
item_at(I,X,Y,T+1) :- putdown(R,I,T), robot_at(R,X,Y,T).

% 7. FRAME AXIOMS (Persistence)
% Robot position persists if no movement
robot_at(R,X,Y,T+1) :- robot_at(R,X,Y,T), time(T+1),
    not move(R,up,T), not move(R,down,T), 
    not move(R,left,T), not move(R,right,T).

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
top(D, P, T) :- on_peg(D, P, T), not on_peg(D2, P, T), D2 < D.

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
on_peg(D, P, T+1) :- on_peg(D, P, T), time(T+1), not move(D, P, _, T).

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
clear(B,T) :- block(B), not on(_, B, T).
on_table_space(T) :- #count { B : on_table(B,T) } < max_table_slots.

% Exactly one move per timestep
1 { move_to_block(B1, B2, T) : clear(B1,T), clear(B2,T), B1 != B2 ;
    move_to_table(B, T) : clear(B,T), on_table_space(T) } 1 :- time(T).
```

**General Takeaway:** For sequential planning, always:
- Define valid actions via auxiliary predicates
- Use `1 { ... } 1 :- time(T)` for exactly one action per timestep
- Avoid generate-and-filter patterns in choice rules
- Eliminate global #count constraints over entire plans

### 4.10 Sequence and Run-Based Constraints

**Critical for**: Puzzles with consecutive sequences (nonograms, nurse scheduling, delivery routes), problems requiring runs of items with specific properties.

Modeling sequences or "runs" of consecutive items is common in constraint problems. A critical distinction exists between problems where ALL items must justify themselves vs. problems where only certain items need to form runs.

#### Understanding "Runs" in Different Problem Types

**Run Definition**: A consecutive sequence of cells/slots with the same property (color, state, assignment).

**Two Fundamentally Different Patterns:**

1. **Simple Binary Problems** (e.g., black-and-white nonogram):
   - Every "on" cell MUST belong to a defined run
   - White cells are just separators
   - Total coverage constraint is VALID

2. **Multi-State Problems** (e.g., colored nonogram, multi-skill nurse scheduling):
   - Cells/slots of one state can act as separators for runs of another state
   - Only run-forming cells need to justify themselves
   - Total coverage constraint is INVALID

#### Anti-Pattern: Incorrect Total Coverage

A common error is applying simple binary logic to multi-state problems:

```asp
% ANTI-PATTERN for multi-color/multi-state problems
% This says: "Every non-empty cell MUST be part of a defined run"
% This is WRONG when cells of one color/state can separate runs of another
:- cell(R, C, Color), Color != empty, not is_in_a_run(R, C, Color).

% Example failure case:
% Row clue: [(red, 2), (red, 2)]
% Valid solution: [red, red, green, red, red]
% The green cell at index 2 is NOT in any green run, but it's REQUIRED
% to separate the two red runs. The constraint incorrectly forbids this!
```

**Why This Fails:**
- In multi-state problems, non-run items serve as separators
- Requiring every item to justify itself over-constrains the model
- Creates false UNSAT or forces invalid solutions

#### Correct Pattern: Constrain Runs, Not Gaps

The correct approach focuses ONLY on enforcing run properties, letting the solver determine separators naturally:

```asp
% CORRECT PATTERN: Only constrain the runs themselves

% 1. Define run positions (generate)
{ run_pos(Row, RunIdx, StartCol) : possible_start(StartCol) } 1
    :- row_clue(Row, RunIdx, _, _).

% 2. Enforce run cells match the required color/state (constrain)
:- run_pos(Row, RunIdx, Start), row_clue(Row, RunIdx, RequiredColor, Len),
   col(C), C >= Start, C < Start + Len,
   cell(Row, C, ActualColor), ActualColor != RequiredColor.

% 3. Enforce ordering (runs appear in sequence)
:- run_pos(R, I1, C1), run_pos(R, I2, C2),
   row_clue(R, I1, _, L1), I2 = I1 + 1, C2 < C1 + L1.

% 4. Enforce separation for same-color/state runs only
%    (Different colors/states CAN be adjacent)
:- run_pos(R, I1, C1), run_pos(R, I2, C2),
   row_clue(R, I1, Color, L1), row_clue(R, I2, Color, _),
   I1 < I2, C2 < C1 + L1 + 1.  % Same color needs gap

% NOTE: NO constraint on cells outside of runs!
% The solver determines their values to satisfy other constraints
```

#### Pattern Comparison Table

| Problem Type | Run Coverage | Separation Rules | Example Domains |
|:-------------|:-------------|:-----------------|:----------------|
| **Binary** | All "on" cells in runs | All runs need separation | B&W Nonogram, Simple scheduling |
| **Multi-State** | Only run-forming cells in runs | Same-state runs need separation<br/>Different-state runs can be adjacent | Colored Nonogram, Multi-skill scheduling, Delivery with vehicle types |

#### Complete Multi-Color Nonogram Pattern

```asp
% Facts
row(1..n). col(1..m).
color(0..k).  % 0 = empty/white
row_clue(Row, RunIdx, Color, Length) :- ...  % From problem data
col_clue(Col, RunIdx, Color, Length) :- ...

% Grid - each cell gets exactly one color
{ cell(R, C, Color) : color(Color) } 1 :- row(R), col(C).

% ROW CONSTRAINTS
% Define possible starting positions for runs
possible_col(C) :- col(C).

% Each run clue gets exactly one starting position
{ row_run_pos(R, RunIdx, StartCol) : possible_col(StartCol) } 1
    :- row_clue(R, RunIdx, _, _).

% Cells within run must match run's color
:- row_run_pos(R, RunIdx, Start), row_clue(R, RunIdx, ReqColor, Len),
   col(C), C >= Start, C < Start + Len,
   cell(R, C, ActualColor), ActualColor != ReqColor.

% Runs must appear in order
:- row_run_pos(R, I1, C1), row_run_pos(R, I2, C2),
   row_clue(R, I1, _, L1), I2 = I1 + 1, C2 < C1 + L1.

% Runs of SAME color must be separated (different colors can touch)
:- row_run_pos(R, I1, C1), row_run_pos(R, I2, C2),
   row_clue(R, I1, SameColor, L1), row_clue(R, I2, SameColor, _),
   I1 < I2, C2 < C1 + L1 + 1.

% Repeat for column constraints...
% (Same pattern with row/col swapped)
```

#### When to Use This Pattern

**Use Correct (Constrain Runs Only) Pattern When:**
- Multiple colors/states can coexist
- Different types can be adjacent
- Separators have meaningful values
- Problem has multi-dimensional constraints

**Examples:**
- Colored nonograms
- Multi-skill nurse scheduling
- Vehicle routing with different vehicle types
- Resource allocation with multiple resource types

#### Key Takeaways

1. **Never require total coverage** in multi-state sequence problems
2. **Only constrain run properties**: position, length, internal consistency
3. **Separate by sameness**: Only runs of identical color/state need gaps
4. **Let solver determine gaps**: Don't over-constrain non-run cells
5. **Different is adjacent**: Cells of different colors/states CAN touch

### 4.11 "Reachable Only After" / Ordered Reachability Constraints

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

## Section 5: Debugging & Advanced Techniques

### 5.1 Efficient Generation: Constrain Choices Early (Anti-Pattern: Generate-and-Filter)

A common performance pitfall is to generate a massive number of candidate solutions in a choice rule and then use separate integrity constraints to filter out the vast majority of invalid ones. This forces the grounder and solver to consider many useless options.

**The Principle: Constrain at the Source.** Build validity checks directly into the generating choice rule itself, so the solver only ever considers valid candidates.

**Anti-Pattern: Generate, then Filter**
This approach generates all combinations and then prunes them.

```asp
% DOMAIN: 100 items, 50 containers. compatible(I,C) defines valid pairs.
item(1..100). container(1..50).
compatible(1,c1). compatible(2,c1). %... many other compatibility facts

% ANTI-PATTERN: Generate all 100*50=5000 possible assignments
{ placed(I,C) : container(C) } 1 :- item(I).

% Then, filter out the invalid ones
:- placed(I,C), not compatible(I,C).
```
**Why it's slow:** The solver must first consider 5,000 potential `placed/2` atoms before eliminating most of them.

**Critical Note: Beyond Performance, This is a Correctness Issue**

The "Generate-and-Filter" anti-pattern is not just slow; it can lead to confusing `UNSATISFIABLE` results. If the choice rule generates a candidate for which the filtering data does not exist, the solver may be forced into an impossible state.

**Example: Assigning Features to Devices**
Suppose you must assign a `feature(F)` to a `device(D)`, but only if a `driver(F, D)` fact exists.

**Anti-Pattern (Brittle and Confusing):**
```asp
% For each feature, choose a device to assign it to.
1 { assign(F, D) : device(D) } 1 :- feature(F).

% Constraint: An assignment is only valid if a driver exists.
:- assign(F,D), not driver(F,D).
```
**The Hidden Problem:** If you have a `feature(f_special)` for which **no** `driver(f_special, _)` facts exist, this model becomes UNSAT. The choice rule forces an assignment `assign(f_special, D)` for some device `D`. But the constraint then makes every possible choice invalid, leading to failure with no obvious reason why.

By constraining at the source, you avoid this problem entirely:
```asp
% CORRECT: Only generate assignments where a driver exists
1 { assign(F, D) : device(D), driver(F, D) } 1 :- feature(F).
% No separate constraint needed - only valid pairs are considered
```

**Good Pattern: Constrain During Generation**
This approach only generates combinations that are already known to be valid.

```asp
% DOMAIN: Same as above.
item(1..100). container(1..50).
compatible(1,c1). compatible(2,c1). %...

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
oneway_pair(R1, R2) :- R1 < R2,
    ( (conn(R1, R2, _), not conn(R2, R1, _)) ;
      (conn(R2, R1, _), not conn(R1, R2, _)) ).

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

| Goal                                   | ASP Pattern                                                         | Explanation                                            |
| :------------------------------------- | :------------------------------------------------------------------ | :----------------------------------------------------- |
| **Assign a unique property**           | `1 { has_property(X, P) : property(P) } 1 :- entity(X).`             | Each entity `X` must have **exactly one** property `P`.|
| **Model an optional, unique connection** | `0 { connected(X, Y, Type) : conn_type(Type) } 1 :- pair(X, Y).`     | There is **at most one** connection between `X` and `Y`.|
| **Select a fixed number of items**     | `k { selected(I) : item(I) } k :- ...`                              | **Exactly `k`** items must be selected from the domain. |
| **Select a minimum number of items**   | `k { selected(I) : item(I) } :- ...`                                | **At least `k`** items must be selected.               |

### 5.3 Avoiding Grounding Explosion: Decompose Complex Arithmetic (Anti-Pattern: Monolithic Arithmetic)

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
sum_ab(S) :- S = #sum { N,V : assign(V,N), (V=a; V=b) }.
sum_cd(S) :- S = #sum { N,V : assign(V,N), (V=c; V=d) }.

% Now, constrain the two resulting sums. This generates very few ground rules.
:- sum_ab(S1), sum_cd(S2), S1 != S2.
```
**Why it's fast:** The `#sum` aggregate is highly optimized. The solver computes each sum just once. The final constraint only compares the two resulting sum values, avoiding the combinatorial explosion.

**General Takeaway:** Instead of writing complex arithmetic expressions with many variables in a single rule body, use aggregates to compute sub-expressions and constrain their results.

### Debugging Temporal Models (CRITICAL)

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
   # Show grounded rules
   ctl.ground([("base", [])])
   for atom in ctl.symbolic_atoms:
       if "item_at" in str(atom):
           print(atom)
   ```

### Common Errors and Solutions

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

### Best Practices

1. **Start with state model**: Define fluents and exclusivity first
2. **Test incrementally**: Add one action type at a time
3. **Use meaningful names**: `robot_at(r1,3,4,5)` not `p(1,3,4,5)`
4. **Document assumptions**: Comment complex constraints
5. **Verify exclusivity**: Always enforce mutual exclusion for fluents