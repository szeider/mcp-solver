# MCP Solver – ASP (clingo) Quick Start Guide

Welcome to the MCP Solver. This document provides detailed guidelines for building and solving Answer Set Programming (ASP) models using the clingo Python API.

## Overview

The MCP Solver integrates ASP solving with the Model Context Protocol, allowing you to create, modify, and solve logic programs incrementally. The following tools are available:

- **clear_model**
- **add_item**
- **replace_item**
- **delete_item**
- **solve_model**

These tools let you construct your model item by item and solve it using clingo.

## ASP Model Items and Structure

- **ASP Item:**
  An ASP item is a complete fact, rule, or constraint (ending with a period). Inline comments are considered part of the same item.

- **No Output Statements:**
  Do not include output formatting in your model. The solver handles only facts, rules, and constraints.

- **Indices Start at 0:**
  Items are added one by one, starting with index 0 (i.e., index=0, index=1, etc.).

## List Semantics for Model Operations

The model items behave like a standard programming list with these exact semantics:

- **add_item(index, content)**: Inserts the item at the specified position, shifting all items at that index and after to the right.
  - Example: If model has items [A, B, C] and you call add_item(1, X), result is [A, X, B, C]
  - Valid index range: 0 to length (inclusive)

- **delete_item(index)**: Removes the item at the specified index, shifting all subsequent items to the left.
  - Example: If model has items [A, B, C, D] and you call delete_item(1), result is [A, C, D]
  - Valid index range: 0 to length-1 (inclusive)

- **replace_item(index, content)**: Replaces the item at the specified index in-place. No shifting occurs.
  - Example: If model has items [A, B, C] and you call replace_item(1, X), result is [A, X, C]
  - Valid index range: 0 to length-1 (inclusive)

**Important**: All indices are 0-based. The first item is at index 0, the second at index 1, etc.

**Critical: Index stability on errors**

- Indices only change when an operation succeeds. If `add_item`, `replace_item`, or `delete_item` returns an error, the model is unchanged and item indices remain exactly the same.
- Specifically for `add_item`: do not advance your intended insertion index after a failed call. Try again with the same index once the cause of the error is fixed.

## Tool Input and Output Details

1. **clear_model**
   - **Input:** No arguments.
   - **Output:** Confirmation that the model has been cleared.

2. **add_item**
   - **Input:**
     - `index` (integer): Position to insert the new ASP statement.
     - `content` (string): The complete ASP statement to add.
   - **Output:** Confirmation and the current (truncated) model.
   - **Index behavior on error:** If the call fails (e.g., invalid index, malformed content), the model is not modified and no indices shift. Do not increment your next `index` based on a failed attempt.

3. **replace_item**
   - **Input:**
     - `index` (integer): Index of the item to replace.
     - `content` (string): The new ASP statement.
   - **Output:** Confirmation and the updated (truncated) model.

4. **delete_item**
   - **Input:**
     - `index` (integer): Index of the item to delete.
   - **Output:** Confirmation and the updated (truncated) model.

5. **solve_model**
   - **Input:**
     - `timeout` (number): Time in seconds allowed for solving (between 1 and 30 seconds).
   - **Output:**
     - A JSON object with:
       - **status:** `"SAT"`, `"UNSAT"`, or `"TIMEOUT"`.
       - **solution:** (If applicable) The solution object when the model is satisfiable.

## Model Solving and Verification

- **Solution Verification:**
  After solving, verify that the returned solution satisfies all specified constraints. If the model is satisfiable (`SAT`), you will receive both the status and the solution; otherwise, only the status is provided.

## Model Modification Guidelines

- **Comments:**
  A comment is not an item by itself. Always combine a comment with the fact, rule, or constraint it belongs to.

- **Combining similar parts:**
  If you have a long list of similar facts or rules, you can put them into the same item.

- **Incremental Changes:**
  Use `add_item`, `replace_item`, and `delete_item` to modify your model incrementally. This allows you to maintain consistency in item numbering without needing to clear the entire model.

- **Making Small Changes:**
  When a user requests a small change to the model (like changing a parameter value or modifying a rule), use `replace_item` to update just the relevant item rather than rebuilding the entire model. This maintains the model structure and is more efficient.

- **When to Clear the Model:**
  Use `clear_model` only when extensive changes are required and starting over is necessary.

## Important: Model Item Indexing

ASP mode uses **0-based indexing** for all model operations:
- First item is at index 0
- Used with add_item, replace_item, delete_item
- Example: `add_item(0, "color(red).")` adds at the beginning
- Example: `replace_item(2, "edge(a,b).")` replaces the third item

## Blueprint: Recommended ASP Model Structure

A typical ASP model for MCP Solver should follow this structure:

1. **Facts and Data**: All problem-specific facts and data.
2. **Domain Declarations**: Define domains, constants, and sets.
3. **Rules**: Logical rules that define relationships and constraints.
4. **Integrity Constraints**: Constraints that must be satisfied (e.g., `:- condition.`).
5. **Optimization Statements** (if any): Use `#minimize` or `#maximize` as needed.

**Example:**
```asp
% Item 0: Facts
graph_node(a). graph_node(b). graph_node(c).
edge(a,b). edge(b,c).

% Item 1: Domain
domain_color(red).
domain_color(green).
domain_color(blue).

% Item 2: Rules
1 { color(N,C) : domain_color(C) } 1 :- graph_node(N).

% Item 3: Integrity Constraints
:- edge(N,M), color(N,C), color(M,C).

% Item 4: Optimization (optional)
#minimize { 1,N,C : color(N,C) }.
```

## Best Practices

- **Use clear, descriptive names** for predicates and variables.
- **Comment complex rules** for clarity.
- **Group related facts and rules** together.
- **Avoid redundant rules** and facts.
- **Test incrementally**: Add and solve small parts before building the full model.
- **Use integrity constraints** to enforce requirements.
- **Use optimization statements** only when required by the problem.

## Common Pitfalls

- **Forgetting periods** at the end of facts/rules.
- **Incorrect variable usage** (e.g., ungrounded variables).
- **Redundant or conflicting rules**.
- **Missing or incorrect integrity constraints**.
- **Improper use of optimization statements**.
- **Not checking for unsatisfiable models**.

## Minimal Working Example

Suppose you want to color a simple graph:

```asp
% Item 0: Facts
graph_node(a). graph_node(b). graph_node(c).
edge(a,b). edge(b,c).

% Item 1: Domain
domain_color(red).
domain_color(green).
domain_color(blue).

% Item 2: Rules
1 { color(N,C) : domain_color(C) } 1 :- graph_node(N).

% Item 3: Integrity Constraints
:- edge(N,M), color(N,C), color(M,C).

% Item 4: Optimization (optional)
#minimize { 1,N,C : color(N,C) }.
```

## Advanced ASP Constructs and Patterns

### Defaults and Exceptions (Negation-as-Failure)

- Encode defaults using `not` and override with explicit exceptions.
- Pattern:
```asp
flies(X) :- bird(X), not abnormal(X).
abnormal(X) :- penguin(X).
:- penguin(X), flies(X).
```
- Tips:
  - Place taxonomy rules first (e.g., `bird(X) :- penguin(X).`).
  - Keep defaults separate from integrity constraints that enforce exceptions.

### Negation-as-Failure for Eligibility Policies

- Derive permissive defaults, then constrain with explicit facts.
```asp
eligible(C) :- customer(C), not excluded(C).
eligible(C) :- vip(C), not blacklisted(C).
excluded(C) :- blacklisted(C).
:- eligible(C), excluded(C).
```
- Use integrity constraints to prevent contradictory conclusions.

### Recursive Aggregates (#sum)

- Aggregate over a recursively defined relation to compute thresholds.
```asp
controls(X,X) :- company(X).
contrib(A,B,A,P) :- owns(A,B,P).
contrib(A,B,C,P) :- controls(A,C), owns(C,B,P), A != C.
sum(A,B,S) :- S = #sum { P,C : contrib(A,B,C,P) }.
controls(A,B) :- sum(A,B,S), S > 50, A != B.
```
- Use helper predicates like `contrib/4` to keep aggregates readable.

### Weak Constraints (Optimization with :~)

- Prefer solutions that minimize penalties using weak constraints.
```asp
1 { assign(T,S) : slot(S) } 1 :- task(T).
:- assign(T,S), conflict(T,S).
:~ prefer(T,S,W), not assign(T,S). [W@1,T,S]
```
- Alternatively, use `#minimize` with weighted literals.
- Keep all hard constraints as `:- ...` and only preferences in weak constraints.

### Modeling UNSAT for Testing

- To intentionally create UNSAT, introduce contradictory defaults with integrity constraints.
```asp
p :- not not_p.
not_p :- not p.
:- p.
:- not p.
```
- Useful for verifying solver correctly reports `UNSAT`.

## Final Notes

- **Review return information** after each tool call.
- **Maintain a consistent structure** for easier debugging and review.
- **Verify solutions** after solving to ensure all constraints are met.

Happy modeling with MCP Solver and ASP (clingo)!
