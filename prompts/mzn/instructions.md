# MCP Solver – Quick Start Guide

Welcome to the MCP Solver. This document provides concise guidelines on how to interact with the MCP Solver. You have access to the essential tools for building and solving MiniZinc models.

## Overview

The MCP Solver integrates MiniZinc constraint solving with the Model Context Protocol, allowing you to create, modify, and solve constraint models. The following tools are available:

- **clear_model**
- **add_item**
- **replace_item**
- **delete_item**
- **solve_model**

These tools let you construct your model incrementally and solve it using the Chuffed solver.

## MiniZinc Items and Model Structure

- **MiniZinc Item:**  
  A MiniZinc item is a complete statement (e.g., a variable declaration or constraint) that typically ends with a semicolon. Inline comments are considered part of the same item.

- **No Output Statements:**  
  Do not include output formatting in your model. The solver handles only declarations and constraints.

- **Truncated Model Display:**  
  The model may be returned in a truncated form for brevity. This is solely for display purposes to keep item indices consistent.

- **Indices Start at 0:**  
  Items are added one by one, starting with index 0 (i.e., index=0, index=1, etc.).

## Tool Input and Output Details

1. **clear_model**  
   - **Input:** No arguments.  
   - **Output:** A confirmation message indicating that the model has been cleared.

2. **add_item**  
   - **Input:**  
     - `index` (integer): The position at which to insert the new MiniZinc statement.  
     - `content` (string): The complete MiniZinc statement to add.
   - **Output:**  
     - A confirmation message that the item was added, along with the current (truncated) model.

3. **replace_item**  
   - **Input:**  
     - `index` (integer): The index of the item to replace.  
     - `content` (string): The new MiniZinc statement that will replace the existing one.
   - **Output:**  
     - A confirmation message that the item was replaced, along with the updated (truncated) model.

4. **delete_item**  
   - **Input:**  
     - `index` (integer): The index of the item to delete.
   - **Output:**  
     - A confirmation message that the item was deleted, along with the updated (truncated) model.

5. **solve_model**  
   - **Input:**  
     - `timeout` (number): Time in seconds allowed for solving (between 1 and 30 seconds).
   - **Output:**  
     - A JSON object with:
       - **status:** `"SAT"`, `"UNSAT"`, or `"TIMEOUT"`.
       - **solution:** (If applicable) The solution object when the model is satisfiable.  
         In unsatisfiable or timeout cases, only the status is returned.

## Model Solving and Verification

- **Solution Verification:**  
  After solving, verify that the returned solution satisfies all specified constraints. If the model is satisfiable (`SAT`), you will receive both the status and the solution; otherwise, only the status is provided.

## Model Modification Guidelines

- **Comments**:
  A comment is not an item by itself. Always combine a comment with the constraint or declaration it belongs to.

- **Combining similar parts:**
  
  If you have a long list of similar parts like constant definitions, you can put them into the same item.
  
- **Incremental Changes:**  
  Use `add_item`, `replace_item`, and `delete_item` to modify your model incrementally. This allows you to maintain consistency in item numbering without needing to clear the entire model.

- **Making Small Changes:**
  When a user requests a small change to the model (like changing a parameter value or modifying a constraint), use `replace_item` to update just the relevant item rather than rebuilding the entire model. This maintains the model structure and is more efficient.

- **When to Clear the Model:**  
  Use `clear_model` only when extensive changes are required and starting over is necessary.

## Final Notes

- **Review Return Information:**  
  Carefully review the confirmation messages and the current model after each tool call.

- **Consistent Structure:**  
  Remember that comments on a MiniZinc statement (on the same line) are considered part of that item, ensuring any context or annotations remain with the statement.

- **Verification:**  
  Always verify the solution after a solve operation by checking that all constraints are met.

Happy modeling with MCP Solver!
