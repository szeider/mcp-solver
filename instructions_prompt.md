# Detailed Guidelines for MCP Solver Usage

Welcome to the MCP Solver Detailed Guidelines. This document provides comprehensive instructions on how to interact with the MCP Solver and its tools. (Place this file in the root folder of your project.)

## Overview

The MCP Solver integrates MiniZinc constraint solving capabilities with the Model Context Protocol, allowing large language models (LLMs) to create, modify, and solve constraint models. This guide explains:
- The structure of a MiniZinc model and what constitutes a MiniZinc item.
- The input and output for each available tool.
- Best practices for model modification and solution verification.

## MiniZinc Items and Model Structure

- **MiniZinc Item:**  
  A MiniZinc item is a single complete statement (for example, a variable declaration or a constraint) in MiniZinc. Each item typically ends with a semicolon and may include inline comments.  
  **Note:** Comments that appear on the same line as a MiniZinc statement are considered part of that item; they annotate the code rather than standing alone.

- **No Output Statements:**  
  Do not add any output statements to your model. The MCP Solver is designed to work only with declarations, constraints, and other definitions—not with output formatting.

- **Truncated Model Display:**  
  When using tools such as `get_model`, `add_item`, `delete_item`, or `replace_item`, the current model is returned in a truncated form. This truncation is solely for display purposes, to keep item indices consistent and the output concise.
  
- **indizes start at 0**
  
  You add items one by one, starting with index 0
  
  index=0, index=1, etc.
  
  

## Tool Input and Output Details

Each tool provided by the MCP Solver has defined inputs and outputs. Here is a summary:

1. **get_model**  
   - **Input:** No arguments.  
   - **Output:** A message containing the current model, with each item truncated to a fixed number of characters. This view is provided to keep item numbering consistent.

2. **add_item**  
   - **Input:**  
     - `index` (integer): The position at which to insert the new MiniZinc statement.  
     - `content` (string): The complete MiniZinc statement to add.  
   - **Output:**  
     - A message confirming that the item was added, plus the current model in truncated form.

3. **delete_item**  
   - **Input:**  
     - `index` (integer): The index of the item to delete.  
   - **Output:**  
     - A message confirming that the item was deleted, plus the updated model in truncated form.

4. **replace_item**  
   - **Input:**  
     - `index` (integer): The index of the item to replace.  
     - `content` (string): The new MiniZinc statement that will replace the existing one.  
   - **Output:**  
     - A message confirming that the item was replaced, along with the current model (truncated).

5. **clear_model**  
   - **Input:** No arguments.  
   - **Output:**  
     - A message indicating that the model has been cleared.

6. **solve_model**  
   - **Input:**  
     - `timeout` (number): Time in seconds allowed for solving (between 1 and 10 seconds).
   - **Output:**  
     - A JSON/dictionary object with:
       - **status:** e.g., `"SAT"`, `"UNSAT"`, or `"TIMEOUT"`.
       - **solution:** (if applicable) The solution object returned by the solver.
       - **solve_time:** The execution time of the solve operation.

7. **get_solution**  
   - **Input:**  
     - `variable_name` (string): The name of the variable whose value is to be retrieved.
     - Optional `indices` (array of integers): For retrieving a specific element if the variable is an array (1-based indexing).  
   - **Output:**  
     - A message with the value of the specified variable (or array element).

8. **get_solve_time**  
   - **Input:** No arguments.  
   - **Output:**  
     - A message with the last recorded solve execution time.

9. **get_memo**  
   - **Input:** No arguments.  
   - **Output:**  
     - The current content of the knowledge base (memo).

10. **edit_memo**  
    - **Input:**  
      - `line_start` (integer): The starting line number (1-based) to edit.
      - Optional `line_end` (integer or null): The ending line number for the edit.
      - `content` (string): The new content that will replace the specified lines.  
    - **Output:**  
      - A message indicating that the memo has been updated.

## Model Solving and Verification

- **Solution Verification:**  
  After a model is solved, the LLM should verify that the returned solution satisfies all constraints mentioned by the user. Ensure that the solver's output is consistent with your expectations.

## Model Modification Guidelines

- **Incremental Changes:**  
  When modifying your model, use `add_item`, `delete_item`, and `replace_item` to alter specific items rather than clearing the entire model. This helps preserve item numbering and consistency.

- **When to Clear the Model:**  
  Only use `clear_model` if the changes to the model are very extensive and starting over is necessary.
  
  You can start adding items without clearing the model first. You will get the current model from the tool as a feedback. You see if there are any unwanted items that you can delete if necessary.

## Final Notes

- **Review Return Information:**  
  Carefully review the return messages from each tool call. They contain both confirmation messages and the current model (when applicable) in a truncated format for clarity.

- **Consistent Structure:**  
  Remember that comments on a MiniZinc statement are part of the same item. This design ensures that any commentary related to a code statement is preserved with that statement.

- **Verification:**  
  Always verify the solution after a solve operation by checking that all user-specified constraints are satisfied.

- **Incremental Modifications:**  
  Modify the existing model rather than starting from scratch unless a complete overhaul is needed.

