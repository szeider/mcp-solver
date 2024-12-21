
# MCP Solver
[![MCP][mcp-badge]][mcp-url][![License: MIT][license-badge]][license-url]![Python Version][python-badge]

A Model Context Protocol (MCP) server that exposes MiniZinc constraint solving capabilities to Large Language Models.

---
## Overview
The MCP Solver integrates MiniZinc constraint programming with LLMs through the Model Context Protocol, enabling AI models to:

* Create, edit and validate constraint models
* Execute constraint solving operations
* Access and update solution knowledge
* Manage solver insights through a memo system

## Features
* Finite domain and global constraint support
* Asynchronous solving with configurable timeouts
* Line-based model editing
* Solution state management
* Knowledge base maintenance

## Available Tools

| Tool Name        | Description                              |
| ---------------- | ---------------------------------------- |
| `get_model`      | Get current constraint model content     |
| `edit_model`     | Edit model with line-based modifications |
| `validate_model` | Check model syntax and semantics         |
| `solve_model`    | Execute solver with timeout management   |
| `get_variable`   | Get variable values from solution        |
| `get_solve_time` | Get solution computation time            |
| `get_memo`       | Access knowledge base content            |
| `edit_memo`      | Update knowledge base entries            |




---

## System Requirements

- Python 3.9+
- [MiniZinc](https://www.minizinc.org) with Chuffed solver
- Operating system: 
  - macOS
  - Windows 
  - Linux (requires an alternative to the Claude Dekstop app)

## Installation

1. Install the Claude Desktop app (or a different MCP client)

	https://claude.ai/download

2. Install the MCP Solver

	```
	git clone https://github.com/szeider/mcp-solver.git
	cd mcp-solver
	uv pip install -e .
	```

3. Create the file 

  ```
  ~/Library/Application/Support/Claude/claude_desktop_config.json
  ```
  containing
  ```
  {
   "mcpServers": {
       "minizinc": {
           "command": "uv",
           "args": ["--directory", 
           "/absolute/path/to/mcp-solver", 
           "run", 
           "mcp-solver"]
       }
     }
  }
   // For Windows, replace "/absolute/path/to/mcp-solver" with "C:\\absolute\\path\\to\\mcp-solver"
  
  ```

4. Path to `memo.md`

The default location can be changed in `pyproject.toml`.

## Examples

### Casting

This is based on an  example by George Johnson (NYT, July 13, 1999).

**User**

> Suppose that a theatrical director feels obligated to cast either his ingenue, Actress Alvarez, or his nephew, Actor Cohen, in a production. But Miss Alvarez won’t be in a play with Mr. Cohen (her former lover), and she demands that the cast include her new flame, Actor Davenport. The producer, with her own favors to repay, insists that Actor Branislavsky have a part. But Mr. Branislavsky won’t be in any play with Miss Alvarez or Mr. Davenport. Can the director cast the play?

**LLM**

> Let me help solve this casting constraint satisfaction problem. Let me break this down systematically and create a MiniZinc model.
>
> The key constraints are:
>
> - Must cast either Alvarez or Cohen
> - If Alvarez is cast, Davenport must be cast
> - If Alvarez is cast, Cohen cannot be cast
> - Branislavsky must be cast
> - If Branislavsky is cast, neither Alvarez nor Davenport can be cast
>
> Let me create a model with boolean variables for each actor:
>
> ```minizinc
> % Variables for each actor (1 if cast, 0 if not)
> var bool: alvarez;  % Actress Alvarez
> var bool: cohen;    % Actor Cohen (nephew)
> var bool: branislavsky;  % Actor Branislavsky
> var bool: davenport;     % Actor Davenport
> 
> % Must cast either Alvarez or Cohen
> constraint alvarez \/ cohen;
> 
> % If Alvarez is cast, Davenport must be cast
> constraint alvarez -> davenport;
> 
> % If Alvarez is cast, Cohen cannot be cast
> constraint not (alvarez /\ cohen);
> 
> % Branislavsky must be cast
> constraint branislavsky;
> 
> % If Branislavsky is cast, neither Alvarez nor Davenport can be cast
> constraint branislavsky -> (not alvarez /\ not davenport);
> 
> solve satisfy;
> ```

**System:** **`edit_model`**

> `Success: Model validated, and parameters initialized dynamically.`

**System:** **`solve_model`**

> `{'status': 'SUCCESS', 'solution': Solution(alvarez=0, branislavsky=1, cohen=1, davenport=0, _checker=''), 'solve_time': 0.0}`

**LLM**

> Yes, the director can cast the play! Here's the solution:
>
> - **Actor Cohen:** Cast ✓
> - **Actor Branislavsky:** Cast ✓
> - **Actress Alvarez:** Not cast
> - **Actor Davenport:** Not cast
>
> This solution satisfies all constraints:
>
> - Cohen is cast, satisfying the requirement to cast either Alvarez or Cohen
> - Since Alvarez is not cast, we don't need to cast Davenport
> - The conflict between Alvarez and Cohen is avoided since Alvarez isn't cast
> - Branislavsky is cast as required
> - Since Branislavsky is cast, neither Alvarez nor Davenport is cast, respecting his preferences
>
> Do you have any questions about this solution or would you like to explore any variations of the casting constraints?

------

### N-Queens

This example shows how the same model can be solved with different parameter values, and the solving times can be obtained.

**User**

> Check whether you can place n Queens on an nxn chessboard. Try n=10,20,30,40 and compare the solving times.EditI'll help solve the N-Queens problem for different board sizes and compare solving times.

**LLM**

> I'll help solve the n-Queens problem for different board sizes and compare solving times.

**System:** **`edit_model`**

> ```minizinc
> int: n;
> array[1..n] of var 1..n: queens; % queens[i] represents the row position of queen in column i
> 
> % No two queens in same row
> constraint alldifferent(queens);
> 
> % No two queens in same diagonal
> constraint forall(i,j in 1..n where i < j) (
>  abs(queens[i] - queens[j]) != j - i
> );
> 
> solve satisfy;
> ```

**System:** **`edit_model`**

> `Model updated`

**LLM**

> Let me validate and solve this model:

**System:** **`validate_model`**

> `{}`  
> `Valid: Model is valid`

**System:** **`solve_model`**

> `{'status': 'SAT', 'solution': Solution(queens=[4, 6, 3, 9, 2, 8, 5, 7, 10, 1], _checker=''), 'solve_time': 0.005}`

**LLM**

> Let's try n=20:

**System:** **`edit_model`**

> ```minizinc
> n = 20;
> int: n;
> array[1..n] of var 1..n: queens;
> 
> constraint alldifferent(queens);
> constraint forall(i,j in 1..n where i < j) (
>  abs(queens[i] - queens[j]) != j - i
> );
> 
> solve satisfy;
> ```

**System:** **`edit_model`**

> `Model updated`

**System:** **`solve_model`**

> `{'status': 'SAT', 'solution': Solution(queens=[5, 12, 8, 11, 17, 6, 16, 2, 4, 9, 19, 14, 3, 10, 18, 13, 15, 7, 20, 1], _checker=''), 'solve_time': 0.055}`

**LLM**

> Let's try n=30:

**System:** **`edit_model`**

> ```minizinc
> n = 30;
> int: n;
> array[1..n] of var 1..n: queens;
> 
> constraint alldifferent(queens);
> constraint forall(i,j in 1..n where i < j) (
>  abs(queens[i] - queens[j]) != j - i
> );
> 
> solve satisfy;
> ```

**System:** **`edit_model`**

> `Model updated`

**System:** **`solve_model`**

> `{'status': 'SAT', 'solution': Solution(queens=[11, 4, 7, 13, 10, 29, 27, 15, 21, 26, 22, 20, 8, 28, 5, 12, 23, 9, 3, 18, 24, 2, 6, 17, 14, 25, 19, 16, 30, 1], _checker=''), 'solve_time': 0.094}`

**LLM**

> Finally, let's try n=40:

**System:** **`edit_model`**

> ```minizinc
> n = 40;
> int: n;
> array[1..n] of var 1..n: queens;
> 
> constraint alldifferent(queens);
> constraint forall(i,j in 1..n where i < j) (
>  abs(queens[i] - queens[j]) != j - i
> );
> 
> solve satisfy;
> ```

**System:** **`edit_model`**

> `Model updated`

**System:** **`solve_model`**

> `{'status': 'SAT', 'solution': Solution(queens=[14, 17, 26, 18, 38, 36, 6, 37, 30, 34, 9, 40, 1, 21, 31, 2, 39, 10, 22, 20, 33, 16, 3, 29, 12, 7, 5, 19, 35, 32, 27, 4, 28, 23, 25, 15, 13, 11, 24, 8], _checker=''), 'solve_time': 0.289}`

**LLM**

> Here are the solving times for different board sizes:
>
> - **n=10:** 0.005 seconds
> - **n=20:** 0.055 seconds
> - **n=30:** 0.094 seconds
> - **n=40:** 0.289 seconds
>
> We can observe that:
>
> - All instances were solvable, which aligns with the knowledge base stating that n-Queens has solutions for all n≥4
> - The solving time increases non-linearly with the board size
> - Even for n=40, the solver found a solution relatively quickly (under 0.3 seconds)
>
> The quick solving times suggest we could try even larger values of N if you're interested. 

---

### TSP

**User:**

> A saleswoman based in Vienna needs to plan her upcoming tour through Austria, visiting each province capital once. Help find the shortest route. Distances in km:
>
> 1=Vienna, 2=St. Pölten, 3=Eisenstadt, 4=Linz, 5=Graz, 6=Klagenfurt, 7=Salzburg, 8=Innsbruck, 9=Bregenz
>
> |      | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
> | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
> | 1    | 0    | 65   | 60   | 184  | 195  | 319  | 299  | 478  | 631  |
> | 2    | 65   | 0    | 125  | 119  | 130  | 254  | 234  | 413  | 566  |
> | 3    | 60   | 125  | 0    | 184  | 157  | 281  | 261  | 440  | 593  |
> | 4    | 184  | 119  | 184  | 0    | 208  | 252  | 136  | 315  | 468  |
> | 5    | 195  | 130  | 157  | 208  | 0    | 136  | 280  | 459  | 629  |
> | 6    | 319  | 254  | 281  | 252  | 136  | 0    | 217  | 391  | 566  |
> | 7    | 299  | 234  | 261  | 136  | 280  | 217  | 0    | 188  | 343  |
> | 8    | 478  | 413  | 440  | 315  | 459  | 391  | 188  | 0    | 157  |
> | 9    | 631  | 566  | 593  | 468  | 629  | 566  | 343  | 157  | 0    |

**LLM:**
> I'll help formulate this as a Traveling Salesperson Problem (TSP) using MiniZinc. We need to find the shortest route that visits each capital exactly once and returns to Vienna.
>
> ```minizinc
> % TSP for Austrian province capitals
> int: n = 9;  % number of cities
> array[1..n,1..n] of int: dist = [| 
> 0, 65, 60, 184, 195, 319, 299, 478, 631|
> 65, 0, 125, 119, 130, 254, 234, 413, 566|
> 60, 125, 0, 184, 157, 281, 261, 440, 593|
> 184, 119, 184, 0, 208, 252, 136, 315, 468|
> 195, 130, 157, 208, 0, 136, 280, 459, 629|
> 319, 254, 281, 252, 136, 0, 217, 391, 566|
> 299, 234, 261, 136, 280, 217, 0, 188, 343|
> 478, 413, 440, 315, 459, 391, 188, 0, 157|
> 631, 566, 593, 468, 629, 566, 343, 157, 0|];
> 
> array[1..n] of var 1..n: tour;  % sequence of cities
> var int: total_distance;  % objective to minimize
> 
> % Each city must be visited exactly once (permutation)
> include "alldifferent.mzn";
> constraint alldifferent(tour);
> 
> % Start from Vienna (city 1)
> constraint tour[1] = 1;
> 
> % Calculate total distance
> constraint total_distance = sum(i in 1..n-1)(
>  dist[tour[i], tour[i+1]]
> ) + dist[tour[n], tour[1]];  % Add return to Vienna
> 
> solve minimize total_distance;
> ```

**System:** **`edit_model`**
> Model updated

**System:** **`validate_model`**
> Valid: Model is valid

**System:** **`solve_model`**
> {'status': 'SAT', 'solution': Solution(objective=1564, tour=[1, 3, 5, 6, 8, 9, 7, 4, 2], total_distance=1564, _checker=''), 'solve_time': 0.185}

**LLM:**
> I've found the optimal route! Here's the sequence of cities to visit:
>
> 1. Vienna (start)
> 2. Eisenstadt
> 3. Graz
> 4. Klagenfurt
> 5. Innsbruck
> 6. Bregenz
> 7. Salzburg
> 8. Linz
> 9. St. Pölten
> 10. Return to Vienna
>
> The total distance for this route is 1,564 km. This solution makes intuitive sense because:
>
> - It starts by visiting the nearby eastern cities first (Eisenstadt)
> - Then moves south (Graz, Klagenfurt)
> - Makes an efficient sweep through the western cities (Innsbruck, Bregenz)
> - Returns via the central route (Salzburg, Linz)
> - Finishes with the closest city to Vienna (St. Pölten)
>
> Would you like me to analyze any specific aspects of this solution or explore alternative routes?

**User:**
> Assume the road between Vienna and Eisenstadt is blocked, what is now the shortest tour?

**LLM:**
> I'll modify the model to exclude the direct connection between Vienna (1) and Eisenstadt (3) by setting it to a very high value.

**System:** **`edit_model`**
> Model updated

**System:** **`validate_model`**
> Valid: Model is valid

**System:** **`solve_model`**
> {'status': 'SAT', 'solution': Solution(objective=1694, tour=[1, 4, 7, 9, 8, 6, 5, 3, 2], total_distance=1694, _checker=''), 'solve_time': 0.166}

**LLM:**
> I've found the new optimal route with the blocked road. Here's the sequence:
>
> 1. Vienna (start)
> 2. Linz
> 3. Salzburg
> 4. Bregenz
> 5. Innsbruck
> 6. Klagenfurt
> 7. Graz
> 8. Eisenstadt
> 9. St. Pölten
> 10. Return to Vienna
>
> The total distance is now 1,694 km, which is 130 km longer than the previous route. The tour now takes a different approach:
>
> - It heads west first through Linz instead of east
> - Makes a similar western loop through Salzburg, Bregenz, and Innsbruck
> - Returns via the southern cities (Klagenfurt, Graz)
> - Reaches Eisenstadt via Graz instead of directly from Vienna
> - Finishes with St. Pölten before returning to Vienna
>
> The solver found this alternative route that avoids the blocked Vienna-Eisenstadt road while minimizing the total distance.
> 

------

**Further sample chat dialogs demonstrating tool usage are included in the `examples` folder of the repository.** 

---

## Disclaimer

This MCP Solver is in its prototype stage and should be used with caution. Users are encouraged to experiment, but any use in critical environments is at their own risk.

---

## Changelog 

See the [CHANGELOG.md](./CHANGELOG.md) for details on version history and changes.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[mcp-badge]: https://img.shields.io/badge/MCP-Compatible-blue.svg
[mcp-url]: https://modelcontextprotocol.io
[license-badge]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://opensource.org/licenses/MIT
[python-badge]: https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue
[python-url]: https://www.python.org

---
