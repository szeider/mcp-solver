
# MCP Solver
[![MCP][mcp-badge]][mcp-url][![License: MIT][license-badge]][license-url]![Python Version][python-badge]

A Model Context Protocol (MCP) server that exposes MiniZinc constraint solving capabilities to Large Language Models.

---

## Overview

The MCP Solver integrates MiniZinc constraint programming with LLMs through the Model Context Protocol, enabling AI models to:
- Submit and validate constraint models.
- Set model parameters.
- Solve constraint satisfaction problems.
- Retrieve and analyze solutions.

---

## Features

- Full MiniZinc constraint model support.
- Asynchronous solving with timeout management.
- Parameter validation and type checking.
- Solution state management.
- Automatic library inclusion based on constraints.
- Progress tracking and solver state monitoring.

---

## Installation

### Standard Installation
Install the package directly from GitHub:
```
uv pip install git+https://github.com/szeider/mcp-solver.git
```

### Development Installation
For local development and testing:
```
git clone https://github.com/szeider/mcp-solver.git
cd mcp-solver
uv pip install -e .
```

---

## System Requirements

- Python 3.9+
- MiniZinc with Chuffed solver.
- Operating system: Linux, macOS, or Windows.

---


## Example 1
This is based on an  example by George Johnson (NYT, July 13, 1999).

**User**

> Suppose that a theatrical director feels obligated to cast either his ingenue, Actress Alvarez, or his nephew, Actor Cohen, in a production. But Miss Alvarez won't be in a play with Mr. Cohen (her former lover), and she demands that the cast include her new flame, Actor Davenport. The producer, with her own favors to repay, insists that Actor Branislavsky have a part. But Mr. Branislavsky won't be in any play with Miss Alvarez or Mr. Davenport. Can the director cast the play?

**LLM**

> Let's solve this step by step using a constraint satisfaction model.
>
> **Theater Casting Constraint Model**
>
> ```minizinc
> % Variables for each actor (1 if cast, 0 if not)
> var 0..1: alvarez;
> var 0..1: branislavsky; 
> var 0..1: cohen;
> var 0..1: davenport;
> 
> % Must cast either Alvarez or Cohen
> constraint alvarez + cohen >= 1;
> 
> % Alvarez won't be in play with Cohen
> constraint alvarez + cohen <= 1;
> 
> % If Alvarez is cast, Davenport must be cast
> constraint alvarez <= davenport;
> 
> % Branislavsky must be cast  
> constraint branislavsky = 1;
> 
> % Branislavsky won't be with Alvarez or Davenport
> constraint branislavsky + alvarez <= 1;
> constraint branislavsky + davenport <= 1;
> 
> solve satisfy;
> ```

**MCP Solver:** **`submit_model`**

> `Success: Model validated, and parameters initialized dynamically.`

**MCP Solver:** **`solve_model`**

> ```json
> {'status': 'SUCCESS', 'solution': Solution(alvarez=0, branislavsky=1, cohen=1, davenport=0, _checker=''), 'solve_time': 0.0}
> ```

**LLM** 

> The director can cast the play by:
>
> * Casting Cohen and Branislavsky 
> * Not casting Alvarez or Davenport
>
> This satisfies all constraints:
>
> 1. Cohen is cast (Alvarez-or-Cohen requirement)  
> 2. Branislavsky is cast (producer's demand)
> 3. No incompatible pairs appear together

---

## Example 2

This example shows how the same model can be solved with different parameter values, and the solving times can be obtained.

**User**

> Check whether you can place n Queens on an nxn chessboard. Try N=10,20,30,40 and compare the solving times.

**LLM**

> ```minizinc
> include "globals.mzn";
> 
> int: n; 
> array[1..n] of var 1..n: queens;
> 
> constraint alldifferent(queens);
> constraint alldifferent([ queens[i] + i | i in 1..n ]);  
> constraint alldifferent([ queens[i] - i | i in 1..n ]);
> ```

**MCP Solver:** **`submit_model`**

> `Success: Model validated, and parameters initialized dynamically.`

**LLM**

> Testing N=10:

**MCP Solver:** **`set_parameter`**

> `Parameter n set successfully`

**MCP Solver:** **`solve_model`**  

> `{'status': 'SUCCESS', 'solution': Solution(queens=[3, 6, 9, 1, 8, 4, 2, 7, 5, 10], _checker=''), 'solve_time': 0.001}`

**LLM**

> N=20:

**MCP Solver:** **`set_parameter`**

> `Parameter n set successfully`

**MCP Solver:** **`solve_model`**

> `{'status': 'SUCCESS', 'solution': Solution(queens=[4, 18, 16, 11, 13, 1, 3, 19, 8, 12, 17, 2, 10, 15, 6, 14, 9, 7, 5, 20], _checker=''), 'solve_time': 0.004}`  

**LLM**

> N=30:

**MCP Solver:** **`set_parameter`**

> `Parameter n set successfully`

**MCP Solver:** **`solve_model`**

> `{'status': 'SUCCESS', 'solution': Solution(queens=[17, 23, 9, 18, 2, 13, 8, 1, 24, 21, 6, 22, 11, 26, 5, 25, 19, 14, 10, 3, 29, 27, 15, 4, 12, 20, 16, 7, 28, 30], _checker=''), 'solve_time': 0.04}`

**LLM** 

> N=40:

**MCP Solver:** **`set_parameter`**

> `Parameter n set successfully`

**MCP Solver:** **`solve_model`**  

> `{'status': 'SUCCESS', 'solution': Solution(queens=[12, 34, 23, 8, 20, 31, 15, 27, 18, 33, 5, 2, 26, 10, 16, 32, 4, 11, 14, 3, 9, 25, 28, 30, 35, 6, 1, 17, 21, 37, 13, 29, 24, 19, 7, 22, 39, 36, 38, 40], _checker=''), 'solve_time': 0.042}`

**LLM**

> Solutions exist for N=10, 20, 30, 40. Solve times:
>
> - N=10: 0.001s 
> - N=20: 0.004s
> - N=30: 0.040s
> - N=40: 0.042s
>
> Time complexity appears roughly quadratic.

## 

**Further sample chat dialogs demonstrating tool usage are included in the `examples` folder of the repository.** 

---

## Available Tools (to be extended)

| Tool Name         | Description                                           |
|--------------------|-------------------------------------------------------|
| `submit-model`     | Submit and validate a MiniZinc model.                |
| `solve-model`      | Solve the current constraint model.                  |
| `get-solution`     | Retrieve the last solution.                          |
| `set-parameter`    | Set a model parameter value.                         |
| `get-variable`     | Get a variable's value from the solution.            |
| `get-solve-time`   | Get solution computation time.                       |
| `get-solver-state` | Check solver status.                                 |

---

## Disclaimer

This MCP Solver is in its prototype stage and should be used with caution. Users are encouraged to experiment, but any use in critical environments is at their own risk.

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
