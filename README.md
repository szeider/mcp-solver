# MCP Solver
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org)

A Model Context Protocol (MCP) server that exposes MiniZinc constraint solving capabilities to Large Language Models.

---

## Overview
The MCP Solver integrates MiniZinc constraint programming with LLMs through the Model Context Protocol, enabling AI models to:

* Create, edit, and validate constraint models
* Execute constraint solving operations
* Access and update solution knowledge
* Manage solver insights through a memo system

For a detailed description of the system architecture and theoretical foundations, see the accompanying research paper: Stefan Szeider, ["MCP-Solver: Integrating Language Models with Constraint Programming Systems"](https://arxiv.org/abs/2501.00539), arXiv:2501.00539, 2024.

## Features
* Finite domain and global constraint support
* Asynchronous solving with configurable timeouts
* Item-based model editing
* Solution state management
* Knowledge base maintenance

## Available Tools

Model modification has transitioned in this version from a line-based to an item-based editing approach, which improves robustness by validating each item independently. Each model modification operation returns the current model with numbered items in truncated form, ensuring consistent tracking of items and their indices. Line-based editing remains for handling the memo.

| Tool Name        | Description                                           |
| ---------------- | ----------------------------------------------------- |
| `get_model`      | Get current model content with numbered items         |
| `add_item`       | Add new item at specific index                        |
| `delete_item`    | Delete item at index                                  |
| `replace_item`   | Replace item at index                                 |
| `clear_model`    | Clear all items in the model                         |
| `solve_model`    | Solve the model with the Chuffed constraint solver   |
| `get_solution`   | Get specific variable value from solution with optional array indices |
| `get_solve_time` | Get last solve execution time                        |
| `get_memo`       | Get current knowledge base                           |
| `edit_memo`      | Edit knowledge base                                  |

---

## System Requirements

- **Python 3.11+**  
  (Note: Python 3.11 is required to support the use of `asyncio.timeout` in the solver code.)
- [MiniZinc](https://www.minizinc.org) with the Chuffed solver
- Operating systems:
  - macOS
  - Windows 
  - Linux (with appropriate adaptations)

---

## Installation

1. Install an MCP-compatible client (e.g., Claude web interface or desktop app)

2. Install the MCP Solver:

   ```bash
   git clone https://github.com/szeider/mcp-solver.git
   cd mcp-solver
   uv pip install -e .
   ```

3. Create the configuration file:

   For macOS/Linux:

   ```bash
   ~/Library/Application/Support/Claude/claude_desktop_config.json
   ```

   For Windows:

   ```bash
   %APPDATA%\Claude\claude_desktop_config.json
   ```

   Add the following content:

   ```json
   {
     "mcpServers": {
       "MCP Solver": {
         "command": "uv",
         "args": [
           "--directory", 
           "/absolute/path/to/mcp-solver",  // Use full path to mcp-solver directory
           "run", 
           "mcp-solver"
         ]
       }
     }
   }
   ```

   Note: For Windows, use backslashes and escape them:

   ```json
   "args": ["--directory", "C:\\absolute\\path\\to\\mcp-solver", "run", "mcp-solver"]
   ```

4. The memo file location defaults to the standard configuration directory for your operating system. The default can be overridden in `pyproject.toml`.

------

## Examples

### Casting

This is based on an  example by George Johnson (NYT, July 13, 1999).

**User**

> Suppose that a theatrical director feels obligated to cast either his ingenue, Actress Alvarez, or his nephew, Actor Cohen, in a production. But Miss Alvarez won't be in a play with Mr. Cohen (her former lover), and she demands that the cast include her new flame, Actor Davenport. The producer, with her own favors to repay, insists that Actor Branislavsky have a part. But Mr. Branislavsky won't be in any play with Miss Alvarez or Mr. Davenport. Can the director cast the play?

**LLM**

> Let me model this casting problem using MiniZinc. We'll use boolean variables to represent whether each actor is cast.

**System:** **`clear_model`**

> ```
> Model cleared
> ```

**System:** **`add_item`** (multiple)

> ```minizinc
> var bool: alvarez;
> var bool: cohen;
> var bool: branislavsky;
> var bool: davenport;
> constraint alvarez \/ cohen;                          % Must cast either Alvarez or Cohen
> constraint not (alvarez /\ cohen);                    % Alvarez won't work with Cohen
> constraint alvarez -> davenport;                      % If Alvarez is cast, Davenport must be cast
> constraint branislavsky;                              % Branislavsky must be cast
> constraint not (branislavsky /\ alvarez);             % Branislavsky won't work with Alvarez
> constraint not (branislavsky /\ davenport);           % Branislavsky won't work with Davenport
> solve satisfy;
> ```

**System:** **`solve_model`**

> ```
> {'status': 'SAT', 'solution': Solution(alvarez=False, cohen=True, branislavsky=True, davenport=False, _checker=''), 'solve_time': 0.0}
> ```

**LLM**

> The problem has a solution: cast Cohen and Branislavsky, but not Alvarez or Davenport. Let's verify:
>
> 1. Either Alvarez or Cohen must be cast ✓ (Cohen is cast)
> 2. Alvarez and Cohen can't both be cast ✓ (only Cohen is cast)
> 3. If Alvarez is cast, Davenport must be cast ✓ (Alvarez isn't cast)
> 4. Branislavsky must be cast ✓ (Branislavsky is cast)
> 5. Branislavsky won't work with Alvarez ✓ (Alvarez isn't cast)
> 6. Branislavsky won't work with Davenport ✓ (Davenport isn't cast)
>
> The director can cast the play with Cohen and Branislavsky.

------

### N-Queens

This example shows how the same model can be solved with different parameter values, and the solving times can be obtained.

**User**

> Check whether you can place n Queens on an nxn chessboard. Try n=10,20,30,40 and compare the solving times

**LLM**

> Let me create a model for the N-Queens problem and test different sizes.

**System:** **`clear_model`**

> ```
> Model cleared
> ```

**System:** **`add_item`** (multiple)

> ```minizinc
> int: n = 10;
> array[1..n] of var 1..n: queens;
> constraint alldifferent(queens);
> constraint alldifferent([queens[i] + i | i in 1..n]);
> constraint alldifferent([queens[i] - i | i in 1..n]);
> solve satisfy;
> ```

**LLM**

> Testing n=10:

**System:** **`solve_model`**

> ```
> {'status': 'SAT', 'solution': Solution(queens=[3, 6, 9, 1, 8, 4, 2, 7, 5, 10], _checker=''), 'solve_time': 0.001}
> ```

**LLM**

> Testing n=20:

**System:** **`edit_model`**

> ```minizinc
> int: n = 20;
> ```

**System:** **`solve_model`**

> ```
> {'status': 'SAT', 'solution': Solution(queens=[4, 18, 16, 11, 13, 1, 3, 19, 8, 12, 17, 2, 10, 15, 6, 14, 9, 7, 5, 20], _checker=''), 'solve_time': 0.005}
> ```

**LLM**

> Testing n=30:

**System:** **`edit_model`**

> ```minizinc
> int: n = 30;
> ```

**System:** **`solve_model`**

> ```
> {'status': 'SAT', 'solution': Solution(queens=[17, 23, 9, 18, 2, 13, 8, 1, 24, 21, 6, 22, 11, 26, 5, 25, 19, 14, 10, 3, 29, 27, 15, 4, 12, 20, 16, 7, 28, 30], _checker=''), 'solve_time': 0.04}
> ```

**LLM**

> Testing n=40:

**System:** **`edit_model`**

> ```minizinc
> int: n = 40;
> ```

**System:** **`solve_model`**

> ```
> {'status': 'SAT', 'solution': Solution(queens=[12, 34, 23, 8, 20, 31, 15, 27, 18, 33, 5, 2, 26, 10, 16, 32, 4, 11, 14, 3, 9, 25, 28, 30, 35, 6, 1, 17, 21, 37, 13, 29, 24, 19, 7, 22, 39, 36, 38, 40], _checker=''), 'solve_time': 0.043}
> ```

**LLM**

> Solve times comparison:
> n=10: 0.001s
> n=20: 0.005s
> n=30: 0.040s
> n=40: 0.043s
>
> Solutions exist for all tested board sizes. Time complexity increases non-linearly but remains manageable up to n=40.

---

### TSP

**User**

> A saleswoman based in Vienna needs to plan her upcoming tour through Austria, visiting each province capital once. Help find the shortest route. Distances in km:
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

**LLM**

> I'll model this as a Traveling Salesperson Problem using a circuit constraint.

**System:** **`add_item`** (multiple)

> ```minizinc
> include "globals.mzn";
> int: n = 9;
> array[1..n,1..n] of int: dist = [|0,65,60,184,195,319,299,478,631|65,0,125,119,130,254,234,413,566|60,125,0,184,157,281,261,440,593|184,119,184,0,208,252,136,315,468|195,130,157,208,0,136,280,459,629|319,254,281,252,136,0,217,391,566|299,234,261,136,280,217,0,188,343|478,413,440,315,459,391,188,0,157|631,566,593,468,629,566,343,157,0|];
> array[1..n] of var 1..n: succ;
> var int: total_dist = sum(i in 1..n)(dist[i,succ[i]]);
> constraint circuit(succ);
> solve minimize total_dist;
> ```

**System:** **`solve_model`**

> ```
> {'status': 'SAT', 'solution': Solution(objective=1564, succ=[2, 4, 1, 7, 3, 5, 9, 6, 8], _checker=''), 'solve_time': 0.002}
> ```

**LLM**

> Optimal route found: Vienna → St. Pölten → Linz → Salzburg → Bregenz → Innsbruck → Klagenfurt → Graz → Eisenstadt → Vienna
>
> Total distance: 1,564 km
>
> Route verification:
>
> - Each city visited exactly once: ✓ (circuit constraint)
> - Distance calculation correct: ✓
> - Starting/ending in Vienna (city 1): ✓

**User**

> Assume the road between Vienna and Eisenstadt is blocked, what is now the shortest tour?

**LLM**

> I'll modify the distance matrix to make the Vienna-Eisenstadt route impossible (infinite distance).

**System:** **`add_item`**

> ```minizinc
> array[1..n,1..n] of int: dist = [|0,65,9999,184,195,319,299,478,631|65,0,125,119,130,254,234,413,566|9999,125,0,184,157,281,261,440,593|184,119,184,0,208,252,136,315,468|195,130,157,208,0,136,280,459,629|319,254,281,252,136,0,217,391,566|299,234,261,136,280,217,0,188,343|478,413,440,315,459,391,188,0,157|631,566,593,468,629,566,343,157,0|];
> ```

**System:** **`solve_model`**

> ```
> {'status': 'SAT', 'solution': Solution(objective=1694, succ=[4, 1, 2, 7, 3, 5, 9, 6, 8], _checker=''), 'solve_time': 0.003}
> ```

**LLM**

> Optimal route: Vienna → Linz → St. Pölten → Salzburg → Bregenz → Innsbruck → Klagenfurt → Graz → Eisenstadt → Vienna
>
> Total distance: 1,694 km (+130 km compared to unblocked route)

------

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
