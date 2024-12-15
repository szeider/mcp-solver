
# MCP Solver

[![MCP][mcp-badge]][mcp-url]  
[![License: MIT][license-badge]][license-url]  
[![Python Version][python-badge]][python-url]

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


## Example

### User Prompt (based on an example by  George  Johnson, NYT, July 13, 1999)

> Suppose that a theatrical director feels obligated to cast either his ingenue, Actress Alvarez, or his nephew, Actor Cohen, in a production. But Miss Alvarez won’t be in a play with Mr. Cohen (her former lover), and she demands that the cast include her new flame, Actor Davenport. The producer, with her own favors to repay, insists that Actor Branislavsky have a part. But Mr. Branislavsky won’t be in any play with Miss Alvarez or Mr. Davenport. Can the director cast the play?

### LLM

I'll solve this constraint satisfaction problem using MiniZinc. Let me model the theatrical casting constraints.



> Let me submit this model:


```minizinc
% Define actors as integers
int: Alvarez = 1;
int: Cohen = 2;
int: Davenport = 3;
int: Branislavsky = 4;

% Decision variable for casting
array[1..4] of var 0..1: cast;

% Constraint: Either Alvarez or Cohen, but not both
constraint cast[Alvarez] + cast[Cohen] = 1;

% Constraint: If Alvarez is in, Davenport must be in
constraint cast[Alvarez] = 1 -> cast[Davenport] = 1;

% Constraint: Branislavsky must be in
constraint cast[Branislavsky] = 1;

% Negative constraints: Cannot be cast together
constraint cast[Alvarez] + cast[Cohen] <= 1;      % Alvarez cannot be with Cohen
constraint cast[Branislavsky] + cast[Alvarez] <= 1;  % Branislavsky cannot be with Alvarez
constraint cast[Branislavsky] + cast[Davenport] <= 1;  % Branislavsky cannot be with Davenport

solve satisfy;

```

>  Now I'll solve the model.

> The model found a valid casting arrangement:
> 
> - Cast: Cohen and Branislavsky
> - Not Cast: Alvarez and Davenport
>
> This satisfies all the constraints:
> 1. Either Alvarez or Cohen is cast (Cohen is chosen)
> 2. Branislavsky must be in the play
> 3. Alvarez and Cohen are not together
> 4. Branislavsky is not with Alvarez or Davenport

> So yes, the director can cast the play under these specific conditions.


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

## Example Chat Dialogs

Further sample chat dialogs demonstrating tool usage are included in the `examples` folder of the repository. 

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
[python-badge]: https://img.shields.io/pypi/pyversions/mcp-solver  
[python-url]: https://www.python.org  

---
