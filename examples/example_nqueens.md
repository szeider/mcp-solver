## N-Queens 

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
>
------
MCP server version 0.2.1
Claude 3.5 Sonnet