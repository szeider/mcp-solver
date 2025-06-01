# Simple MaxSAT Test Problem

I need to solve a simple MaxSAT optimization problem with just one variable.

- Variable x should be true (with weight 1)
- There are no hard constraints

Please create a MaxSAT formulation and solve this trivial problem. Show the optimal solution where x should be true.

Important: Make sure to use the correct syntax for adding soft constraints to the WCNF:
```python
# Correct:
wcnf.append([x], weight=1)  # Note the square brackets around the variable

# Incorrect:
wcnf.append(x, weight=1)    # Missing square brackets
wcnf.append(, weight=1)     # Incomplete
```