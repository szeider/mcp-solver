# ASP Example Problem: Simple Path Coloring

## Problem Description
Given three nodes labeled 1, 2, and 3, and two available colors (red and green), color each node so that:
- Each node is assigned exactly one color (red or green).
- No two adjacent nodes share the same color.
- The nodes are connected in a path: node 1 is connected to node 2, and node 2 is connected to node 3.

## ASP Encoding
```
node(1..3).
color(red;green).

1 { assign(N,C) : color(C) } 1 :- node(N).
:- assign(N,C), assign(M,C), edge(N,M), N < M.

edge(1,2). edge(2,3).

#show assign/2.
```

## Expected Output
At least one answer set where each node is assigned a color and no two adjacent nodes share the same color. For example:

```
assign(1,red) assign(2,green) assign(3,red)
``` 