# ASP Example Problem: Simple Path Coloring

## Problem Description
Color the nodes of a path with 3 nodes using 2 colors such that no adjacent nodes have the same color.

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