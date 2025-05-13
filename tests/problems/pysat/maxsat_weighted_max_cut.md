# Weighted Maximum Cut Problem

The Maximum Cut problem is a classic optimization problem in graph theory. Given a weighted undirected graph, the goal is to find a partition of the vertices into two sets such that the total weight of edges crossing the partition is maximized.

## Problem Description

You are given an undirected graph with 8 vertices and the following weighted edges:

| Edge | Weight |
|------|--------|
| 1-2  | 10     |
| 1-3  | 8      |
| 1-4  | 15     |
| 2-3  | 7      |
| 2-6  | 9      |
| 3-4  | 12     |
| 3-5  | 6      |
| 4-7  | 14     |
| 5-6  | 11     |
| 5-8  | 5      |
| 6-7  | 13     |
| 7-8  | 4      |

## Task

Use MaxSAT with the RC2 solver to find a partition of the vertices into two sets (S and its complement) that maximizes the sum of weights of edges crossing the partition.

1. Define a variable for each vertex, where True means the vertex is in set S and False means it's in the complement
2. For each edge (i,j) with weight w, add a soft constraint with weight w that the variables for vertices i and j have different truth values
3. Use the RC2 solver to compute the optimal solution
4. Report:
   - Which vertices are in each set
   - The total weight of the cut (sum of weights of edges crossing the partition)
   - Verify your solution by listing all edges that cross the partition

## Hints

- For each edge (i,j), you need to create a soft clause that encourages the variables for vertices i and j to have different values
- To represent "variables have different values" in CNF, you can use XOR: (xi ⊕ xj) which expands to (xi ∨ xj) ∧ (¬xi ∨ ¬xj)
- In MaxSAT with RC2, you want to maximize the weight of satisfied soft constraints