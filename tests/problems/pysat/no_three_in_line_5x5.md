# No-Three-in-Line Problem: Verification for n=5

## Problem Statement

Consider a 5×5 grid of points with coordinates (i,j) where i,j ∈ {0,1,2,3,4}. This gives us 25 possible positions arranged in a square grid pattern.

**Task:** Verify that it is possible to place exactly 10 points on this 5×5 grid such that no three of the selected points are collinear (lie on the same straight line).

## Constraints

- Points must be placed at integer grid coordinates only
- "No three in line" means no three selected points lie on the same straight line, regardless of the line's slope
- This includes horizontal lines, vertical lines, diagonal lines, and lines with any other slope (like 1/2, 2/1, 1/3, 1/4, 3/2, etc.)

## What to Verify

1. **Existence:** Find a configuration of 10 points on the 5×5 grid that satisfies the no-three-in-line constraint
2. **Completeness:** Check all possible lines that could pass through three or more points in your configuration to confirm none contain three selected points
3. **Optimality:** Verify that 10 is indeed the maximum number of points that can be placed (optional)

## Example Grid Layout

```
(0,4)  (1,4)  (2,4)  (3,4)  (4,4)
(0,3)  (1,3)  (2,3)  (3,3)  (4,3)
(0,2)  (1,2)  (2,2)  (3,2)  (4,2)
(0,1)  (1,1)  (2,1)  (3,1)  (4,1)
(0,0)  (1,0)  (2,0)  (3,0)  (4,0)
```

## Lines to Consider

The 5×5 grid contains 32 distinct lines with 3 or more collinear points:

- 5 horizontal lines
- 5 vertical lines
- 2 main diagonal lines
- 20 other diagonal lines with various slopes

Some examples of lines that must be avoided:

- **Horizontal:** [(0,0), (1,0), (2,0), (3,0), (4,0)]
- **Vertical:** [(0,0), (0,1), (0,2), (0,3), (0,4)]
- **Diagonal:** [(0,0), (1,1), (2,2), (3,3), (4,4)]
- **Slope 1/2:** [(0,0), (2,1), (4,2)]
- **Slope 2:** [(0,0), (1,2), (2,4)]

**Question:** Can you find such a configuration of 10 points and prove that no three of them are collinear?