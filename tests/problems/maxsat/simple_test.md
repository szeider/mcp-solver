# Very Simple MaxSAT Test

Create a MaxSAT formulation for a simple optimization problem:

- We have three boolean variables: A, B, and C
- Hard constraint: A and B cannot both be true (mutually exclusive)
- Soft constraint: We prefer A to be true (weight 2)
- Soft constraint: We prefer B to be true (weight 3)
- Soft constraint: We prefer C to be true (weight 1)

Find the assignment that maximizes the sum of weights of satisfied soft constraints while respecting the hard constraint.