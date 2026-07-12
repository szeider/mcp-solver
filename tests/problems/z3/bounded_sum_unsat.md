Find an assignment of integers to variables a, b, c, d, and e that:

1. Each variable must be between 1 and 10 inclusive
2. All variables must have different values
3. The sum of all variables must be exactly 36
4. The product of a and b must be less than 20
5. The product of c, d, and e must be at least 200
6. Any two variables must sum to at least 11

or decide that this is impossible.

## Output Format

Return a single JSON object on stdout with this schema:

- `satisfiable` (boolean): `true` if an assignment satisfying all six
  constraints exists, `false` otherwise.
- `assignment` (object): maps `a, b, c, d, e` to integer values. Required
  only when `satisfiable` is `true`.

This instance is unsatisfiable, so the expected answer is:

```json
{"satisfiable": false}
```