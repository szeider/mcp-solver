Consider this simple program that computes the sum of cubes from 1 to n:

```
result = 0
i = 1
while i <= n:
    result = result + (i * i * i)
    i = i + 1
```

Using mathematical induction, prove or disprove that for every nonnegative integer n, the result equals [n²(n+1)²/4]

Show the algebraic steps in your proof.

## Output Format

Return a single JSON object on stdout with this schema:

- `verified` (boolean): `true` if the identity
  `sum(i^3, i=1..n) = n^2(n+1)^2/4` is proven to hold for all n, `false`
  otherwise.
- `conclusion` (string): the proven statement. Optional.
- `algebraic_steps` (list of strings): the induction steps when `verified` is
  `true`, or the computation that refutes the identity when it is `false`.
  Optional.
- `counterexample` (integer): an n for which the identity fails. Required
  when `verified` is `false`.

Example (this is a dummy showing the shape only, **not** the real answer):

```json
{
  "verified": false,
  "conclusion": "...",
  "algebraic_steps": ["..."],
  "counterexample": 1
}
```