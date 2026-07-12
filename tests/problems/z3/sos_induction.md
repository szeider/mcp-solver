Consider this simple program that computes the sum of cubes from 1 to n:

```
result = 0
i = 1
while i <= n:
    result = result + (i * i * i)
    i = i + 1
```

Using mathematical induction, prove that for any n, the result equals [n²(n+1)²/4]

Show the algebraic steps in your proof.

## Output Format

Return a single JSON object on stdout with this schema:

- `verified` (boolean): `true` if the identity
  `sum(i^3, i=1..n) = n^2(n+1)^2/4` is proven to hold for all n, `false`
  otherwise.
- `conclusion` (string): the proven identity. Optional.
- `algebraic_steps` (list of strings): the induction steps. Optional.

The identity is a true theorem, so the expected verdict is `verified: true`.

Example:

```json
{
  "verified": true,
  "conclusion": "sum(i^3, i=1..n) = n^2(n+1)^2/4",
  "algebraic_steps": [
    "Base n=0: sum = 0 and 0^2(0+1)^2/4 = 0.",
    "Induction hypothesis: S_k = k^2(k+1)^2/4.",
    "S_(k+1) = S_k + (k+1)^3 = (k+1)^2(k+2)^2/4."
  ]
}
```