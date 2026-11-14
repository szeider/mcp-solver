A package has been marked both as lost and delivered. A package is considered lost if it is not delivered. A package is considered delivered if it is not lost. Consider that a package cannot be lost and delivered at the same time.

## Output Format

Return a single JSON object on stdout with this schema:

- `satisfiable` (boolean): `true` if the program has an answer set, `false`
  otherwise.
- `lost` and `delivered` (booleans): the package's status in the answer set.
  Required when `satisfiable` is `true`; omit both when it is `false`, so the
  output is then `{"satisfiable": false}`.

Example (this is a dummy showing the shape only, **not** the real answer):

```json
{"satisfiable": true, "lost": true, "delivered": true}
```
