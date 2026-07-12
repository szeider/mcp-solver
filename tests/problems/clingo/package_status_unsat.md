A package has been marked both as lost and delivered. A package is considered lost if it is not delivered. A package is considered delivered if it is not lost. Consider that a package cannot be lost and delivered at the same time.

## Output Format

Return a single JSON object on stdout with this schema:

- `satisfiable` (boolean): `true` if the program has an answer set, `false`
  otherwise.

The package is asserted to be both lost and delivered, while an integrity
constraint forbids being both at once, so the program has no answer set. The
expected answer is:

```json
{"satisfiable": false}
```
