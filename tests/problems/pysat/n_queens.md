Solve the n-queens problem for n=8.

## Output Format

Return a single JSON object. On success, `satisfiable` is `true` and `queens`
is a list of 8 `[column, row]` pairs (each coordinate in 0..7): one queen per
column, with no two queens sharing a row or a diagonal.

```json
{"satisfiable": true, "queens": [[0, 3], [1, 1], [2, 6], [3, 2], [4, 5], [5, 7], [6, 4], [7, 0]]}
```

If no placement exists (not the case for n=8): `{"satisfiable": false}`.