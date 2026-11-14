Solve the n-queens problem for n=8.

## Output Format

Return a single JSON object. On success, `satisfiable` is `true` and `queens`
is a list of 8 `[column, row]` pairs (each coordinate in 0..7): one queen per
column, with no two queens sharing a row or a diagonal.

Example (this is a dummy showing the shape only, **not** the real answer):

```json
{"satisfiable": true, "queens": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0]]}
```

If no placement exists: `{"satisfiable": false}`.