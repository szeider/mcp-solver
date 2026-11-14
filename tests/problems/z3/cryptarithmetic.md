# Cryptarithmetic Puzzle

Can you solve this cryptarithmetic puzzle using Z3?

```
  SEND
+ MORE
------
 MONEY
```

Each letter represents a unique digit (0-9). Find the digit assignment that makes this sum equation valid.

The leading digit of any number cannot be 0 (so S and M cannot be 0).

## Output Format

Return a single JSON object on stdout with this schema:

- `satisfiable` (boolean): `true` if a valid assignment exists.
- `assignment` (object): maps each of the 8 letters `S, E, N, D, M, O, R, Y`
  to its digit (integer 0-9). Required when `satisfiable` is `true`.
- `SEND`, `MORE`, `MONEY` (integers): the numeric values of the three words
  under the assignment. Optional; validated if present.

Example (this is a dummy showing the shape only, **not** the real answer):

```json
{
  "satisfiable": true,
  "assignment": {"S": 0, "E": 0, "N": 0, "D": 0, "M": 0, "O": 0, "R": 0, "Y": 0},
  "SEND": 0,
  "MORE": 0,
  "MONEY": 0
}
```
