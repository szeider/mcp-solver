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

The puzzle has a unique solution.

Example:

```json
{
  "satisfiable": true,
  "assignment": {"S": 9, "E": 5, "N": 6, "D": 7, "M": 1, "O": 0, "R": 8, "Y": 2},
  "SEND": 9567,
  "MORE": 1085,
  "MONEY": 10652
}
```
