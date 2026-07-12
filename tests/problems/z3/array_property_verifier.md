Consider an integer array with 8 empty slots that needs to satisfy these conditions:

1. The array must be sorted in ascending order
2. Each value must be between 1 and 15 (inclusive)
3. The number 7 must appear exactly once in the array
4. The sum of all elements must equal 60
5. No two adjacent elements can both be even numbers

Find a valid array that meets all these requirements.

## Output Format

Return a single JSON object on stdout with this schema:

- `satisfiable` (boolean): `true` if a valid array exists.
- `array` (list of 8 integers): the array, in ascending order. Required when
  `satisfiable` is `true`.

Example (any array meeting all five conditions is accepted):

```json
{"satisfiable": true, "array": [3, 5, 5, 5, 5, 7, 15, 15]}
```