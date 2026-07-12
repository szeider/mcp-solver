A carpet installer needs to efficiently cut rectangular carpet pieces from a single standard-sized roll, minimizing waste.

### Given:

- 1 standard carpet roll measuring 12ft × 50ft
- Need to cut the following rectangular pieces:
  1. 6ft × 8ft (quantity: 1)
  2. 4ft × 5ft (quantity: 1)
  3. 5ft × 7ft (quantity: 1)
  4. 8ft × 10ft (quantity: 1)

### Constraints:

1. All pieces must be placed without overlap
2. Pieces can optionally be rotated 90 degrees
3. All pieces must fit within the roll dimensions

### Objective:

Minimize the total length of carpet roll used

## Output Format

The roll is 12 ft wide (the `x` axis, `0 <= x <= 12`) and extends along its
length (the `y` axis). Each placement gives the lower-left corner `(x, y)` of
a piece together with its placed dimensions `width` (along the roll width) and
`height` (along the roll length). A piece may be used in its original
orientation or rotated 90 degrees.

Return a single JSON object on stdout with this schema:

- `satisfiable` (boolean): `true` (a packing always exists).
- `roll_length_used` (integer): the length of roll consumed, i.e. the maximum
  `y + height` over all pieces.
- `placements` (list of 4 objects): one per piece, each with:
  - `piece` (integer 1-4): the piece index, matching the numbered list above
    (1 = 6x8, 2 = 4x5, 3 = 5x7, 4 = 8x10).
  - `x`, `y` (integers): lower-left corner.
  - `width`, `height` (integers): placed dimensions; `{width, height}` must
    equal the piece's `{a, b}` (rotation allowed).

The optimal (minimum) roll length is 18 ft.

Example:

```json
{
  "satisfiable": true,
  "roll_length_used": 18,
  "placements": [
    {"piece": 1, "x": 0, "y": 0, "width": 6, "height": 8},
    {"piece": 2, "x": 8, "y": 7, "width": 4, "height": 5},
    {"piece": 3, "x": 6, "y": 0, "width": 5, "height": 7},
    {"piece": 4, "x": 0, "y": 8, "width": 8, "height": 10}
  ]
}
```

