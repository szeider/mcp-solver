A courier van can carry at most **700** kilograms. There are 40 parcels
available, each with a weight (kg) and a profit (the fee earned for delivering
it). Each parcel is either loaded whole or left behind — parcels cannot be
split. Choose a subset of parcels whose total weight does not exceed the 700 kg
capacity and whose **total profit is as large as possible**.

The 40 parcels are indexed `0`–`39`. Weights and profits, index-aligned:

```json
{
  "capacity": 700,
  "weights": [93, 30, 82, 117, 53, 24, 20, 38, 104, 95, 80, 117, 114, 67, 60, 118, 22, 54, 82, 45, 113, 72, 88, 89, 107, 32, 44, 92, 90, 109, 113, 53, 104, 98, 107, 31, 74, 62, 31, 66],
  "profits": [114, 74, 123, 189, 34, 60, 189, 172, 84, 34, 21, 160, 61, 177, 102, 134, 59, 141, 157, 175, 188, 138, 17, 172, 102, 72, 164, 120, 87, 101, 160, 40, 32, 138, 183, 144, 60, 39, 165, 179]
}
```

## Output Format

Return a single JSON object on stdout with this schema:

- `selected` (list of integers): the indices of the chosen parcels, sorted in
  increasing order, each in `0..39` and appearing at most once.
- `total_profit` (integer): the summed profit of the chosen parcels.
- `total_weight` (integer): the summed weight of the chosen parcels
  (must be `<= 700`).

Any selection achieving the maximum possible total profit is accepted.

Example (this is a dummy showing the shape only, **not** the real answer):

```json
{
  "selected": [0, 3, 7, 12, 19, 26, 35],
  "total_profit": 1234,
  "total_weight": 555
}
```
