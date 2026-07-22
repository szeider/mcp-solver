A courier starts the day at the depot (node `0`) and must deliver to 10
customers (nodes `1`–`10`), then return to the depot. The van leaves the depot
at time `t = 0`. Travelling between two nodes takes the number of minutes given
in the matrix below. Each customer will only accept the delivery inside a time
window `[ready, due]`: if the van reaches a customer **before** its window
opens it simply waits until `ready` (waiting is allowed and free), but if it
arrives **after** `due` the delivery fails, so that ordering is not allowed.

Plan a route that starts at the depot, visits every customer exactly once, and
returns to the depot, respecting every time window. Among all valid routes,
choose one whose **total travel time is smallest**. Travel time counts only the
minutes spent driving between nodes; time spent waiting for a window to open
does **not** count toward the objective.

Travel-time matrix (row `i`, column `j` = minutes to drive from node `i` to
node `j`; symmetric, and the diagonal is `0`):

```json
[
  [0, 32, 33, 60, 42, 24, 45, 51, 39, 42, 44],
  [32, 0, 65, 36, 28, 55, 56, 21, 47, 16, 14],
  [33, 65, 0, 86, 73, 10, 47, 82, 48, 75, 75],
  [60, 36, 86, 0, 62, 76, 55, 22, 46, 47, 24],
  [42, 28, 73, 62, 0, 66, 80, 42, 71, 16, 38],
  [24, 55, 10, 76, 66, 0, 40, 73, 40, 66, 65],
  [45, 56, 47, 55, 80, 40, 0, 63, 10, 72, 58],
  [51, 21, 82, 22, 42, 73, 63, 0, 53, 26, 8],
  [39, 47, 48, 46, 71, 40, 10, 53, 0, 63, 48],
  [42, 16, 75, 47, 16, 66, 72, 26, 63, 0, 23],
  [44, 14, 75, 24, 38, 65, 58, 8, 48, 23, 0]
]
```

Time windows for the 10 customers (node `0` is the depot and has no window;
the van departs the depot at time `0`):

```json
{
  "1":  {"ready": 413, "due": 554},
  "2":  {"ready": 259, "due": 419},
  "3":  {"ready": 176, "due": 290},
  "4":  {"ready": 24,  "due": 200},
  "5":  {"ready": 334, "due": 454},
  "6":  {"ready": 121, "due": 257},
  "7":  {"ready": 448, "due": 559},
  "8":  {"ready": 298, "due": 433},
  "9":  {"ready": 459, "due": 636},
  "10": {"ready": 398, "due": 516}
}
```

## Output Format

Return a single JSON object on stdout with this schema:

- `tour` (list of 12 integers): the visiting order, starting and ending at the
  depot `0`, listing each of the 10 customers exactly once in between (so the
  first and last entries are both `0` and the length is 12).
- `cost` (integer): the total travel time (driving minutes only, waiting
  excluded) of the reported tour.

Any route achieving the minimum total travel time is accepted.

Example (this is a dummy showing the shape only, **not** the real answer):

```json
{
  "tour": [0, 3, 1, 5, 2, 8, 4, 9, 6, 10, 7, 0],
  "cost": 999
}
```

If no valid route exists, return `{"error": "No solution exists"}` instead.
