In an L(2, 1)-coloring of a graph the vertices are assigned color numbers in such a way that adjacent vertices get labels that differ by at least two, and the vertices that are at a distance of two from each other get labels that differ by at least one.

Check whether the Petersen graph (5-cycle + pentagram + perfect matching) has an L(2,1) coloring with 9 colors.

## Output Format

Return a single JSON object with a boolean `satisfiable` field. The 9 colors
are the labels `0..8`.

- If such a coloring exists: `satisfiable` is `true` and `coloring` maps each
  vertex to its label, `{"satisfiable": true, "coloring": {"0": label, ..., "9": label}}`,
  where the keys are the vertices `0`..`9` (outer 5-cycle 0-4, inner pentagram
  5-9, spokes `i`-`i+5`) and each label lies in `0..8`.
- If no such coloring exists: `{"satisfiable": false}`.

Example (this is a dummy showing the shape only, **not** the real answer):

```json
{"satisfiable": true, "coloring": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}}
```