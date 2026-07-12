In an L(2, 1)-coloring of a graph the vertices are assigned color numbers in such a way that adjacent vertices get labels that differ by at least two, and the vertices that are at a distance of two from each other get labels that differ by at least one.

Check whether the Petersen graph (5-cycle + pentagram + perfect matching) has an L(2,1) coloring with 9 colors.

## Output Format

Return a single JSON object with a boolean `satisfiable` field. The Petersen
graph has L(2,1)-labeling number 9 (its labels must span 10 distinct values,
0..9), so no L(2,1) coloring exists with only 9 colors, and the expected
answer is:

```json
{"satisfiable": false}
```

If instead a coloring is claimed, report it as
`{"satisfiable": true, "coloring": {"0": label, ..., "9": label}}`, where the
keys are the vertices `0`..`9` (outer 5-cycle 0-4, inner pentagram 5-9, spokes
`i`-`i+5`) and each label lies in `0..8`.