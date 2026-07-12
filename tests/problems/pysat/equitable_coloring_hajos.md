Find an equitable 3-coloring of the Hajós join of three C₅ cycles (cycle graphs with 5 vertices each).

## Output Format

Return a single JSON object. On success, `satisfiable` is `true` and
`coloring` maps each of the 13 vertices of the Hajós-join graph to a color in
`{0, 1, 2}`. Vertices are named `c{cycle}_{index}`; after the two Hajós joins
the graph has vertices `c0_0`..`c0_4`, `c1_1`..`c1_4`, and `c2_1`..`c2_4`. The
coloring must be proper (adjacent vertices differ) and equitable (the three
color-class sizes differ by at most one, i.e. sizes 4, 4, 5).

```json
{"satisfiable": true, "coloring": {"c0_0": 1, "c0_1": 2, "c0_2": 1, "c0_3": 1, "c0_4": 0, "c1_1": 1, "c1_2": 2, "c1_3": 0, "c1_4": 2, "c2_1": 2, "c2_2": 0, "c2_3": 2, "c2_4": 0}}
```

If no such coloring exists: `{"satisfiable": false}`.