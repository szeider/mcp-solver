Find an equitable 3-coloring of the Hajós join of three C₅ cycles (cycle graphs with 5 vertices each).

The graph is built from three 5-cycles — cycle `k` has vertices `ck_0`..`ck_4`
and edges `ck_i`–`ck_{(i+1) mod 5}` — by two Hajós joins (join of G with edge
(a,b) and H with edge (u,v): delete both edges, identify u with a, add the
edge (b,v)):

1. Join cycle 0 with edge (`c0_0`,`c0_1`) and cycle 1 with edge
   (`c1_0`,`c1_1`): delete both edges, identify `c1_0` with `c0_0`, add the
   edge (`c0_1`,`c1_1`).
2. Join the result with edge (`c0_2`,`c0_3`) and cycle 2 with edge
   (`c2_0`,`c2_1`): delete both edges, identify `c2_0` with `c0_2`, add the
   edge (`c0_3`,`c2_1`).

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