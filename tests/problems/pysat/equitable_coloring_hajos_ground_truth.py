#!/usr/bin/env python3
"""Semantic validator for the equitable 3-coloring of the Hajos join.

The graph is the Hajos join of three C5 cycles. Its edge set is rebuilt
here with plain stdlib (no solver dependency). A valid answer is a proper
3-coloring whose color-class sizes differ by at most one (an equitable
coloring of the 13 vertices, hence classes of sizes 4, 4, 5).

Reads a candidate solution as JSON from stdin, prints
{"valid": bool, "message": str}, exits 0 on valid, 1 otherwise.
"""

import json
import sys


def hajos_join(graph, edge_a, edge_b, name_a, name_b):
    """Perform a Hajos join using the indicated edges and identified ends."""
    graph = {tuple(sorted(edge)) for edge in graph}
    graph.remove(tuple(sorted(edge_a)))
    graph.remove(tuple(sorted(edge_b)))
    other_a = edge_a[1] if edge_a[0] == name_a else edge_a[0]
    other_b = edge_b[1] if edge_b[0] == name_b else edge_b[0]

    joined = set()
    for u, v in graph:
        u = name_a if u == name_b else u
        v = name_a if v == name_b else v
        if u != v:
            joined.add(tuple(sorted((u, v))))
    joined.add(tuple(sorted((other_a, other_b))))
    return joined


def build_graph():
    cycles = []
    for cycle in range(3):
        vertices = [f"c{cycle}_{index}" for index in range(5)]
        edges = {
            tuple(sorted((vertices[index], vertices[(index + 1) % 5])))
            for index in range(5)
        }
        cycles.append(edges)

    first_join = hajos_join(
        cycles[0] | cycles[1],
        ("c0_0", "c0_1"),
        ("c1_0", "c1_1"),
        "c0_0",
        "c1_0",
    )
    return hajos_join(
        first_join | cycles[2],
        ("c0_2", "c0_3"),
        ("c2_0", "c2_1"),
        "c0_2",
        "c2_0",
    )


def validate(data):
    edges = build_graph()
    vertices = sorted({vertex for edge in edges for vertex in edge})

    if data.get("satisfiable") is not True:
        return False, "Problem is satisfiable, but candidate reports otherwise"

    coloring = data.get("coloring")
    if not isinstance(coloring, dict):
        return False, "Missing 'coloring' object"

    if set(coloring) != set(vertices):
        return False, (
            f"Coloring must assign exactly the {len(vertices)} graph "
            f"vertices {vertices}"
        )

    for vertex, color in coloring.items():
        if not isinstance(color, int):
            return False, f"Color of {vertex} must be an integer, got {color}"

    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False, (f"Adjacent vertices {u} and {v} share color {coloring[u]}")

    classes = {}
    for color in coloring.values():
        classes[color] = classes.get(color, 0) + 1

    if len(classes) != 3:
        return False, (
            f"An equitable 3-coloring must use exactly 3 colors, got {len(classes)}"
        )

    sizes = sorted(classes.values())
    if sizes[-1] - sizes[0] > 1:
        return False, f"Color classes not equitable: sizes {sizes}"

    return True, f"Valid equitable 3-coloring (class sizes {sizes})"


def main():
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"valid": False, "message": "No input provided"}))
        sys.exit(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"valid": False, "message": f"Invalid JSON: {exc}"}))
        sys.exit(1)

    ok, message = validate(data)
    print(json.dumps({"valid": ok, "message": message}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
