#!/usr/bin/env python3
"""Semantic validator for the Petersen L(2,1)-coloring (UNSAT by design).

The Petersen graph has L(2,1)-labeling number 9, meaning the minimum span
is 9 and a valid labeling needs 10 distinct label values (0..9). With only
9 colors available no L(2,1)-labeling exists, so the correct answer is
{"satisfiable": false}.

A candidate that claims satisfiable=true is checked against the L(2,1)
constraints and rejected, naming the violated constraint.

Reads a candidate solution as JSON from stdin, prints
{"valid": bool, "message": str}, exits 0 on valid, 1 otherwise.
"""

import json
import sys
from collections import deque

NUM_COLORS = 9

# Petersen graph: outer 5-cycle (0-4), inner pentagram (5-9), spokes.
EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 0),  # outer 5-cycle
    (0, 5),
    (1, 6),
    (2, 7),
    (3, 8),
    (4, 9),  # spokes (perfect matching)
    (5, 7),
    (7, 9),
    (9, 6),
    (6, 8),
    (8, 5),  # inner pentagram
]
VERTICES = list(range(10))


def adjacency():
    adj = {v: set() for v in VERTICES}
    for u, v in EDGES:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def distances():
    adj = adjacency()
    dist = {}
    for src in VERTICES:
        seen = {src: 0}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if nb not in seen:
                    seen[nb] = seen[node] + 1
                    queue.append(nb)
        dist[src] = seen
    return dist


def parse_labels(coloring):
    """Return {vertex: label} for vertices 0..9, or None if unrecognized."""
    if isinstance(coloring, list):
        if len(coloring) != 10:
            return None
        labels = {}
        for index, value in enumerate(coloring):
            if not isinstance(value, int):
                return None
            labels[index] = value
        return labels
    if isinstance(coloring, dict):
        labels = {}
        for key, value in coloring.items():
            try:
                vertex = int(key)
            except (TypeError, ValueError):
                return None
            if vertex not in VERTICES or not isinstance(value, int):
                return None
            labels[vertex] = value
        if set(labels) != set(VERTICES):
            return None
        return labels
    return None


def validate(data):
    satisfiable = data.get("satisfiable")

    if satisfiable is False:
        return True, "Correctly reports the instance as unsatisfiable"

    if satisfiable is not True:
        return False, "Missing or invalid 'satisfiable' field"

    # Candidate claims SAT; the instance is provably UNSAT. Find the flaw.
    labels = parse_labels(data.get("coloring"))
    if labels is None:
        return False, (
            "Claimed satisfiable, but the Petersen graph has no L(2,1)-"
            "labeling with 9 colors (its L(2,1) number is 9, requiring 10 "
            "label values)"
        )

    used = set(labels.values())
    if len(used) > NUM_COLORS:
        return False, (
            f"Uses {len(used)} distinct colors, exceeding the budget of {NUM_COLORS}"
        )

    dist = distances()
    for u in VERTICES:
        for v in VERTICES:
            if u >= v:
                continue
            d = dist[u][v]
            if d == 1 and abs(labels[u] - labels[v]) < 2:
                return False, (
                    f"Adjacent vertices {u} and {v} have labels "
                    f"{labels[u]} and {labels[v]} differing by less than 2"
                )
            if d == 2 and labels[u] == labels[v]:
                return False, (
                    f"Distance-2 vertices {u} and {v} share label {labels[u]}"
                )

    # A consistent L(2,1)-labeling with <=9 colors cannot exist.
    return False, (
        "Claimed a valid 9-color L(2,1)-labeling, but the Petersen graph "
        "admits none (contradiction with its L(2,1) number of 9)"
    )


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
