#!/usr/bin/env python3
"""Semantic validator for the TSPTW (traveling salesperson with time windows).

Depot is node 0; customers are nodes 1..10. The van leaves the depot at t = 0,
visits every customer exactly once inside its [ready, due] window (waiting until
`ready` is allowed and free; arriving after `due` is forbidden), and returns to
the depot. Objective: minimize total travel time (waiting excluded from cost).

This validator is self-contained: it (1) replays the reported tour, checking
structure and every time window and recomputing the cost, then (2) runs its own
independent branch-and-bound DFS oracle to find the true optimum, and requires
the reported cost to equal that optimum. Any optimal feasible tour is accepted.
The instance is feasible, so an {"error": ...} answer is rejected.

Reads JSON on stdin, prints {"valid": bool, "message": str}, exits 0/1.
Plain stdlib only.
"""

import json
import sys

# Travel-time matrix, symmetric, node 0 = depot.
DIST = [
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
    [44, 14, 75, 24, 38, 65, 58, 8, 48, 23, 0],
]

# Time windows [ready, due] for nodes 0..10 (node 0 is the depot, unconstrained).
READY = [0, 413, 259, 176, 24, 334, 121, 448, 298, 459, 398]
DUE = [0, 554, 419, 290, 200, 454, 257, 559, 433, 636, 516]

N = 11  # depot + 10 customers
CUSTOMERS = list(range(1, N))


def replay(tour):
    """Recompute arrival times/cost for a tour, enforcing windows.

    Returns (ok, cost_or_message). Assumes tour structure already validated.
    """
    t = 0
    cost = 0
    prev = 0
    for node in tour[1:]:  # first entry is the starting depot
        cost += DIST[prev][node]
        t += DIST[prev][node]
        if node != 0:
            if t > DUE[node]:
                return False, (
                    f"customer {node} reached at t={t}, after its due time {DUE[node]}"
                )
            if t < READY[node]:
                t = READY[node]  # wait until the window opens (free)
        prev = node
    return True, cost


def oracle_optimum():
    """Independent branch-and-bound DFS for the true optimum travel time."""
    best = [None]

    def dfs(cur, t, visited, cost):
        if best[0] is not None and cost >= best[0]:
            return  # prune: cannot beat the incumbent
        if len(visited) == len(CUSTOMERS):
            total = cost + DIST[cur][0]  # drive back to the depot
            if best[0] is None or total < best[0]:
                best[0] = total
            return
        for c in CUSTOMERS:
            if c in visited:
                continue
            arr = t + DIST[cur][c]
            if arr > DUE[c]:
                continue  # prune: window violated
            start = arr if arr >= READY[c] else READY[c]
            dfs(c, start, visited | {c}, cost + DIST[cur][c])

    dfs(0, 0, frozenset(), 0)
    return best[0]


OPT = oracle_optimum()


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"

    if "error" in data:
        return False, (
            f"reported no solution, but the instance is feasible (optimum is {OPT})"
        )

    tour = data.get("tour")
    if not isinstance(tour, list) or len(tour) != 12:
        return False, "tour must be a list of 12 nodes (0 ... 0)"
    for x in tour:
        if not isinstance(x, int) or isinstance(x, bool):
            return False, f"tour entries must be integers, got {x!r}"
    if tour[0] != 0 or tour[-1] != 0:
        return False, "tour must start and end at the depot (node 0)"
    interior = tour[1:-1]
    if sorted(interior) != CUSTOMERS:
        return False, (
            f"the 10 interior nodes must be a permutation of {CUSTOMERS}, "
            f"got {interior}"
        )

    ok, res = replay(tour)
    if not ok:
        return False, f"time-window violation: {res}"
    cost = res

    total = data.get("cost")
    if not isinstance(total, int) or isinstance(total, bool):
        return False, f"cost must be an integer, got {total!r}"
    if total != cost:
        return False, (
            f"reported cost {total} does not match recomputed travel time {cost}"
        )

    if cost != OPT:
        return False, f"tour cost {cost} is not optimal (optimum is {OPT})"

    return True, f"valid optimal tour with total travel time {cost}"


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

    valid, message = validate(data)
    print(json.dumps({"valid": valid, "message": message}))
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
