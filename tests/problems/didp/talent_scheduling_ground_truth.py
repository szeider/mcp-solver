#!/usr/bin/env python3
"""Semantic validator for talent scheduling (9 scenes, 6 actors).

Scenes are shot one per day in a chosen order. Each actor is on location from the
day of their first scheduled scene through the day of their last scheduled scene
(inclusive) and is paid their daily wage for every such day, idle days included.
A schedule's total cost is sum over actors of rate[a] * (last_day - first_day + 1).
Minimize the total cost.

This validator is self-contained: it (1) replays the reported order, checking it
is a permutation of 0..8 and recomputing the cost directly from the span
definition, then (2) runs its own independent exact oracle -- a subset DP over
the 2^9 subsets of already-shot scenes -- and requires the reported cost to equal
both the recomputed cost and that optimum. Any optimal order is accepted.

Oracle transition. In state S (set of scenes already shot) we shoot scene s next,
on day |S|. The actors on location that day, whose wage is incurred, are exactly:
  - every actor required in scene s, plus
  - every actor who has a scene in S (already appeared) AND a scene outside
    S u {s} (still to appear) -- i.e. on location but idle today.
Summing these transition costs over a full schedule reproduces, for each actor,
rate[a] paid once for each day from their first to their last scene inclusive,
which is exactly the span cost. dp[S] = minimal cost to reach state S; the answer
is dp[full]. (This equivalence is checked against brute-force enumeration over all
9! orders in the project's end-to-end tests.)

Reads JSON on stdin, prints {"valid": bool, "message": str}, exits 0/1.
Plain stdlib only.
"""

import json
import sys

# Requirements matrix: row a = actor, column s = scene; 1 means actor a is in scene s.
# fmt: off
REQUIREMENTS = [
    [1, 0, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 0, 1, 1]
]
RATE = [4, 1, 3, 6, 3, 3]
# fmt: on

A = len(REQUIREMENTS)  # actors
S = len(REQUIREMENTS[0])  # scenes

# actor -> bitmask over the scenes that require the actor
ACTOR_SCENES = [sum(1 << s for s in range(S) if REQUIREMENTS[a][s]) for a in range(A)]


def replay(order):
    """Recompute total cost directly from actor spans (structure pre-checked)."""
    day = {scene: d for d, scene in enumerate(order)}
    total = 0
    for a in range(A):
        days = [day[s] for s in range(S) if REQUIREMENTS[a][s]]
        if days:
            total += RATE[a] * (max(days) - min(days) + 1)
    return total


def oracle_optimum():
    """Exact subset DP over the 2^S sets of already-shot scenes."""
    full = (1 << S) - 1
    INF = float("inf")
    dp = [INF] * (1 << S)
    dp[0] = 0
    for St in range(1 << S):
        base = dp[St]
        if base == INF:
            continue
        rem = full & ~St
        m = rem
        while m:
            low = m & -m
            s = low.bit_length() - 1
            m ^= low
            newS = St | low
            future = full & ~newS  # scenes still unshot after this one
            cost = 0
            for a in range(A):
                asc = ACTOR_SCENES[a]
                on_today = REQUIREMENTS[a][s] or (
                    (asc & St) != 0 and (asc & future) != 0
                )
                if on_today:
                    cost += RATE[a]
            cand = base + cost
            if cand < dp[newS]:
                dp[newS] = cand
    return dp[full]


OPT = oracle_optimum()


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"

    order = data.get("order")
    if not isinstance(order, list) or len(order) != S:
        return False, f"order must be a list of {S} scene indices"
    for x in order:
        if not isinstance(x, int) or isinstance(x, bool):
            return False, f"order entries must be integers, got {x!r}"
    if sorted(order) != list(range(S)):
        return False, f"order must be a permutation of 0..{S - 1}, got {order}"

    total = replay(order)

    reported = data.get("total_cost")
    if not isinstance(reported, int) or isinstance(reported, bool):
        return False, f"total_cost must be an integer, got {reported!r}"
    if reported != total:
        return False, (
            f"reported total_cost {reported} does not match recomputed cost {total}"
        )

    if total != OPT:
        return False, f"total cost {total} is not optimal (optimum is {OPT})"

    return True, f"valid optimal schedule with total cost {total}"


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
