#!/usr/bin/env python3
"""Semantic validator for the 40-item 0/1 knapsack (maximization).

Checks the reported selection is a set of distinct in-range indices, recomputes
its weight and profit from the instance data hardcoded below, verifies the
reported totals match and that weight <= capacity, then computes the true
optimum with an O(n*capacity) dynamic program inside this validator and requires
total_profit to equal that optimum. Any optimal selection is accepted.

Reads JSON on stdin, prints {"valid": bool, "message": str}, exits 0/1.
Plain stdlib only.
"""

import json
import sys

CAPACITY = 700
# fmt: off
WEIGHTS = [
    93, 30, 82, 117, 53, 24, 20, 38, 104, 95, 80, 117, 114, 67, 60, 118,
    22, 54, 82, 45, 113, 72, 88, 89, 107, 32, 44, 92, 90, 109, 113, 53,
    104, 98, 107, 31, 74, 62, 31, 66,
]
PROFITS = [
    114, 74, 123, 189, 34, 60, 189, 172, 84, 34, 21, 160, 61, 177, 102, 134,
    59, 141, 157, 175, 188, 138, 17, 172, 102, 72, 164, 120, 87, 101, 160, 40,
    32, 138, 183, 144, 60, 39, 165, 179,
]
# fmt: on

N = len(WEIGHTS)


def optimum():
    """True optimum via O(n*capacity) 0/1-knapsack DP."""
    dp = [0] * (CAPACITY + 1)
    for i in range(N):
        wi, pi = WEIGHTS[i], PROFITS[i]
        for c in range(CAPACITY, wi - 1, -1):
            cand = dp[c - wi] + pi
            if cand > dp[c]:
                dp[c] = cand
    return max(dp)


OPT = optimum()


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"

    selected = data.get("selected")
    if not isinstance(selected, list):
        return False, "selected must be a list of item indices"
    for x in selected:
        if not isinstance(x, int) or isinstance(x, bool):
            return False, f"selected entries must be integers, got {x!r}"
    if len(set(selected)) != len(selected):
        return False, "selected contains duplicate indices"
    if selected != sorted(selected):
        return False, "selected must be sorted in increasing order"
    for x in selected:
        if x < 0 or x >= N:
            return False, f"index {x} out of range 0..{N - 1}"

    weight = sum(WEIGHTS[i] for i in selected)
    profit = sum(PROFITS[i] for i in selected)

    total_weight = data.get("total_weight")
    if not isinstance(total_weight, int) or isinstance(total_weight, bool):
        return False, f"total_weight must be an integer, got {total_weight!r}"
    if total_weight != weight:
        return False, (
            f"reported total_weight {total_weight} does not match recomputed "
            f"weight {weight}"
        )

    total_profit = data.get("total_profit")
    if not isinstance(total_profit, int) or isinstance(total_profit, bool):
        return False, f"total_profit must be an integer, got {total_profit!r}"
    if total_profit != profit:
        return False, (
            f"reported total_profit {total_profit} does not match recomputed "
            f"profit {profit}"
        )

    if weight > CAPACITY:
        return False, f"total weight {weight} exceeds capacity {CAPACITY}"

    if profit != OPT:
        return False, f"total profit {profit} is not optimal (optimum is {OPT})"

    return True, f"valid optimal selection with profit {profit}, weight {weight}"


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
