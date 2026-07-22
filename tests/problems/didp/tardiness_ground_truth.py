#!/usr/bin/env python3
"""Semantic validator for single-machine total weighted tardiness (10 jobs).

One machine, no preemption, no idle time: a job's completion time is the running
sum of processing times up to and including it. Job j finishing at C[j] costs
w[j] * max(0, C[j] - d[j]); the objective is the sum over all jobs, minimized.

This validator is self-contained: it (1) replays the reported order, checking it
is a permutation of 0..9 and recomputing the objective, then (2) runs its own
independent exact oracle -- a Held-Karp-style subset DP over the 2^10 subsets --
to find the true optimum, and requires the reported value to equal both the
recomputed objective and that optimum. Any optimal order is accepted.

The DP is exact because the completion time of any prefix depends only on the
SET of jobs in it (it equals their total processing time), not on their internal
order, so dp[S] = minimal total weighted tardiness of scheduling the jobs in S
first, in the best internal order.

Reads JSON on stdin, prints {"valid": bool, "message": str}, exits 0/1.
Plain stdlib only.
"""

import json
import sys

# fmt: off
P = [7, 5, 11, 6, 18, 17, 18, 15, 9, 6]
D = [8, 7, 8, 88, 74, 6, 53, 92, 32, 59]
W = [8, 1, 7, 7, 1, 8, 5, 4, 2, 6]
# fmt: on

N = len(P)


def replay(order):
    """Recompute total weighted tardiness of an order (structure pre-checked)."""
    t = 0
    total = 0
    for j in order:
        t += P[j]
        total += W[j] * max(0, t - D[j])
    return total


def oracle_optimum():
    """Exact subset DP over 2^N subsets scheduled as a prefix."""
    full = (1 << N) - 1
    # completion time of a subset = sum of its processing times
    psum = [0] * (1 << N)
    for S in range(1, 1 << N):
        low = S & -S
        j = low.bit_length() - 1
        psum[S] = psum[S ^ low] + P[j]

    INF = float("inf")
    dp = [INF] * (1 << N)
    dp[0] = 0
    for S in range(1, 1 << N):
        C = psum[S]  # time at which the last job of this prefix finishes
        best = INF
        m = S
        while m:
            low = m & -m
            j = low.bit_length() - 1
            m ^= low
            cand = dp[S ^ low] + W[j] * max(0, C - D[j])
            if cand < best:
                best = cand
        dp[S] = best
    return dp[full]


OPT = oracle_optimum()


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"

    order = data.get("order")
    if not isinstance(order, list) or len(order) != N:
        return False, f"order must be a list of {N} job indices"
    for x in order:
        if not isinstance(x, int) or isinstance(x, bool):
            return False, f"order entries must be integers, got {x!r}"
    if sorted(order) != list(range(N)):
        return False, f"order must be a permutation of 0..{N - 1}, got {order}"

    total = replay(order)

    reported = data.get("total_weighted_tardiness")
    if not isinstance(reported, int) or isinstance(reported, bool):
        return False, (f"total_weighted_tardiness must be an integer, got {reported!r}")
    if reported != total:
        return False, (
            f"reported total_weighted_tardiness {reported} does not match "
            f"recomputed value {total}"
        )

    if total != OPT:
        return False, (
            f"total weighted tardiness {total} is not optimal (optimum is {OPT})"
        )

    return True, f"valid optimal schedule with total weighted tardiness {total}"


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
