"""MaxSAT helper library: hard cardinality constraints that mutate a WCNF.

Unlike the clause-returning helpers in :mod:`mcp_solver.helpers.pysat`, these
functions append HARD clauses directly to a ``WCNF`` in place (weight ``None``
in pysat terms), so they express constraints that must hold in every model. For
weighted SOFT versions, use the ``soft_*`` helpers in the pysat module.

Example:
    >>> from pysat.formula import WCNF
    >>> wcnf = WCNF()
    >>> exactly_k(wcnf, [1, 2, 3], 1)
    >>> wcnf.append([1], weight=1)  # a soft preference
"""

import itertools

from pysat.formula import WCNF


def at_most_k(wcnf: WCNF, variables: list[int], k: int) -> None:
    """Append hard clauses so at most ``k`` of ``variables`` are true.

    Uses a direct pairwise encoding (no auxiliary variables): each (k+1)-subset
    gets a clause requiring at least one false member. ``k < 0`` forces every
    variable false; ``k >= len(variables)`` adds nothing.

    Args:
        wcnf: WCNF to mutate in place.
        variables: Signed variable IDs.
        k: Upper bound.
    """
    if k >= len(variables):
        return

    if k < 0:
        for var in variables:
            wcnf.append([-var])
        return

    for subset in itertools.combinations(variables, k + 1):
        wcnf.append([-var for var in subset])


def at_least_k(wcnf: WCNF, variables: list[int], k: int) -> None:
    """Append hard clauses so at least ``k`` of ``variables`` are true.

    Each (n-k+1)-subset gets a clause requiring at least one true member.
    ``k <= 0`` adds nothing; an impossible ``k > len(variables)`` appends a
    single empty clause, making the WCNF unsatisfiable.

    Args:
        wcnf: WCNF to mutate in place.
        variables: Signed variable IDs.
        k: Lower bound.
    """
    if k <= 0:
        return

    n = len(variables)
    if k > n:
        wcnf.append([])
        return

    for subset in itertools.combinations(range(n), n - k + 1):
        wcnf.append([variables[i] for i in subset])


def exactly_k(wcnf: WCNF, variables: list[int], k: int) -> None:
    """Append hard clauses so exactly ``k`` of ``variables`` are true.

    Combines :func:`at_least_k` and :func:`at_most_k`.

    Args:
        wcnf: WCNF to mutate in place.
        variables: Signed variable IDs.
        k: Exact count required.
    """
    at_least_k(wcnf, variables, k)
    at_most_k(wcnf, variables, k)
