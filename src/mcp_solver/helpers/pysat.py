"""PySAT helper library: variable mapping, clause builders, cardinality
constraints, and MaxSAT soft-clause helpers.

All cardinality helpers return plain clause lists (``list[list[int]]``) that
you can feed straight to a solver or append to a ``CNF``/``WCNF``. Literals are
DIMACS-style signed integers: ``v`` means "variable v true", ``-v`` means
"variable v false".

Example:
    >>> from pysat.solvers import Glucose3
    >>> vm = VariableMap()
    >>> a, b, c = (vm.get_id(n) for n in ("a", "b", "c"))
    >>> solver = Glucose3()
    >>> for clause in exactly_one([a, b, c]):
    ...     solver.add_clause(clause)
    >>> solver.solve()
    True
    >>> vm.interpret_model(solver.get_model())  # doctest: +SKIP
    {'a': True, 'b': False, 'c': False}
"""

import itertools

from pysat.card import CardEnc, EncType
from pysat.formula import WCNF

# Threshold below which at_most_k uses a direct pairwise encoding instead of
# pysat's CardEnc. Pairwise adds no auxiliary variables and behaves identically
# across pysat versions, at the cost of O(n^(k+1)) clauses.
_PAIRWISE_MAX_VARS = 20

# Threshold below which at_least_k reduces to at_most_k via De Morgan rather
# than calling CardEnc directly.
_DE_MORGAN_MAX_VARS = 100


class VariableMap:
    """Bidirectional mapping between meaningful names and SAT variable IDs.

    IDs are assigned sequentially from 1 as names are first requested, matching
    the DIMACS convention (variable 0 is reserved). Use :meth:`interpret_model`
    to turn a solver model back into ``{name: bool}``: a positive literal in the
    model means the variable is true.

    Example:
        >>> vm = VariableMap()
        >>> vm.get_id("x")
        1
        >>> vm.get_id("y")
        2
        >>> vm.get_id("x")  # stable
        1
        >>> vm.interpret_model([1, -2])
        {'x': True, 'y': False}
    """

    def __init__(self) -> None:
        self.var_to_id: dict[str, int] = {}
        self.id_to_var: dict[int, str] = {}
        self.next_id: int = 1

    def get_id(self, var_name: str) -> int:
        """Return the ID for ``var_name``, allocating a fresh one if new."""
        if var_name not in self.var_to_id:
            self.var_to_id[var_name] = self.next_id
            self.id_to_var[self.next_id] = var_name
            self.next_id += 1
        return self.var_to_id[var_name]

    def get_name(self, var_id: int) -> str:
        """Return the name for ``var_id`` (sign ignored).

        Unknown IDs yield ``"unknown_<id>"`` rather than raising.
        """
        return self.id_to_var.get(abs(var_id), f"unknown_{abs(var_id)}")

    def interpret_model(self, model: list[int]) -> dict[str, bool]:
        """Convert a SAT model (list of signed literals) to ``{name: bool}``.

        Only variables known to this map appear in the result; auxiliary IDs
        introduced by cardinality encodings are skipped.
        """
        result: dict[str, bool] = {}
        for lit in model:
            var_id = abs(lit)
            if var_id in self.id_to_var:
                result[self.id_to_var[var_id]] = lit > 0
        return result

    def get_mapping(self) -> dict[str, int]:
        """Return a copy of the current name-to-ID mapping."""
        return self.var_to_id.copy()


# ---------------------------------------------------------------------------
# Cardinality constraints (return clause lists)
# ---------------------------------------------------------------------------


def at_most_k(
    variables: list[int],
    k: int,
    *,
    encoding: int = EncType.seqcounter,
    top_id: int | None = None,
) -> list[list[int]]:
    """Clauses enforcing that at most ``k`` of ``variables`` are true.

    For up to ``_PAIRWISE_MAX_VARS`` (20) variables a direct pairwise encoding
    is used: it introduces no auxiliary variables and is identical across pysat
    versions. For larger sets pysat's ``CardEnc.atmost`` is used (falling back
    to pairwise if it raises), which allocates auxiliary variable IDs above
    ``max(abs(v))`` unless you pass ``top_id`` to reserve a higher starting
    point.

    Args:
        variables: Signed variable IDs (usually all positive).
        k: Upper bound, ``k >= 0``.
        encoding: pysat ``EncType`` used only on the CardEnc path.
        top_id: Highest variable ID already in use, so CardEnc allocates
            auxiliary variables above it (CardEnc path only).

    Returns:
        A list of clauses; empty if the constraint is trivially satisfied
        (``k >= len(variables)``).
    """
    if k >= len(variables):
        return []

    if len(variables) <= _PAIRWISE_MAX_VARS:
        return _pairwise_at_most(variables, k)

    try:
        return CardEnc.atmost(variables, k, encoding=encoding, top_id=top_id).clauses
    except Exception:
        return _pairwise_at_most(variables, k)


def _pairwise_at_most(variables: list[int], k: int) -> list[list[int]]:
    """Direct pairwise at-most-k: each (k+1)-subset has a false member."""
    return [[-v for v in combo] for combo in itertools.combinations(variables, k + 1)]


def at_least_k(
    variables: list[int],
    k: int,
    *,
    encoding: int = EncType.seqcounter,
    top_id: int | None = None,
) -> list[list[int]]:
    """Clauses enforcing that at least ``k`` of ``variables`` are true.

    Uses De Morgan's law for up to ``_DE_MORGAN_MAX_VARS`` (100) variables:
    "at least k true" == "at most (n - k) of the negated variables true", which
    routes through :func:`at_most_k` and its pairwise encoding. Larger sets try
    ``CardEnc.atleast`` first, falling back to the De Morgan reduction.

    Args:
        variables: Signed variable IDs.
        k: Lower bound.
        encoding: pysat ``EncType`` used only on the CardEnc path.
        top_id: Passed through to the CardEnc auxiliary-variable allocation.

    Returns:
        A list of clauses.
    """
    n = len(variables)
    if n <= _DE_MORGAN_MAX_VARS:
        neg_vars = [-v for v in variables]
        return at_most_k(neg_vars, n - k, encoding=encoding, top_id=top_id)

    try:
        return CardEnc.atleast(variables, k, encoding=encoding, top_id=top_id).clauses
    except Exception:
        neg_vars = [-v for v in variables]
        return at_most_k(neg_vars, n - k, encoding=encoding, top_id=top_id)


def exactly_k(
    variables: list[int],
    k: int,
    *,
    encoding: int = EncType.seqcounter,
    top_id: int | None = None,
) -> list[list[int]]:
    """Clauses enforcing that exactly ``k`` of ``variables`` are true.

    ``k == 0`` forbids every variable and ``k == 1`` combines a single
    at-least-one clause with pairwise at-most-one (both auxiliary-free). The
    general case is :func:`at_most_k` plus :func:`at_least_k`; above the
    pairwise threshold those allocate CardEnc auxiliary variables, with the
    IDs threaded so the two encodings never collide with each other or with
    ``variables``. Pass ``top_id`` when your formula already uses IDs above
    ``max(variables)``. An impossible request (``k < 0`` or
    ``k > len(variables)``) returns ``[[]]`` (a single empty clause), which
    any solver reads as UNSAT.

    Args:
        variables: Signed variable IDs.
        k: Exact count required.
        encoding: pysat ``EncType`` used only on the CardEnc path.
        top_id: Highest variable ID already in use, so CardEnc allocates
            auxiliary variables above it (CardEnc path only).

    Returns:
        A list of clauses.
    """
    if k > len(variables) or k < 0:
        return [[]]

    if k == 0:
        return [[-v] for v in variables]

    if k == 1:
        at_least = [variables.copy()]
        at_most = [
            [-variables[i], -variables[j]]
            for i in range(len(variables))
            for j in range(i + 1, len(variables))
        ]
        return at_least + at_most

    at_most = at_most_k(variables, k, encoding=encoding, top_id=top_id)
    used = max(
        (abs(lit) for clause in at_most for lit in clause),
        default=0,
    )
    used = max(used, max(abs(v) for v in variables), top_id or 0)
    return at_most + at_least_k(variables, k, encoding=encoding, top_id=used)


def at_most_one(variables: list[int]) -> list[list[int]]:
    """Clauses enforcing that at most one of ``variables`` is true (pairwise)."""
    return at_most_k(variables, 1)


def exactly_one(variables: list[int]) -> list[list[int]]:
    """Clauses enforcing that exactly one of ``variables`` is true."""
    return exactly_k(variables, 1)


def one_hot_encoding(variables: list[int]) -> list[list[int]]:
    """Clauses for a one-hot encoding (exactly one variable true).

    Semantically identical to :func:`exactly_one`; kept as a named alias for
    readability in models that speak of one-hot vectors.
    """
    return exactly_one(variables)


def mutually_exclusive(variables: list[int]) -> list[list[int]]:
    """Clauses ensuring at most one of ``variables`` is true (alias)."""
    return at_most_one(variables)


def implies(a: int, b: int) -> list[list[int]]:
    """Clauses for the implication ``a -> b`` (equivalently ``!a OR b``)."""
    return [[-a, b]]


def if_then_else(condition: int, then_var: int, else_var: int) -> list[list[int]]:
    """Clauses for ``if condition then then_var else else_var``.

    Enforces ``condition -> then_var`` and ``!condition -> else_var``.
    """
    return [
        [-condition, then_var],
        [condition, else_var],
    ]


# ---------------------------------------------------------------------------
# MaxSAT soft-clause helpers (return (clause, weight) pairs)
# ---------------------------------------------------------------------------


def soft_clause(literals: int | list[int], weight: int = 1) -> tuple[list[int], int]:
    """Build a single soft clause ``(clause, weight)`` for a WCNF.

    A soft clause may be violated at a cost equal to ``weight``. A bare int is
    wrapped into a unit clause.
    """
    if isinstance(literals, int):
        literals = [literals]
    return (literals, weight)


def soft_at_most_k(
    variables: list[int], k: int, weight: int = 1
) -> list[tuple[list[int], int]]:
    """Soft at-most-k as weighted pairwise clauses for a WCNF.

    Each (k+1)-subset contributes a soft clause "at least one is false"; each
    violated clause costs ``weight``. Returns ``[]`` when trivially satisfied.
    """
    if k >= len(variables):
        return []
    return [
        ([-var for var in combo], weight)
        for combo in itertools.combinations(variables, k + 1)
    ]


def soft_at_least_k(
    variables: list[int], k: int, weight: int = 1
) -> list[tuple[list[int], int]]:
    """Soft at-least-k as weighted clauses for a WCNF.

    Built by De Morgan over the negated variables. Returns ``[]`` when
    ``k <= 0`` (trivially satisfied).
    """
    if k <= 0:
        return []
    negated_vars = [-var for var in variables]
    return [
        ([-lit for lit in combo], weight)
        for combo in itertools.combinations(negated_vars, len(variables) - k + 1)
    ]


def soft_exactly_k(
    variables: list[int], k: int, weight: int = 1
) -> list[tuple[list[int], int]]:
    """Soft exactly-k: the union of soft at-most-k and soft at-least-k."""
    return soft_at_most_k(variables, k, weight) + soft_at_least_k(variables, k, weight)


def soft_implies(a: int, b: int, weight: int = 1) -> list[tuple[list[int], int]]:
    """Soft implication ``a -> b`` as a single weighted clause."""
    return [([-a, b], weight)]


def soft_if_then_else(
    condition: int, then_var: int, else_var: int, weight: int = 1
) -> list[tuple[list[int], int]]:
    """Soft if-then-else as two weighted clauses."""
    return [
        ([-condition, then_var], weight),
        ([condition, else_var], weight),
    ]


def add_soft_clauses_to_wcnf(wcnf: WCNF, clauses: list[tuple[list[int], int]]) -> None:
    """Append ``(clause, weight)`` pairs to ``wcnf`` in place."""
    for clause, weight in clauses:
        wcnf.append(clause, weight=weight)
