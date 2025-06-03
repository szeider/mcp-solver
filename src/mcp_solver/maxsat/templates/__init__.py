"""
MaxSAT templates package.

This package provides basic cardinality constraint helpers for MaxSAT problems.
"""

from .cardinality_constraints import at_least_k, at_most_k, exactly_k


__all__ = [
    "at_most_k",
    "at_least_k",
    "exactly_k",
]
