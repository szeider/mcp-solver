"""
Template functions for Z3 quantifiers and common constraint patterns.

This module provides pre-built patterns for common Z3 quantifier use cases,
making it easier for users to express complex constraints without needing to
write detailed quantifier logic.
"""

from .z3_templates import (
    array_is_sorted,
    all_distinct,
    array_contains,
    exactly_k,
    at_most_k,
    at_least_k,
    function_is_injective,
    function_is_surjective
)

__all__ = [
    'array_is_sorted',
    'all_distinct',
    'array_contains',
    'exactly_k',
    'at_most_k',
    'at_least_k',
    'function_is_injective',
    'function_is_surjective'
] 