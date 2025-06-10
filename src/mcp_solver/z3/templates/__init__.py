"""
Template functions for Z3 quantifiers and common constraint patterns.

This module provides pre-built patterns for common Z3 quantifier use cases,
making it easier for users to express complex constraints without needing to
write detailed quantifier logic.
"""

# Import and expose quantifier templates
# Import and expose function templates
from .function_templates import (
    array_template,
    constraint_satisfaction_template,
    demo_template,
    optimization_template,
    quantifier_template,
)

# Import and expose subset templates
from .subset_templates import smallest_subset_with_property
from .z3_templates import (
    all_distinct,
    array_contains,
    array_is_sorted,
    at_least_k,
    at_most_k,
    exactly_k,
    function_is_injective,
    function_is_surjective,
)


__all__ = [
    # Quantifier templates
    "array_is_sorted",
    "all_distinct",
    "array_contains",
    "exactly_k",
    "at_most_k",
    "at_least_k",
    "function_is_injective",
    "function_is_surjective",
    # Function templates
    "constraint_satisfaction_template",
    "optimization_template",
    "array_template",
    "quantifier_template",
    "demo_template",
    # Subset templates
    "smallest_subset_with_property",
]
