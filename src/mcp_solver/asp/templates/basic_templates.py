"""
Basic templates for ASP (clingo).

This module provides template functions for common ASP patterns.
"""
from typing import List

def facts(atoms: List[str]) -> str:
    """
    Generate facts from a list of atoms.

    Args:
        atoms: List of atoms (strings)

    Returns:
        A string containing the ASP facts.
    """
    return "\n".join([f"{atom}." for atom in atoms])

def rule(head: str, body: List[str]) -> str:
    """
    Generate a simple ASP rule.

    Args:
        head: The head of the rule.
        body: A list of atoms in the body of the rule.

    Returns:
        A string containing the ASP rule.
    """
    if not body:
        return f"{head}."
    return f"{head} :- {', '.join(body)}." 