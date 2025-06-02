"""
Assignment problem helpers for MaxSAT.

This module provides helper functions for common assignment patterns
like workers to tasks, one-to-one mappings, and preference optimization.
"""

from typing import List, Dict, Tuple

from pysat.formula import WCNF

from .cardinality_constraints import exactly_k
from .objective_helpers import maximize_sum


def one_to_one_assignment(wcnf: WCNF, workers: List[str], tasks: List[str], 
                         assignment_vars: Dict[Tuple[str, str], int]) -> None:
    """
    Create constraints for one-to-one assignment between workers and tasks.
    
    Each worker gets exactly one task, each task gets exactly one worker.
    This is a common pattern in assignment problems.
    
    Args:
        wcnf: WCNF formula to modify
        workers: List of worker names/IDs
        tasks: List of task names/IDs
        assignment_vars: Dictionary mapping (worker, task) tuples to variable IDs
    """
    # Each worker assigned to exactly one task
    for worker in workers:
        worker_vars = [assignment_vars[(worker, task)] for task in tasks 
                      if (worker, task) in assignment_vars]
        exactly_k(wcnf, worker_vars, 1)
    
    # Each task assigned to exactly one worker
    for task in tasks:
        task_vars = [assignment_vars[(worker, task)] for worker in workers
                    if (worker, task) in assignment_vars]
        exactly_k(wcnf, task_vars, 1)


def assignment_with_preferences(wcnf: WCNF, assignment_vars: Dict[Tuple[str, str], int],
                               preferences: Dict[Tuple[str, str], int]) -> None:
    """
    Add soft constraints to maximize total preference scores for assignments.
    
    This helper makes it easy to optimize assignments based on preference scores,
    enthusiasm levels, or other positive weights.
    
    Args:
        wcnf: WCNF formula to modify
        assignment_vars: Dictionary mapping (worker, task) tuples to variable IDs
        preferences: Dictionary mapping (worker, task) tuples to preference scores
    """
    var_value_pairs = [(assignment_vars[key], value) 
                       for key, value in preferences.items() 
                       if key in assignment_vars]
    maximize_sum(wcnf, var_value_pairs)


def partial_assignment(wcnf: WCNF, entities: List[str], slots: List[str],
                      assignment_vars: Dict[Tuple[str, str], int],
                      min_per_entity: int = 0, max_per_entity: int = 1,
                      min_per_slot: int = 0, max_per_slot: int = 1) -> None:
    """
    Create constraints for partial assignment where not all entities/slots need to be filled.
    
    More flexible than one_to_one_assignment, allowing for unassigned entities or slots.
    
    Args:
        wcnf: WCNF formula to modify
        entities: List of entity names (e.g., workers, items)
        slots: List of slot names (e.g., tasks, positions)
        assignment_vars: Dictionary mapping (entity, slot) tuples to variable IDs
        min_per_entity: Minimum slots per entity (default 0)
        max_per_entity: Maximum slots per entity (default 1)
        min_per_slot: Minimum entities per slot (default 0)
        max_per_slot: Maximum entities per slot (default 1)
    """
    # Constraints for entities
    for entity in entities:
        entity_vars = [assignment_vars[(entity, slot)] for slot in slots
                      if (entity, slot) in assignment_vars]
        if entity_vars:
            if min_per_entity > 0:
                at_least_k(wcnf, entity_vars, min_per_entity)
            if max_per_entity < len(entity_vars):
                at_most_k(wcnf, entity_vars, max_per_entity)
    
    # Constraints for slots
    for slot in slots:
        slot_vars = [assignment_vars[(entity, slot)] for entity in entities
                    if (entity, slot) in assignment_vars]
        if slot_vars:
            if min_per_slot > 0:
                at_least_k(wcnf, slot_vars, min_per_slot)
            if max_per_slot < len(slot_vars):
                at_most_k(wcnf, slot_vars, max_per_slot)


def many_to_many_assignment(wcnf: WCNF, entities_a: List[str], entities_b: List[str],
                           assignment_vars: Dict[Tuple[str, str], int],
                           min_b_per_a: int, max_b_per_a: int,
                           min_a_per_b: int, max_a_per_b: int) -> None:
    """
    Create constraints for many-to-many assignments with cardinality bounds.
    
    Useful for problems like shift scheduling where workers can have multiple shifts
    and shifts can have multiple workers.
    
    Args:
        wcnf: WCNF formula to modify
        entities_a: First set of entities (e.g., workers)
        entities_b: Second set of entities (e.g., shifts)
        assignment_vars: Dictionary mapping (a, b) tuples to variable IDs
        min_b_per_a: Minimum number of B entities per A entity
        max_b_per_a: Maximum number of B entities per A entity
        min_a_per_b: Minimum number of A entities per B entity
        max_a_per_b: Maximum number of A entities per B entity
    """
    from .cardinality_constraints import at_least_k, at_most_k
    
    # Constraints for entities in set A
    for a in entities_a:
        a_vars = [assignment_vars[(a, b)] for b in entities_b
                 if (a, b) in assignment_vars]
        if a_vars:
            if min_b_per_a > 0:
                at_least_k(wcnf, a_vars, min_b_per_a)
            if max_b_per_a < len(a_vars):
                at_most_k(wcnf, a_vars, max_b_per_a)
    
    # Constraints for entities in set B
    for b in entities_b:
        b_vars = [assignment_vars[(a, b)] for a in entities_a
                 if (a, b) in assignment_vars]
        if b_vars:
            if min_a_per_b > 0:
                at_least_k(wcnf, b_vars, min_a_per_b)
            if max_a_per_b < len(b_vars):
                at_most_k(wcnf, b_vars, max_a_per_b)