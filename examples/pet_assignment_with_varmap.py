"""
Pet Assignment Example using VariableMap

This example shows how to use the VariableMap helper class to solve a pet assignment problem.
"""

from pysat.formula import CNF
from pysat.solvers import Glucose3
from pysat.templates.mapping import VariableMap  # Using VariableMap from templates

# Create a simple logical puzzle:
# Three friends (Alice, Bob, Charlie) each have one pet (cat, dog, rabbit)
# We need to determine who has which pet based on these clues:
# 1. Alice doesn't have a cat
# 2. If Bob has a dog, Charlie has a rabbit
# 3. At least one person has a dog

# Create variable mapping
vars = VariableMap()

# Create variables with meaningful names
def has_pet(person, pet):
    return vars.get_id(f"{person}_has_{pet}")

people = ["Alice", "Bob", "Charlie"]
pets = ["cat", "dog", "rabbit"]

# Create CNF formula
formula = CNF()

# Each person has exactly one pet
for person in people:
    # At least one pet per person
    formula.append([has_pet(person, pet) for pet in pets])
    
    # At most one pet per person
    for i in range(len(pets)):
        for j in range(i+1, len(pets)):
            formula.append([-has_pet(person, pets[i]), -has_pet(person, pets[j])])

# Each pet is owned by exactly one person
for pet in pets:
    # At least one person per pet
    formula.append([has_pet(person, pet) for person in people])
    
    # At most one person per pet
    for i in range(len(people)):
        for j in range(i+1, len(people)):
            formula.append([-has_pet(people[i], pet), -has_pet(people[j], pet)])

# Clue 1: Alice doesn't have a cat
formula.append([-has_pet("Alice", "cat")])

# Clue 2: If Bob has a dog, Charlie has a rabbit
formula.append([-has_pet("Bob", "dog"), has_pet("Charlie", "rabbit")])

# Clue 3: At least one person has a dog
formula.append([has_pet("Alice", "dog"), has_pet("Bob", "dog"), has_pet("Charlie", "dog")])

# Create solver and add formula
solver = Glucose3()
solver.append_formula(formula)

# Solve and interpret results
if solver.solve():
    model = solver.get_model()
    solution = vars.interpret_model(model)
    
    # Create a more readable result
    assignments = {}
    for person in people:
        for pet in pets:
            if solution.get(f"{person}_has_{pet}", False):
                assignments[person] = pet
    
    # Display the results
    export_solution({
        "satisfiable": True,
        "assignments": assignments,
        "variable_mapping": vars.get_mapping()
    })
else:
    export_solution({
        "satisfiable": False,
        "message": "No solution exists"
    })

# Free solver memory
solver.delete() 