# Simple MaxSAT Test Problem

I need to solve a simple MaxSAT optimization problem where we have four potential items to select, each with a different value. We have constraints on which items can be selected together, and we want to maximize the total value.

Here are the details:
- Four items: Item 1 (value 10), Item 2 (value 5), Item 3 (value 7), Item 4 (value 12)
- Hard constraint: Items 1 and 4 are mutually exclusive (can't select both)
- Hard constraint: Can't select all items (budget limit)
- Hard constraint: If Item 3 is selected, Item 2 must also be selected (dependency)
- Goal: Maximize the total value of selected items

Please create a MaxSAT formulation and solve this problem. Show which items should be selected to maximize value while satisfying all constraints.