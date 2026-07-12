# Furniture Arrangement Problem

You are tasked with arranging 5 pieces of furniture (Sofa, TV, Bookshelf, Dining Table, and Desk) in 4 rooms (Living Room, Bedroom, Study, and Dining Room) of a house.

Each piece of furniture must be placed in exactly one room. The Living Room can have at most 3 pieces of furniture. The Bedroom can have at most 2 pieces of furniture. The TV and Sofa must be placed in the same room. The Desk and Bookshelf must be placed in the Study. The Dining Table must be placed in the Dining Room. The Sofa cannot be placed in the Study because it won't fit.

Find a valid arrangement of all furniture items that satisfies all constraints.

## Output Format

Return a single JSON object. On success, `satisfiable` is `true` and
`arrangement` maps each of the five furniture items ("Sofa", "TV",
"Bookshelf", "Dining Table", "Desk") to one of the four rooms ("Living Room",
"Bedroom", "Study", "Dining Room").

```json
{"satisfiable": true, "arrangement": {"Sofa": "Living Room", "TV": "Living Room", "Bookshelf": "Study", "Dining Table": "Dining Room", "Desk": "Study"}}
```

If no arrangement exists: `{"satisfiable": false}`.