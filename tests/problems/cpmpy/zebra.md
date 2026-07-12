There are four desks. The student studying Biology is sitting at Desk 3. The student who likes Pizza is sitting next to the student who is studying Chemistry. The student studying Physics is not sitting at Desk 2. The student sitting at Desk 2 likes Pasta. Anna likes apples. Ben is not studying Biology. Emily is sitting at Desk 1. The student sitting at Desk 4 likes ice cream. The name of student who is studying Physics has an even number of letters (i.e., Anna or Liam). What food does Liam like and who is studying Literature?

## Output Format

The four students are Emily, Anna, Ben, and Liam; the four subjects are
Biology, Chemistry, Physics, and Literature; the four foods are Pizza, Pasta,
apples, and ice cream; the desks are numbered 1-4 (adjacent desks differ by
one).

Return a single JSON object on stdout with this schema:

- `satisfiable` (boolean): `true` (the puzzle has a unique solution).
- `assignment` (list of 4 objects): one per desk, each with:
  - `desk` (integer 1-4)
  - `name` (string): one of `Emily`, `Anna`, `Ben`, `Liam`.
  - `subject` (string): one of `Biology`, `Chemistry`, `Physics`,
    `Literature`.
  - `food` (string): one of `Pizza`, `Pasta`, `apples`, `ice cream`.

Names, subjects, and foods are compared case-insensitively.

Expected (unique) solution:

```json
{
  "satisfiable": true,
  "assignment": [
    {"desk": 1, "name": "Emily", "subject": "Literature", "food": "Pizza"},
    {"desk": 2, "name": "Ben", "subject": "Chemistry", "food": "Pasta"},
    {"desk": 3, "name": "Anna", "subject": "Biology", "food": "apples"},
    {"desk": 4, "name": "Liam", "subject": "Physics", "food": "ice cream"}
  ]
}
```