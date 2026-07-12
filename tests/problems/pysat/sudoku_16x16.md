Solve the following 16x16 Sudoku puzzle, where the letter x represents empty fields: line 1: (x,x,4,7,x,x,A,x,x,x,x,x,3,x,x,5), line 2: (x,6,x,x,x,x,x,9,x,x,B,x,x,x,x,x), line 3: (x,x,x,x,8,x,x,x,7,x,x,x,x,D,x,x), line 4: (G,x,x,x,x,3,x,x,x,x,x,1,x,x,x,x), line 5: (x,x,2,x,x,x,x,x,x,F,x,x,x,x,6,x), line 6: (x,x,x,x,x,x,7,x,x,x,x,C,x,x,x,x), line 7: (x,5,x,x,x,x,x,x,D,x,x,x,x,x,x,8), line 8: (x,x,x,x,x,9,x,x,x,x,x,x,4,x,x,x), line 9: (x,x,x,E,x,x,x,x,x,x,x,8,x,x,x,x), line 10: (x,x,1,x,x,x,x,F,x,x,x,x,x,7,x,x), line 11: (x,x,x,x,x,x,x,x,B,x,x,x,x,x,x,x), line 12: (x,8,x,x,x,x,x,x,x,x,x,x,E,x,x,x), line 13: (x,x,x,x,x,x,x,D,x,x,9,x,x,x,x,x), line 14: (x,x,x,x,F,x,x,x,x,x,x,x,x,x,2,x), line 15: (x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x), line 16: (x,x,x,2,x,x,x,x,x,x,x,x,x,x,x,x). Use the numbers 1-9 and the letters A-G as values.

## Output Format

Return a single JSON object. On success, `satisfiable` is `true` and
`solution` is a 16×16 array (list of 16 rows, each a list of 16 single-
character strings over `1`-`9`, `A`-`G`). Every row, column, and 4×4 box
contains each of the 16 symbols exactly once, and every filled clue in the
puzzle is preserved.

```json
{"satisfiable": true, "solution": [["1", "D", "4", "7", "B", "2", "A", "E", "6", "9", "8", "F", "3", "C", "G", "5"], ["...15 more rows..."]]}
```

If the puzzle has no solution: `{"satisfiable": false}`.