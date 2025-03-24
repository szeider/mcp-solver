# Queens and Knights Problem (6x6)

Create a MiniZinc model to place 6 queens and 5 knights on a 6x6 chessboard in one complete formulation. The following constraints must be satisfied:

1. No two queens may threaten each other (no two queens in the same row, column, or diagonal)
2. No knight may threaten any queen
3. No queen may threaten any knight
4. No two knights may threaten each other

For your model:
- Represent positions of the 6 queens and 5 knights using appropriate arrays and decision variables
- Define all necessary constraints to ensure the requirements above are met
- Add appropriate constraints to handle the chess piece movement rules:
  * Queens attack along rows, columns, and diagonals
  * Knights move in an L-shape (2 squares in one direction and 1 square perpendicular)
- Use a solve satisfy directive to find a valid solution

Formulate this model in one complete piece, not incrementally building up the solution. 