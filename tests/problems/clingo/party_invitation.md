#### Suppose you are organizing a party and you want to invite people under the following rules

- You will invite Alice unless you know she is not coming.
- You will invite Bob if Alice is not invited.
- You will invite Carol only if both Alice and Bob are invited.

## Output Format

Return a single JSON object on stdout with this schema:

- `satisfiable` (boolean): `true` (the program has an answer set).
- `invited` (list of strings): the lowercase names of the people invited in
  the answer set. Order does not matter.

Nothing tells us Alice is not coming, so Alice is invited by default; Bob is
invited only if Alice is not, so Bob is out; Carol needs both Alice and Bob,
so Carol is out. Expected result:

```json
{"satisfiable": true, "invited": ["alice"]}
```
