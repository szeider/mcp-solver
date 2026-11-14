#### Suppose you are organizing a party and you want to invite people under the following rules

- You will invite Alice unless you know she is not coming.
- You will invite Bob if Alice is not invited.
- You will invite Carol only if both Alice and Bob are invited.

## Output Format

Return a single JSON object on stdout with this schema:

- `satisfiable` (boolean): `true` if the program has an answer set, `false`
  otherwise.
- `invited` (list of strings): the lowercase names of the people invited in
  the answer set. Order does not matter.

Nothing tells us that Alice is not coming.

Example (this is a dummy showing the shape only, **not** the real answer):

```json
{"satisfiable": true, "invited": ["bob", "carol"]}
```
