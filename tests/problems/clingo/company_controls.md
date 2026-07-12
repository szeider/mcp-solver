### There are four companies: c1, c2, c3, and c4

- Company c1 owns 60% of company c2, giving it direct control over c2.
- Company c1 also owns 20% of company c3.
- Company c2 owns 40% of company c3.
- Company c3 owns 51% of company c4, giving it direct control over c4.

I want to know all the pair of companies X, Y where X is different than Y such that X controls Y, given the previous information on stock possessions of companies.

Company X controls company Y when the shares of Y that X owns directly, plus
the shares of Y owned by companies that X (transitively) controls, together
exceed 50%.

## Output Format

Return a single JSON object on stdout with this schema:

- `satisfiable` (boolean): `true` (the program has an answer set).
- `controls` (list of pairs): every `[X, Y]` with `X != Y` such that X
  controls Y. Each pair is a two-element list of company names (strings).
  Order of pairs does not matter.

Expected result:

```json
{
  "satisfiable": true,
  "controls": [["c1", "c2"], ["c1", "c3"], ["c1", "c4"], ["c3", "c4"]]
}
```
