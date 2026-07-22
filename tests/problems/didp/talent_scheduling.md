A film crew must shoot 9 scenes, indexed `0`–`8`, one scene per day, in some
chosen order. There are 6 actors, indexed `0`–`5`. Each scene requires a known
set of actors to be present on set that day.

An actor is booked continuously: they arrive on location on the day of their
**first** scheduled scene and stay until the day of their **last** scheduled
scene, inclusive. They must be paid their daily wage for **every** day in that
span -- including any idle days in the middle when they are on location but not
actually needed for that day's scene. An actor who never appears in the chosen
span before or after a given day is not on location and is not paid for it.

So for each actor `a` with daily wage `rate[a]`, if `first_day[a]` and
`last_day[a]` are the days of their earliest and latest scheduled scenes, that
actor costs `rate[a] * (last_day[a] - first_day[a] + 1)`. The **total cost** of a
schedule is the sum of this quantity over all actors.

Choose the order in which to shoot the 9 scenes so that the total cost is as
small as possible. (Shooting scenes with overlapping casts close together shrinks
everyone's span; a bad order strands expensive actors on location across long
idle stretches.)

Which actors each scene needs, as a boolean matrix (row `a` = actor, column `s` =
scene; a `1` means actor `a` is required in scene `s`):

```json
{
  "requirements": [
    [1, 0, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 0, 1, 1]
  ],
  "rate": [4, 1, 3, 6, 3, 3]
}
```

## Output Format

Return a single JSON object on stdout with this schema:

- `order` (list of 9 integers): the order in which the scenes are shot, one per
  day, a permutation of `0..8` (each scene appears exactly once).
- `total_cost` (integer): the total actor-payment cost of the reported order.

Any order achieving the minimum total cost is accepted.

Example (this is a dummy showing the shape only, **not** the real answer):

```json
{
  "order": [2, 0, 5, 8, 1, 7, 3, 6, 4],
  "total_cost": 999
}
```
