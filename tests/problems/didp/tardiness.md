A single machine must process 10 jobs, indexed `0`–`9`. The machine handles one
job at a time, with no preemption: once a job starts it runs to completion before
the next one begins, and the machine is never idle between jobs. Processing the
jobs in some order, the **completion time** `C[j]` of a job is the moment it
finishes, i.e. the sum of the processing times of that job and of every job
scheduled before it.

Each job `j` has a **due date** `d[j]`. If it finishes late its **tardiness** is
`max(0, C[j] - d[j])`; finishing early or on time gives tardiness `0`. Being late
on job `j` costs `w[j]` per unit of time late, so job `j` contributes
`w[j] * max(0, C[j] - d[j])` to the bill.

Choose the order in which to run the 10 jobs so that the **total weighted
tardiness**, summed over all jobs, is as small as possible.

Job data, index-aligned (`p` = processing time, `d` = due date, `w` = weight):

```json
{
  "p": [7, 5, 11, 6, 18, 17, 18, 15, 9, 6],
  "d": [8, 7, 8, 88, 74, 6, 53, 92, 32, 59],
  "w": [8, 1, 7, 7, 1, 8, 5, 4, 2, 6]
}
```

## Output Format

Return a single JSON object on stdout with this schema:

- `order` (list of 10 integers): the sequence in which the jobs are run, a
  permutation of `0..9` (each job appears exactly once).
- `total_weighted_tardiness` (integer): the total weighted tardiness of the
  reported order.

Any order achieving the minimum total weighted tardiness is accepted.

Example (this is a dummy showing the shape only, **not** the real answer):

```json
{
  "order": [3, 7, 1, 5, 0, 9, 2, 8, 4, 6],
  "total_weighted_tardiness": 999
}
```
