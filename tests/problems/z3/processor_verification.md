You are given a simplified 8-bit processor model with the following components:

- 4 registers (R0-R3), each storing 8-bit values
- A small memory array with 8 locations (addressable by 3 bits)
- A zero flag that gets set when certain operations produce a zero result

The processor executes the following instruction sequence:

```
1. LOAD R1, [R0]       # Load memory at address in R0 into R1
2. XOR R2, R1, R0      # R2 = R1 XOR R0
3. AND R3, R2, #1      # R3 = R2 & 1 (extract lowest bit)
4. STORE R3, [R0+1]    # Store R3 to memory at address R0+1
5. COND(ZERO) OR R2, R2, #1  # If zero flag set, set lowest bit of R2
```

The zero flag is updated after instructions 1-3 based on whether the result is zero.

Using Z3 SMT solver with bitvector theory, determine whether the following property holds:

**After executing this instruction sequence, does register R3 always contain the parity bit of register R0?**

The parity bit of a value is defined as 1 if the number of 1 bits in its binary representation is odd, and 0 if the number is even.

Provide a clear answer with evidence supporting your conclusion. If the property does not hold, provide a specific counterexample showing register and memory values.

## Output Format

Return a single JSON object on stdout with this schema:

- `property_holds` (boolean): `true` if R3 always equals the parity bit of
  R0, `false` otherwise.
- `counterexample` (object): **required when `property_holds` is `false`.**
  Must contain enough of the initial state to reproduce the falsifying run:
  - `initial_registers` (object): at least `R0` (integer 0-255).
  - `initial_memory` (list of 8 integers 0-255): memory contents before
    execution.
  Additional fields (`final_registers`, `final_memory`, `zero_flags`,
  `evidence`) are optional and ignored by validation.

The property does **not** hold: R3 ends up as the low bit of
`memory[R0] XOR R0`, which is not the bit-count parity of R0. The expected
verdict is therefore `property_holds: false` with a valid counterexample.

Example:

```json
{
  "property_holds": false,
  "counterexample": {
    "initial_registers": {"R0": 253, "R1": 0, "R2": 0, "R3": 0},
    "initial_memory": [1, 1, 1, 1, 1, 1, 1, 1]
  }
}
```