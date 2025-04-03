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