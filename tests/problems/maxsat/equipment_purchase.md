# Equipment Purchase Optimization

A laboratory needs to purchase equipment with a budget of $7000. Each piece of equipment has a cost and provides research capabilities.

## Available Equipment
| Equipment | Cost | Capability Value |
|-----------|------|------------------|
| Analyzer | $3500 | 9 points |
| Bench | $2500 | 6 points |
| Computer | $2000 | 5 points |
| Desk | $1500 | 4 points |

## Constraints
1. **Budget**: Total cost must not exceed $7000
2. **Dependencies**: Analyzer requires Computer to function
3. **Synergy**: Bench and Desk together provide an additional 2 points

## Task
Find the optimal set of equipment that maximizes total capability value while satisfying all constraints.

## Expected Output
- Equipment selected
- Total cost
- Total value (including any synergy bonus)