# MaxSAT Feature Selection Problem

You are designing a software product with multiple optional features. Each feature has a profit value but may depend on other features being included. Your goal is to select a subset of features that maximizes the total profit while respecting all dependencies.

## Features and their profit values:

- Base product (required): $0
- Premium upgrade: $10
- Cloud storage: $15
- Sync capability: $7 (depends on Cloud storage)
- Mobile app: $12
- Analytics dashboard: $20 (depends on Premium)

## Dependencies:

1. Base product is required
2. Premium upgrade requires Base product
3. Sync capability requires Cloud storage
4. Analytics dashboard requires Premium upgrade
5. Mobile app requires Base product

## Task

Find the optimal feature selection that maximizes total profit while respecting all dependencies. Use MaxSAT with the RC2 solver to solve this optimization problem.

1. Define variables for each feature
2. Add hard constraints for dependencies
3. Add soft constraints with weights equal to the profit values
4. Use the RC2 solver to compute the optimal solution
5. Report which features are selected and the total profit