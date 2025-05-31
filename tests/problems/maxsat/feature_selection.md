# Feature Selection Optimization Problem

I need to select features for a software product to maximize customer value while respecting constraints. Each feature has a value (benefit to customers) and there are dependencies between features.

## Features and Values:
- Base product (required, value: 0)
- Premium upgrade (value: 10)
- Cloud storage (value: 15)
- Sync capability (value: 7)
- Mobile app (value: 12)
- Analytics dashboard (value: 20)

## Constraints:
1. The base product is required
2. Premium upgrade requires the base product
3. Sync capability requires cloud storage
4. Analytics dashboard requires premium upgrade
5. Budget limitations: We cannot include all features (need at least one to be excluded)

## Goal:
Find the optimal set of features to include that maximizes the total value while respecting all constraints.

Use MaxSAT optimization to solve this problem and show which features should be included to maximize value.