# Network Monitoring (Simplified)

A company needs to place monitoring software on servers to ensure all network connections are monitored. Each server has a cost and monitoring value.

## Network Structure

### Servers (6 total)
| Server | Type | Value | Cost |
|--------|------|-------|------|
| Core1 | Core | 10 | 3 |
| Core2 | Core | 10 | 3 |
| Web1 | Web | 6 | 2 |
| Web2 | Web | 6 | 2 |
| DB1 | Database | 8 | 3 |
| Edge1 | Edge | 5 | 1 |

### Network Connections (8 edges)
- Core1 connects to: Core2, Web1, DB1
- Core2 connects to: Web2, DB1, Edge1
- Web1 connects to: Web2
- Web2 connects to: Edge1

## Constraints

### Hard Constraints
1. **Edge Coverage**: Every network connection must be monitored by at least one server on either end
2. **Critical Server**: At least one Core server (Core1 or Core2) must have monitoring

### Soft Constraints
1. **Cost Minimization**: Minimize total deployment cost (weight = cost)
2. **Value Maximization**: Maximize monitoring value (weight = 20 - value)

## Task
Find the optimal set of servers to monitor that:
- Covers all network connections
- Includes at least one Core server
- Balances cost and value

## Expected Output
- List of servers selected for monitoring
- Total cost
- Total monitoring value
- Verification that all edges are covered