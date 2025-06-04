# Software Package Selection

## Problem Description

Select software packages for a development project, considering dependencies and value.

## Available Packages
| Package | Value | Description |
|---------|-------|-------------|
| Core | 0 | Basic framework (required) |
| UI | 6 | User interface library |
| Auth | 8 | Authentication module |
| API | 7 | API integration tools |
| Analytics | 5 | Analytics dashboard |

## Hard Constraints
1. Core package must be selected (it's the foundation)
2. UI requires Core (dependency)
3. Auth requires Core (dependency)
4. Analytics requires both UI and API (complex dependency)

## Soft Constraints
- Maximize total value of selected packages
- Prefer to have at least 3 packages selected (penalty of 4 if fewer)

## Task
Find the optimal set of packages that:
1. Satisfies all dependencies
2. Maximizes total value while considering the package count preference

Output which packages to select and the total value achieved.