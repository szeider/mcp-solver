"""Solver helper library, importable inside solve-time kernels.

Submodules are imported explicitly by the agent (``from mcp_solver.helpers
import pysat``), never eagerly here: each submodule requires its solver
package (python-sat, z3-solver), which is only present in kernels launched
with the matching ``--with`` list.
"""
