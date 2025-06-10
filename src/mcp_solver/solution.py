"""
Solution module for convenience imports.

This module simplifies imports by re-exporting the export_maxsat_solution function
from the maxsat.solution module, making it available with a simpler import path.
"""

# Re-export the export_maxsat_solution function from the maxsat.solution module
# This allows users to import it directly with "from solution import export_maxsat_solution"
try:
    from .maxsat.solution import export_maxsat_solution
except ImportError:
    # Define a placeholder function in case the import fails
    def export_maxsat_solution(*args, **kwargs):
        """
        Placeholder function when the real implementation is not available.
        This should never be called in normal operation.
        """
        raise NotImplementedError(
            "export_maxsat_solution is not available. "
            "This might indicate that the MaxSAT mode is not properly installed or initialized."
        )
