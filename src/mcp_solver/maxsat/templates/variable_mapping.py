"""
Helper classes for mapping between meaningful variable names and MaxSAT variable IDs.

This module provides utilities for creating and managing variable mappings in MaxSAT problems,
making it easier to translate between human-readable variable names and numeric variable IDs.
"""


class VariableMap:
    """
    Helper class for mapping between meaningful variable names and MaxSAT variable IDs.
    
    This class provides methods for creating variables with meaningful names,
    retrieving variable IDs, and interpreting MaxSAT models in terms of the named variables.
    """

    def __init__(self):
        """Initialize an empty variable mapping."""
        self.var_to_id = {}
        self.id_to_var = {}
        self.next_id = 1

    def get_id(self, var_name):
        """
        Get or create variable ID for a named variable.
        
        Args:
            var_name: Name of the variable
            
        Returns:
            Integer variable ID for use in MaxSAT formulas
        """
        if var_name not in self.var_to_id:
            self.var_to_id[var_name] = self.next_id
            self.id_to_var[self.next_id] = var_name
            self.next_id += 1
        return self.var_to_id[var_name]

    def get_name(self, var_id):
        """
        Get variable name from ID.
        
        Args:
            var_id: Variable ID (positive or negative)
            
        Returns:
            Name of the variable, or "unknown_X" if not found
        """
        return self.id_to_var.get(abs(var_id), f"unknown_{abs(var_id)}")

    def interpret_model(self, model):
        """
        Convert MaxSAT model to dictionary of variable assignments.
        
        Args:
            model: MaxSAT model (list of integers)
            
        Returns:
            Dictionary mapping variable names to boolean values
        """
        result = {}
        for lit in model:
            var_id = abs(lit)
            if var_id in self.id_to_var:
                result[self.id_to_var[var_id]] = lit > 0
        return result

    def get_mapping(self):
        """
        Return a copy of the current variable mapping.
        
        Returns:
            Dictionary mapping variable names to their IDs
        """
        return self.var_to_id.copy()
    
    def create_var(self, name):
        """
        Create a new variable with the given name.
        
        This is a convenience method that is equivalent to get_id(name).
        
        Args:
            name: Name of the variable
            
        Returns:
            Integer variable ID for use in MaxSAT formulas
        """
        return self.get_id(name)
    
    def create_vars(self, names):
        """
        Create multiple variables at once.
        
        Args:
            names: List or iterable of variable names
            
        Returns:
            Dictionary mapping names to variable IDs
        """
        return {name: self.get_id(name) for name in names}


def create_variable_map():
    """
    Create and return a new VariableMap instance.
    
    This is a convenience function to create a variable mapping.
    
    Returns:
        New VariableMap instance
    """
    return VariableMap()