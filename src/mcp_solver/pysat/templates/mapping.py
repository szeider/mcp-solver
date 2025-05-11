"""
Helper classes for mapping between meaningful variable names and SAT variable IDs.
"""


class VariableMap:
    """
    Helper class for mapping between meaningful variable names and SAT variable IDs.
    """

    def __init__(self):
        self.var_to_id = {}
        self.id_to_var = {}
        self.next_id = 1

    def get_id(self, var_name):
        """Get or create variable ID for a named variable"""
        if var_name not in self.var_to_id:
            self.var_to_id[var_name] = self.next_id
            self.id_to_var[self.next_id] = var_name
            self.next_id += 1
        return self.var_to_id[var_name]

    def get_name(self, var_id):
        """Get variable name from ID"""
        return self.id_to_var.get(abs(var_id), f"unknown_{abs(var_id)}")

    def interpret_model(self, model):
        """Convert SAT model to dictionary of variable assignments"""
        result = {}
        for lit in model:
            var_id = abs(lit)
            if var_id in self.id_to_var:
                result[self.id_to_var[var_id]] = lit > 0
        return result

    def get_mapping(self):
        """Return a copy of the current variable mapping"""
        return self.var_to_id.copy()
