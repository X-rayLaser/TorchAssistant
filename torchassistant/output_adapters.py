class OutputAdapter:
    def __call__(self, all_outputs):
        raise NotImplementedError


class IdentityAdapter(OutputAdapter):
    def __call__(self, all_outputs):
        return all_outputs


class SelectVariables(OutputAdapter):
    def __init__(self, variable_names):
        self.variable_names = variable_names

    def __call__(self, all_outputs):
        return {var_name: all_outputs[var_name] for var_name in self.variable_names}
