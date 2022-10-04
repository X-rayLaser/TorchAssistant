class DefaultAdapter:
    def __init__(self, model, target_names):
        self.model = model
        self.target_names = target_names

    def adapt(self, *args):
        # todo: consider returning 2 dicts: one for inputs, one for targets
        # also leave out associated model names from representation; this can be automatically determined

        inputs_dict = {}

        unused_inputs = list(reversed(args))
        prev_outputs = set()
        for module in self.model:
            module_dict = {name: unused_inputs.pop()
                           for name in module.inputs if name not in prev_outputs}

            prev_outputs = prev_outputs.union(module.outputs)
            inputs_dict[module.name] = module_dict

        targets_dict = {name: unused_inputs.pop() for name in self.target_names}

        return {
            "inputs": inputs_dict,
            "targets": targets_dict
        }
