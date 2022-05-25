class DefaultAdapter:
    def adapt(self, *args):
        # todo: need to know information about model parts and their inputs/outputs
        # if model pipeline was built before data pipeline, this information could be passed to the init method
        return {
            "inputs": {
            },
            "targets": {

            }
        }
