class DefaultInputAdapter:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, dataframe: dict) -> dict:
        return {
            self.model_name: {
                "input_1": dataframe["input_1"]
            }
        }
