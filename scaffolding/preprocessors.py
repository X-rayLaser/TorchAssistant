class ValuePreprocessor:
    def fit(self, dataset):
        pass

    def process(self, value):
        pass

    def __call__(self, value):
        return self.process(value)


class ExamplePreprocessor:
    def process(self, values):
        pass
