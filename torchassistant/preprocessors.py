class ValuePreprocessor:
    def fit(self, dataset):
        pass

    def process(self, value):
        pass

    def __call__(self, value):
        return self.process(value)

    def wrap_preprocessor(self, preprocessor):
        """Wraps a given preprocessor with self.

        Order of preprocessing: pass a value through a new preprocessor,
        then preprocess the result with self

        :param preprocessor: preprocessor to wrap
        :return: A callable
        """
        return lambda value: self.process(preprocessor(value))


class ExamplePreprocessor:
    def process(self, values):
        pass


class NullProcessor(ValuePreprocessor):
    def process(self, value):
        return value


class SimpleNormalizer(ValuePreprocessor):
    def process(self, value):
        print('yes simple preprocessor')
        return value / 255
