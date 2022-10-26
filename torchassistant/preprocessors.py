import torch

from torchvision.transforms import ToTensor


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


class ImagePreprocessor(ValuePreprocessor):
    def __init__(self, fit_index):
        self.fit_index = fit_index
        self.mu = []
        self.sd = []

    def fit(self, dataset):
        to_tensor = ToTensor()

        mu = []
        sd = []
        # todo: shuffle indices and use 1/10 first ones to fit
        for i, example in enumerate(dataset):
            if i > 1000:
                break

            pil_image = example[self.fit_index]
            tensor = to_tensor(pil_image)
            mu.append(tensor.mean(dim=[1, 2]))  # channel-wise statistics
            sd.append(tensor.std(dim=[1, 2]))

        # todo: maybe find sd in terms of found mu in its formula
        self.mu = torch.stack(mu).mean(dim=0).tolist()
        self.sd = torch.stack(sd).mean(dim=0).tolist()

    def process(self, value):
        to_tensor = ToTensor()
        tensor = to_tensor(value)
        num_channels = len(tensor)
        for c in range(num_channels):
            tensor[c] = (tensor[c] - self.mu[c]) / self.sd[c]
        return tensor

    def state_dict(self):
        return self.__dict__.copy()

    def load_state_dict(self, state_dict):
        self.__dict__ = state_dict.copy()


class TextPreprocessor(ValuePreprocessor):
    def __init__(self, fit_index):
        self.fit_index = fit_index

    def fit(self, dataset):
        pass

    def process(self, value):
        pass
