from random import shuffle
import torch

from torchvision.transforms import ToTensor
from torchassistant.formatters import ProgressBar


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
        return value / 255


class TrainablePreprocessor(ValuePreprocessor):
    def fit(self, dataset):
        progress_bar = ProgressBar()

        indices = self._try_shuffling_indices(dataset)

        total_steps = int(len(dataset) * 0.1)
        indices = indices[:total_steps]
        gen = self._do_fit(dataset, indices)
        for i, idx in enumerate(gen):
            if (i + 1) % 100 == 0:
                step_number = i + 1
                progress = progress_bar.updated(step_number, total_steps, cols=50)
                print(f'\rFitting preprocessor: {progress} {step_number}/{total_steps}', end='')

        whitespaces = ' ' * 100
        print(f'\r{whitespaces}\rDone!')

    def _try_shuffling_indices(self, dataset):
        num_examples = len(dataset)
        if num_examples < 10 ** 7:
            indices = list(range(num_examples))
            shuffle(indices)
        else:
            indices = range(num_examples)
        return indices

    def _do_fit(self, dataset, indices):
        """A generator object that performs fitting and yields index of currently fit example"""
        yield 0


class ImagePreprocessor(TrainablePreprocessor):
    def __init__(self, fit_index):
        self.fit_index = fit_index
        self.mu = []
        self.sd = []

    def _do_fit(self, dataset, indices):
        to_tensor = ToTensor()

        mu = []
        sd = []

        for idx in indices:
            example = dataset[idx]
            pil_image = example[self.fit_index]
            tensor = to_tensor(pil_image)
            mu.append(tensor.mean(dim=[1, 2]))  # channel-wise statistics
            sd.append(tensor.std(dim=[1, 2]))
            yield idx

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
