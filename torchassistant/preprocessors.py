import re
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
    def __init__(self, fit_index):
        self.fit_index = fit_index

    def fit(self, dataset):
        progress_bar = ProgressBar()

        indices = self._try_shuffling_indices(dataset)

        total_steps = self._get_truncated_steps(len(dataset))
        indices = indices[:total_steps]

        examples = (dataset[idx][self.fit_index] for idx in indices)
        gen = self._do_fit(examples)
        for i, data in enumerate(gen):
            if (i + 1) % 100 == 0:
                step_number = i + 1
                progress = progress_bar.updated(step_number, total_steps, cols=50)
                print(f'\rFitting preprocessor: {progress} {step_number}/{total_steps}', end='')

        whitespaces = ' ' * 100
        print(f'\r{whitespaces}\rDone!')

    def _get_truncated_steps(self, num_examples):
        if num_examples > 10 ** 6:
            fraction = 0.01
        elif num_examples > 10 ** 4:
            fraction = 0.1
        elif num_examples > 10 ** 3:
            fraction = 0.5
        else:
            fraction = 1.0

        return int(num_examples * fraction)

    def _try_shuffling_indices(self, dataset):
        num_examples = len(dataset)
        if num_examples < 10 ** 7:
            indices = list(range(num_examples))
            shuffle(indices)
        else:
            indices = range(num_examples)
        return indices

    def _do_fit(self, column):
        """A generator object that performs fitting and yields index of currently fit data"""
        yield 0


class ImagePreprocessor(TrainablePreprocessor):
    def __init__(self, fit_index):
        super().__init__(fit_index)
        self.mu = []
        self.sd = []

    def _do_fit(self, column):
        to_tensor = ToTensor()

        mu = []
        sd = []

        for pil_image in column:
            tensor = to_tensor(pil_image)
            mu.append(tensor.mean(dim=[1, 2]))  # channel-wise statistics
            sd.append(tensor.std(dim=[1, 2]))
            yield pil_image

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


class TextPreprocessor(TrainablePreprocessor):
    sentence_start = '<start>'
    sentence_end = '<end>'
    out_of_vocab = '<OOV>'

    def __init__(self, fit_index):
        super().__init__(fit_index)
        self.word_indices = {}
        self.index2word = {}
        self.num_words = 0

    @property
    def num_classes(self):
        return self.num_words

    def _do_fit(self, column):
        self._add_word(self.sentence_start)
        self._add_word(self.sentence_end)
        self._add_word(self.out_of_vocab)

        for text in column:
            self._fit_text(text)
            yield text

    def _fit_text(self, text):
        words = re.findall(r'\w+', text)
        for w in words:
            self._add_word(w)

    def _add_word(self, word):
        self.index2word[self.num_words] = word
        self.word_indices[word] = self.num_words
        self.num_words += 1

    def process(self, value):
        words = re.findall(r'\w+', value)
        start_token = self.encode_word(self.sentence_start)
        end_token = self.encode_word(self.sentence_end)
        text_tokens = [self.encode_word(w) for w in words]
        return [start_token] + text_tokens + [end_token]

    def encode_word(self, word):
        return self.word_indices.get(word, self.word_indices[self.out_of_vocab])

    def decode_token(self, token):
        return self.index2word.get(token, self.out_of_vocab)

    def decode(self, tokens):
        words = [self.decode_token(token) for token in tokens]
        return ' '.join(words)

    def state_dict(self):
        return self.__dict__.copy()

    def load_state_dict(self, state_dict):
        self.__dict__ = state_dict.copy()
        self.index2word = {int(k): v for k, v in self.index2word.items()}
