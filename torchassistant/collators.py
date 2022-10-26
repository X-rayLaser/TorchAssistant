import torch
from torchvision.transforms import ToTensor

from torchassistant.utils import Serializable, pad_sequences


class BaseCollator(Serializable):
    def __call__(self, batch):
        raise NotImplementedError

    def collate_inputs(self, *inputs):
        return self(inputs)


class BatchDivide(BaseCollator):
    """Divide batch into a tuple of lists"""
    def __call__(self, batch):
        num_vars = len(batch[0])
        res = [[] for _ in range(num_vars)]

        for example in batch:
            for i, inp in enumerate(example):
                res[i].append(inp)

        return res


class StackTensors(BatchDivide):
    def __call__(self, batch):
        tensor_lists = super().__call__(batch)
        # todo: this is too naive implementation; handle other cases; raise errors for wrong data types/shapes
        return [torch.stack(lst) if isinstance(lst[0], torch.Tensor) else torch.tensor(lst) for lst in tensor_lists]


class ColumnWiseCollator(BatchDivide):
    def __init__(self, column_transforms):
        self.column_transforms = column_transforms

    def __call__(self, batch):
        columns = super().__call__(batch)

        if len(self.column_transforms) != len(columns):
            raise CollationError(
                f'Expects equal # of columns and transforms. '
                f'Got {len(columns)} columns and {len(self.column_transforms)} transforms'
            )

        cols_with_transforms = zip(columns, self.column_transforms)
        return tuple(transform(col) for col, transform in cols_with_transforms)


class CollationError(Exception):
    pass


def to_long_tensor(numbers):
    return torch.LongTensor(numbers)


def to_float_tensor(numbers):
    return torch.FloatTensor(numbers)


def seqs_to_tensor(token_lists):
    filler = token_lists[0][-1]
    inputs, mask = pad_sequences(token_lists, filler)
    return torch.LongTensor(inputs)


def stack_tensors(tensors):
    return torch.stack(tensors)


def no_transform(x):
    return x


class SeqsToOneHot:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, token_lists):
        # todo: we need to access mask later when computing loss
        filler = token_lists[0][-1]
        inputs, mask = pad_sequences(token_lists, filler)
        return one_hot_tensor(inputs, self.num_classes)


def images_to_tensor(images):
    # todo: add cropping
    to_tensor = ToTensor()
    return torch.stack([to_tensor(image) for image in images])


def one_hot_tensor(classes, num_classes):
    """Form a 1-hot tensor from a list of class sequences

    :param classes: list of class sequences (list of lists)
    :param num_classes: total number of available classes
    :return: torch.tensor of shape (batch_size, max_seq_len, num_classes)
    :raise ValueError: if there is any class value that is negative or >= num_classes
    """

    # todo: clean this up (make this function into a callable class)

    if not hasattr(one_hot_tensor, 'eye'):
        one_hot_tensor.eye = {}

    if num_classes not in one_hot_tensor.eye:
        one_hot_tensor.eye[num_classes] = torch.eye(num_classes, dtype=torch.float32)
    eye = one_hot_tensor.eye[num_classes]
    try:
        tensors = [eye[class_seq] for class_seq in classes]
    except IndexError:
        msg = f'Every class must be a non-negative number less than num_classes={num_classes}. ' \
              f'Got classes {classes}'
        raise ValueError(msg)

    return torch.stack(tensors)

