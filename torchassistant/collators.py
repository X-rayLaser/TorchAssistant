import torch
from torchassistant.utils import Serializable


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

        return tuple(transform(col) for col, transform in zip(columns, self.column_transforms))


class CollationError(Exception):
    pass


def to_long_tensor(numbers):
    return torch.LongTensor(numbers)


def to_float_tensor(numbers):
    return torch.FloatTensor(numbers)


def seq_to_tensor(int_lists):
    # todo: finish this
    return int_lists


def img_to_tensor(images):
    # todo: add cropping
    from torchvision.transforms import ToTensor
    to_tensor = ToTensor()
    return torch.stack([to_tensor(image) for image in images])
