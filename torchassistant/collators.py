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
