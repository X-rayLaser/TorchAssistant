import torch
from scaffolding.utils import Serializable


class BatchDivide(Serializable):
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
        return [torch.stack(lst) for lst in tensor_lists]
