import torch
from torchassistant.collators import BatchDivide


class ZeroPadImages(BatchDivide):
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, batch):
        parts = super().__call__(batch)
        if type(self.pad_index) is int:
            parts[self.pad_index] = self.zero_pad(parts[self.pad_index])
        elif type(self.pad_index) is list or type(self.pad_index) is tuple:
            for idx in self.pad_index:
                parts[idx] = self.zero_pad(parts[idx])
        else:
            raise Exception('Invalid padding index')

        return parts

    def zero_pad(self, tensors):
        max_height = max([t.shape[1] for t in tensors])
        max_width = max([t.shape[2] for t in tensors])

        padded_tensors = []
        for t in tensors:
            channels, height, width = t.shape
            padding_top, padding_bottom = self.calculate_padding(max_height, height)
            padding_left, padding_right = self.calculate_padding(max_width, width)
            padding = (padding_top, padding_bottom, padding_left, padding_right)
            new_tensor = torch.nn.ConstantPad2d(padding, 0)(t)
            padded_tensors.append(new_tensor)

        return torch.stack(padded_tensors)

    def calculate_padding(self, max_length, length):
        len_diff = max_length - length
        padding1 = len_diff // 2
        padding2 = len_diff - padding1
        return padding1, padding2
