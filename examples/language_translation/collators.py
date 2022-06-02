import torch
from scaffolding.utils import Serializable


class MyCollator(Serializable):
    """Simple collator that only works with batch size = 1"""
    def __init__(self, num_french_words, num_english_words):
        self.num_french_words = num_french_words
        self.num_english_words = num_english_words

    def __call__(self, batch):
        inputs = [x for x, y in batch]
        targets = [y for x, y in batch]

        return torch.LongTensor(inputs), torch.LongTensor(targets)

    def collate_inputs(self, *inputs):
        return torch.LongTensor(inputs)
