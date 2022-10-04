import torch
from torchassistant.collators import BaseCollator, BatchDivide


class MyCollator(BatchDivide):
    """Simple collator that only works with batch size = 1"""
    def __init__(self, num_french_words, num_english_words):
        self.num_french_words = num_french_words
        self.num_english_words = num_english_words

    def __call__(self, batch):
        batch = super().__call__(batch)
        return [torch.LongTensor(inputs) for inputs in batch]

    def collate_inputs(self, *inputs):
        return torch.LongTensor(inputs)


def build_collator(session):
    return MyCollator(
        session.preprocessors["french_encoder"].num_french_words,
        session.preprocessors["english_encoder"].num_english_words
    )
