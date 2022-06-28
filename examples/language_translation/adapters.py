import torch
from examples.language_translation.datasets import normalize_string


class InputAdapter:
    def __call__(self, x):
        return normalize_string(x)


class BatchAdapter:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def adapt(self, french_batch, english_batch):
        english_input = english_batch[:, :-1]
        english_target = english_batch[:, 1:]

        hidden = torch.zeros(1, 1, self.hidden_size, device="cpu")

        return {
            "inputs": {
                "encoder_model": {
                    "x": french_batch,
                    "h": hidden
                },
                "decoder_model": {
                    "y_shifted": english_input
                }
            },
            "targets": {
                "y": english_target
            }
        }


class InferenceAdapter:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def adapt(self, french_batch):
        hidden = torch.zeros(1, 1, self.hidden_size, device="cpu")

        return {
            "inputs": {
                "encoder_model": {
                    "x": french_batch,
                    "h": hidden
                },
                "decoder_model": {
                    "sos": torch.LongTensor([[1]])
                }
            }
        }
