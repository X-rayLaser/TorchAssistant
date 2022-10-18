import torch
from examples.language_translation.datasets import normalize_string


class InputConverter:
    def __call__(self, x):
        return normalize_string(x)


class BatchAdapter:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def __call__(self, batch: dict) -> dict:
        french_batch = batch["french_batch"]
        english_batch = batch["english_batch"]

        english_input = english_batch[:, :-1]

        hidden = torch.zeros(1, 1, self.hidden_size, device="cpu")

        return {
            "encoder_model": {
                "x": french_batch,
                "h": hidden
            },
            "decoder_model": {
                "y_shifted": english_input
            }
        }


class NullAdapter:
    def __call__(self, batch):
        return batch


class OutputAdapter:
    def __call__(self, batch):
        english_batch = batch["english_batch"]
        batch["y"] = english_batch[:, 1:]
        return batch


class BatchInferenceAdapter:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def __call__(self, batch: dict):
        french_batch = batch["french_batch"]
        hidden = torch.zeros(1, 1, self.hidden_size, device="cpu")

        return {
            "encoder_model": {
                "x": french_batch,
                "h": hidden
            },
            "decoder_model": {
                "sos": torch.LongTensor([[1]])
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
