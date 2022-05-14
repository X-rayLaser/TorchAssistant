import torch


class BatchAdapter:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def adapt(self, french_batch, english_batch):
        # todo: support gpu or other devices
        english_input = english_batch[:, :-1]
        english_target = english_batch[:, 1:]

        hidden = torch.zeros(1, 1, self.hidden_size, device="cpu")

        return {
            "inputs": {
                "encoder": {
                    "x": french_batch,
                    "h": hidden
                },
                "decoder": {
                    "y_shifted": english_input
                }
            },
            "targets": {
                "y": english_target
            }
        }
