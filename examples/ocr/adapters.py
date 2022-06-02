import torch
from PIL import Image


class InputAdapter:
    def __call__(self, image_path):
        return Image.open(image_path)


class InferenceAdapter:
    def __init__(self, alphabet_size, decoder_hidden_size):
        self.alphabet_size = alphabet_size
        self.hidden_size = decoder_hidden_size
        self.eye = torch.eye(self.alphabet_size)

    def adapt(self, images_batch):
        if not hasattr(self, 'eye'):
            self.eye = torch.eye(self.alphabet_size)

        images_batch = torch.stack(images_batch)

        hidden = torch.zeros(1, 1, self.hidden_size, device="cpu")

        sos = torch.zeros(1, self.alphabet_size)
        sos[0, 2] = 1.0

        return {
            "inputs": {
                "encoder": {
                    "x": images_batch
                },
                "decoder": {
                    "h_d": hidden,
                    "sos": sos
                }
            }
        }


class BatchAdapter:
    def __init__(self, alphabet_size, decoder_hidden_size):
        self.alphabet_size = alphabet_size
        self.hidden_size = decoder_hidden_size
        self.eye = torch.eye(self.alphabet_size)

    def adapt(self, images_batch, transcriptions_batch):
        if not hasattr(self, 'eye'):
            self.eye = torch.eye(self.alphabet_size)

        images_batch = torch.stack(images_batch)

        transcriptions_input = self.eye[transcriptions_batch].unsqueeze(0)[:, :-1]
        transcriptions_target = torch.LongTensor(transcriptions_batch)[:, 1:]

        hidden = torch.zeros(1, 1, self.hidden_size, device="cpu")

        return {
            "inputs": {
                "encoder": {
                    "x": images_batch
                },
                "decoder": {
                    "h_d": hidden,
                    "y_shifted": transcriptions_input
                }
            },
            "targets": {
                "y": transcriptions_target
            }
        }

    def state_dict(self):
        return dict(alphabet_size=self.alphabet_size, hidden_size=self.hidden_size)
