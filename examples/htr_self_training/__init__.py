from torch import nn


class ConsistencyLoss:
    def __init__(self):
        self.loss_function = MaskedCrossEntropy()

    def __call__(self, y_hat_weak, y_hat_strong, ground_true, mask):
        weak_loss = self.loss_function(y_hat_weak, ground_true, mask).sum()
        strong_loss = self.loss_function(y_hat_strong, ground_true, mask).sum()
        return weak_loss + strong_loss

    def swap_axes(self, t):
        return t.transpose(1, 2)


class MaskedCrossEntropy:
    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, y_hat, ground_true, mask):
        losses = self.loss_function(self.swap_axes(y_hat), ground_true)
        return losses[mask.mask]

    def swap_axes(self, t):
        return t.transpose(1, 2)


class PostProcessor:
    def __init__(self, session, *args, **kwargs):
        self.tokenizer = session.preprocessors["tokenize"]

    def __call__(self, predictions_dict):
        return {k: self.to_text(v) for k, v in predictions_dict.items()}

    def to_text(self, pmf):
        assert pmf.shape[0] == 1
        pmf = pmf.squeeze(0)
        tokens = pmf.argmax(dim=1).tolist()

        return self.tokenizer.decode_to_string(tokens, clean_output=True)
