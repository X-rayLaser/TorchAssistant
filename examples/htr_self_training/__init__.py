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
