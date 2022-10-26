from torch import nn


class MaskedCrossEntropy:
    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, y_hat, ground_true, mask):
        losses = self.loss_function(self.swap_axes(y_hat), ground_true)
        return losses[mask.mask].mean()

    def swap_axes(self, t):
        return t.transpose(1, 2)
