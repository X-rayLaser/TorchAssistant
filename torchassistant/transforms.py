import torch


def reverse_onehot(y_hat, ground_true):
    return y_hat.argmax(dim=-1), torch.LongTensor(ground_true)
