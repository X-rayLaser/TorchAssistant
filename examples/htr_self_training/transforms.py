import torch
from torchassistant.collators import pad_sequences


def pad_targets(*args):
    *y_hat, ground_true = args
    filler = ground_true[0][-1]
    seqs, mask = pad_sequences(ground_true, filler)
    target = torch.LongTensor(seqs)
    return y_hat + [target] + [mask]
