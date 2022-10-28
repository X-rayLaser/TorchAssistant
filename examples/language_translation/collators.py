from torchassistant.collators import ColumnWiseCollator, seqs_to_tensor, no_transform


def build_collator2(session):
    collator = ColumnWiseCollator([seqs_to_tensor, no_transform])
    return collator


def build_inference_collator(session):
    collator = ColumnWiseCollator([seqs_to_tensor])
    return collator
