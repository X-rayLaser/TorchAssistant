def transform(y_hat, y):
    return y_hat.transpose(1, 2), y


def reverse_onehot(y_hat):
    return (y_hat.argmax(dim=2)[0], )
