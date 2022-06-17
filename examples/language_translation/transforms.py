def transform(y_hat, y):
    return y_hat.transpose(1, 2), y


def reverse_onehot(y_hat):
    return (y_hat.argmax(dim=2)[0], )


class DecodeClassesTransform:
    def __init__(self, session):
        self.session = session

    def __call__(self, y_hat, ground_true):
        english_decoder = self.session.preprocessors['english_encoder']

        y_hat = y_hat.argmax(dim=2)[0].tolist()
        ground_true = ground_true[0].tolist()

        predicted_text = ''.join([english_decoder.index2word.get(idx, 'OOV') for idx in y_hat])
        actual_text = ''.join([english_decoder.index2word.get(idx, 'OOV') for idx in ground_true])

        return [predicted_text], [actual_text]
