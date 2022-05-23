def transform(y_hat, y):
    return y_hat.transpose(1, 2), y


def reverse_onehot(y_hat):
    return (y_hat.argmax(dim=2)[0], )


class DecodeClassesTransform:
    def __init__(self, data_pipeline):
        self.data_pipeline = data_pipeline

    def __call__(self, y_hat, ground_true):
        english_decoder = self.data_pipeline.preprocessors[1]

        y_hat = y_hat.argmax(dim=2)[0].tolist()
        ground_true = ground_true[0].tolist()
        #print(y_hat, ground_true)
        predicted_text = ''.join([english_decoder.index2word.get(str(idx), 'OOV') for idx in y_hat])
        actual_text = ''.join([english_decoder.index2word.get(str(idx), 'OOV') for idx in ground_true])
        #print(actual_text, '->', predicted_text)
        return [predicted_text], [actual_text]
