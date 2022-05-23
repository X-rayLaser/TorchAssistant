def transform(y_hat, y):
    return y_hat.transpose(1, 2), y


class DecodeClassesTransform:
    def __init__(self, data_pipeline):
        self.data_pipeline = data_pipeline

    def __call__(self, y_hat, ground_true):
        decoder = self.data_pipeline.preprocessors[1]

        y_hat = y_hat.argmax(dim=2)[0]
        ground_true = ground_true[0]

        predicted_text = ''.join([decoder.decode_char(code_point) for code_point in y_hat])
        actual_text = ''.join([decoder.decode_char(code_point) for code_point in ground_true])

        return [predicted_text], [actual_text]
