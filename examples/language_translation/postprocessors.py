class PostProcessor:
    def __init__(self, data_pipeline, *args, **kwargs):
        self.decoder = data_pipeline.preprocessors[1]

    def __call__(self, predictions_dict):
        return {k: self.to_text(v) for k, v in predictions_dict.items()}

    def to_text(self, tensors):
        output = tensors
        try:
            eos_index = output.index(2)
        except ValueError:
            eos_index = len(output)
        output = output[:eos_index]

        return ''.join([self.decoder.index2word.get(str(idx), 'OOV') for idx in output])
