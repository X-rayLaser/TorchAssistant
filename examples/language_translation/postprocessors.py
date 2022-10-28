class PostProcessor:
    def __init__(self, session, *args, **kwargs):
        self.decoder = session.preprocessors["autogenerated_preprocessor_2"]

    def __call__(self, predictions_dict):
        return {k: self.to_text(v) for k, v in predictions_dict.items()}

    def to_text(self, tokens):
        output = tokens
        try:
            eos_index = output.index(self.decoder.encode_word(self.decoder.sentence_end))
        except ValueError:
            eos_index = len(output)

        output = output[:eos_index]

        return ''.join([self.decoder.index2word.get(idx, 'OOV') for idx in output])
