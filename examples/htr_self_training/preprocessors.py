import torch
from torchassistant.preprocessors import ValuePreprocessor


class WeakAugmentorNormalizer(ValuePreprocessor):
    def process(self, value):
        return value


class StrongAugmentorNormalizer(ValuePreprocessor):
    def process(self, value):
        return value


class Normalizer(ValuePreprocessor):
    def process(self, value):
        return value / 255


class CharacterTokenizer(ValuePreprocessor):
    start = '<s>'
    end = r'<\s>'
    out_of_charset = '<?>'

    english = "english"

    def __init__(self, charset):
        if charset == self.english:
            letters = "abcdefghijklmnopqrstuvwxyz"
            digits = "0123456789"
            punctuation = ".,?!:;-()'\""
            letters.upper()
            charset = letters + letters.upper() + digits + punctuation

        self.char2index = {}
        self.index2char = {}
        self.charset = charset
        self._build_char_table(charset)

    def process(self, text):
        start_token = self._encode(self.start)
        end_token = self._encode(self.end)
        tokens = [self._encode(ch) for ch in text]
        return [start_token] + tokens + [end_token]

    @property
    def charset_size(self):
        return len(self.char2index)

    def _build_char_table(self, charset):
        self._add_char(self.start)
        self._add_char(self.end)
        self._add_char(self.out_of_charset)

        for ch in charset:
            self._add_char(ch)

    def _add_char(self, ch):
        if ch not in self.char2index:
            num_chars = self.charset_size
            self.char2index[ch] = num_chars
            self.index2char[num_chars] = ch

    def _encode(self, ch):
        default_char = self.char2index[self.out_of_charset]
        return self.char2index.get(ch, default_char)

    def _decode(self, token):
        return self.index2char.get(token, self.out_of_charset)

    def decode_to_string(self, tokens, clean_output=False):
        s = ''.join([self._decode(token) for token in tokens[1:-1]])

        first_char = self._decode(tokens[0])
        last_char = self._decode(tokens[-1])

        if first_char != self.start:
            s += first_char

        if last_char != self.end:
            s += last_char

        if clean_output:
            s = s.replace(self.start, '')
            s = s.replace(self.end, '')
            s = s.replace(self.out_of_charset, '')

        return s


class DecodeCharacterString:
    def __init__(self, session):
        self.tokenizer = session.preprocessors["tokenize"]

    def __call__(self, y_hat, ground_true):
        if type(ground_true) is torch.Tensor:
            ground_true = ground_true.tolist()

        y_hat = y_hat.argmax(dim=2).tolist()
        tokenizer = self.tokenizer
        predicted_texts = []
        actual_texts = []

        for predicted_tokens, true_tokens in zip(y_hat, ground_true):
            predicted_texts.append(tokenizer.decode_to_string(predicted_tokens))
            actual_texts.append(tokenizer.decode_to_string(true_tokens))

        return predicted_texts, actual_texts
