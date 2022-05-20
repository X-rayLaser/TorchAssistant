from scaffolding.preprocessors import ValuePreprocessor
from scaffolding.utils import Serializable
from torchvision.transforms import ToTensor


class ImagePreProcessor(ValuePreprocessor, Serializable):
    def process(self, pillow_image):
        return ToTensor()(pillow_image) / 255.


class TextPreProcessor(ValuePreprocessor, Serializable):
    def process(self, text):
        sos = 1
        eos = 2
        return [sos] + [self.encode_char(c) for c in text] + [eos]

    def encode_char(self, char):
        return ord(char)

    def decode_char(self, code_point):
        if code_point == 0:
            s = '<?>'
        elif code_point == 1:
            s = '<SOS>'
        elif code_point == 2:
            s = '<EOS>'
        else:
            s = chr(code_point)
        return s
