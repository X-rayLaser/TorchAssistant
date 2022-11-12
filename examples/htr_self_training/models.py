import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg19_bn


class VGG19Truncated(nn.Module):
    def __init__(self):
        super().__init__()
        original_vgg = vgg19_bn()
        self.truncated_vgg = nn.Sequential(*list(original_vgg.children())[:-2])

    def forward(self, x):
        return self.truncated_vgg(x)


class ImageEncoder(nn.Module):
    def __init__(self, image_height, hidden_size):
        super().__init__()
        out_feature_maps = 512
        feature_height = image_height // 2**5
        input_size = feature_height * out_feature_maps
        self.hidden_size = hidden_size
        self.vgg = VGG19Truncated()
        self.gru1 = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x):
        output = self.vgg(x)
        batch_size, f_maps, h, w = output.size()
        x = output.transpose(1, 3).resize(batch_size, w, h * f_maps)

        x, hidden = self.gru1(x)

        x, hidden = self.gru2(x)

        return x, hidden

    def run_inference(self, x):
        return self.forward(x)


# todo: wrong implementation; use location based attention mentioned in unsupervised adaptation paper
class AttentionNetwork(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        """
        :param embedding_size: length of vectors of encoder outputs
        :param hidden_size: number of hidden units in the decoder GRU
        """
        super().__init__()

        inner_dim = 32

        self.net = nn.Sequential(
            nn.Linear(embedding_size + hidden_size, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, 1)
        )

    def forward(self, decoder_hidden, encoder_outputs):
        """Computes attention context vectors as a weighted sum of encoder_outputs
        :param decoder_hidden: hidden state of the decoder
        :type decoder_hidden: tensor of shape (1, batch_size, num_cells)
        :param encoder_outputs: outputs of the encoder
        :type encoder_outputs: tensor of shape (batch_size, num_steps, embedding_size)
        :return: context vectors
        :rtype: tensor of shape (batch_size, embedding_size)
        """

        batch_size, num_steps, embedding_size = encoder_outputs.shape

        h = decoder_hidden.squeeze(0).unsqueeze(1).expand(batch_size, num_steps, -1)

        x = torch.cat([encoder_outputs, h], dim=2)

        scores = self.net(x)

        weights = F.softmax(scores.squeeze(2), dim=1)
        return torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)


class AttendingDecoder(nn.Module):
    def __init__(self, sos_token, context_size, y_size, hidden_size=256):
        super().__init__()

        # start of sequence token
        self.sos_token = sos_token

        self.y_size = y_size

        self.hidden_size = hidden_size

        self.attention = AttentionNetwork(context_size, hidden_size)
        self.decoder_gru = nn.GRU(context_size + y_size, hidden_size, batch_first=True)

        self.linear = nn.Linear(hidden_size, y_size)

    def forward(self, encodings, y_shifted=None):
        decoder_hidden = torch.zeros(1, len(encodings), self.hidden_size, device=encodings.device)

        if y_shifted is None:
            # close loop inference
            sos = torch.zeros(len(encodings), self.y_size, dtype=encodings.dtype, device=encodings.device)
            sos[:, self.sos_token] = 1
            return self.run_inference(encodings, decoder_hidden, sos)

        # teacher forcing
        batch_size, num_steps, num_classes = y_shifted.size()

        outputs = []

        for t in range(0, num_steps):
            y_hat_prev = y_shifted[:, t, :]
            log_pmf, decoder_hidden = self.predict_next(decoder_hidden, encodings, y_hat_prev)
            outputs.append(log_pmf)

        y_hat = torch.stack(outputs, dim=1)
        return [y_hat]

    def run_inference(self, encodings, decoder_hidden, sos):
        outputs = []

        y_hat_prev = sos

        batch_size = len(encodings)
        batch_indices = range(batch_size)

        for t in range(20):
            scores, decoder_hidden = self.predict_next(decoder_hidden, encodings, y_hat_prev)

            top = scores.argmax(dim=1)

            y_hat_prev = torch.zeros(batch_size, self.y_size)
            y_hat_prev[batch_indices, top] = 1.0
            outputs.append(scores)

        return [torch.stack(outputs, dim=1)]

    def predict_next(self, decoder_hidden, encoder_outputs, y_hat_prev):
        c = self.attention(decoder_hidden, encoder_outputs)

        v = torch.cat([y_hat_prev, c], dim=1).unsqueeze(1)

        h, hidden = self.decoder_gru(v, decoder_hidden)
        h = h.squeeze(1)
        return self.linear(h), hidden


def build_encoder(session, *args, **kwargs):
    return ImageEncoder(*args, **kwargs)


def build_decoder(session):
    encoder_model = next(iter(session.models.values()))
    context_size = encoder_model.hidden_size * 2
    decoder_hidden_size = encoder_model.hidden_size

    tokenizer = session.preprocessors["tokenize"]
    sos_token = tokenizer.char2index[tokenizer.start]
    return AttendingDecoder(sos_token, context_size, y_size=tokenizer.charset_size,
                            hidden_size=decoder_hidden_size)


# todo: edit spec format: support doing extra work before/after training in a stage
# or consider to cycle through epoch tasks (each task runs a pipeline for 1 epoch either trainable or not)
# todo: pipelines can be made trainable or not; also instead of calculating losses you can pass outputs to
# some post processing functions (e. g. to save predictions for an image)
# this post processing function needs to access original path to the image file to save
# corresponding prediction to the text file next to the image location path
# this can be done by saving each example state before/after every preprocessing
# in a context (e.g. list-like object); alternatively, you can only keep original state and current one,
# the original examples will thus be at index 0;
# this context can be added under a special key to every data frame
