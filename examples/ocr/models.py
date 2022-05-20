import torch
from torch import nn
from torch.nn import functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_features, mid_features=128, k=32):
        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_features, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, mid_features, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(mid_features, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_features, k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        )

    def forward(self, x):
        return self.layers(x)


class DenseBlock(nn.Module):
    def __init__(self, in_features, block_size=6, k=32):
        super().__init__()

        in_sizes = [in_features + k * i for i in range(block_size)]
        self.dense_layers = nn.ModuleList(
            [DenseLayer(in_features=num_features, k=32) for num_features in in_sizes]
        )

    def forward(self, x):
        input_feature_maps = x
        for i, layer in enumerate(self.dense_layers):
            feature_maps = layer(input_feature_maps)
            input_feature_maps = torch.cat([input_feature_maps, feature_maps], dim=1)
        return input_feature_maps


class InitialBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.initial_block = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(out_features, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )

    def forward(self, x):
        return self.initial_block(x)


class ImageEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.network = nn.Sequential(
            InitialBlock(input_channels, 64),
            DenseBlock(64, 6),
            nn.BatchNorm2d(256, affine=True)
        )

    def forward(self, x):
        output = self.network(x)
        batch_size, f_maps, h, w = output.size()

        return (output.resize(batch_size, f_maps, h * w).transpose(1, 2),)

    def run_inference(self, x):
        return self.forward(x)


class AttentionNetwork(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        """

        :param embedding_size: length of vectors of encoder outputs
        :param hidden_size: number of hidden units in the decoder GRU
        """
        super().__init__()

        print(embedding_size, hidden_size, embedding_size + hidden_size)
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
    def __init__(self, context_size=1024, y_size=128, hidden_size=512, inner_dim=128):
        super().__init__()

        self.y_size = y_size
        self.instance_params = dict(context_size=context_size, y_size=y_size,
                                    hidden_size=hidden_size, inner_dim=inner_dim)

        self.hidden_size = hidden_size

        self.attention = AttentionNetwork(context_size, hidden_size)
        self.decoder_gru = nn.GRU(context_size + y_size, hidden_size, batch_first=True)

        self.linear = nn.Linear(hidden_size, y_size)

    def forward(self, encodings, decoder_hidden, y_shifted):
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

        for t in range(40):
            scores, decoder_hidden = self.predict_next(decoder_hidden, encodings, y_hat_prev)

            top = scores[0].argmax()

            y_hat_prev = torch.zeros(1, self.y_size)
            y_hat_prev[0, top] = 1.0
            outputs.append(top)
        return [outputs]

    def predict_next(self, decoder_hidden, encoder_outputs, y_hat_prev):
        c = self.attention(decoder_hidden, encoder_outputs)

        v = torch.cat([y_hat_prev, c], dim=1).unsqueeze(1)

        h, hidden = self.decoder_gru(v, decoder_hidden)
        h = h.squeeze(1)
        return self.linear(h), hidden
