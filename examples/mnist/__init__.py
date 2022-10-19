import torch
from torch import nn
from torchvision.transforms import ToTensor
from torchassistant.collators import BatchDivide


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, padding="same")
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return [x]


def reverse_onehot(y_hat, ground_true):
    return y_hat.argmax(dim=-1), torch.LongTensor(ground_true)


def convert_labels(y_hat, labels):
    labels = torch.LongTensor(labels)
    return y_hat, labels


class MyCollator(BatchDivide):
    def __call__(self, batch):
        to_tensor = ToTensor()

        images, labels = super().__call__(batch)

        x = torch.stack([to_tensor(image) for image in images])
        return x, labels


class InputConverter:
    def __call__(self, image_path):
        from PIL import Image
        with Image.open(image_path) as im:
            return im.copy(), None
