import os
from PIL import Image

from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms import Resize


class SyntheticDataset:
    def __init__(self, path):
        self.root_path = path
        self.files = os.listdir(path)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        path = os.path.join(self.root_path, file_name)
        transcript, _ = os.path.splitext(file_name)
        image = Image.open(path)
        image = rgb_to_grayscale(image)
        return image, transcript

    def __len__(self):
        return len(self.files)


class IAMWordsDataset:
    def __init__(self, index_path, target_height=64):
        self.index_path = index_path
        self.iam_index = []
        self.target_height = target_height

        self.re_build()

    def re_build(self):
        with open(self.index_path) as f:
            for line in f:
                path, gray_level, transcript = line.split(',')
                path = path.strip()
                transcript = transcript.strip()
                gray_level = int(gray_level.strip())
                self.iam_index.append((path, gray_level, transcript))

    def __getitem__(self, idx):
        path, gray_level, transcript = self.iam_index[idx]
        image = Image.open(path)
        image = clean_image(image, gray_level)

        w = image.width
        h = image.height

        scaler = self.target_height / h

        target_width = int(round(scaler * w))

        resizer = Resize((self.target_height, target_width))
        image = resizer(image)
        return path, gray_level, image, transcript

    def __len__(self):
        return len(self.iam_index)


class UnlabeledDataset(IAMWordsDataset):
    def __getitem__(self, idx):
        path, gray_level, image, transcript = super().__getitem__(idx)
        return path, gray_level, image


class LabeledDataset(IAMWordsDataset):
    def __getitem__(self, idx):
        path, gray_level, image, transcript = super().__getitem__(idx)
        return image, transcript


def clean_image(image, gray_level):
    return image.point(lambda p: 255 if p > gray_level else p)
