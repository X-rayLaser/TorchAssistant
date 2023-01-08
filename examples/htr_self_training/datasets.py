import os
from PIL import Image

from torchvision.transforms import Resize
from examples.htr_self_training.data_generator import SimpleRandomWordGenerator


class SyntheticOnlineDataset:
    def __init__(self, fonts_dir, size, image_height=75):
        self.size = size
        self.fonts_dir = fonts_dir
        self.image_height = image_height

        dictionary = os.path.join("examples/htr_self_training/words.txt")
        simple_generator = SimpleRandomWordGenerator(dictionary, self.fonts_dir,
                                                     bg_range=(255, 255),
                                                     color_range=(0, 100),
                                                     font_size_range=(64, 86), rotation_range=(0, 0))
        self.iterator = iter(simple_generator)

    def __iter__(self):
        for i in range(len(self)):
            yield self.generate_example()

    def __getitem__(self, idx):
        return self.generate_example()

    def generate_example(self):
        im, word = next(self.iterator)
        if im.height > self.image_height:
            im = scale_image(im, target_height=self.image_height)
        return im, word

    def __len__(self):
        return self.size


class SyntheticDataset:
    def __init__(self, path):
        self.root_path = path
        self.files = os.listdir(path)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        path = os.path.join(self.root_path, file_name)
        transcript, _ = os.path.splitext(file_name)
        transcript = transcript.split('_')[0]

        image = Image.open(path)
        return image, transcript

    def __len__(self):
        return len(self.files)


class IAMWordsDataset:
    def __init__(self, index_path, target_height=75):
        self.index_path = index_path
        self.iam_index = []
        self.target_height = target_height

        self.re_build()

    def re_build(self):
        self.iam_index = []
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
        if image.height > self.target_height:
            image = scale_image(image, self.target_height)
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


def scale_image(image, target_height):
    w = image.width
    h = image.height

    scaler = target_height / h

    target_width = int(round(scaler * w))

    resizer = Resize((target_height, target_width))
    return resizer(image)
