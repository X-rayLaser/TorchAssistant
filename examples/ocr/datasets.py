import random
import os

from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont


class DatasetGenerator:
    def __init__(self, font_size, num_examples, dictionary_path):
        self.font_size = font_size
        self.num_examples = num_examples
        self.dictionary_path = dictionary_path

    def __iter__(self):
        with open(self.dictionary_path) as f:
            words = f.read().split('\n')
        random.shuffle(words)
        return (self.next_example(w) for w in words[:self.num_examples])

    def next_example(self, text):
        padding = 10

        font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", self.font_size)
        width, height = font.getsize(text)
        image = Image.new("L", (width + padding, self.font_size + padding), 255)

        d = ImageDraw.Draw(image)

        d.text((padding // 2, padding // 2), text, font=font, fill=0)
        return image, text


def save_example(example, output_dir, index):
    image, text = example

    images_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    image_path = os.path.join(images_dir, f'{index}.jpg')

    texts_path = os.path.join(output_dir, 'texts.txt')

    with open(texts_path, 'a', encoding='utf-8') as f:
        f.write(text + '\n')

    image.save(image_path)


class SyntheticDataset(Dataset):
    def __init__(self, path):
        self.path = path

        self.images_dir = os.path.join(path, 'images')
        self.texts_path = os.path.join(path, 'texts.txt')

        with open(self.texts_path, encoding='utf-8') as f:
            self.texts = [line for line in f.read().split('\n') if line]

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, f'{idx}.jpg')
        return Image.open(image_path), self.texts[idx]

    def __len__(self):
        return len(self.texts)
