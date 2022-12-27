from PIL import Image, ImageDraw, ImageFont
import time

import os
import random


class SimpleRandomWordGenerator:
    def __init__(self, dictionary, font_dir, font_size_range=(64, 64),
                 bg_range=(240, 255), color_range=(0, 15), stroke_width_range=(0, 2),
                 stroke_fill_range=(0, 15), max_word_len=14):
        if isinstance(dictionary, list):
            self.dictionary = dictionary
        else:
            with open(dictionary) as f:
                self.dictionary = [word.strip() for word in f
                                   if word.strip() and len(word.strip()) <= max_word_len]

        self.font_dir = font_dir
        self.font_files = [os.path.join(font_dir, font_file) for font_file in os.listdir(font_dir)]

        self.font_size_range = font_size_range
        self.bg_range = bg_range
        self.color_range = color_range
        self.stroke_width_range = stroke_width_range
        self.stroke_fill_range = stroke_fill_range

    def __iter__(self):
        while True:
            font_size = random.randint(*self.font_size_range)
            font_file = random.choice(self.font_files)
            font = ImageFont.truetype(font_file, size=font_size)
            word = random.choice(self.dictionary)
            background = random.randint(*self.bg_range)
            color = random.randint(*self.color_range)
            stroke_fill = random.randint(*self.stroke_fill_range)
            stroke_width = random.randint(*self.stroke_width_range)
            try:
                image = self.create_image(word, font, font_size,
                                          background=background, color=color,
                                          stroke_width=stroke_width, stroke_fill=stroke_fill)
                yield image, word
            except Exception:
                pass

    def create_image(self, word, font, size=64, background=255, color=0, stroke_width=1, stroke_fill=0):
        char_size = size
        num_chars = len(word)
        width = char_size * num_chars
        with Image.new("L", (width, size + 20)) as im:
            draw = ImageDraw.Draw(im)
            bbox = draw.textbbox((0, 0), word, font=font)
            draw.rectangle(bbox, fill=background)
            draw.text((0, 0), word, fill=color, font=font, stroke_width=stroke_width, stroke_fill=stroke_fill)
            return im.crop(bbox)


if __name__ == '__main__':
    # simple benchmark
    word_gen = SimpleRandomWordGenerator("examples/htr_self_training/words.txt",
                                         "examples/htr_self_training/fonts", font_size_range=(58, 70))

    it = iter(word_gen)
    t = time.time()

    for i in range(1000):
        im, tr = next(it)
        #print(tr)
        #im.show()

    print(time.time() - t, (time.time() - t) / 100 / 10)
