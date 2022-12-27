from PIL import Image, ImageDraw, ImageFont
import time

import os
import random


class SimpleRandomWordGenerator:
    def __init__(self, dictionary, font_dir, size=64):
        if isinstance(dictionary, list):
            self.dictionary = dictionary
        else:
            with open(dictionary) as f:
                self.dictionary = [word.strip() for word in f if word.strip()]

        self.font_dir = font_dir
        font_files = [os.path.join(font_dir, font_file) for font_file in os.listdir(font_dir)]
        self.fonts = [ImageFont.truetype(file, size=size) for file in font_files]

    def __iter__(self):
        while True:
            font = random.choice(self.fonts)
            word = random.choice(self.dictionary)
            background = random.randint(240, 255)
            color = random.randint(0, 15)
            stroke_fill = random.randint(0, 15)
            stroke_width = random.randint(0, 2)
            try:
                image = self.create_image(word, font, 64, background=background, color=color,
                                          stroke_width=stroke_width, stroke_fill=stroke_fill)
                yield image, word
            except Exception as e:
                print("Error:", repr(e))

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
                                         "examples/htr_self_training/fonts")

    it = iter(word_gen)
    t = time.time()

    for i in range(1):
        im, tr = next(it)
        print(tr)
        im.show()

    print(time.time() - t, (time.time() - t) / 100 / 10)
