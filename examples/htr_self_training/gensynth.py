import random
import os
from trdg.generators import GeneratorFromDict


def int2hex(v):
    hex_val = hex(v)[2:]
    if len(hex_val) == 1:
        hex_val = '0' + hex_val
    return hex_val


def random_color(ceiling):
    intensity = random.randint(0, ceiling)
    hex_intensity = int2hex(intensity)
    text_color = f'#{hex_intensity}{hex_intensity}{hex_intensity}'
    return text_color


def create_generator(fonts_dir):
    char_spacing = random.randint(0, 10)
    stroke_width = random.randint(0, 2)
    max_intensity = 50
    color = random_color(max_intensity)

    fonts = [
        os.path.join(fonts_dir, p)
        for p in os.listdir(fonts_dir)
        if os.path.splitext(p)[1] == ".ttf"
    ]

    return GeneratorFromDict(
        stroke_width=stroke_width,
        character_spacing=char_spacing,
        skewing_angle=10,
        random_skew=True,
        distorsion_type=0,
        text_color=color,
        stroke_fill=color,
        background_type=0,
        blur=1,
        random_blur=True,
        fonts=fonts,
        size=64,
        image_mode='L'
    )


def generate_data(fonts_dir, num_words, reset_ivl=40):
    generator = create_generator(fonts_dir)

    for i in range(num_words):
        if i % reset_ivl == 0:
            generator = create_generator(fonts_dir)

        try:
            yield next(generator)
        except StopIteration:
            generator = create_generator(fonts_dir)
        except Exception as e:
            print('exception', repr(e))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate synthetic dataset'
    )
    parser.add_argument('fonts_dir', type=str, help='Path to the fonts directory')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')

    parser.add_argument('max_words', type=int, default=None, help='Total # of examples to generate')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for i, (img, label) in enumerate(generate_data(args.fonts_dir, args.max_words)):
        img.save(os.path.join(args.output_dir, f'{label}_{i}.jpg'))
        if i % 100 == 0:
            print(f'progress: {i} out of {args.max_words}')
