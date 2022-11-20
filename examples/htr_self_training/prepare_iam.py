import os
import re
import shutil
import argparse
import random
import PIL
from PIL import Image


def prepare_iam_dataset(iam_location, output_dir, max_words=None, train_fraction=0.8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    transcripts_file = os.path.join(iam_location, 'ascii', 'words.txt')
    words_dir = os.path.join(iam_location, 'words')

    lines_skipped = 0
    paths_with_transcripts = []

    train_file = os.path.join(output_dir, 'iam_train.txt')
    val_file = os.path.join(output_dir, 'iam_val.txt')

    pseudo_labels_file = os.path.join(output_dir, 'pseudo_labels.txt')
    with open(pseudo_labels_file, 'w') as f:
        pass

    with open(transcripts_file) as f:
        for i, line in enumerate(f):
            if max_words and len(paths_with_transcripts) >= max_words:
                break

            if line.lstrip().startswith('#'):
                print('skipping line', line)
                lines_skipped += 1
                continue

            parts = re.findall(r'[\w-]+', line)
            image_id = parts[0]
            status = parts[1]
            gray_level = parts[2]

            if status != 'ok':
                lines_skipped += 1
                continue

            transcript = parts[-1]
            path = locate_image(words_dir, image_id)
            if not path:
                lines_skipped += 1
                raise Exception(f'File not found: {path}. Line {line}')

            try:
                Image.open(path)
            except PIL.UnidentifiedImageError:
                print('Bad image file')
                continue

            if i + 1 % 10000:
                print('Words processed:', i + 1)

            paths_with_transcripts.append(f'{path}, {gray_level}, {transcript}')

    print(f'total lines processed {i}, lines skipped {lines_skipped}')

    indices = list(range(len(paths_with_transcripts)))
    random.shuffle(indices)
    train_size = int(len(indices) * train_fraction)

    training_words = [paths_with_transcripts[idx] for idx in indices[:train_size]]
    val_words = [paths_with_transcripts[idx] for idx in indices[train_size:]]

    with open(train_file, 'w') as f:
        f.write('\n'.join(training_words))

    with open(val_file, 'w') as f:
        f.write('\n'.join(val_words))


def locate_image(words_dir, image_id):
    parts = image_id.split('-')
    dir_name = parts[0]
    sub_dir_name = f'{parts[0]}-{parts[1]}'
    file_name = f'{image_id}.png'
    image_path = os.path.join(words_dir, dir_name, sub_dir_name, file_name)
    if os.path.isfile(image_path):
        return image_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ML pipeline according to a specified configuration file'
    )
    parser.add_argument('iam_home', type=str, help='Path to the location of IAM database directory')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    parser.add_argument('--max-words', type=int, default=None, help='Path to the output directory')

    args = parser.parse_args()
    prepare_iam_dataset(args.iam_home, args.output_dir, max_words=args.max_words)
