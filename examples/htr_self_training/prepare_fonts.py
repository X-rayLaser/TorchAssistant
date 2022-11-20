if __name__ == '__main__':
    import argparse
    import os
    import shutil

    parser = argparse.ArgumentParser(
        description='Extracts pseudo-handwritten fonts from Google fonts'
    )
    parser.add_argument('fonts_dir', type=str, help='Path to the location of google fonts directory')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    parser.add_argument('--num_fonts', type=int, default=100, help='Maximum number of fonts to take')
    parser.add_argument('--lang', type=str, default='latin',
                        help='Specifies target language that fonts are required to support')

    args = parser.parse_args()

    ofl_dir = os.path.join(args.fonts_dir, 'ofl')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    found_fonts = 0
    for font_dir in os.listdir(ofl_dir):
        if found_fonts >= args.num_fonts:
            break

        dir_path = os.path.join(ofl_dir, font_dir)
        ttf_files = [f for f in os.listdir(dir_path) if f.endswith('.ttf')]
        if not ttf_files:
            print(f'No ttf files found', os.listdir(dir_path))
            continue

        metadata_path = os.path.join(dir_path, 'METADATA.pb')
        if not os.path.exists(metadata_path):
            print('no metadata file, directory', font_dir, 'dir path', dir_path, 'dir content', os.listdir(dir_path))
            continue

        with open(metadata_path) as f:
            s = f.read()

        if 'category: "HANDWRITING"' in s and f'subsets: "{args.lang}"' in s:
            print('found font', font_dir)
            found_fonts += 1

            for name in ttf_files:
                src = os.path.join(dir_path, name)
                dest = os.path.join(args.output_dir, name)
                shutil.copyfile(src, dest)
