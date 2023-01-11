import os
import argparse
import sys
from PIL import Image, ImageDraw

sys.path.insert(0, '.')
from torchassistant.session import SessionSaver
from examples.htr_self_training.datasets import SyntheticOnlineDataset
from examples.htr_self_training.adapters import TrainableProcessorInputAdapter


def create_attention_overlay(original, attention_weights, scaler=32):
    overlay = Image.new('RGBA', original.size, (255, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, w in enumerate(attention_weights):
        intensity = int((1 - w) * 255)
        filling = (intensity, intensity, intensity, 128)
        draw.rectangle((i * scaler, 0, i * scaler + scaler, overlay.height), fill=filling)

    original.paste(overlay, (0, 0), overlay)
    overlay.close()


def debug_attention():
    parser = argparse.ArgumentParser(
        description='Train ML pipeline according to a specified configuration file'
    )
    parser.add_argument('session_path', type=str, help='Path to the session file')

    cmd_args = parser.parse_args()
    path = cmd_args.session_path

    saver = SessionSaver(path)
    session = saver.load_from_latest_checkpoint()

    encoder = session.models["encoder"]
    decoder = session.models["decoder"]

    tokenizer = session.preprocessors["tokenize"]
    num_classes = tokenizer.charset_size
    ds = SyntheticOnlineDataset("examples/htr_self_training/fonts", 1, 64)

    im, label = ds[0]

    im.show()

    adapter = TrainableProcessorInputAdapter(num_classes)
    batch = [im]
    actual_tokens = tokenizer.process(label)
    inputs = adapter({"input_1": batch, "input_2": [actual_tokens]})
    x = inputs["encoder"]["x"]

    encodings, encoder_state = encoder(x)
    outputs, attention = decoder.debug_attention(encodings)
    tokens = outputs.argmax(dim=2).tolist()[0]
    predicted_text = tokenizer.decode_to_string(tokens, clean_output=True)
    print('prediction:', predicted_text)

    os.makedirs('attention_vis', exist_ok=True)

    for i in range(len(tokens)):
        token = tokens[i]
        attention_weights = attention[i][0]
        decoded_token = tokenizer._decode(token)
        im_copy = im.copy()
        create_attention_overlay(im_copy, attention_weights)

        image_path = os.path.join('attention_vis', f'{i}_{decoded_token}.png')
        im_copy.save(image_path)


if __name__ == '__main__':
    debug_attention()
