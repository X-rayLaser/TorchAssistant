import argparse
import json

from scaffolding.training import train
from scaffolding import parse


def load_config(path):
    with open(path) as f:
        s = f.read()

    return json.loads(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ML pipeline according to a specified configuration file'
    )
    parser.add_argument('config', type=str, help='Path to the configuration file')

    cmd_args = parser.parse_args()
    path = cmd_args.config

    config = load_config(path)
    config = config["pipeline"]

    epochs = parse.parse_epochs(config)
    checkpoints_dir = parse.parse_checkpoint_dir(config)

    train_loader, test_loader = parse.parse_data_pipeline(config)

    metrics = parse.parse_metrics(config)

    train_pipeline = parse.parse_model(config)

    criterion = parse.parse_loss(config)

    train(train_pipeline, train_loader, test_loader, criterion, metrics, epochs, checkpoints_dir)


# todo: saving and loading
# todo: support inference mode (as opposed to training mode) and use it for validation metrics calculations;
# greedy search vs beam search decoding for seq2seq inference
# todo: extra scripts to evaluate model (possibly on some benchmarks), run inference, fine tune and export
# todo: debug tool (show predictions for input as well as all inputs, outputs and transformations)
# todo: ensure other examples work
# todo: refactor code
# todo: support batch sizes > 1 (this will involve some extra transformations like padding, etc.)
# todo: think of a better way to implement dynamic arguments and the way that one component (data pipeline) may
# affect other components (models)
# todo: support training seq2seq model with attention
