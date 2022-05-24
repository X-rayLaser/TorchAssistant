import argparse
import json
import os

import torch

from scaffolding.training import train
from scaffolding import parse
from scaffolding.utils import load_session, save_data_pipeline, load_data_pipeline, change_model_device


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

    data_pipeline_path = os.path.join(checkpoints_dir, 'data_pipeline.json')
    store_path = os.path.join(checkpoints_dir, 'store.json')

    epochs_dir = os.path.join(checkpoints_dir, 'epochs')

    if checkpoints_dir and os.path.isfile(data_pipeline_path):
        data_pipeline = load_data_pipeline(data_pipeline_path)
        last_epoch = sorted(os.listdir(epochs_dir), key=lambda dir_name: int(dir_name), reverse=True)[0]
        start_epoch = int(last_epoch)
        train_pipeline = load_session(epochs_dir, last_epoch, torch.device(data_pipeline.device_str))
    else:
        os.makedirs(checkpoints_dir, exist_ok=True)
        data_pipeline = parse.parse_data_pipeline(config)

        save_data_pipeline(data_pipeline, data_pipeline_path)

        start_epoch = 0
        train_pipeline = parse.parse_model(config)
        change_model_device(train_pipeline, data_pipeline.device_str)

    metrics = parse.parse_metrics(config, data_pipeline)

    criterion = parse.parse_loss(config)

    train(data_pipeline, train_pipeline, criterion, metrics, epochs, start_epoch, epochs_dir)


# todo: refactor code
# todo: choose device (CPU vs GPU, optionally TPU)
# todo: consider making a batch adapter a part of prediction pipeline (rather than data pipeline)
# todo: support batch sizes > 1 (this will involve some extra transformations like padding, etc.)
# todo: ensure other examples work
# todo: greedy search vs beam search decoding for seq2seq inference
# todo: extra scripts to fine tune and export
# todo: debug tool (show predictions for input as well as all inputs, outputs and transformations)
# todo: think of a better way to implement dynamic arguments and the way that one component (data pipeline) may
# affect other components (models)
# todo: test suite
