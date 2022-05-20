import argparse
import json
import os

from scaffolding.utils import load_data_pipeline, load_session_from_last_epoch
from scaffolding.training import evaluate
from scaffolding import parse


def load_config(path):
    with open(path) as f:
        s = f.read()

    return json.loads(s)


if __name__ == '__main__':
    # todo: fix parsing evaluation.json (it should not containing key "training")
    parser = argparse.ArgumentParser(
        description='Evaluate a model using a specified set of metrics'
    )
    parser.add_argument('config', type=str, help='Path to the configuration file')

    cmd_args = parser.parse_args()
    path = cmd_args.config

    config = load_config(path)
    config = config["pipeline"]

    checkpoints_dir = parse.parse_checkpoint_dir(config)

    data_pipeline_path = os.path.join(checkpoints_dir, 'data_pipeline.json')

    epochs_dir = os.path.join(checkpoints_dir, 'epochs')

    data_pipeline = load_data_pipeline(data_pipeline_path)

    prediction_pipeline = load_session_from_last_epoch(epochs_dir, inference_mode=True)

    train_loader, test_loader = data_pipeline.get_data_loaders()

    loss_fn = parse.parse_loss(config)

    metrics = parse.parse_metrics(config)

    if 'loss' in metrics:
        metrics['loss'] = loss_fn

    s = ''
    computed_metrics = evaluate(prediction_pipeline, train_loader, metrics, num_batches=64)
    for name, value in computed_metrics.items():
        s += f'{name}: {value}; '

    computed_metrics = evaluate(prediction_pipeline, test_loader, metrics, num_batches=64)
    for name, value in computed_metrics.items():
        s += f'val {name}: {value}; '

    print(s)
