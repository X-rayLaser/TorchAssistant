import argparse
import json

from scaffolding.training import evaluate
from scaffolding import parse
from scaffolding.session import TrainingSession


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

    pretrained_dir = parse.parse_checkpoint_dir(config)

    session = TrainingSession(pretrained_dir)

    data_pipeline = session.data_pipeline
    prediction_pipeline = session.restore_from_last_checkpoint()
    loss_fn = session.criterion
    metrics = session.metrics

    # overrides from config
    if "loss" in config["training"]:
        loss_fn = parse.parse_loss(config["training"]["loss"], session.device)

    if "metrics" in config["training"]:
        metrics = parse.parse_metrics(config["training"]["metrics"], data_pipeline, session.device)

    train_loader, test_loader = data_pipeline.get_data_loaders()

    if 'loss' in metrics:
        metrics['loss'] = session.criterion

    s = ''
    computed_metrics = evaluate(prediction_pipeline, train_loader, metrics, num_batches=64)
    for name, value in computed_metrics.items():
        s += f'{name}: {value}; '

    computed_metrics = evaluate(prediction_pipeline, test_loader, metrics, num_batches=64)
    for name, value in computed_metrics.items():
        s += f'val {name}: {value}; '

    print(s)
