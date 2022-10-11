import argparse
import json

from torchassistant.training import evaluate_pipeline
from torchassistant.session import SessionSaver
from torchassistant.formatters import Formatter


def load_config(path):
    with open(path, encoding='utf-8') as f:
        s = f.read()

    return json.loads(s)


# todo: support overrides
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a model using a specified set of metrics'
    )
    parser.add_argument('config', type=str, help='Path to the configuration file')

    cmd_args = parser.parse_args()
    path = cmd_args.config

    config = load_config(path)

    session_dir = config["session_dir"]

    saver = SessionSaver(session_dir)
    session = saver.load_from_latest_checkpoint(new_spec=config)

    new_pipelines_spec = config.get("pipelines", {})

    metrics = {}
    for name in config["validation_pipelines"]:
        pipeline = session.pipelines[name]
        metrics.update(evaluate_pipeline(pipeline))

    formatter = Formatter()
    final_metrics_string = formatter.format_metrics(metrics, validation=False)
    print(final_metrics_string)
