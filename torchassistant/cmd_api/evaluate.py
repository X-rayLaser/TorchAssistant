import argparse

from torchassistant.training import evaluate_pipeline
from torchassistant.session import SessionSaver
from torchassistant.formatters import Formatter
from torchassistant.utils import add_cwd_to_python_path, load_config


def run_evaluate():
    # todo: support overrides

    add_cwd_to_python_path()

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

    num_eval_steps = config.get("eval_steps", 1.0)

    metrics = {}
    for name in config["validation_pipelines"]:
        pipeline = session.pipelines[name]
        metrics.update(evaluate_pipeline(pipeline, num_eval_steps))

    formatter = Formatter()
    final_metrics_string = formatter.format_metrics(metrics, validation=False)
    spaces = ' ' * 150
    print(f'\r{spaces}\r{final_metrics_string}')
