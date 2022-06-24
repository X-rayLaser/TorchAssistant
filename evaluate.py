import argparse
import json

from scaffolding.training import evaluate, evaluate_pipeline
from scaffolding.session import SessionSaver, load_json
from scaffolding.session.parse import PipelineLoader
from scaffolding.formatters import Formatter


def load_config(path):
    with open(path, encoding='utf-8') as f:
        s = f.read()

    return json.loads(s)


def override_pipelines(session, old_spec, new_pipelines_spec):
    pipeline_loader = PipelineLoader()

    old_pipelines_spec = old_spec["initialize"]["pipelines"]

    pipeline_spec = {}
    for name, override_options in new_pipelines_spec.items():
        pipeline_spec[name] = old_pipelines_spec[name]
        pipeline_spec[name].update(override_options)

    return [pipeline_loader.load(session, options)
            for _, options in pipeline_spec.items()]


def print_metrics(pipelines, formatter):
    metric_strings = []

    for pipeline in pipelines:
        train_metrics, val_metrics = evaluate_pipeline(pipeline)
        train_metric_string = formatter.format_metrics(train_metrics, validation=False)
        val_metric_string = formatter.format_metrics(val_metrics, validation=True)
        metric_string = f'{train_metric_string}; {val_metric_string}'
        metric_strings.append(metric_string)

    final_metrics_string = '  |  '.join(metric_strings)
    print(f'\r{final_metrics_string}')


if __name__ == '__main__':
    # todo: fix parsing evaluation.json (it should not containing key "training")
    parser = argparse.ArgumentParser(
        description='Evaluate a model using a specified set of metrics'
    )
    parser.add_argument('config', type=str, help='Path to the configuration file')

    cmd_args = parser.parse_args()
    path = cmd_args.config

    config = load_config(path)

    session_dir = config["session_dir"]

    saver = SessionSaver(session_dir)
    session = saver.load_from_latest_checkpoint()
    old_spec = load_json(saver.spec_path)

    new_pipelines_spec = config["pipelines"]
    pipelines = override_pipelines(session, old_spec, new_pipelines_spec)
    print_metrics(pipelines, Formatter())
