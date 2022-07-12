import argparse
import json

from scaffolding.training import evaluate_pipeline
from scaffolding.session import SessionSaver, load_json
from scaffolding.session.parse import PipelineLoader
from scaffolding.formatters import Formatter


def load_config(path):
    with open(path, encoding='utf-8') as f:
        s = f.read()

    return json.loads(s)


def override_pipelines(session, old_spec, new_pipelines_spec):
    pipeline_loader = PipelineLoader()

    pipeline_spec = old_spec["initialize"]["pipelines"]

    for name, override_options in new_pipelines_spec.items():
        pipeline_spec.setdefault(name, {}).update(override_options)

    return {name: pipeline_loader.load(session, options)
            for name, options in pipeline_spec.items()}


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
    session = saver.load_from_latest_checkpoint()
    old_spec = load_json(saver.spec_path)

    new_pipelines_spec = config.get("pipelines", {})
    pipelines = override_pipelines(session, old_spec, new_pipelines_spec)

    metrics = {}
    for name in config["validation_pipelines"]:
        pipeline = pipelines[name]
        metrics.update(evaluate_pipeline(pipeline))

    formatter = Formatter()
    final_metrics_string = formatter.format_metrics(metrics, validation=False)
    print(final_metrics_string)
