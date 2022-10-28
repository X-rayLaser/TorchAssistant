import argparse
import torch

from torchassistant.utils import instantiate_class, add_cwd_to_python_path, load_config
from torchassistant.session import SessionSaver
from torchassistant.data import WrappedDataset, InputInjector
from torchassistant.output_devices import Printer


class DefaultPostProcessor:
    def __call__(self, predictions):
        return predictions


def parse_input_converter(config_dict):
    spec = config_dict["input_converter"]

    return instantiate_class(
        spec["class"], *spec.get("args", []), **spec.get("kwargs", {})
    )


def parse_post_processor(session, config_dict):
    if "post_processor" not in config_dict:
        return DefaultPostProcessor()

    post_processor_dict = config_dict["post_processor"]

    post_processor_args = [session] + post_processor_dict.get("args", [])

    return instantiate_class(post_processor_dict["class"],
                             *post_processor_args,
                             **post_processor_dict.get("kwargs", {}))


def parse_output_device(config_dict):
    if "output_device" not in config_dict:
        return Printer()

    device_dict = config_dict["output_device"]
    return instantiate_class(
        device_dict["class"], *device_dict.get("args", []), **device_dict.get("kwargs", {})
    )


def run_infer():
    add_cwd_to_python_path()

    parser = argparse.ArgumentParser(
        description='Run inference using pretrained ML pipeline according to a specified configuration file'
    )
    parser.add_argument('config', type=str, help='Path to the configuration file for inference')
    parser.add_argument('input_strings', nargs='+', type=str, help='Input strings for which to make predictions')

    cmd_args = parser.parse_args()
    path = cmd_args.config
    input_strings = cmd_args.input_strings

    config = load_config(path)

    session_dir = config["session_dir"]

    saver = SessionSaver(session_dir)
    session = saver.load_from_latest_checkpoint(new_spec=config)

    if "inference_pipeline" in config:
        pipeline = session.pipelines[config["inference_pipeline"]]
    else:
        pipeline = next(iter(session.pipelines.values()))

    graph = pipeline.graph
    graph.eval_mode()
    input_injector = InputInjector(pipeline.input_loaders)

    inputs_meta = config["inputs_meta"]

    new_datasets = {}

    for s, meta in zip(input_strings, inputs_meta):
        input_converter = parse_input_converter(meta)
        input_port = meta.get("input_port")
        if not input_port:
            input_port = next(iter(graph.batch_input_names))
        new_datasets[input_port] = WrappedDataset([input_converter(s)], [])

    input_injector.override_datasets(new_datasets)

    graph_inputs = next(iter(input_injector))

    post_processor = parse_post_processor(session, config)
    output_device = parse_output_device(config)

    outputs_keys = config.get("results", ["y_hat"])

    print('Running inference on: ', input_strings)

    # todo: refactor this
    class InferenceMode:
        def __enter__(self):
            for node in graph.nodes.values():
                node.inference_mode = True

        def __exit__(self, exc_type, exc_val, exc_tb):
            for node in graph.nodes.values():
                node.inference_mode = False

    with torch.no_grad(), InferenceMode():
        leaf_outputs = graph(graph_inputs)

    leaf_outputs = {k: v for d in leaf_outputs.values() for k, v in d.items()}
    predictions = {k: leaf_outputs[k] for k in outputs_keys}

    output_data = post_processor(predictions)

    output_device(output_data)
