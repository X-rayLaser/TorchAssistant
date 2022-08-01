import argparse
from scaffolding.utils import instantiate_class
import torch
from train import load_config
from scaffolding.session import SessionSaver
from scaffolding.utils import WrappedDataset
from scaffolding.processing_graph import DataBlueprint


def parse_input_adapter(config_dict):
    adapter_dict = config_dict["input_adapter"]

    return instantiate_class(
        adapter_dict["class"], *adapter_dict.get("args", []), **adapter_dict.get("kwargs", {})
    )


def parse_post_processor(session, config_dict):
    post_processor_dict = config_dict["post_processor"]

    post_processor_args = [session] + post_processor_dict.get("args", [])
    return instantiate_class(post_processor_dict["class"],
                             *post_processor_args,
                             **post_processor_dict.get("kwargs", {}))


def parse_output_device(config_dict):
    device_dict = config_dict["output_device"]
    return instantiate_class(
        device_dict["class"], *device_dict.get("args", []), **device_dict.get("kwargs", {})
    )


def process_input(pipeline, input_adapter, input_string):
    ds = [input_adapter(input_string)]

    if pipeline.preprocessors:
        ds = WrappedDataset(ds, pipeline.preprocessors)

    return pipeline.collator.collate_inputs(ds[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run inference using pretrained ML pipeline according to a specified configuration file'
    )
    parser.add_argument('config', type=str, help='Path to the configuration file for inference')
    parser.add_argument('input_strings', nargs='+', type=str, help='Input strings for which to make predictions')

    cmd_args = parser.parse_args()
    path = cmd_args.config
    input_strings = cmd_args.input_strings
    print(input_strings)
    config = load_config(path)

    session_dir = config["session_dir"]

    saver = SessionSaver(session_dir)
    session = saver.load_from_latest_checkpoint(new_spec=config)

    pipeline = session.pipelines[config["inference_pipeline"]]

    graph = pipeline.graph
    loaders = pipeline.input_loaders
    data_generator = DataBlueprint(loaders)

    inputs_meta = config["inputs_meta"]

    new_datasets = {}

    for s, meta in zip(input_strings, inputs_meta):
        input_adapter = parse_input_adapter(meta)
        input_alias = meta["input_alias"]
        new_datasets[input_alias] = WrappedDataset([input_adapter(s)], [])

    data_generator.override_datasets(new_datasets)

    graph_inputs = next(iter(data_generator))

    post_processor = parse_post_processor(session, config)
    output_device = parse_output_device(config)

    outputs_keys = config["results"]

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
