import argparse
from scaffolding.utils import instantiate_class
import torch
from train import load_config
from evaluate import load_json, override_pipelines
from scaffolding.session_v2 import SessionSaver
from scaffolding.training import PredictionPipeline
from scaffolding.utils import WrappedDataset


def parse_input_adapter(config_dict):
    adapter_dict = config_dict["input_adapter"]

    return instantiate_class(
        adapter_dict["class"], *adapter_dict.get("args", []), **adapter_dict.get("kwargs", {})
    )


def parse_post_processor(pipeline, config_dict):
    post_processor_dict = config_dict["post_processor"]
    # todo: consider to pass dynamic_kwargs instead of data pipeline instance
    post_processor_args = [pipeline] + post_processor_dict.get("args", [])
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
    parser.add_argument('input_string', type=str, help='Input string for which to make predictions')

    cmd_args = parser.parse_args()
    path = cmd_args.config
    input_string = cmd_args.input_string

    config = load_config(path)

    session_dir = config["session_dir"]

    saver = SessionSaver(session_dir)
    session = saver.load_from_latest_checkpoint()
    old_spec = load_json(saver.spec_path)
    new_pipelines_spec = config["pipelines"]
    pipelines = override_pipelines(session, old_spec, new_pipelines_spec)

    pipeline = pipelines[0]

    prediction_pipeline = PredictionPipeline(
        pipeline.neural_graph, pipeline.device, pipeline.batch_adapter
    )

    input_adapter = parse_input_adapter(config)
    post_processor = parse_post_processor(pipeline, config)
    output_device = parse_output_device(config)

    outputs_keys = config["results"]

    print('Running inference on: ', input_string)

    batch = process_input(pipeline, input_adapter, input_string)

    with torch.no_grad():
        inputs, _ = prediction_pipeline.adapt_batch(batch)
        outputs = prediction_pipeline(inputs, inference_mode=True)

    predictions = {k: outputs[k] for k in outputs_keys}

    output_data = post_processor(predictions)

    output_device(output_data)
