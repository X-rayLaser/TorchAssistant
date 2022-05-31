import argparse
import os
from scaffolding.utils import load_data_pipeline, load_session_from_last_epoch, instantiate_class
from scaffolding.parse import parse_batch_adapter, build_generic_serializable_instance
import torch
from train import load_config
from scaffolding.training import PredictionPipeline
from init import TrainingSession


def parse_input_adapter(config_dict):
    adapter_dict = config_dict["data"]["input_adapter"]

    return instantiate_class(
        adapter_dict["class"], *adapter_dict.get("args", []), **adapter_dict.get("kwargs", {})
    )


def parse_post_processor(config_dict):
    post_processor_dict = config_dict["post_processor"]
    # todo: consider to pass dynamic_kwargs instead of data pipeline instance
    post_processor_args = [data_pipeline] + post_processor_dict.get("args", [])
    return instantiate_class(post_processor_dict["class"],
                             *post_processor_args,
                             **post_processor_dict.get("kwargs", {}))


def parse_output_device(config_dict):
    device_dict = config_dict["output_device"]
    return instantiate_class(
        device_dict["class"], *device_dict.get("args", []), **device_dict.get("kwargs", {})
    )


def override_from_config(prediction_pipeline, config):
    # todo: should be able to override device

    for i, node in enumerate(prediction_pipeline.model):
        node.inputs = config["model"][i]["inputs"]
        node.outputs = config["model"][i]["outputs"]

    batch_adapter_config = config.get("batch_adapter")
    if batch_adapter_config:
        prediction_pipeline.batch_adapter = build_generic_serializable_instance(batch_adapter_config)


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
    config = config["pipeline"]
    pretrained_dir = config["checkpoints_dir"]

    session = TrainingSession(pretrained_dir)

    data_pipeline = session.data_pipeline

    batch_adapter = session.batch_adapter

    prediction_pipeline = session.restore_from_last_checkpoint(inference_mode=True)
    override_from_config(prediction_pipeline, config)

    input_adapter = parse_input_adapter(config)
    post_processor = parse_post_processor(config)
    output_device = parse_output_device(config)

    outputs_keys = config["results"]

    print('Running inference on: ', input_string)

    batch = data_pipeline.process_raw_input(input_string, input_adapter)

    with torch.no_grad():
        inputs, _ = prediction_pipeline.adapt_batch(batch)
        outputs = prediction_pipeline(inputs, inference_mode=True)

    predictions = {k: outputs[k] for k in outputs_keys}

    output_data = post_processor(predictions)

    output_device(output_data)
