import csv
import json
import os

import torch

from scaffolding import parse
from scaffolding.training import PredictionPipeline
from scaffolding.utils import instantiate_class, load_data_pipeline, change_model_device, \
    save_data_pipeline


def parse_adapter(batch_adapter_config):
    def instantiate_fn(class_name, *args, **kwargs):
        args = (model,) + args
        return instantiate_class(class_name, *args, *kwargs)

    if batch_adapter_config:
        return parse.build_generic_serializable_instance(batch_adapter_config)
    else:
        # todo: default adapter
        pass


class TrainingSession:
    def __init__(self, path):
        self.path = path

        self.checkpoints_dir = os.path.join(path, 'checkpoints')
        self.history_path = os.path.join(path, 'history.csv')

        self.data_pipeline_path = os.path.join(path, 'data_pipeline.json')
        self.batch_adapter_path = os.path.join(path, 'batch_adapter.json')
        self.extra_params_path = os.path.join(path, 'extra_params.json')

        self.data_pipeline = load_data_pipeline(self.data_pipeline_path)

        d = load_json(self.batch_adapter_path)
        self.batch_adapter = parse.GenericSerializableInstance.from_dict(d)

        self.extra_params = load_json(self.extra_params_path)
        self.device = torch.device(self.extra_params["device"])
        self.num_epochs = self.extra_params["num_epochs"]

        metrics_dict = self.extra_params.get("metrics", {})
        self.metrics = parse.parse_metrics(metrics_dict, self.data_pipeline, self.device)

        loss_config = self.extra_params.get("loss", {})
        self.criterion = parse.parse_loss(loss_config, self.device)

    @property
    def epochs_trained(self):
        _, dir_names, _ = next(os.walk(self.checkpoints_dir))
        # first checkpoint is for untrained model, therefore we subtract it
        return len(dir_names) - 1

    def restore_from_last_checkpoint(self, inference_mode=False):
        model = load_last_checkpoint(self.checkpoints_dir, self.device, inference_mode)
        change_model_device(model, self.data_pipeline.device_str)

        train_pipeline = PredictionPipeline(model, self.device, self.batch_adapter)

        return train_pipeline

    def make_checkpoint(self, train_pipeline, epoch):
        save_model_pipeline(train_pipeline, epoch, self.checkpoints_dir)

    def log_metrics(self, epoch, train_metrics, val_metrics):
        # todo: log metrics to csv file
        history = TrainingHistory(self.history_path)
        history.add_entry(epoch, train_metrics, val_metrics)

    @classmethod
    def create_session(cls, config, save_path):
        checkpoints_dir = os.path.join(save_path, 'checkpoints')
        history_path = os.path.join(save_path, 'history.csv')
        data_pipeline_path = os.path.join(save_path, 'data_pipeline.json')
        batch_adapter_path = os.path.join(save_path, 'batch_adapter.json')
        extra_params_path = os.path.join(save_path, 'extra_params.json')

        os.makedirs(checkpoints_dir, exist_ok=True)

        epochs = config.num_epochs

        data_pipeline = config.data_pipeline
        save_data_pipeline(data_pipeline, data_pipeline_path)

        model = config.model
        change_model_device(model, data_pipeline.device_str)
        save_model_pipeline(model, 0, checkpoints_dir)

        batch_adapter = config.batch_adapter
        save_as_json(batch_adapter.to_dict(), batch_adapter_path)

        extra_params = {}

        extra_params["device"] = config.device_str

        if config.loss:
            extra_params["loss"] = config.loss

        if config.metrics:
            extra_params["metrics"] = config.metrics

        extra_params["num_epochs"] = epochs
        save_as_json(extra_params, extra_params_path)

        metrics_dict = extra_params.get("metrics", [])

        field_names = list(metrics_dict.keys()) + [f'val {name}' for name in metrics_dict.keys()]

        history = TrainingHistory.create(history_path, field_names)
        # todo: calculate metrics for 0-th epoch (before any training)


class ConfigParser:
    def __init__(self, settings):
        self.settings = settings
        self.training_config = settings["training"]

    def get_config(self):
        config = Configuration(
            session_dir=self.parse_checkpoint_dir(),
            num_epochs=self.parse_epochs(),
            data_pipeline=self.parse_data_pipeline(),
            model=self.parse_model_pipeline(),
            batch_adapter=self.parse_batch_adapter(),
            device_str=self.parse_device(),
            loss=self.parse_loss_fn(),
            metrics=self.parse_metrics()
        )

        return config

    def parse_checkpoint_dir(self):
        return parse.parse_checkpoint_dir(self.settings)

    def parse_epochs(self):
        return parse.parse_epochs(self.settings)

    def parse_data_pipeline(self):
        return parse.DataPipeline.create(self.settings)

    def parse_model_pipeline(self):
        return parse.parse_model(self.settings)

    def parse_batch_adapter(self):
        return parse_adapter(self.settings["training"].get("batch_adapter"))

    def parse_device(self):
        return self.training_config.get("device", "cpu")

    def parse_loss_fn(self):
        return self.training_config.get("loss")

    def parse_metrics(self):
        return self.training_config.get("metrics")


class Configuration:
    def __init__(self, *, session_dir, num_epochs, data_pipeline, model, batch_adapter,
                 device_str, loss, metrics):
        self.session_dir = session_dir
        self.num_epochs = num_epochs
        self.data_pipeline = data_pipeline
        self.model = model
        self.batch_adapter = batch_adapter
        self.device_str = device_str
        self.loss = loss
        self.metrics = metrics


class TrainingHistory:
    def __init__(self, file_path):
        self.file_path = file_path

    def add_entry(self, epoch, train_metrics, val_metrics):
        # todo: make sure the ordering is right
        val_metrics = {f'val {k}': v for k, v in val_metrics.items()}

        all_metrics = {}
        all_metrics.update(train_metrics)
        all_metrics.update(val_metrics)

        row_dict = {'epoch': epoch}
        row_dict.update({k: self.scalar(v) for k, v in all_metrics.items()})

        with open(self.file_path, 'a', encoding='utf-8', newline='') as csvfile:
            fieldnames = list(row_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row_dict)

    def scalar(self, t):
        return t.item() if hasattr(t, 'item') else t

    @classmethod
    def create(cls, path, field_names):
        with open(path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = ['Epoch #'] + field_names
            writer.writerow(row)
        return cls(path)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        s = f.read()

    return json.loads(s)


def save_as_json(d, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(d))


def load_checkpoint(checkpoints_dir, epoch, device, inference_mode=False):
    from scaffolding.parse import Node, SerializableModel, SerializableOptimizer

    epoch_dir = os.path.join(checkpoints_dir, str(epoch))

    nodes_with_numbers = []
    for file_name in os.listdir(epoch_dir):
        path = os.path.join(epoch_dir, file_name)
        checkpoint = torch.load(path)

        serializable_model = SerializableModel.from_dict(checkpoint)
        serializable_model.instance.to(device)
        serializable_optimizer = SerializableOptimizer.from_dict(
            checkpoint, serializable_model.instance
        )

        # todo: consider doing this outside the function call
        if inference_mode:
            serializable_model.instance.eval()
        else:
            serializable_model.instance.train()

        node = Node(name=checkpoint["name"], serializable_model=serializable_model,
                    serializable_optimizer=serializable_optimizer, inputs=checkpoint["inputs"],
                    outputs=checkpoint["outputs"])
        nodes_with_numbers.append((node, checkpoint["number"]))

    nodes_with_numbers.sort(key=lambda t: t[1])
    return [t[0] for t in nodes_with_numbers]


def load_last_checkpoint(epochs_dir, device, inference_mode=False):
    last_epoch = sorted(os.listdir(epochs_dir), key=lambda d: int(d), reverse=True)[0]
    return load_checkpoint(epochs_dir, int(last_epoch), device, inference_mode)


def save_model_pipeline(train_pipeline, epoch, checkpoints_dir):
    epoch_dir = os.path.join(checkpoints_dir, str(epoch))

    os.makedirs(epoch_dir, exist_ok=True)

    for number, pipe in enumerate(train_pipeline, start=1):
        save_path = os.path.join(epoch_dir, pipe.name)
        d = {
            'name': pipe.name,
            'number': number,
            'inputs': pipe.inputs,
            'outputs': pipe.outputs,
            'epoch': epoch
        }
        d.update(pipe.net.to_dict())
        d.update(pipe.optimizer.to_dict())

        torch.save(d, save_path)
