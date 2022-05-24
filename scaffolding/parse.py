import os.path
import os
import inspect

import torch
from torch import nn

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import torchmetrics

from scaffolding.metrics import metric_functions, Metric
from scaffolding.utils import SimpleSplitter, instantiate_class, import_function, import_entity, \
    AdaptedCollator, WrappedDataset, DecoratedInstance, GenericSerializableInstance, change_batch_device
from scaffolding.store import store
from scaffolding.exceptions import InvalidParameterError


def parse_transform(transform_dict):
    name = transform_dict["name"]
    args_list = transform_dict.get("args", [])

    if name == "totensor":
        return transforms.ToTensor()
    if name == "normalize":
        return transforms.Normalize(tuple(args_list[0]), tuple(args_list[1]))


def get_transform_pipeline(config_dict):
    transform_config = config_dict["data"].get("transform")
    if transform_config:
        return transforms.Compose([parse_transform(t) for t in transform_config])
    return transforms.Compose([])


def parse_data_pipeline(config_dict):
    generate_data(config_dict)
    splitter = SimpleSplitter()
    train_set, test_set = parse_datasets(config_dict, splitter)
    preprocessors = fit_preprocessors(train_set, config_dict)

    collate_fn = parse_collator(config_dict)
    batch_adapter = parse_batch_adapter(config_dict)
    batch_size = config_dict["data"]["batch_size"]
    device_str = config_dict["training"].get("device", "cpu")

    ds_name = config_dict["data"]["dataset_name"]
    ds_kwargs = config_dict["data"].get('dataset_kwargs', {})
    ds = SerializableDataset(class_name=ds_name, args=[], kwargs=ds_kwargs)

    splitter = GenericSerializableInstance(
        splitter, class_name='scaffolding.utils.SimpleSplitter', args=[], kwargs={}
    )

    preprocessors_config = config_dict["data"]["preprocessors"]
    preprocessors = [GenericSerializableInstance(p, conf["class"], [], {})
                     for conf, p in zip(preprocessors_config, preprocessors)]

    return DataPipeline(dataset=ds, splitter=splitter,
                        preprocessors=preprocessors, collator=collate_fn,
                        batch_adapter=batch_adapter, batch_size=batch_size,
                        device_str=device_str)


def generate_data(config_dict):
    if "data_generator" not in config_dict["data"]:
        return

    generator_config = config_dict["data"]["data_generator"]

    class_name = generator_config["class"]
    output_dir = generator_config["output_dir"]

    if os.path.isdir(output_dir):
        return

    os.makedirs(output_dir)

    args = generator_config.get('args', [])
    kwargs = generator_config.get('kwargs', {})
    save_example_name = generator_config.get('save_example_fn')

    save_example = import_function(save_example_name) if save_example_name else default_save_example

    generator = instantiate_class(class_name, *args, **kwargs)

    for i, inputs in enumerate(generator):
        save_example(inputs, output_dir, i)


def default_save_example(example, output_dir, index):
    for j, elem in enumerate(example):
        path = os.path.join(output_dir, str(j))
        with open(path, 'a', encoding='utf-8') as f:
            f.write(f'{elem}\n')


class DataPipeline:
    def __init__(self, dataset, splitter, preprocessors, collator, batch_adapter, batch_size, device_str):
        self.dataset = dataset
        self.splitter = splitter
        self.preprocessors = preprocessors
        self.collator = collator
        self.batch_adapter = batch_adapter
        self.batch_size = batch_size
        self.device_str = device_str

    def get_data_loaders(self):
        # todo: this is a quick fix, refactor later
        # todo: fix code to access instance attribute on serializable objects or add getattr to delegate
        config_dict = {
            'data': {
                'dataset_name': self.dataset.class_name,
                'dataset_kwargs': self.dataset.kwargs
            }
        }
        train_set, test_set = parse_datasets(config_dict, self.splitter)

        train_set = WrappedDataset(train_set, self.preprocessors)
        test_set = WrappedDataset(test_set, self.preprocessors)

        final_collate_fn = AdaptedCollator(self.collator, self.batch_adapter)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size,
                                                   shuffle=True, num_workers=2, collate_fn=final_collate_fn)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size,
                                                  shuffle=False, num_workers=2, collate_fn=final_collate_fn)
        return LoaderWithDevice(train_loader, self.device_str), LoaderWithDevice(test_loader, self.device_str)

    def process_raw_input(self, raw_data, input_adapter):
        class MyDataset:
            def __getitem__(self, idx):
                return input_adapter(raw_data)

            def __len__(self):
                return 1

        ds = MyDataset()
        ds = WrappedDataset(ds, self.preprocessors)
        final_collate_fn = AdaptedCollator(self.collator, self.batch_adapter)

        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                                             collate_fn=final_collate_fn)

        batch = next(iter(loader))
        change_batch_device(batch, torch.device(self.device_str))
        return batch

    def to_dict(self):
        return {
            'dataset': self.dataset.to_dict(),
            'splitter': self.splitter.to_dict(),
            'preprocessors': [p.to_dict() for p in self.preprocessors],
            'collator': self.collator.to_dict(),
            'batch_adapter': self.batch_adapter.to_dict(),
            'batch_size': self.batch_size,
            'device_str': self.device_str
        }

    @classmethod
    def from_dict(cls, state_dict):
        dataset = SerializableDataset.from_dict(state_dict['dataset'])
        splitter = GenericSerializableInstance.from_dict(state_dict['splitter'])

        preprocessors = [GenericSerializableInstance.from_dict(d) for d in state_dict['preprocessors']]

        collator = GenericSerializableInstance.from_dict(state_dict['collator'])
        batch_adapter = GenericSerializableInstance.from_dict(state_dict['batch_adapter'])
        batch_size = state_dict['batch_size']
        device_str = state_dict['device_str']
        return cls(dataset, splitter, preprocessors, collator, batch_adapter, batch_size, device_str)


class LoaderWithDevice:
    def __init__(self, loader, device_str):
        self.loader = loader
        self.device_str = device_str

    def __iter__(self):
        for batch in self.loader:
            change_batch_device(batch, torch.device(self.device_str))
            yield batch


def parse_datasets(config_dict, splitter):
    ds_class_name = config_dict["data"]["dataset_name"]

    if '.' in ds_class_name:
        # assume that we got a fully-qualified path to a custom class
        ds_kwargs = config_dict["data"].get("dataset_kwargs", {})
        dataset = instantiate_class(ds_class_name, **ds_kwargs)
        splitter.prepare(dataset)
        train_set = splitter.train_ds
        test_set = splitter.val_ds
    else:
        dataset_class = getattr(torchvision.datasets, ds_class_name)
        transform = get_transform_pipeline(config_dict)
        train_set = dataset_class(root='./data', train=True, download=True, transform=transform)
        test_set = dataset_class(root='./data', train=False, download=True, transform=transform)
    return train_set, test_set


def fit_preprocessors(train_set, config_dict):
    preprocessors_config = config_dict["data"]["preprocessors"]
    preprocessors = []
    for d in preprocessors_config:
        class_name = d["class"]
        args = d.get("args", [])
        kwargs = d.get("kwargs", {})
        preprocessor = instantiate_class(class_name, *args, **kwargs)

        preprocessor.fit(train_set)

        for attr_name in d.get('expose_attributes', []):
            attr_value = getattr(preprocessor, attr_name)
            setattr(store, attr_name, attr_value)

        wrapped_preprocessor = GenericSerializableInstance(preprocessor, class_name, args=args, kwargs=kwargs)
        preprocessors.append(wrapped_preprocessor)

    return preprocessors


def parse_collator(config_dict):
    collator_config = config_dict["data"]["collator"]
    collator_class_name = collator_config["class"]

    collator_args = collator_config.get("args", [])
    collator_kwargs = collator_config.get("kwargs", {})

    dynamic_kwargs = {}
    for collator_arg_name, store_arg_name in collator_config.get("dynamic_kwargs", {}).items():
        dynamic_kwargs[collator_arg_name] = getattr(store, store_arg_name)

    collator_kwargs.update(dynamic_kwargs)

    collator = instantiate_class(collator_class_name, *collator_args, **collator_kwargs)
    return GenericSerializableInstance(collator, collator_class_name, collator_args, collator_kwargs)


def parse_batch_adapter(config_dict):
    adapter_config = config_dict["data"]["batch_adapter"]
    adapter_class_name = adapter_config["class"]
    adapter_args = adapter_config.get("args", [])
    adapter_kwargs = adapter_config.get("kwargs", {})

    dynamic_kwargs = {}
    for adapter_arg_name, store_arg_name in adapter_config.get("dynamic_kwargs", {}).items():
        dynamic_kwargs[adapter_arg_name] = getattr(store, store_arg_name)

    adapter_kwargs.update(dynamic_kwargs)
    adapter = instantiate_class(adapter_class_name, *adapter_args, **adapter_kwargs)
    return GenericSerializableInstance(adapter, adapter_class_name, adapter_args, adapter_kwargs)


def parse_metrics(config_dict, data_pipeline):
    metrics_config = config_dict["training"]["metrics"]
    metrics = {}

    try:
        error = False
        exc_args = None
        metrics = {name: parse_single_metric(name, metrics_config[name], data_pipeline)
                   for name in metrics_config}
    except KeyError as exc:
        error = True
        exc_args = exc.args

    if error:
        allowed_metrics = list(metric_functions.keys())
        error_message = f'Unknown metric "{exc_args[0]}". ' \
                        f'Must be either a metric from TorchMetrics or one of {allowed_metrics}'
        raise InvalidParameterError(error_message)
    return metrics


def parse_single_metric(metric_name, metric_dict, data_pipeline):
    if "transform" in metric_dict:
        transform_fn = import_entity(metric_dict["transform"])
        if inspect.isclass(transform_fn):
            transform_fn = transform_fn(data_pipeline)
    else:
        def transform_fn(*fn_args):
            return fn_args

    if hasattr(torchmetrics, metric_name):
        metric = instantiate_class(f'torchmetrics.{metric_name}')
    else:
        metric = metric_functions[metric_name]

    return Metric(metric_name, metric, metric_dict["inputs"], transform_fn)


def parse_model(config_dict):
    model_config = config_dict["training"]["model"]
    return [parse_submodel(config) for config in model_config]


def parse_submodel(config):
    """Returns a tuple of (nn.Module instance, optimizer, inputs, outputs)"""
    path = config["arch"]
    model_args = config.get("args", [])
    model_kwargs = config.get("kwargs", {})

    dynamic_kwargs = config.get("dynamic_kwargs", {})
    total_kwargs = model_kwargs.copy()

    for k, v in dynamic_kwargs.items():
        total_kwargs[k] = getattr(store, v)

    sub_model = instantiate_class(path, *model_args, **total_kwargs)

    optimizer_config = config["optimizer"]
    optimizer_class_name = optimizer_config["class"]

    # todo: this is obviously an error, it should read args and kwargs, not params
    optimizer_args = optimizer_config.get("kwargs", [])
    optimizer_kwargs = optimizer_config.get("params", {})

    optimizer_class = getattr(optim, optimizer_class_name)
    optimizer = optimizer_class(sub_model.parameters(), **optimizer_kwargs)

    serializable_model = SerializableModel(sub_model, path, model_args, total_kwargs)
    serializable_optimizer = SerializableOptimizer(optimizer, optimizer_class_name,
                                                   optimizer_args, optimizer_kwargs)
    return Node(name=config["name"], serializable_model=serializable_model,
                serializable_optimizer=serializable_optimizer,
                inputs=config["inputs"], outputs=config["outputs"])


def parse_loss(config_dict):
    loss_config = config_dict["training"]["loss"]
    loss_class_name = loss_config["class"]
    if "transform" in loss_config:
        transform_fn = import_function(loss_config["transform"])
    else:
        transform_fn = lambda *fn_args: fn_args

    criterion_class = getattr(nn, loss_class_name)
    args = loss_config.get("args", [])
    kwargs = loss_config.get("kwargs", {})

    return Metric('loss', criterion_class(*args, **kwargs), loss_config["inputs"], transform_fn)


def parse_epochs(config_dict):
    return config_dict["training"]["num_epochs"]


def parse_checkpoint_dir(config_dict):
    return config_dict["training"]["checkpoints_dir"]


class Node:
    def __init__(self, name, serializable_model, serializable_optimizer, inputs, outputs):
        self.name = name
        self.net = serializable_model
        self.optimizer = serializable_optimizer
        self.inputs = inputs
        self.outputs = outputs


class SerializableModel(DecoratedInstance):
    def to_dict(self):
        return {
            'model_state_dict': self.instance.state_dict(),
            'model_class': self.class_name,
            'model_args': self.args,
            'model_kwargs': self.kwargs,
        }

    @classmethod
    def from_dict(cls, d):
        model_class_path = d['model_class']
        args = d['model_args']
        kwargs = d['model_kwargs']
        model = instantiate_class(model_class_path, *args, **kwargs)
        model.load_state_dict(d['model_state_dict'])

        return cls(instance=model, class_name=model_class_path, args=args, kwargs=kwargs)


class SerializableOptimizer(DecoratedInstance):
    def to_dict(self):
        return {
            'optimizer_state_dict': self.instance.state_dict(),
            'optimizer_class': self.class_name,
            'optimizer_args': self.args,
            'optimizer_kwargs': self.kwargs,
        }

    @classmethod
    def from_dict(cls, d, model):
        optimizer_class_name = d['optimizer_class']
        args = d['optimizer_args']
        kwargs = d['optimizer_kwargs']

        optimizer_class = getattr(optim, optimizer_class_name)
        optimizer = optimizer_class(model.parameters(), *args, **kwargs)
        optimizer.load_state_dict(d['optimizer_state_dict'])

        return cls(instance=optimizer, class_name=optimizer_class_name, args=args, kwargs=kwargs)


class SerializableDataset(DecoratedInstance):
    def __init__(self, class_name, args, kwargs):
        super().__init__(None, class_name, args, kwargs)

    def to_dict(self):
        return {
            'dataset_class': self.class_name,
            'dataset_args': self.args,
            'dataset_kwargs': self.kwargs,
        }

    @classmethod
    def from_dict(cls, state_dict):
        return cls(state_dict["dataset_class"], state_dict["dataset_args"], state_dict["dataset_kwargs"])
