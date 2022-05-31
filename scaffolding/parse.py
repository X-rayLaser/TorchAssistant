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


def get_transform_pipeline(transform_config):
    if transform_config:
        return transforms.Compose([parse_transform(t) for t in transform_config])
    return transforms.Compose([])


def generate_data(data_dict):
    if "data_generator" not in data_dict:
        return

    generator_config = data_dict["data_generator"]

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
    def __init__(self, dataset, transform, splitter, preprocessors, collator, batch_size, device_str):
        self.dataset = dataset
        self.transform = transform
        self.splitter = splitter
        self.preprocessors = preprocessors
        self.collator = collator
        self.batch_size = batch_size
        self.device_str = device_str

    def get_data_loaders(self):
        # todo: this is a quick fix, refactor later
        data_dict = {
            'dataset_name': self.dataset.class_name,
            'dataset_kwargs': self.dataset.kwargs,
            'transform': self.transform
        }

        train_set, test_set = build_data_split(data_dict, self.splitter)

        if self.preprocessors:
            train_set = WrappedDataset(train_set, self.preprocessors)
            test_set = WrappedDataset(test_set, self.preprocessors)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size,
                                                   shuffle=True, num_workers=2, collate_fn=self.collator)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size,
                                                  shuffle=False, num_workers=2, collate_fn=self.collator)

        return train_loader, test_loader

    def process_raw_input(self, raw_data, input_adapter):
        class MyDataset:
            def __getitem__(self, idx):
                return input_adapter(raw_data)

            def __len__(self):
                return 1

        ds = MyDataset()
        if self.preprocessors:
            ds = WrappedDataset(ds, self.preprocessors)

        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                                             collate_fn=self.collator)

        batch = next(iter(loader))
        return batch

    def to_dict(self):
        return {
            'dataset': self.dataset.to_dict(),
            'transform': self.transform,
            'splitter': self.splitter.to_dict(),
            'preprocessors': [p.to_dict() for p in self.preprocessors],
            'collator': self.collator.to_dict(),
            'batch_size': self.batch_size,
            'device_str': self.device_str
        }

    @classmethod
    def create(cls, config_dict):
        generate_data(config_dict["data"])
        splitter = SimpleSplitter()
        train_set, test_set = build_data_split(config_dict["data"], splitter)

        preprocessors = build_preprocessors(config_dict)
        fit_preprocessors(preprocessors, train_set)

        preprocessors_config = config_dict["data"].get("preprocessors", [])
        exports = [d.get('expose_attributes', []) for d in preprocessors_config]
        for preprocessor, attr_list in zip(preprocessors, exports):
            export_attributes(preprocessor.instance, attr_list)

        collate_fn = parse_collator(config_dict)

        batch_size = config_dict["data"]["batch_size"]
        device_str = config_dict["training"].get("device", "cpu")

        ds_name = config_dict["data"]["dataset_name"]

        ds_args = config_dict["data"].get('dataset_args', [])

        ds_kwargs = config_dict["data"].get('dataset_kwargs', {})
        ds = SerializableDataset(class_name=ds_name, args=ds_args, kwargs=ds_kwargs)

        transform = config_dict["data"].get("transform", [])
        splitter = GenericSerializableInstance(
            splitter, class_name='scaffolding.utils.SimpleSplitter', args=[], kwargs={}
        )

        return DataPipeline(dataset=ds, transform=transform, splitter=splitter,
                            preprocessors=preprocessors, collator=collate_fn,
                            batch_size=batch_size, device_str=device_str)

    @classmethod
    def from_dict(cls, state_dict):
        dataset = SerializableDataset.from_dict(state_dict['dataset'])
        transform = state_dict['transform']
        splitter = GenericSerializableInstance.from_dict(state_dict['splitter'])

        preprocessors = [GenericSerializableInstance.from_dict(d) for d in state_dict['preprocessors']]

        collator = GenericSerializableInstance.from_dict(state_dict['collator'])
        batch_size = state_dict['batch_size']
        device_str = state_dict['device_str']
        return cls(dataset, transform, splitter, preprocessors, collator, batch_size, device_str)


class LoaderWithDevice:
    def __init__(self, loader, device_str):
        self.loader = loader
        self.device_str = device_str

    def __iter__(self):
        for batch in self.loader:
            change_batch_device(batch, torch.device(self.device_str))
            yield batch


def build_data_split(data_dict, splitter):
    ds_class_name = data_dict["dataset_name"]

    if '.' in ds_class_name:
        # assume that we got a fully-qualified path to a custom class
        ds_kwargs = data_dict.get("dataset_kwargs", {})
        dataset = instantiate_class(ds_class_name, **ds_kwargs)
        splitter.prepare(dataset)
        train_set = splitter.train_ds
        test_set = splitter.val_ds
    else:
        dataset_class = getattr(torchvision.datasets, ds_class_name)
        transform_list = data_dict.get("transform", [])
        transform = get_transform_pipeline(transform_list)

        train_set = dataset_class(root='./data', train=True, download=True, transform=transform)
        test_set = dataset_class(root='./data', train=False, download=True, transform=transform)

    return train_set, test_set


def build_preprocessors(config_dict):
    preprocessors_config = config_dict["data"].get("preprocessors", [])
    preprocessors = []
    for d in preprocessors_config:
        wrapped_preprocessor = build_generic_serializable_instance(d)
        preprocessors.append(wrapped_preprocessor)

    return preprocessors


def fit_preprocessors(preprocessors, train_set):
    for p in preprocessors:
        p.instance.fit(train_set)


def export_attributes(obj, exported_attrs):
    for attr_name in exported_attrs:
        attr_value = getattr(obj, attr_name)
        setattr(store, attr_name, attr_value)


def parse_collator(config_dict):
    collator_config = config_dict["data"].get("collator")
    if collator_config:
        return build_generic_serializable_instance(collator_config)
    else:
        from .collators import StackTensors
        batch_divide = StackTensors()
        return GenericSerializableInstance(batch_divide, 'scaffolding.collators.StackTensors', [], {})


def parse_batch_adapter(config_dict):
    adapter_config = config_dict["data"].get("batch_adapter")
    if adapter_config:
        return build_generic_serializable_instance(adapter_config)
    else:
        # todo: return default adapter
        pass


def build_generic_serializable_instance(object_dict, instantiate_fn=None, wrapper_class=None):
    instantiate_fn = instantiate_fn or instantiate_class
    wrapper_class = wrapper_class or GenericSerializableInstance

    class_name = object_dict["class"]
    args = object_dict.get("args", [])
    kwargs = object_dict.get("kwargs", {})

    dynamic_kwargs = {}
    for arg_name, store_arg_name in object_dict.get("dynamic_kwargs", {}).items():
        dynamic_kwargs[arg_name] = getattr(store, store_arg_name)

    kwargs.update(dynamic_kwargs)

    instance = instantiate_fn(class_name, *args, **kwargs)
    return wrapper_class(instance, class_name, args, kwargs)


def parse_metrics(metrics_config, data_pipeline, device):
    metrics = {}

    try:
        error = False
        exc_args = None
        metrics = {name: parse_single_metric(name, metrics_config[name], data_pipeline, device)
                   for name in metrics_config}
    except KeyError as exc:
        error = True
        exc_args = exc.args

    if error:
        print(exc_args)
        allowed_metrics = list(metric_functions.keys())
        error_message = f'Unknown metric "{exc_args[0]}". ' \
                        f'Must be either a metric from TorchMetrics or one of {allowed_metrics}'
        raise InvalidParameterError(error_message)
    return metrics


def parse_single_metric(metric_name, metric_dict, data_pipeline, device):
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

    return Metric(metric_name, metric, metric_dict["inputs"], transform_fn, device)


def parse_model(config_dict):
    model_config = config_dict["training"]["model"]
    return [parse_submodel(config) for config in model_config]


def parse_submodel(config):
    """Returns a tuple of (nn.Module instance, optimizer, inputs, outputs)"""
    arch_config = config["arch"]
    serializable_model = build_generic_serializable_instance(arch_config, wrapper_class=SerializableModel)

    optimizer_config = config["optimizer"]

    def instantiate_fn(class_name, *args, **kwargs):
        optimizer_class = getattr(optim, class_name)
        return optimizer_class(serializable_model.instance.parameters(), *args, **kwargs)

    serializable_optimizer = build_generic_serializable_instance(
        optimizer_config, instantiate_fn, wrapper_class=SerializableOptimizer
    )

    return Node(name=config["name"], serializable_model=serializable_model,
                serializable_optimizer=serializable_optimizer,
                inputs=config["inputs"], outputs=config["outputs"])


def parse_loss(loss_config, device):
    loss_class_name = loss_config["class"]

    if "transform" in loss_config:
        transform_fn = import_function(loss_config["transform"])
    else:
        transform_fn = lambda *fn_args: fn_args

    criterion_class = getattr(nn, loss_class_name)
    args = loss_config.get("args", [])
    kwargs = loss_config.get("kwargs", {})

    return Metric('loss', criterion_class(*args, **kwargs), loss_config["inputs"], transform_fn, device)


def parse_device(config_dict):
    device_str = config_dict["training"].get("device", "cpu")
    return torch.device(device_str)


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
