import importlib
import os
from collections import namedtuple

import torch
from torch import optim
from torch.utils.data import Dataset

from scaffolding.exceptions import ClassImportError, FunctionImportError


class DataSplitter:
    def __init__(self, dataset, train_fraction=0.8):
        self.dataset = dataset
        self.fraction = train_fraction

        self.num_train = int(train_fraction * len(dataset))
        self.num_val = len(dataset) - self.num_train

    @property
    def train_ds(self):
        return []

    @property
    def val_ds(self):
        return []


class SimpleSplitter(DataSplitter):
    @property
    def train_ds(self):
        return DatasetSlice(self.dataset, 0, self.num_train)

    @property
    def val_ds(self):
        offset = self.num_train
        return DatasetSlice(self.dataset, offset, offset + self.num_val)


class DatasetSlice(Dataset):
    def __init__(self, ds, index_from, index_to):
        """Create a dataset slice

        :param ds: original dataset
        :type ds: type of sequence
        :param index_from: start index of the slice (included in a slice)
        :param index_to: last index of the slice (excluded from a slice)
        """
        self.ds = ds
        self.index_from = index_from
        self.index_to = index_to

    def __getitem__(self, idx):
        if not (0 <= idx < len(self)):
            raise IndexError(f'DatasetSlice: Index out of bounds: {idx}')
        return self.ds[idx + self.index_from]

    def __len__(self):
        return self.index_to - self.index_from


def instantiate_class(dotted_path, *args, **kwargs):
    parts = dotted_path.split('.')
    module_path = '.'.join(parts[:-1])
    class_name = parts[-1]

    error_msg = f'Failed to import a class "{class_name}" from "{module_path}"'

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        raise ClassImportError(error_msg)

    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise ClassImportError(error_msg)

    return cls(*args, **kwargs)


def import_function(dotted_path):
    parts = dotted_path.split('.')
    module_path = '.'.join(parts[:-1])
    fn_name = parts[-1]

    error_msg = f'Failed to import a function "{fn_name}" from "{module_path}"'

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        raise FunctionImportError(error_msg)

    try:
        return getattr(module, fn_name)
    except AttributeError:
        raise FunctionImportError(error_msg)


def load_pipeline(pipeline_dict, inference_mode=False):
    data_pipeline = pipeline_dict["data"]

    def parse_submodel(config):
        """Returns a tuple of (nn.Module instance, optimizer, inputs, outputs)"""
        path = config["arch"]
        weights_path = config["weights"]
        checkpoint_path = config["checkpoint"]
        model_args = config.get("args", [])
        model_kwargs = config.get("kwargs", {})

        sub_model = instantiate_class(path, *model_args, **model_kwargs)

        optimizer_config = config["optimizer"]
        optimizer_class_name = optimizer_config["class"]
        optimizer_params = optimizer_config["params"]

        optimizer_class = getattr(optim, optimizer_class_name)
        optimizer = optimizer_class(sub_model.parameters(), **optimizer_params)

        if inference_mode:
            sub_model.load_state_dict(torch.load(weights_path))
            sub_model.eval()
        else:
            checkpoint = torch.load(checkpoint_path)
            sub_model.train()
            sub_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']

        cls = namedtuple('SubModel', ['name', 'net', 'optimizer', 'inputs', 'outputs'])
        return cls(config["name"], sub_model, optimizer, config["inputs"], config["outputs"])

    model_pipeline = [parse_submodel(config) for config in pipeline_dict["inference"]]
    return data_pipeline, model_pipeline


def save_session(train_pipeline, epoch, checkpoints_dir):
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


def load_session(checkpoints_dir, epoch):
    from scaffolding.parse import Node, SerializableModel, SerializableOptimizer

    epoch_dir = os.path.join(checkpoints_dir, str(epoch))

    nodes_with_numbers = []
    for file_name in os.listdir(epoch_dir):
        path = os.path.join(epoch_dir, file_name)
        checkpoint = torch.load(path)

        serializable_model = SerializableModel.from_dict(checkpoint)
        serializable_optimizer = SerializableOptimizer.from_dict(
            checkpoint, serializable_model.instance
        )

        node = Node(name=checkpoint["name"], serializable_model=serializable_model,
                    serializable_optimizer=serializable_optimizer, inputs=checkpoint["inputs"],
                    outputs=["outputs"])
        nodes_with_numbers.append((node, checkpoint["number"]))

    nodes_with_numbers.sort(key=lambda t: t[1])
    return [t[0] for t in nodes_with_numbers]


# todo: implement a proper pseudo random yet deterministic splitter class based on seed
class AdaptedCollator:
    def __init__(self, collator, batch_adapter):
        self.collator = collator
        self.adapter = batch_adapter

    def __call__(self, *args):
        batch = self.collator(*args)
        return self.adapter.adapt(*batch)


class WrappedDataset:
    def __init__(self, dataset, preprocessors):
        self.dataset = dataset
        self.preprocessors = preprocessors

    def __getitem__(self, idx):
        example = self.dataset[idx]
        if isinstance(example, list) or isinstance(example, tuple):
            if len(example) == len(self.preprocessors):
                return [preprocessor(v) for v, preprocessor in zip(example, self.preprocessors)]
            elif len(self.preprocessors) == 1:
                preprocessor = self.preprocessors[0]
                return [preprocessor(v) for v in example]
            else:
                raise Exception(f'# of preprocessors must be 1 or equal to the size of example list/tuple')
        else:
            if len(self.preprocessors) == 1:
                preprocessor = self.preprocessors[0]
                return preprocessor(example)
            raise Exception(f'# of preprocessors must be equal to 1')

    def __len__(self):
        return len(self.dataset)
