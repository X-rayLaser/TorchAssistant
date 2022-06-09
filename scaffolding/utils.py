import importlib

from torch.utils.data import Dataset

from scaffolding.exceptions import ClassImportError, FunctionImportError, EntityImportError


class Serializable:
    def state_dict(self):
        return {}

    def load(self, state_dict):
        pass


class DecoratedInstance:
    def __init__(self, instance, class_name, args, kwargs):
        self.class_name = class_name
        self.instance = instance
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.instance(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.instance, attr)

    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, *args, **kwargs):
        raise NotImplementedError


class GenericSerializableInstance(DecoratedInstance):
    def to_dict(self):
        try:
            state_dict = self.instance.state_dict()
        except (AttributeError, TypeError):
            state_dict = self.instance.__dict__

        return {
            'state_dict': state_dict,
            'class': self.class_name,
            'args': self.args,
            'kwargs': self.kwargs,
        }

    @classmethod
    def from_dict(cls, d):
        class_path = d['class']
        args = d['args']
        kwargs = d['kwargs']
        state_dict = d['state_dict']
        instance = instantiate_class(class_path, *args, **kwargs)

        if hasattr(instance, 'load'):
            instance.load(state_dict)
        else:
            instance.__dict__ = state_dict

        return cls(instance=instance, class_name=class_path, args=args, kwargs=kwargs)


class DataSplitter(Serializable):
    def __init__(self, train_fraction=0.8):
        self.dataset = None
        self.train_fraction = train_fraction

        self.num_train = None
        self.num_val = None

    def prepare(self, dataset):
        self.dataset = dataset
        self.num_train = int(self.train_fraction * len(dataset))
        self.num_val = len(dataset) - self.num_train

    @property
    def train_ds(self):
        return []

    @property
    def val_ds(self):
        return []

    def state_dict(self):
        return {}

    def load(self, state):
        pass


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


def import_entity(dotted_path):
    parts = dotted_path.split('.')
    module_path = '.'.join(parts[:-1])
    name = parts[-1]

    error_msg = f'Failed to import an entity "{name}" from "{module_path}"'

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        raise EntityImportError(error_msg)

    try:
        return getattr(module, name)
    except AttributeError:
        raise EntityImportError(error_msg)


def switch_to_train_mode(prediction_pipeline):
    for node in prediction_pipeline:
        node.net.instance.train()


def switch_to_evaluation_mode(prediction_pipeline):
    for node in prediction_pipeline:
        node.net.instance.eval()


def change_model_device(train_pipeline, device):
    for model in train_pipeline:
        model.net.instance.to(device)


def change_batch_device(batch, device):
    inputs_dict = batch["inputs"]

    for k, mapping in inputs_dict.items():
        for tensor_name, value in mapping.items():
            mapping[tensor_name] = value.to(device)

    targets_dict = batch.get("targets")
    if targets_dict:
        for tensor_name, value in targets_dict.items():
            targets_dict[tensor_name] = value.to(device)


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
        if not (isinstance(example, list) or isinstance(example, tuple)):
            example = [example]

        if isinstance(example, list) or isinstance(example, tuple):
            from itertools import zip_longest

            if len(example) > len(self.preprocessors):
                # when number of inputs > number of preprocessors, leave redundant ones as is
                pairs = zip_longest(example, self.preprocessors)
                return [p(v) if p else v for v, p in pairs]
            else:
                # when number of inputs <= number of preprocessors, ignore redundant preprocessors
                return [preprocessor(v) for v, preprocessor in zip(example, self.preprocessors)]

    def __len__(self):
        return len(self.dataset)
