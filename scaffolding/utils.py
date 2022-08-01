import importlib
from itertools import zip_longest
from collections import UserDict, UserList

from scaffolding.exceptions import ClassImportError, FunctionImportError, EntityImportError


class Serializable:
    def state_dict(self):
        return {}

    def load(self, state_dict):
        pass


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
        #node.net.instance.train()
        node.net.train()


def switch_to_evaluation_mode(prediction_pipeline):
    for node in prediction_pipeline:
        #node.net.instance.eval()
        node.net.eval()


def change_model_device(train_pipeline, device):
    for model in train_pipeline:
        #model.net.instance.to(device)
        model.net.to(device)


def change_batch_device(batch, device):
    inputs_dict = batch["inputs"]

    for k, mapping in inputs_dict.items():
        for tensor_name, value in mapping.items():
            mapping[tensor_name] = value.to(device)

    targets_dict = batch.get("targets")
    if targets_dict:
        for tensor_name, value in targets_dict.items():
            targets_dict[tensor_name] = value.to(device)


class AdaptedCollator:
    def __init__(self, collator, batch_adapter):
        self.collator = collator
        self.adapter = batch_adapter

    def __call__(self, *args):
        batch = self.collator(*args)
        return self.adapter.adapt(*batch)


class BaseDataset:
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_preprocessors(self):
        raise NotImplementedError


class WrappedDataset(BaseDataset):
    def __init__(self, dataset, preprocessors):
        self.dataset = dataset
        self.preprocessors = preprocessors

    def __getitem__(self, idx):
        example = self.dataset[idx]
        if not (isinstance(example, list) or isinstance(example, tuple)):
            example = [example]

        # todo: top if statement seems to be redundant
        if isinstance(example, list) or isinstance(example, tuple):
            if len(example) > len(self.preprocessors):
                # when number of inputs > number of preprocessors, leave redundant ones as is
                pairs = zip_longest(example, self.preprocessors)
                return [p(v) if p else v for v, p in pairs]
            else:
                # when number of inputs <= number of preprocessors, ignore redundant preprocessors
                return [preprocessor(v) for v, preprocessor in zip(example, self.preprocessors)]

    def __len__(self):
        return len(self.dataset)

    def get_preprocessors(self):
        wrapped_preprocessors = []

        if self.dataset.get_preprocessors() and self.preprocessors:
            for old_one, new_one in zip(self.dataset.get_preprocessors(), self.preprocessors):
                wrapped_preprocessors.append(new_one.wrap_preprocessor(old_one))
        else:
            wrapped_preprocessors = self.dataset.get_preprocessors() or self.preprocessors
        return wrapped_preprocessors


class MergedDataset(BaseDataset):
    class SemiInterval:
        def __init__(self, a, b):
            self.a = a
            self.b = b

        def __contains__(self, n):
            return self.a <= n < self.b

    def __init__(self, *datasets):
        self.datasets = datasets
        sizes = [len(ds) for ds in self.datasets]

        s = 0
        self.intervals = []
        for size in sizes:
            self.intervals.append(self.SemiInterval(s, s + size))
            s += size

    def __getitem__(self, idx):
        datasets_with_intervals = zip(self.intervals, self.datasets)
        for ivl, dataset in datasets_with_intervals:
            if idx in ivl:
                offset = idx - ivl.a
                return dataset[offset]

        raise IndexError('dataset index out of range')

    def __len__(self):
        return sum([len(ds) for ds in self.datasets])

    def get_preprocessors(self):
        return self.datasets[0].get_preprocessors()


def override_spec(old_spec: dict, new_spec: dict) -> dict:
    new_spec = parse_object(new_spec)
    d = override_dict(old_spec, new_spec)
    return remove_meta_data(d)


def remove_meta_data(obj):
    """Recursively traverse the data structure, convert every instance of MetaDict to dict,
    convert every instance of MetaList to list"""
    if isinstance(obj, dict) or isinstance(obj, MetaDict):
        return {k: remove_meta_data(v) for k, v in obj.items()}

    if isinstance(obj, list) or isinstance(obj, MetaList):
        return [remove_meta_data(v) for v in obj]

    return obj


class MetaDict(UserDict):
    pass


class MetaList(UserList):
    pass


def parse_object(obj):
    """Recursively parse deep nested dict or list structure into override specification.

    Resulting object will contain original entries + some metadata at every level
    of depth/nesting (this will be used to control overriding behavior by override_dict
    and override_list functions).

    dict -> MetaDict
    dict with metadata -> MetaDict or MetaList
    list -> MetaList
    values of other types are kept unchanged
    """
    if isinstance(obj, list):
        alist = MetaList([parse_object(item) for item in obj])
        alist.replace_strategy = "replace"
        return alist
    elif isinstance(obj, dict) and "options" not in obj:
        adict = MetaDict({k: parse_object(v) for k, v in obj.items()})
        adict.replace_strategy = "override"
        return adict
    elif isinstance(obj, dict):
        strategy = obj.get("replace_strategy", "replace")
        options = obj["options"]

        items = parse_object(options)
        items.replace_strategy = strategy
        if isinstance(options, list):
            items.override_key = obj["override_key"]
        return items
    else:
        return obj


def override_dict(old_spec: dict, new_spec) -> dict:
    if new_spec.replace_strategy == "replace":
        return dict(new_spec)

    old_spec = dict(old_spec)
    new_spec = dict(new_spec)

    for k, v in new_spec.items():
        old_value = old_spec.get(k)
        if not old_value:
            old_spec[k] = v
        else:
            both_dicts = isinstance(old_value, dict) and isinstance(v, MetaDict)
            both_lists = isinstance(old_value, list) and isinstance(v, MetaList)

            if both_dicts:
                old_spec[k] = override_dict(old_spec[k], v)
            elif both_lists:
                old_spec[k] = override_list(old_value, v)
            else:
                old_spec[k] = v

    return old_spec


def override_list(old_list: list, new_list) -> list:
    if new_list.replace_strategy == 'replace':
        return list(new_list)

    key_set = new_list.override_key
    old_list = list(old_list)
    new_list = list(new_list)

    for new_item in new_list:
        if isinstance(new_item, dict) or isinstance(new_item, MetaDict):
            idx = find_first_dict_index(old_list, new_item, key_set)
            if idx is not None:
                old_list[idx] = override_dict(old_list[idx], new_item)
            else:
                old_list.append(new_item)
        else:
            # there is no way to override, replace the whole list with a new one
            return new_list

    return old_list


def find_first_dict_index(items: list, d: dict, key_set):
    item_indices = [i for i, item in enumerate(items)
                    if dicts_equal(item, d, key_set)]

    if len(item_indices) > 1:
        raise InvalidNumberOfMatchesError()
    return item_indices[0] if len(item_indices) > 0 else None


class InvalidNumberOfMatchesError(Exception):
    pass


def dicts_equal(d1: dict, d2: dict, key_set):
    if not isinstance(d1, dict):
        raise NotDictError(f'Expects d1 to be dictionary. Got {type(d1)}')

    if not (isinstance(d2, dict) or isinstance(d2, MetaDict)):
        raise NotDictError(f'Expects d2 to be dictionary. Got {type(d2)}')

    try:
        for key in key_set:
            if d1[key] != d2[key]:
                return False
    except KeyError:
        return False

    return True


class NotDictError(Exception):
    pass
