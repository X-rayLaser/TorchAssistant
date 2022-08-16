import importlib

import torch

from scaffolding.data import DataBlueprint
from scaffolding.exceptions import ClassImportError, FunctionImportError, EntityImportError


class Serializable:
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


def instantiate_class(dotted_path, *args, **kwargs):
    if '.' not in dotted_path:
        raise ClassImportError(f'Invalid import path: "{dotted_path}"')

    module_path, class_name = split_import_path(dotted_path)
    error_msg = f'Failed to import and instantiate a class "{class_name}" from "{module_path}": '
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)
    except Exception as e:
        raise ClassImportError(error_msg + str(e))


def import_function(dotted_path):
    try:
        return import_entity(dotted_path)
    except EntityImportError as e:
        raise FunctionImportError(str(e))


def import_entity(dotted_path):
    if '.' not in dotted_path:
        raise EntityImportError(f'Invalid import path: "{dotted_path}"')

    module_path, name = split_import_path(dotted_path)

    error_msg = f'Failed to import an entity "{name}" from "{module_path}": '

    try:
        module = importlib.import_module(module_path)
        return getattr(module, name)
    except Exception as e:
        raise EntityImportError(error_msg + str(e))


def split_import_path(dotted_path):
    idx = dotted_path.rindex('.')
    module_path = dotted_path[:idx]
    name = dotted_path[idx + 1:]
    return module_path, name


class GradientClipper:
    def __init__(self, model, clip_value=None, clip_norm=None):
        self.model = model
        self.clip_value = clip_value
        self.clip_norm = clip_norm

    def __call__(self):
        if self.clip_value:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)

        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)


class HookInstaller:
    def __init__(self, model, hook_factory_fn):
        self.model = model
        self.hook_factory_fn = hook_factory_fn

    def __call__(self):
        for name, param in self.model.named_parameters():
            hook = self.hook_factory_fn(name)
            self.register_hook(param, hook)

    def register_hook(self, param, hook):
        raise NotImplementedError


class BackwardHookInstaller(HookInstaller):
    def register_hook(self, param, hook):
        param.register_hook(hook)


class OptimizerWithLearningRateScheduler:
    def __init__(self, optimizer, lr_scheduler):
        self.optimizer = optimizer
        self.scheduler = lr_scheduler

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
        self.scheduler.step()


def get_dataset(session, dataset_name):
    if '.' in dataset_name:
        splitter_name, slice_name = dataset_name.split('.')
        splitter = session.splits[splitter_name]
        split = splitter.split(session.datasets[splitter.dataset_name])
        dataset = getattr(split, slice_name)
    else:
        dataset = session.datasets[dataset_name]
    return dataset


class Debugger:
    def __init__(self, pipeline):
        self.graph = pipeline.graph
        self.postprocessor = pipeline.postprocessor
        self.output_device = pipeline.output_device
        self.pipeline = pipeline

    def __call__(self, log_entries):
        entry = log_entries[0]
        interval = self.pipeline.interval

        if entry.iteration % interval == interval - 1:
            with torch.no_grad():
                self.debug()

    def debug(self):
        it = iter(DataBlueprint(self.pipeline.input_loaders))

        for _ in range(self.pipeline.num_iterations):

            graph_inputs = next(it)
            results = self.graph(graph_inputs)
            all_results = {k: v for batches in results.values() for k, v in batches.items()}

            predictions = {k: all_results[k] for k in self.pipeline.output_keys}

            if self.postprocessor:
                predictions = self.postprocessor(predictions)

            self.output_device(predictions)


class InvalidImportPathError(Exception):
    pass
