from collections import namedtuple
import inspect
from random import shuffle

import torch
import torchmetrics

from scaffolding.utils import instantiate_class, import_function, import_entity, MultiSplitter
from scaffolding.metrics import metric_functions, Metric


class Learner:
    def __init__(self, preprocessor_name, dataset_name):
        self.preprocessor_name = preprocessor_name
        self.dataset_name = dataset_name


class SpecParser:
    def __init__(self, factory=None):
        self.factory = factory or instantiate_class

    def parse(self, spec_dict):
        if not isinstance(spec_dict, dict):
            raise BadSpecificationError(f'SpecParser: spec_dict should be a dictionary. Got {type(spec_dict)}.')

        factory_name = spec_dict.get("class") or spec_dict.get("factory_fn")
        if not factory_name:
            raise BadSpecificationError('SpecParser: missing "class" key')

        # splitter_spec = spec.get("splitter")
        args = spec_dict.get("args", [])
        kwargs = spec_dict.get("kwargs", {})

        if "class" in spec_dict:
            if not isinstance(factory_name, str):
                raise BadSpecificationError(f'SpecParser: class must be a string. Got {type(factory_name)}')

            return SingletonFactory(factory_name, args, kwargs, instantiate_fn=self.factory)
        else:
            if not isinstance(factory_name, str):
                raise BadSpecificationError(f'SpecParser: "factory_fn" must be a string. Got {type(factory_name)}')

            return BuildFromFunctionFactory(factory_name, args, kwargs)


class SingletonFactory:
    def __init__(self, class_name, args, kwargs, instantiate_fn=None):
        self.class_name = class_name
        self.args = args
        self.kwargs = kwargs

        self.instance = None
        self.instantiate_fn = instantiate_fn or instantiate_class

    def get_instance(self, session):
        if not self.instance:
            self.instance = self.instantiate_fn(self.class_name, *self.args, **self.kwargs)

        return self.instance


class BuildFromFunctionFactory:
    def __init__(self, function_name, args, kwargs):
        self.function_name = function_name
        self.args = args
        self.kwargs = kwargs

        self.instance = None

    def get_instance(self, session):
        if not self.instance:
            factory_fn = import_function(self.function_name)
            self.instance = factory_fn(session, *self.args, **self.kwargs)

        return self.instance


class BadSpecificationError(Exception):
    pass


class Loader:
    def load(self, session, spec, object_name=None):
        parser = SpecParser()
        factory = parser.parse(spec)
        return factory.get_instance(session)


class SplitLoader(Loader):
    def load(self, session, spec, object_name=None):
        ds_name = spec["dataset_name"]
        ratio = spec["ratio"]
        ds = session.datasets[ds_name]

        shuffled_indices = list(range(len(ds)))
        shuffle(shuffled_indices)
        splitter = MultiSplitter(ratio, shuffled_indices)
        splitter.split(ds)
        return splitter


class LearnerLoader(Loader):
    def __init__(self, fit_preprocessors):
        self.fit_preprocessors = fit_preprocessors

    def load(self, session, spec, object_name=None):
        proc_name = spec["preprocessor_name"]
        ds_name = spec["dataset_name"]

        if self.fit_preprocessors:
            if '.' in ds_name:
                split_name, split_part = ds_name.split('.')
                dataset = getattr(session.splits[split_name], split_part)
            else:
                dataset = session.datasets[ds_name]

            preprocessor = session.preprocessors[proc_name]
            preprocessor.fit(dataset)

        return Learner(proc_name, ds_name)


class OptimizerLoader(Loader):
    def load(self, session, spec_dict, object_name=None):
        # todo: support setting the same optimizer for more than 1 model
        class_name = spec_dict.get("class")
        args = spec_dict.get("args", [])
        kwargs = spec_dict.get("kwargs", {})

        cls = getattr(torch.optim, class_name)
        model = session.models[object_name]
        return cls(model.parameters(), *args, **kwargs)


class LossLoader(Loader):
    def load(self, session, spec, object_name=None):
        from torch import nn
        loss_class_name = spec["class"]

        if "transform" in spec:
            transform_fn = import_function(spec["transform"])
            if inspect.isclass(transform_fn):
                transform_fn = transform_fn(session)
        else:
            transform_fn = lambda *fn_args: fn_args

        criterion_class = getattr(nn, loss_class_name)
        args = spec.get("args", [])
        kwargs = spec.get("kwargs", {})

        # todo: set device later
        return Metric('loss', criterion_class(*args, **kwargs), spec["inputs"], transform_fn, 'cpu')


class MetricLoader(Loader):
    def load(self, session, spec, object_name=None):
        if "transform" in spec:
            transform_fn = import_entity(spec["transform"])
            if inspect.isclass(transform_fn):
                transform_fn = transform_fn(session)
        else:
            def transform_fn(*fn_args):
                return fn_args

        metric_class = spec["class"]
        if hasattr(torchmetrics, metric_class):
            metric = instantiate_class(f'torchmetrics.{metric_class}')
        else:
            metric = metric_functions[metric_class]

        # todo: set device later
        return Metric(object_name, metric, spec["inputs"], transform_fn, device='cpu')


class PipelineLoader(Loader):
    def load(self, session, pipeline_spec, object_name=None):
        dataset = session.datasets[pipeline_spec.ds_name]
        preprocessors = [session.preprocessors[name]
                         for name in pipeline_spec.preprocessor_names]

        collator = session.collators[pipeline_spec.collator_name]
        batch_size = pipeline_spec.batch_size
        batch_adapter = session.batch_adapters[pipeline_spec.batch_adapter_name]

        neural_graph = self.parse_neural_graph(session, pipeline_spec['neural_graph'])
        loss_fn = session.losses[pipeline_spec.loss_name]
        metric_fns = [session.metrics[name] for name in pipeline_spec.metric_names]

        return Pipeline(
            dataset=dataset,
            preprocessors=preprocessors,
            collator=collator,
            batch_size=batch_size,
            batch_adapter=batch_adapter,
            neural_graph=neural_graph,
            loss_fn=loss_fn,
            metric_fns=metric_fns
        )

    def parse_neural_graph(self, session, graph_spec):
        nodes = []
        for node_dict in graph_spec:
            model_name = node_dict['model_name']
            inputs = node_dict['inputs']
            outputs = node_dict['outputs']
            optimizer_name = node_dict['optimizer_name']

            NodeSpec = namedtuple('NodeSpec', ['model_name', 'inputs', 'outputs', 'optimizer_name'])
            node_spec = NodeSpec(model_name, inputs, outputs, optimizer_name)
            node = self.node_from_spec(session, node_spec)
            nodes.append(node)
        return nodes

    def node_from_spec(self, session, node_spec):
        model = session.models[node_spec.model_name]
        optimizer = session.optimizers[node_spec.optimizer_name]
        return Node(name=node_spec.model_name,
                    serializable_model=model, serializable_optimizer=optimizer,
                    inputs=node_spec.inputs, outputs=node_spec.outputs)


class Pipeline:
    def __init__(self, *,
                 dataset,
                 preprocessors,
                 collator,
                 batch_size,
                 batch_adapter,
                 neural_graph,
                 loss_fn,
                 metric_fns):
        self.dataset = dataset
        self.preprocessors = preprocessors
        self.collator = collator
        self.batch_size = batch_size
        self.batch_adapter = batch_adapter
        self.neural_graph = neural_graph
        self.loss_fn = loss_fn
        self.metric_fns = metric_fns


class Node:
    def __init__(self, name, serializable_model, serializable_optimizer, inputs, outputs):
        self.name = name
        self.net = serializable_model
        self.optimizer = serializable_optimizer
        self.inputs = inputs
        self.outputs = outputs

    def get_dependencies(self, batch_inputs, prev_outputs):
        lookup_table = batch_inputs[self.name].copy()
        lookup_table.update(prev_outputs)
        return [lookup_table[var_name] for var_name in self.inputs]

    def predict(self, *args, inference_mode=False):
        # todo: consider to change args device here (need to store device as attribute)
        if inference_mode:
            return self.net.run_inference(*args)
        else:
            return self.net(*args)

    def __call__(self, batch_inputs, prev_outputs, inference_mode=False):
        args = self.get_dependencies(batch_inputs, prev_outputs)
        return self.predict(*args, inference_mode=inference_mode)


# todo: introduce Action and Definition abstractions and new syntax (list of actions and definitions)
# todo: