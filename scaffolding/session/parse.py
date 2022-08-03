from collections import namedtuple
import inspect

import torch
from torch import nn
import torchmetrics

from scaffolding.utils import instantiate_class, import_function, import_entity, GradientClipper, BackwardHookInstaller, \
    OptimizerWithLearningRateScheduler, get_dataset
from .data_classes import DebugPipeline, Stage, InputLoader, TrainingPipeline
from ..data import WrappedDataset, MergedDataset, MultiSplitter, LoaderFactory
from scaffolding.preprocessors import NullProcessor
from scaffolding.metrics import metric_functions, Metric
from scaffolding.output_devices import Printer
from scaffolding.output_adapters import IdentityAdapter
from .registry import register
from ..processing_graph import NeuralBatchProcessor, Node, BatchProcessingGraph


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

    def __call__(self, session, spec, object_name=None):
        return self.load(session, spec, object_name)


@register("datasets")
def load_dataset(session, spec, object_name=None):
    if 'class' in spec or 'factory_fn' in spec:
        dataset = Loader().load(session, spec, object_name)
    elif 'link' in spec:
        # in that case, we are building a composite dataset
        referred_ds = spec["link"]
        dataset = get_dataset(session, referred_ds)
    elif 'merge' in spec:
        merge_names = spec['merge']
        merge_datasets = [get_dataset(session, name) for name in merge_names]
        dataset = MergedDataset(*merge_datasets)
    else:
        raise BadSpecificationError(f'Either one of these keys must be present: "class", "link" or "merge"')

    preprocessors = spec.get("preprocessors", [])
    preprocessors = [session.preprocessors.get(name, NullProcessor()) for name in preprocessors]
    return WrappedDataset(dataset, preprocessors)


@register("splits")
def load_data_split(session, spec, object_name=None):
    dataset_name = spec["dataset_name"]
    ratio = spec["ratio"]
    splitter = MultiSplitter(dataset_name, ratio)
    return splitter


@register("preprocessors")
def load_preprocessor(session, spec, object_name=None):
    instance = Loader().load(session, spec, object_name)
    return instance


@register("data_loaders")
def load_data_loader(session, spec, object_name=None):
    dataset = session.datasets[spec["dataset"]]
    collator = session.collators[spec["collator"]]

    kwargs = dict(shuffle=True, num_workers=2)
    kwargs.update(spec.get("kwargs", {}))

    return torch.utils.data.DataLoader(dataset, collate_fn=collator, **kwargs)


@register("loader_factories")
def load_data_factory(session, spec, object_name=None):
    dataset = session.datasets[spec["dataset"]]
    collator = session.collators[spec["collator"]]

    kwargs = dict(shuffle=True, num_workers=2)
    kwargs.update(spec.get("kwargs", {}))

    return LoaderFactory(dataset, collator, **kwargs)


class BatchProcessorLoader(Loader):
    def load(self, session, spec, object_name=None):
        if "class" in spec or "factory_fn" in spec:
            return Loader().load(session, spec, object_name)

        input_adapter = Loader().load(session, spec["input_adapter"], object_name)

        if "output_adapter" in spec:
            output_adapter = Loader().load(session, spec["output_adapter"], object_name)
        else:
            output_adapter = IdentityAdapter()

        graph_spec = spec["neural_graph"]
        neural_graph = self.parse_neural_graph(session, graph_spec)
        device = torch.device(spec.get("device", "cpu"))
        return NeuralBatchProcessor(neural_graph, input_adapter, output_adapter, device)

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
        optimizer = session.optimizers.get(node_spec.optimizer_name)
        return Node(name=node_spec.model_name,
                    model=model, optimizer=optimizer,
                    inputs=node_spec.inputs, outputs=node_spec.outputs)


@register("batch_processors")
def load_batch_processor(session, spec, object_name=None):
    return BatchProcessorLoader().load(session, spec, object_name)


@register("batch_graphs")
def load_batch_processing_graph(session, spec, object_name=None):
    nodes = {}
    for node_name in spec["nodes"]:
        nodes[node_name] = session.batch_processors[node_name]

    batch_input_names = spec["input_aliases"]
    graph = BatchProcessingGraph(batch_input_names, **nodes)
    for destination, sources in spec["ingoing_edges"].items():
        for src in sources:
            graph.make_edge(src, destination)
    return graph


class PipelineLoader(Loader):
    def load(self, session, spec, object_name=None):
        graph = session.batch_graphs[spec["graph"]]
        input_loaders = []
        for d in spec["input_factories"]:
            kwargs = dict(d)
            kwargs["loader_factory"] = session.loader_factories[kwargs["loader_factory"]]
            input_loaders.append(InputLoader(**kwargs))

        metric_fns = self.parse_metrics(session, spec)

        loss_fns = {}
        for loss_spec in spec.get("losses", []):
            loss_name = loss_spec["loss_name"]
            node_name = loss_spec["node_name"]

            loss_fn = (node_name, session.losses[loss_name])
            loss_fns[loss_name] = loss_fn

            if 'loss_display_name' in loss_spec:
                loss_display_name = loss_spec.get('loss_display_name', loss_name)
                renamed_loss_fn = loss_fn[1].rename_and_clone(loss_display_name)
                metric_fns[loss_display_name] = (node_name, renamed_loss_fn)

        return TrainingPipeline(graph, input_loaders, loss_fns, metric_fns)

    def parse_metrics(self, session, pipeline_spec):
        metrics = {}
        for spec in pipeline_spec.get("metrics", []):
            name = spec["metric_name"]
            display_name = spec["display_name"]
            node_name = spec["node_name"]
            metric = session.metrics[name].rename_and_clone(display_name)
            metrics[display_name] = (node_name, metric)

        return metrics


@register("gradient_clippers")
def load_clipper(session, spec, object_name=None):
    model_name = spec["model"]
    clip_value = spec.get("clip_value")
    clip_norm = spec.get("clip_norm")
    model = session.models[model_name]
    return GradientClipper(model, clip_value, clip_norm)


class BackwardHookLoader(Loader):
    hook_installer = BackwardHookInstaller

    def load(self, session, spec, object_name=None):
        model_name = spec["model"]
        function_name = spec["factory_fn"]

        model = session.models[model_name]
        create_hook = import_function(function_name)
        return self.hook_installer(model, create_hook)


@register("backward_hooks")
def load_backward_hook(session, spec, object_name=None):
    return BackwardHookLoader().load(session, spec, object_name)


@register("optimizers")
def load_optimizer(session, spec_dict, object_name=None):
    # todo: support setting the same optimizer for more than 1 model
    class_name = spec_dict.get("class")
    args = spec_dict.get("args", [])
    kwargs = spec_dict.get("kwargs", {})
    model_name = spec_dict["model"]

    cls = getattr(torch.optim, class_name)
    model = session.models[model_name]
    optimizer = cls(model.parameters(), *args, **kwargs)

    scheduler_spec = spec_dict.get('lr_scheduler')

    if scheduler_spec:
        scheduler = instantiate_class(scheduler_spec["class"], optimizer,
                                      **scheduler_spec.get("kwargs", {}))
        optimizer = OptimizerWithLearningRateScheduler(optimizer, scheduler)

    return optimizer


@register("losses")
def load_loss(session, spec, object_name=None):
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


@register("metrics")
def load_metric(session, spec, object_name=None):
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


class DebuggerLoader(Loader):
    def load(self, session, pipeline_spec, object_name=None):
        graph = session.batch_graphs[pipeline_spec["graph"]]
        input_loaders = []
        for d in pipeline_spec["input_factories"]:
            kwargs = dict(d)
            kwargs["loader_factory"] = session.loader_factory[kwargs["loader_factory"]]
            input_loaders.append(InputLoader(**kwargs))

        num_iterations = pipeline_spec["num_iterations"]
        interval = pipeline_spec["interval"]

        def default_postprocessor(x):
            return x

        postprocessor_spec = pipeline_spec.get("postprocessor")
        if postprocessor_spec:
            postprocessor = Loader().load(session, postprocessor_spec)
        else:
            postprocessor = default_postprocessor

        output_device_spec = pipeline_spec.get("output_device")
        if output_device_spec:
            output_device = Loader().load(session, output_device_spec)
        else:
            output_device = Printer()

        output_keys = pipeline_spec.get("output_keys", ["y_hat"])

        pipeline = DebugPipeline(
            graph=graph, input_loaders=input_loaders, num_iterations=num_iterations,
            interval=interval, output_keys=output_keys, postprocessor=postprocessor,
            output_device=output_device
        )
        from ..utils import Debugger
        return Debugger(pipeline)


class StageLoader(Loader):
    def load(self, session, spec, object_name=None):
        mode = spec.get("mode", "interleave")
        stop_condition_dict = spec.get("stop_condition")

        training_pipelines = spec["training_pipelines"]
        training_pipelines = [session.pipelines[name] for name in training_pipelines]

        validation_pipelines = spec.get("validation_pipelines", [])
        validation_pipelines = [session.pipelines[name] for name in validation_pipelines]

        debug_pipelines = spec.get("debug_pipelines", [])
        debug_pipelines = [session.debug_pipelines[name] for name in debug_pipelines]

        stop_condition = Loader().load(session, stop_condition_dict)
        return Stage(mode, training_pipelines, validation_pipelines, debug_pipelines, stop_condition)
