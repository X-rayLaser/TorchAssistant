import inspect

import torch
from torch import nn
import torchmetrics

from torchassistant.utils import instantiate_class, import_function, import_entity, GradientClipper, \
    OptimizerWithLearningRateScheduler, get_dataset
from .loaders import Loader, BadSpecificationError, Loader, BatchProcessorLoader, BackwardHookLoader
from ...data import WrappedDataset, MergedDataset, MultiSplitter, LoaderFactory
from torchassistant.preprocessors import NullProcessor
from torchassistant.metrics import metric_functions, Metric
from ..registry import register
from ...processing_graph import BatchProcessingGraph


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
def load_data_factory(session, spec, object_name=None):
    dataset = session.datasets[spec["dataset"]]
    collator = session.collators[spec["collator"]]

    kwargs = dict(shuffle=True, num_workers=2)
    kwargs.update(spec.get("kwargs", {}))

    return LoaderFactory(dataset, collator, **kwargs)


@register("batch_processors")
def load_batch_processor(session, spec, object_name=None):
    return BatchProcessorLoader().load(session, spec, object_name)


@register("batch_graphs")
def load_batch_processing_graph(session, spec, object_name=None):
    nodes = {}
    for node_name in spec["nodes"]:
        nodes[node_name] = session.batch_processors[node_name]

    batch_input_names = spec["input_ports"]
    graph = BatchProcessingGraph(batch_input_names, **nodes)
    for destination, sources in spec["ingoing_edges"].items():
        for src in sources:
            graph.make_edge(src, destination)
    return graph


@register("gradient_clippers")
def load_clipper(session, spec, object_name=None):
    model_name = spec["model"]
    clip_value = spec.get("clip_value")
    clip_norm = spec.get("clip_norm")
    model = session.models[model_name]
    return GradientClipper(model, clip_value, clip_norm)


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

    args = spec.get("args", [])
    kwargs = spec.get("kwargs", {})

    if '.' in loss_class_name:
        instance = instantiate_class(loss_class_name, *args, **kwargs)
    else:
        criterion_class = getattr(nn, loss_class_name)
        instance = criterion_class(*args, **kwargs)

    return Metric('loss', instance, spec["inputs"], transform_fn)


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

    return Metric(object_name, metric, spec["inputs"], transform_fn)
