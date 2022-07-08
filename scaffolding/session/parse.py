from collections import namedtuple
import inspect

import torch
from torch import nn
import torchmetrics

from scaffolding.utils import instantiate_class, import_function, import_entity, MergedDataset, \
    WrappedDataset, switch_to_train_mode, switch_to_evaluation_mode, change_model_device
from scaffolding.preprocessors import NullProcessor
from scaffolding.data_splitters import MultiSplitter
from scaffolding.metrics import metric_functions, Metric
from scaffolding.output_devices import Printer
from scaffolding.output_adapters import IdentityAdapter
from .registry import register


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
        return Loader().load(session, spec, object_name)

    # in that case, we are building a composite dataset
    if 'link' in spec:
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


class NeuralBatchProcessor:
    def __init__(self, neural_graph, input_adapter, output_adapter, device):
        self.neural_graph = neural_graph
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.device = device

    def __call__(self, batch, inference_mode=False):
        inputs = self.input_adapter(batch)
        change_model_device(self.neural_graph, self.device)
        self.inputs_to(inputs)

        all_outputs = {}
        for node in self.neural_graph:
            outputs = node(inputs, all_outputs, inference_mode)
            all_outputs.update(
                dict(zip(node.outputs, outputs))
            )

        result_dict = dict(batch)
        result_dict.update(all_outputs)
        res = self.output_adapter(result_dict)
        return res

    def inputs_to(self, inputs):
        for k, mapping in inputs.items():
            for tensor_name, value in mapping.items():
                if hasattr(value, 'device') and value.device != self.device:
                    mapping[tensor_name] = value.to(self.device)

    def prepare(self):
        for model in self.neural_graph:
            if model.optimizer:
                model.optimizer.zero_grad()

    def update(self):
        for node in self.neural_graph:
            if node.optimizer:
                node.optimizer.step()

    def train_mode(self):
        switch_to_train_mode(self.neural_graph)

    def eval_mode(self):
        switch_to_evaluation_mode(self.neural_graph)


@register("batch_pipelines")
def load_batch_pipeline(session, spec, object_name=None):
    if "mix" in spec:
        mixed_pipelines = [session.batch_pipelines[name] for name in spec["mix"]]
        batch_generator = BatchPipelineMixer(mixed_pipelines)
    else:
        batch_generator = session.data_loaders[spec["data_loader"]]
    object_names = spec.get("batch_processors", [])
    batch_processors = [session.batch_processors[name] for name in object_names]
    variable_names = spec.get("variable_names", [])
    return BatchPipeline(batch_generator, batch_processors, variable_names)


class BatchPipeline:
    def __init__(self, batch_generator, batch_processors, variable_names):
        self.batch_generator = batch_generator
        self.batch_processors = batch_processors
        self.variable_names = variable_names

    def __iter__(self):
        for batch in self.batch_generator:
            batch = self.batch_to_dict(batch)
            processors = list(reversed(self.batch_processors))

            while processors:
                batch_processor = processors.pop()
                batch_processor.prepare()
                batch = batch_processor(batch)

            yield batch

    def __len__(self):
        return len(self.batch_generator)

    def batch_to_dict(self, batch):
        if isinstance(batch, dict):
            return batch
        return dict(zip(self.variable_names, batch))

    def update(self):
        for processor in self.batch_processors:
            processor.update()

    def train_mode(self):
        for processor in self.batch_processors:
            processor.train_mode()

    def eval_mode(self):
        for processor in self.batch_processors:
            processor.eval_mode()


class BatchPipelineMixer:
    def __init__(self, batch_pipelines):
        self.batch_pipelines = batch_pipelines

    def __iter__(self):
        for batches in zip(*self.batch_pipelines):
            batch = self.mix(batches)
            yield batch

    def __len__(self):
        return min(len(pipeline) for pipeline in self.batch_pipelines)

    def mix(self, batches):
        # todo: randomly shuffle rows
        any_batch = batches[0]
        result = {}
        for k in any_batch.keys():
            tensors = [batch[k].to(torch.device("cpu")) for batch in batches]
            concatenation = torch.cat(tensors)
            result[k] = concatenation
        return result

    def train_mode(self):
        for pipeline in self.batch_pipelines:
            pipeline.train_mode()

    def eval_mode(self):
        for pipeline in self.batch_pipelines:
            pipeline.eval_mode()


class ActualTrainingPipelineLoader(Loader):
    def load(self, session, spec, object_name=None):
        batch_pipeline = session.batch_pipelines[spec["batch_pipeline"]]

        loss_name = spec["loss_name"]
        loss_fn = session.losses[loss_name]

        metric_fns = self.parse_metrics(session, spec)

        if 'loss_display_name' in spec:
            loss_display_name = spec.get('loss_display_name', loss_name)
            metric_fns[loss_display_name] = loss_fn.rename_and_clone(loss_display_name)

        TrainingPipeline = namedtuple('TrainingPipeline', ["batch_pipeline", "loss_fn", "metric_fns"])
        return TrainingPipeline(batch_pipeline, loss_fn, metric_fns)

    def parse_metrics(self, session, pipeline_spec):
        metric_names = pipeline_spec.get("metric_names", [])
        display_names = pipeline_spec.get("metric_display_names", metric_names)
        names = zip(metric_names, display_names)
        return {display_name: session.metrics[name].rename_and_clone(display_name)
                for name, display_name in names}


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


@register("gradient_clippers")
def load_clipper(session, spec, object_name=None):
    model_name = spec["model"]
    clip_value = spec.get("clip_value")
    clip_norm = spec.get("clip_norm")
    model = session.models[model_name]
    return GradientClipper(model, clip_value, clip_norm)


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


class DebugPipelineLoader(Loader):
    def load(self, session, pipeline_spec, object_name=None):
        batch_pipeline = session.batch_pipelines[pipeline_spec["batch_pipeline"]]
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

        return DebugPipeline(
            batch_pipeline=batch_pipeline, num_iterations=num_iterations, interval=interval,
            output_keys=output_keys, postprocessor=postprocessor, output_device=output_device
        )


class StageLoader(Loader):
    def load(self, session, spec, object_name=None):
        mode = spec.get("mode", "interleave")
        stop_condition_dict = spec.get("stop_condition")

        training_pipelines = spec["training_pipelines"]
        training_pipelines = [session.pipelines[name] for name in training_pipelines]

        validation_pipelines = spec["validation_pipelines"]
        validation_pipelines = [session.pipelines[name] for name in validation_pipelines]

        debug_pipelines = spec["debug_pipelines"]
        debug_pipelines = [session.debug_pipelines[name] for name in debug_pipelines]

        stop_condition = Loader().load(session, stop_condition_dict)
        return Stage(mode, training_pipelines, validation_pipelines, debug_pipelines, stop_condition)


class DebugPipeline:
    def __init__(self, *, batch_pipeline, num_iterations, interval,
                 output_keys, postprocessor, output_device):
        self.batch_pipeline = batch_pipeline
        self.num_iterations = num_iterations
        self.interval = interval
        self.output_keys = output_keys
        self.postprocessor = postprocessor
        self.output_device = output_device


class Node:
    def __init__(self, name, model, optimizer, inputs, outputs):
        self.name = name
        self.net = model
        self.optimizer = optimizer
        self.inputs = inputs
        self.outputs = outputs

    def get_dependencies(self, batch_inputs, prev_outputs):
        # todo: double check this line
        lookup_table = batch_inputs.get(self.name, {}).copy()
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


class Stage:
    def __init__(self, mode, training_pipelines,
                 validation_pipelines, debug_pipelines, stop_condition):
        self.mode = mode
        self.training_pipelines = training_pipelines
        self.validation_pipelines = validation_pipelines
        self.debug_pipelines = debug_pipelines
        self.stop_condition = stop_condition


def get_dataset(session, dataset_name):
    if '.' in dataset_name:
        splitter_name, slice_name = dataset_name.split('.')
        splitter = session.splits[splitter_name]
        split = splitter.split(session.datasets[splitter.dataset_name])
        dataset = getattr(split, slice_name)
    else:
        dataset = session.datasets[dataset_name]
    return dataset
