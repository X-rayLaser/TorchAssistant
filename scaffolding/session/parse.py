from collections import namedtuple
import inspect

import torch
import torchmetrics

from scaffolding.utils import instantiate_class, import_function, import_entity, MergedDataset, WrappedDataset
from scaffolding.preprocessors import NullProcessor
from scaffolding.data_splitters import MultiSplitter
from scaffolding.metrics import metric_functions, Metric
from scaffolding.output_devices import Printer


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


class DatasetLoader(Loader):
    def load(self, session, spec, object_name=None):
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
        preprocessors = [session.preprocessors.get(name) or session.neural_maps.get(name, NullProcessor())
                         for name in preprocessors]
        return WrappedDataset(dataset, preprocessors)


class SplitLoader(Loader):
    def load(self, session, spec, object_name=None):
        dataset_name = spec["dataset_name"]
        ratio = spec["ratio"]
        splitter = MultiSplitter(dataset_name, ratio)
        return splitter


class PreProcessorLoader(Loader):
    def load(self, session, spec, object_name=None):
        instance = super().load(session, spec, object_name)
        return instance


class NeuralMap:
    def __init__(self, model):
        self.model = model

    def process(self, x):
        with torch.no_grad():
            res = self.model(x.unsqueeze(0))[0]
            res = res.squeeze(0)

        return res

    def __call__(self, x):
        return self.process(x)


class NeuralMapLoader(Loader):
    def load(self, session, spec, object_name=None):
        model_name = spec["mapper_model"]
        model = session.models[model_name]
        instance = NeuralMap(model)
        return instance


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


class GradientClipperLoader(Loader):
    def load(self, session, spec, object_name=None):
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


class OptimizerLoader(Loader):
    def load(self, session, spec_dict, object_name=None):
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


class CommonPipelineeLoader(Loader):
    def load(self, session, pipeline_spec, object_name=None):
        pipeline = BasePipeline()

        pipeline.train_dataset = get_dataset(session, pipeline_spec["train_dataset"])

        if "val_dataset" in pipeline_spec:
            val_dataset = get_dataset(session, pipeline_spec["val_dataset"])
        else:
            val_dataset = pipeline.train_dataset
        pipeline.val_dataset = val_dataset

        pipeline.preprocessors = [session.preprocessors[name]
                                  for name in pipeline_spec["preprocessor_names"]]

        pipeline.collator = session.collators[pipeline_spec["collator_name"]]
        pipeline.batch_size = pipeline_spec["batch_size"]
        if "batch_adapter" in pipeline_spec:
            pipeline.batch_adapter = Loader().load(session, pipeline_spec["batch_adapter"])
        else:
            pipeline.batch_adapter = session.batch_adapters[pipeline_spec["batch_adapter_name"]]

        pipeline.neural_graph = self.parse_neural_graph(session, pipeline_spec['neural_graph'])

        pipeline.device = pipeline_spec.get("device", "cpu")

        return pipeline

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
                    serializable_model=model, serializable_optimizer=optimizer,
                    inputs=node_spec.inputs, outputs=node_spec.outputs)


class PipelineLoader(CommonPipelineeLoader):
    def load(self, session, pipeline_spec, object_name=None):
        base_pipeline = super().load(session, pipeline_spec, object_name)

        loss_name = pipeline_spec["loss_name"]
        loss_fn = session.losses[loss_name]

        metric_fns = self.parse_metrics(session, pipeline_spec)

        if 'loss_display_name' in pipeline_spec:
            loss_display_name = pipeline_spec.get('loss_display_name', loss_name)
            metric_fns[loss_display_name] = loss_fn.rename_and_clone(loss_display_name)

        return Pipeline.from_base_pipeline(
            base_pipeline, loss_fn=loss_fn, metric_fns=metric_fns
        )

    def parse_metrics(self, session, pipeline_spec):
        metric_names = pipeline_spec.get("metric_names", [])
        display_names = pipeline_spec.get("metric_display_names", metric_names)
        names = zip(metric_names, display_names)
        return {display_name: session.metrics[name].rename_and_clone(display_name)
                for name, display_name in names}


class DebugPipelineLoader(CommonPipelineeLoader):
    def load(self, session, pipeline_spec, object_name=None):
        base_pipeline = super().load(session, pipeline_spec, object_name)

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

        return DebugPipeline.from_base_pipeline(
            base_pipeline, num_iterations=num_iterations, interval=interval,
            output_keys=output_keys, postprocessor=postprocessor, output_device=output_device
        )


class StageLoader(Loader):
    def load(self, session, spec, object_name=None):
        mode = spec.get("mode", "interleave")
        stop_condition_dict = spec.get("stop_condition")

        pipelines = spec["pipelines"]
        pipelines = [session.pipelines[name] for name in pipelines]

        debug_pipelines = spec["debug_pipelines"]
        debug_pipelines = [session.debug_pipelines[name] for name in debug_pipelines]

        stop_condition = Loader().load(session, stop_condition_dict)
        return Stage(mode, pipelines, debug_pipelines, stop_condition)


class BasePipeline:
    def __init__(self):
        self.train_dataset = None
        self.val_dataset = None
        self.preprocessors = None
        self.collator = None
        self.batch_size = None
        self.batch_adapter = None
        self.neural_graph = None
        self.device = None


class Pipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        self.loss_fn = None
        self.metric_fns = None

    @classmethod
    def from_base_pipeline(cls, pipeline, *, loss_fn, metric_fns):
        result = cls()
        result.__dict__ = pipeline.__dict__

        result.loss_fn = loss_fn
        result.metric_fns = metric_fns
        return result


class DebugPipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        self.num_iterations = 1
        self.interval = 100
        self.output_keys = []
        self.postprocessor = None
        self.output_device = None

    @classmethod
    def from_base_pipeline(cls, pipeline, *,
                           num_iterations, interval, output_keys, postprocessor, output_device):
        result = cls()
        result.__dict__ = pipeline.__dict__

        result.num_iterations = num_iterations
        result.interval = interval
        result.output_keys = output_keys
        result.postprocessor = postprocessor
        result.output_device = output_device
        return result


class Node:
    def __init__(self, name, serializable_model, serializable_optimizer, inputs, outputs):
        self.name = name
        self.net = serializable_model
        self.optimizer = serializable_optimizer
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
    def __init__(self, mode, pipelines, debug_pipelines, stop_condition):
        self.mode = mode
        self.pipelines = pipelines
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


# todo: introduce Action and Definition abstractions and new syntax (list of actions and definitions)
