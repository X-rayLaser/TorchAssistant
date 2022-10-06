from collections import namedtuple

import torch

from torchassistant.output_adapters import IdentityAdapter
from torchassistant.output_devices import Printer
from torchassistant.processing_graph import NeuralBatchProcessor, BatchProcessingGraph, Node
from torchassistant.session.data_classes import InputLoader, TrainingPipeline, DebugPipeline, Stage
from torchassistant.utils import instantiate_class, import_function, BackwardHookInstaller, Debugger


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


class PipelineLoader(Loader):
    def load(self, session, spec, object_name=None):
        input_loaders = []
        for d in spec["input_factories"]:
            kwargs = dict(d)
            kwargs["loader_factory"] = session.loader_factories[kwargs["loader_factory"]]
            input_loaders.append(InputLoader(**kwargs))

        if "graph" not in spec and len(session.batch_graphs) > 1:
            raise BadSpecificationError(
                f'You must specify which graph to use for the pipeline. '
                f'Options are: {list(session.batch_graphs.keys())}'
            )
        elif "graph" not in spec and session.batch_graphs:
            graph = next(iter(session.batch_graphs.values()))
        elif "graph" not in spec:
            # create a default graph here
            graph = self.infer_graph(session, input_loaders)
        else:
            graph = session.batch_graphs[spec["graph"]]

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

    def infer_graph(self, session, input_loaders):
        input_ports = [loader.input_alias for loader in input_loaders]

        if len(session.batch_processors) > 1 or len(set(input_ports)) != 1:
            raise BadSpecificationError(
                f'Cannot automatically infer processing graph topology. Please, specify it.'
            )

        if len(session.batch_processors) == 0:
            raise BadSpecificationError(
                f'Cannot automatically infer processing graph topology. Missing batch proceessors.'
            )

        node_name = next(iter(session.batch_processors))
        nodes = {node_name: session.batch_processors[node_name]}

        graph = BatchProcessingGraph(input_ports, **nodes)
        graph.make_edge(input_ports[0], node_name)
        return graph

    def parse_metrics(self, session, pipeline_spec):
        metrics = {}
        for spec in pipeline_spec.get("metrics", []):
            name = spec["metric_name"]
            display_name = spec["display_name"]
            node_name = spec["node_name"]
            metric = session.metrics[name].rename_and_clone(display_name)
            metrics[display_name] = (node_name, metric)

        return metrics


class BackwardHookLoader(Loader):
    hook_installer = BackwardHookInstaller

    def load(self, session, spec, object_name=None):
        model_name = spec["model"]
        function_name = spec["factory_fn"]

        model = session.models[model_name]
        create_hook = import_function(function_name)
        return self.hook_installer(model, create_hook)


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
