from collections import namedtuple

from torchassistant.output_adapters import IdentityAdapter
from torchassistant.output_devices import Printer
from torchassistant.processing_graph import NeuralBatchProcessor, BatchProcessingGraph, Node
from torchassistant.session.data_classes import InputLoader, TrainingPipeline, DebugPipeline, Stage
from torchassistant.utils import instantiate_class, import_function, BackwardHookInstaller, Debugger
from torchassistant.adapters import DefaultInputAdapter
from torchassistant.stop_conditions import EarlyStopping, EpochsCompleted


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

        device = session.device
        return NeuralBatchProcessor(neural_graph, input_adapter, output_adapter, device=device)

    def parse_neural_graph(self, session, graph_spec):
        nodes = []
        for node_dict in graph_spec:
            model_name = node_dict['model_name']
            inputs = node_dict['inputs']
            outputs = node_dict['outputs']
            optimizer_name = node_dict.get('optimizer_name', '')

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
        input_loaders = self.parse_input_loaders(session, spec)

        graph = self.parse_graph(session, spec, input_loaders)

        metric_fns = self.parse_metrics(session, spec)

        loss_fns = {}
        self.parse_loss_functions(session, spec, loss_fns, metric_fns)

        return TrainingPipeline(graph, input_loaders, loss_fns, metric_fns)

    def parse_input_loaders(self, session, spec):
        input_loaders = []
        for d in spec["input_injector"]:
            kwargs = dict(d)
            kwargs["loader_factory"] = session.data_loaders[kwargs["data_loader"]]
            del kwargs["data_loader"]

            if "input_port" not in kwargs:
                kwargs["input_port"] = self.guess_input_port(session)

            input_loaders.append(InputLoader(**kwargs))
        return input_loaders

    def guess_input_port(self, session):
        input_ports = set()
        for g in session.batch_graphs.values():
            input_ports = input_ports.union(g.batch_input_names)

        if not input_ports:
            return "default_port"
        elif len(input_ports) == 1:
            return input_ports.pop()
        else:
            raise BadSpecificationError(
                f'Cannot build a pipeline. The value for "input_port" missing.'
            )

    def parse_graph(self, session, spec, input_loaders):
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
        return graph

    def infer_graph(self, session, input_loaders):
        input_ports = [loader.input_port for loader in input_loaders]

        if len(session.batch_processors) > 1 or len(set(input_ports)) != 1:
            raise BadSpecificationError(
                f'Cannot automatically infer processing graph topology. Please, specify it.'
            )

        if session.batch_processors:
            node_name = next(iter(session.batch_processors))
            nodes = {node_name: session.batch_processors[node_name]}
        else:
            node_name = "default_processor"
            nodes = {node_name: self.infer_batch_processor(session)}

        graph = BatchProcessingGraph(input_ports, **nodes)
        graph.make_edge(input_ports[0], node_name)
        return graph

    def infer_batch_processor(self, session):
        if len(session.models) > 1:
            raise BadSpecificationError(
                f'Cannot automatically infer batch processor when multiple models are defined.'
            )

        if len(session.optimizers) > 1:
            raise BadSpecificationError(
                f'Cannot automatically infer batch processor when multiple optimizers are defined.'
            )

        if not session.models:
            raise BadSpecificationError(
                f'Cannot automatically infer batch processor. Please, define model.'
            )

        model_name = next(iter(session.models))
        model = session.models[model_name]

        optimizer = next(iter(session.optimizers.values())) if session.optimizers else None

        inputs = ["input_1"]
        outputs = ["y_hat"]
        neural_nodes = [
            Node(name=model_name, model=model, optimizer=optimizer, inputs=inputs, outputs=outputs)
        ]

        input_adapter = DefaultInputAdapter(model_name)
        output_adapter = IdentityAdapter()

        device = session.device
        return NeuralBatchProcessor(neural_nodes, input_adapter, output_adapter, device)

    def parse_metrics(self, session, pipeline_spec):
        metrics = {}

        for spec in pipeline_spec.get("metrics", []):
            name = spec["metric_name"]
            display_name = spec.get("display_name", name)
            node_name = self.get_node_name(session, spec)
            metric = session.metrics[name].rename_and_clone(display_name)
            metrics[display_name] = (node_name, metric)

        return metrics

    def parse_loss_functions(self, session, spec, loss_fns, metric_fns):
        for loss_spec in spec.get("losses", []):
            loss_name = loss_spec["loss_name"]
            node_name = self.get_node_name(session, loss_spec)

            loss_fn = (node_name, session.losses[loss_name])
            loss_fns[loss_name] = loss_fn

            loss_display_name = loss_spec.get('loss_display_name', loss_name)
            renamed_loss_fn = loss_fn[1].rename_and_clone(loss_display_name)
            metric_fns[loss_display_name] = (node_name, renamed_loss_fn)

    def get_node_name(self, session, spec):
        options = list(session.batch_processors.keys())
        if len(session.batch_processors) > 1 and "node_name" not in spec:
            raise BadSpecificationError(
                f'You must specify which batch processor outputs to use to compute loss/metric.'
                f'Please, provide a value for "node_name". Options: {options}'
            )
        elif len(session.batch_processors) > 1:
            choice = spec["node_name"]
        elif session.batch_processors:
            default_choice = next(iter(session.batch_processors))
            options = [default_choice]
            choice = spec.get("node_name", default_choice)
        else:
            options = ["default_processor"]
            choice = spec.get("node_name", "default_processor")

        if choice not in options:
            raise BadSpecificationError(
                f'Invalid value for "node_name". Must be one of: {options}'
            )
        return choice


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

        training_pipelines = spec["training_pipelines"]
        training_pipelines = [session.pipelines[name] for name in training_pipelines]

        validation_pipelines = spec.get("validation_pipelines", [])
        validation_pipelines = [session.pipelines[name] for name in validation_pipelines]

        debug_pipelines = spec.get("debug_pipelines", [])
        debug_pipelines = [session.debug_pipelines[name] for name in debug_pipelines]

        stop_condition_dict = spec.get("stop_condition")
        if stop_condition_dict:
            stop_condition = Loader().load(session, stop_condition_dict)
        else:
            stop_condition = self.build_stop_condition(session)

        eval_steps = spec.get("eval_steps", self.default_eval_steps(session))

        return Stage(mode, training_pipelines, validation_pipelines, debug_pipelines,
                     stop_condition, eval_steps)

    def default_eval_steps(self, session):
        """Simple heuristics to auto select appropriate value for eval_steps"""
        min_steps = min(len(loader_factory.build())
                        for loader_factory in session.data_loaders.values())
        if min_steps > 10 ** 6:
            fraction = 0.01
        elif min_steps > 10 ** 4:
            fraction = 0.1
        elif min_steps > 10 ** 3:
            fraction = 0.2
        elif min_steps > 10 ** 2:
            fraction = 0.5
        else:
            fraction = 1.0
        return fraction

    def build_stop_condition(self, session):
        if session.pipelines:
            val_loss_fns = []
            for pipeline in session.pipelines.values():
                if len(pipeline.input_loaders) != 1:
                    return EpochsCompleted(20)

                data_loader = pipeline.input_loaders[0].loader_factory
                for i, k in enumerate(session.data_loaders.values()):
                    loss_names = self.get_loss_fns(pipeline)

                    if k == data_loader and i != 0:
                        val_loss_fns.extend(loss_names)

            if not val_loss_fns:
                return EpochsCompleted(20)

            monitored_loss = val_loss_fns[0]
            return EarlyStopping(monitored_loss)
        else:
            return EpochsCompleted(20)

    def get_loss_fns(self, pipeline):
        res = set()
        for metric_name, (_, metric_obj) in pipeline.metric_fns.items():
            for _, loss_obj in pipeline.loss_fns.values():
                if metric_obj.metric_fn is loss_obj.metric_fn:
                    res.add(metric_name)

        return list(res)
