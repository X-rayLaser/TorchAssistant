from collections import namedtuple


class DebugPipeline:
    def __init__(self, *, graph, input_loaders, num_iterations, interval,
                 output_keys, postprocessor, output_device):
        self.graph = graph
        self.input_loaders = input_loaders
        self.num_iterations = num_iterations
        self.interval = interval
        self.output_keys = output_keys
        self.postprocessor = postprocessor
        self.output_device = output_device


class Stage:
    def __init__(self, mode, training_pipelines,
                 validation_pipelines, debug_pipelines, stop_condition, eval_steps):
        self.mode = mode
        self.training_pipelines = training_pipelines
        self.validation_pipelines = validation_pipelines
        self.debug_pipelines = debug_pipelines
        self.stop_condition = stop_condition
        self.eval_steps = eval_steps


class InputLoader:
    def __init__(self, input_port, loader_factory, variable_names=None):
        self.input_port = input_port
        self.loader_factory = loader_factory
        self.variable_names = variable_names


TrainingPipeline = namedtuple('TrainingPipeline', ["graph", "input_loaders", "loss_fns", "metric_fns"])
