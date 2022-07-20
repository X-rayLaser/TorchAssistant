import itertools

import torch
from .utils import switch_to_train_mode, switch_to_evaluation_mode
from .metrics import MovingAverage
from .formatters import Formatter
from scaffolding.utils import WrappedDataset
from scaffolding.session import StopTrainingError
from .processing_graph import Trainer


def train(session, log_metrics, save_checkpoint, stat_ivl=1):
    while True:
        try:
            stage_number = session.progress.get_current_stage_id()
            print(f'Training stage {stage_number} in progress')
            train_stage(session, stage_number, log_metrics, save_checkpoint, stat_ivl)
            print(f'Training stage {stage_number} is completed')
        except StopTrainingError:
            break

    print("Training is completed")


def train_stage(session, stage_number, log_metrics, save_checkpoint, stat_ivl=10):
    stage = session.stages[stage_number]

    stop_condition = stage.stop_condition

    formatter = Formatter()
    start_epoch = session.progress.epochs_done_total + 1

    training_pipelines = stage.training_pipelines

    metric_dicts = [pipeline.metric_fns for pipeline in training_pipelines]

    debuggers = [Debugger(pipeline) for pipeline in stage.debug_pipelines]
    print(debuggers)
    for epoch in range(start_epoch, start_epoch + 1000):
        calculators = [MetricsSetCalculator(metrics, stat_ivl) for metrics in metric_dicts]

        for log_entries in interleave_training(session, training_pipelines):
            all_running_metrics = {}
            for i, log in enumerate(log_entries):
                all_running_metrics.update(calculators[i](log))

            an_entry = log_entries[0]
            stats = formatter(
                epoch, an_entry.iteration + 1,
                an_entry.num_iterations, all_running_metrics
            )
            print(f'\r{stats}', end='')

            # run debuggers/visualization code
            for debug in debuggers:
                debug(log_entries)

        # todo: choose which datasets/split slices to use for evaluation and with which metrics

        all_train_metrics = {}
        for pipeline in stage.validation_pipelines:
            train_metrics = evaluate_pipeline(pipeline)
            all_train_metrics.update(train_metrics)

        final_metrics_string = formatter.format_metrics(all_train_metrics, validation=False)

        epoch_str = formatter.format_epoch(epoch)

        whitespaces = ' ' * 150
        print(f'\r{whitespaces}')
        print(f'\r{epoch_str} {final_metrics_string}')

        log_metrics(stage_number, epoch, all_train_metrics)

        session.progress.increment_progress()

        # todo; more sophisticated stop condition that can look at number of iterations and more
        should_stop = stop_condition(session.progress[stage_number].epochs_done, [])

        if should_stop:
            session.progress.mark_completed()

        save_checkpoint(epoch)

        if should_stop:
            break


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
        from .processing_graph import DataGenerator

        it = iter(DataGenerator(self.pipeline.input_loaders))

        for _ in range(self.pipeline.num_iterations):

            graph_inputs = next(it)
            results = self.graph(graph_inputs)
            all_results = {k: v for batches in results.values() for k, v in batches.items()}

            predictions = {k: all_results[k] for k in self.pipeline.output_keys}

            if self.postprocessor:
                predictions = self.postprocessor(predictions)

            self.output_device(predictions)


def interleave_training(session, pipelines):
    training_loops = prepare_trainers(session, pipelines)

    entries_gen = itertools.zip_longest(*training_loops)

    def fill_missing(original_tuple, filler_tuple):
        if not filler_tuple:
            return original_tuple

        pairs = zip(original_tuple, filler_tuple)
        return tuple(value if value else filler for value, filler in pairs)

    entries = None
    for log_entries in entries_gen:
        entries = fill_missing(log_entries, entries)
        yield entries


def prepare_trainers(session, pipelines):
    trainers = []
    for pipeline in pipelines:
        trainers.append(
            Trainer(pipeline.graph, pipeline.input_loaders, pipeline.loss_fns, session.gradient_clippers)
        )
    return trainers


def evaluate_pipeline(pipeline, num_batches=10):
    from .processing_graph import DataGenerator
    graph = pipeline.graph
    metrics = pipeline.metric_fns

    graph.eval_mode()

    #batch_pipeline.eval_mode()

    data_generator = DataGenerator(pipeline.input_loaders)

    moving_averages = {name: MovingAverage() for name in metrics}

    with torch.no_grad():
        for i, graph_inputs in enumerate(data_generator):
            if i >= num_batches:
                break

            results_batch = graph(graph_inputs)

            for name, (leaf_name, metric_fn) in metrics.items():
                outputs = results_batch[leaf_name]
                moving_averages[name].update(metric_fn(outputs, outputs))
            #update_running_metrics(moving_averages, metrics, all_outputs, targets)

    return {metric_name: avg.value for metric_name, avg in moving_averages.items()}


def update_running_metrics(moving_averages, metrics, outputs, targets):
    for metric_name, metric in metrics.items():
        moving_averages[metric.name].update(metric(outputs, targets))


class MetricsSetCalculator:
    def __init__(self, metrics, interval):
        """
        :param metrics: {'name': ('graph_leaf', Metric())}
        """
        self.metrics = metrics
        self.interval = interval
        self.running_metrics = {name: MovingAverage() for name in metrics}

    def __call__(self, iteration_log):
        if iteration_log.iteration % self.interval == 0:
            self.reset()

        with torch.no_grad():
            for name, (leaf_name, metric_fn) in self.metrics.items():
                outputs = iteration_log.outputs[leaf_name]
                self.running_metrics[name].update(metric_fn(outputs, outputs))
        return self.running_metrics

    def reset(self):
        for metric_avg in self.running_metrics.values():
            metric_avg.reset()


class IterationLogEntry:
    def __init__(self, iteration, num_iterations, inputs, outputs, targets, loss):
        self.iteration = iteration
        self.num_iterations = num_iterations
        self.inputs = inputs
        self.outputs = outputs
        self.targets = targets
        self.loss = loss


class PredictionPipeline:
    def __init__(self, model, device, batch_adapter):
        self.model = model
        self.device = device
        self.batch_adapter = batch_adapter

    def adapt_batch(self, batch):
        batch = self.batch_adapter.adapt(*batch)
        return batch["inputs"], batch.get("targets")

    def __iter__(self):
        return iter(self.model)

    def __call__(self, inputs, inference_mode=False):
        self.inputs_to(inputs)

        all_outputs = {}
        for node in self.model:
            outputs = node(inputs, all_outputs, inference_mode)
            all_outputs.update(
                dict(zip(node.outputs, outputs))
            )

        return all_outputs

    def inputs_to(self, inputs):
        for k, mapping in inputs.items():
            for tensor_name, value in mapping.items():
                if hasattr(value, 'device') and value.device != self.device:
                    mapping[tensor_name] = value.to(self.device)


# todo: remove dead code, make evaluation scripts work
