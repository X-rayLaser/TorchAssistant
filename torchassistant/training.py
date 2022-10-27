import itertools

import torch
from .metrics import MovingAverage
from .formatters import Formatter
from torchassistant.session import StopTrainingError
from .data import InputInjector
from .utils import Debugger
from torchassistant.formatters import ProgressBar


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

    start_epoch = session.progress.epochs_done_total + 1

    metric_dicts = [pipeline.metric_fns for pipeline in stage.training_pipelines]

    debuggers = [Debugger(pipeline) for pipeline in stage.debug_pipelines]

    history = []
    for epoch in range(start_epoch, start_epoch + 1000):
        calculators = [MetricsSetCalculator(metrics, stat_ivl) for metrics in metric_dicts]

        train_on_data(session, stage.training_pipelines, debuggers, calculators, epoch)

        computed_metrics = compute_and_log_metrics(stage)
        history.append(computed_metrics)

        log_computed_metrics(computed_metrics, stage_number, epoch, log_metrics)

        session.progress.increment_progress()

        # todo; more sophisticated stop condition that can look at number of iterations and more
        should_stop = stage.stop_condition(session.progress[stage_number].epochs_done, history)

        if should_stop:
            session.progress.mark_completed()

        save_checkpoint(epoch)

        if should_stop:
            break


def train_on_data(session, training_pipelines, debuggers, metric_calculators, epoch):
    formatter = Formatter()

    for log_entries in interleave_training(session, training_pipelines):
        all_running_metrics = {}
        for i, log in enumerate(log_entries):
            all_running_metrics.update(metric_calculators[i](log))

        an_entry = log_entries[0]
        stats = formatter(
            epoch, an_entry.iteration + 1,
            an_entry.num_iterations, all_running_metrics
        )
        print(f'\r{stats}', end='')

        # run debuggers/visualization code
        for debug in debuggers:
            debug(log_entries)


def compute_and_log_metrics(stage):
    computed_metrics = {}
    for pipeline in stage.validation_pipelines:
        metrics = evaluate_pipeline(pipeline, stage.eval_steps)
        computed_metrics.update(metrics)

    return computed_metrics


def log_computed_metrics(computed_metrics, stage_number, epoch, log_fn):
    formatter = Formatter()

    final_metrics_string = formatter.format_metrics(computed_metrics, validation=False)

    epoch_str = formatter.format_epoch(epoch)

    whitespaces = ' ' * 150
    print(f'\r{whitespaces}\r{epoch_str} {final_metrics_string}')

    log_fn(stage_number, epoch, computed_metrics)


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
        data_generator = InputInjector(pipeline.input_loaders)
        trainers.append(
            Trainer(pipeline.graph, data_generator, pipeline.loss_fns, session.gradient_clippers)
        )
    return trainers


def evaluate_pipeline(pipeline, num_batches=1.0):
    """Compute metrics specified for this pipeline.

    :param pipeline: instance of TrainingPipeline
    :param num_batches: Number of batches used to compute metrics on.
    Integer value is interpreted as precise number batches.
    Floating point number in 0-1 range is interpreted as a fraction of all batches in the pipeline.
    :return: a dictionary mapping names of metrics to their computed values
    """
    graph = pipeline.graph
    metrics = pipeline.metric_fns

    graph.eval_mode()

    data_generator = InputInjector(pipeline.input_loaders)

    if isinstance(num_batches, float):
        fraction = num_batches
        batches_total = len(data_generator)
        num_batches = int(round(batches_total * fraction))
        num_batches = max(1, num_batches)

    moving_averages = {name: MovingAverage() for name in metrics}

    whitespaces = ' ' * 150
    print(f'\r{whitespaces}', end='')

    progress_bar = ProgressBar()

    with torch.no_grad():
        for i, graph_inputs in enumerate(data_generator):
            if i >= num_batches:
                break

            results_batch = graph(graph_inputs)
            update_running_metrics(moving_averages, metrics, results_batch)

            step_number = i + 1
            progress = progress_bar.updated(step_number, num_batches, cols=50)
            print(f'\rEvaluating metrics: {progress} {step_number}/{num_batches}', end='')

    return {metric_name: avg.value for metric_name, avg in moving_averages.items()}


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
            update_running_metrics(self.running_metrics, self.metrics, iteration_log.outputs)

        return self.running_metrics

    def reset(self):
        for metric_avg in self.running_metrics.values():
            metric_avg.reset()


def update_running_metrics(moving_averages, metrics, results_batch):
    for name, (leaf_name, metric_fn) in metrics.items():
        moving_averages[name].update(
            metric_fn(results_batch[leaf_name])
        )


class IterationLogEntry:
    def __init__(self, iteration, num_iterations, inputs, outputs, targets, loss):
        self.iteration = iteration
        self.num_iterations = num_iterations
        self.inputs = inputs
        self.outputs = outputs
        self.targets = targets
        self.loss = loss


class Trainer:
    def __init__(self, graph, data_generator, losses: dict, gradient_clippers):
        self.graph = graph
        self.data_generator = data_generator
        self.losses = losses
        self.gradient_clippers = gradient_clippers

        # calls train() on every model object in the graph
        self.graph.train_mode()

    def __iter__(self):
        num_iterations = len(self.data_generator)

        inputs = []
        for i, batches in enumerate(self.data_generator):
            losses, results = self.train_one_iteration(batches)
            outputs = results
            targets = results
            # todo: fix this (may get rid of inputs and targets)
            yield IterationLogEntry(i, num_iterations, inputs, outputs, targets, losses)

    def train_one_iteration(self, graph_inputs):
        results = self.graph(graph_inputs)

        losses = {}
        for name, (node_name, loss_fn) in self.losses.items():
            batch = results[node_name]
            loss = loss_fn(batch)

            # invoke zero_grad for each neural network
            self.graph.prepare()
            loss.backward()

            for clip_gradients in self.gradient_clippers.values():
                clip_gradients()

            # invoke optimizer.step() for every neural network if there is one
            self.graph.update()

            losses[node_name] = loss

        return losses, results
