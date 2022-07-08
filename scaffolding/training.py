import itertools

import torch
from .utils import switch_to_train_mode, switch_to_evaluation_mode
from .metrics import MovingAverage
from .formatters import Formatter
from scaffolding.utils import WrappedDataset
from scaffolding.session import StopTrainingError


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
        print_metrics = PrintMetrics(metric_dicts, stat_ivl, epoch, formatter)

        for log_entries in interleave_training(session, training_pipelines):
            print_metrics(log_entries)
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
        self.batch_pipeline = pipeline.batch_pipeline
        self.postprocessor = pipeline.postprocessor
        self.output_device = pipeline.output_device
        self.pipeline = pipeline

    def __call__(self, log_entries):
        entry = log_entries[0]
        interval = self.pipeline.interval
        print('called debugger')
        if entry.iteration % interval == interval - 1:
            with torch.no_grad():
                self.debug()

    def debug(self):
        it = iter(self.batch_pipeline)
        print('start debugging')
        for _ in range(self.pipeline.num_iterations):
            print('iteration')
            batch = next(it)
            predictions = {k: batch[k] for k in self.pipeline.output_keys}

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
            ActualTrainer(batch_pipeline=pipeline.batch_pipeline,
                          loss_fn=pipeline.loss_fn,
                          gradient_clippers=session.gradient_clippers)
        )
    return trainers


def evaluate_pipeline(pipeline, num_batches=10):
    batch_pipeline = pipeline.batch_pipeline
    metrics = pipeline.metric_fns
    batch_pipeline.eval_mode()

    moving_averages = {metric.name: MovingAverage() for _, metric in metrics.items()}

    with torch.no_grad():
        for i, results_batch in enumerate(batch_pipeline):
            if i >= num_batches:
                break

            all_outputs = results_batch
            targets = results_batch
            update_running_metrics(moving_averages, metrics, all_outputs, targets)

    return {metric_name: avg.value for metric_name, avg in moving_averages.items()}


def update_running_metrics(moving_averages, metrics, outputs, targets):
    for metric_name, metric in metrics.items():
        moving_averages[metric.name].update(metric(outputs, targets))


def get_data_loaders(pipeline):
    train_set, test_set = pipeline.train_dataset, pipeline.val_dataset

    if pipeline.preprocessors:
        train_set = WrappedDataset(train_set, pipeline.preprocessors)
        test_set = WrappedDataset(test_set, pipeline.preprocessors)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=pipeline.batch_size,
                                               shuffle=True, num_workers=2, collate_fn=pipeline.collator)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=pipeline.batch_size,
                                              shuffle=False, num_workers=2, collate_fn=pipeline.collator)

    return train_loader, test_loader


class PrintMetrics:
    def __init__(self, metric_dicts, ivl, epoch, format_fn):
        self.interval = ivl
        self.formatters = []
        for metrics in metric_dicts:
            self.formatters.append(
                RunningMetricsSetFormatter(metrics, ivl, epoch, format_fn)
            )

    def __call__(self, log_entries):
        columns = [self.formatters[i](entry) for i, entry in enumerate(log_entries)]

        iteration = log_entries[0].iteration

        if iteration % self.interval == self.interval - 1:
            s = '  |  '.join(columns)
            print(f'\r{s}', end='')
            for formatter in self.formatters:
                formatter.reset()


class RunningMetricsSetFormatter:
    def __init__(self, metrics, ivl, epoch, format_fn):
        self.metrics = metrics
        self.interval = ivl
        self.epoch = epoch
        self.format_fn = format_fn

        self.running_metrics = {name: MovingAverage() for name in metrics}

    def __call__(self, iteration_log):
        iteration = iteration_log.iteration

        with torch.no_grad():
            update_running_metrics(self.running_metrics, self.metrics,
                                   iteration_log.outputs, iteration_log.targets)

        return self.format_fn(self.epoch, iteration + 1, iteration_log.num_iterations,
                              self.metrics, self.running_metrics)

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


class ActualTrainer:
    def __init__(self, batch_pipeline, loss_fn, gradient_clippers):
        self.batch_pipeline = batch_pipeline
        self.loss_fn = loss_fn
        self.gradient_clippers = gradient_clippers

    def __iter__(self):
        self.batch_pipeline.train_mode()
        num_iterations = len(self.batch_pipeline)

        for i, results_batch in enumerate(self.batch_pipeline):
            loss = self.train_on_batch(results_batch)

            # todo: retrieve somehow
            inputs = []

            # todo: this hack should work for now, but fix this later
            outputs = results_batch
            targets = results_batch
            yield IterationLogEntry(i, num_iterations, inputs, outputs, targets, loss)

    def get_tensors(self, batch, var_names):
        return {var_name: batch[var_name] for var_name in var_names}

    def train_on_batch(self, batch):
        loss = self.loss_fn(batch)
        loss.backward()

        for clip_gradients in self.gradient_clippers.values():
            clip_gradients()

        self.batch_pipeline.update()
        return loss
