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

    metric_dicts = [pipeline.metric_fns for pipeline in stage.pipelines]
    for epoch in range(start_epoch, start_epoch + 1000):
        print_metrics = PrintMetrics(metric_dicts, stat_ivl, epoch, formatter)

        for log_entries in interleave_training(stage.pipelines):
            print_metrics(log_entries)

        # todo: choose which datasets/split slices to use for evaluation and with which metrics

        metric_strings = []
        all_train_metrics = {}
        all_val_metrics = {}
        for pipeline in stage.pipelines:
            train_metrics, val_metrics = evaluate_pipeline(pipeline)
            all_train_metrics.update(train_metrics)
            all_val_metrics.update(val_metrics)
            train_metric_string = formatter.format_metrics(train_metrics, validation=False)
            val_metric_string = formatter.format_metrics(val_metrics, validation=True)
            metric_string = f'{train_metric_string}; {val_metric_string}'
            metric_strings.append(metric_string)

        epoch_str = formatter.format_epoch(epoch)

        final_metrics_string = '  |  '.join(metric_strings)
        print(f'\r{epoch_str} {final_metrics_string}')

        log_metrics(stage_number, epoch, all_train_metrics, all_val_metrics)

        session.progress.increment_progress()

        # todo; more sophisticated stop condition that can look at number of iterations and more
        should_stop = stop_condition(session.progress[stage_number].epochs_done, [])

        if should_stop:
            session.progress.mark_completed()

        save_checkpoint(epoch)

        if should_stop:
            break


def interleave_training(pipelines):
    training_loops = []
    for pipeline in pipelines:
        train_pipeline = PredictionPipeline(
            pipeline.neural_graph, pipeline.device, pipeline.batch_adapter
        )
        train_loader, _ = get_data_loaders(pipeline)
        training_loops.append(TrainingLoop(train_loader, train_pipeline, pipeline.loss_fn))

    iterators = [iter(loop) for loop in training_loops]
    while True:
        try:
            log_entries = [next(it) for it in iterators]
            yield log_entries
        except StopIteration:
            break


def evaluate_pipeline(pipeline):
    train_loader, test_loader = get_data_loaders(pipeline)
    train_pipeline = PredictionPipeline(
        pipeline.neural_graph, pipeline.device, pipeline.batch_adapter
    )

    metrics = pipeline.metric_fns

    switch_to_evaluation_mode(train_pipeline)

    train_metrics, val_metrics = compute_epoch_metrics(train_pipeline, train_loader, test_loader, metrics)

    return train_metrics, val_metrics


def compute_epoch_metrics(train_pipeline, train_loader, test_loader, metrics):
    train_metrics = evaluate(train_pipeline, train_loader, metrics, num_batches=32)
    val_metrics = evaluate(train_pipeline, test_loader, metrics, num_batches=32)
    return train_metrics, val_metrics


def evaluate(val_pipeline, dataloader, metrics, num_batches):
    moving_averages = {metric.name: MovingAverage() for _, metric in metrics.items()}

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            inputs, targets = val_pipeline.adapt_batch(batch)
            all_outputs = val_pipeline(inputs, inference_mode=False)
            update_running_metrics(moving_averages, metrics, all_outputs, targets)

    return {metric_name: avg.value for metric_name, avg in moving_averages.items()}


def update_running_metrics(moving_averages, metrics, outputs, targets):
    for metric_name, metric in metrics.items():
        moving_averages[metric.name].update(metric(outputs, targets))


class TrainingLoop:
    def __init__(self, data_loader, prediction_pipeline, loss_fn):
        self.data_loader = data_loader
        self.prediction_pipeline = prediction_pipeline
        self.loss_fn = loss_fn

    def __iter__(self):
        switch_to_train_mode(self.prediction_pipeline)
        num_iterations = len(self.data_loader)

        for i, batch in enumerate(self.data_loader):
            inputs, targets = self.prediction_pipeline.adapt_batch(batch)
            loss, outputs = self.train_on_batch(inputs, targets)
            yield IterationLogEntry(i, num_iterations, inputs, outputs, targets, loss)

    def train_on_batch(self, inputs, targets):
        for node in self.prediction_pipeline:
            # todo: refactor this if statement here and below
            if node.optimizer:
                node.optimizer.zero_grad()

        outputs = self.prediction_pipeline(inputs, inference_mode=False)

        loss = self.loss_fn(outputs, targets)
        loss.backward()

        for node in self.prediction_pipeline:
            if node.optimizer:
                node.optimizer.step()

        return loss, outputs


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
