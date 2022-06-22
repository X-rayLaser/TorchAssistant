import torch
from .utils import switch_to_train_mode, switch_to_evaluation_mode
from .metrics import MovingAverage
from .formatters import Formatter
from scaffolding.utils import WrappedDataset
from scaffolding.session_v2 import StopTrainingError


def train(session, log_metrics, save_checkpoint, stat_ivl=10):
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
    pipeline = stage.pipelines[0]

    train_pipeline = PredictionPipeline(pipeline.neural_graph, pipeline.device, pipeline.batch_adapter)

    metrics = session.metrics

    metrics['loss'] = pipeline.loss_fn

    train_loader, test_loader = get_data_loaders(pipeline)
    formatter = Formatter()
    start_epoch = session.progress.epochs_done_total + 1

    for epoch in range(start_epoch, start_epoch + 1000):
        trainer = Trainer(train_loader, train_pipeline, pipeline.loss_fn)
        print_metrics = PrintMetrics(metrics, stat_ivl, epoch, formatter)
        trainer.add_callback(print_metrics)
        trainer.run_epoch()

        switch_to_evaluation_mode(train_pipeline)

        train_metrics, val_metrics = compute_epoch_metrics(train_pipeline, train_loader, test_loader, metrics)
        log_metrics(epoch, train_metrics, val_metrics)
        epoch_str = formatter.format_epoch(epoch)
        train_metrics_str = formatter.format_metrics(train_metrics, validation=False)
        val_metrics_str = formatter.format_metrics(val_metrics, validation=True)

        print(f'\r{epoch_str} {train_metrics_str}; {val_metrics_str}')

        session.progress.increment_progress()

        should_stop = stop_condition(session.progress[stage_number].epochs_done, [])

        if should_stop:
            session.progress.mark_completed()

        save_checkpoint(epoch)

        if should_stop:
            break


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
    def __init__(self, metrics, ivl, epoch, format_fn):
        self.metrics = metrics
        self.interval = ivl
        self.epoch = epoch
        self.format_fn = format_fn

        self.running_loss = MovingAverage()
        self.running_metrics = {name: MovingAverage() for name in metrics}

    def __call__(self, iteration_log):
        iteration = iteration_log.iteration

        self.running_loss.update(iteration_log.loss.item())
        update_running_metrics(self.running_metrics, self.metrics,
                               iteration_log.outputs, iteration_log.targets)

        if iteration % self.interval == self.interval - 1:
            s = self.format_fn(self.epoch, iteration + 1, iteration_log.num_iterations,
                               self.metrics, self.running_loss, self.running_metrics)
            print(s, end='')

            for metric_avg in self.running_metrics.values():
                metric_avg.reset()

            self.running_loss.reset()


def compute_epoch_metrics(train_pipeline, train_loader, test_loader, metrics):
    train_metrics = evaluate(train_pipeline, train_loader, metrics, num_batches=32)
    val_metrics = evaluate(train_pipeline, test_loader, metrics, num_batches=32)
    return train_metrics, val_metrics


def evaluate(val_pipeline, dataloader, metrics, num_batches):
    moving_averages = {metric_name: MovingAverage() for metric_name in metrics}

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
        moving_averages[metric_name].update(metric(outputs, targets))


class Trainer:
    def __init__(self, data_loader, prediction_pipeline, loss_fn):
        self.data_loader = data_loader
        self.prediction_pipeline = prediction_pipeline
        self.loss_fn = loss_fn
        self.callbacks = []

    def add_callback(self, cb):
        self.callbacks.append(cb)

    def run_epoch(self):
        switch_to_train_mode(self.prediction_pipeline)

        num_iterations = len(self.data_loader)
        for i, batch in enumerate(self.data_loader):
            inputs, targets = self.prediction_pipeline.adapt_batch(batch)
            loss, outputs = self.train_on_batch(inputs, targets)
            self.invoke_callbacks(
                IterationLogEntry(i, num_iterations, inputs, outputs, targets, loss)
            )

    def invoke_callbacks(self, log_entry):
        for cb in self.callbacks:
            cb(log_entry)

    def train_on_batch(self, inputs, targets):
        for node in self.prediction_pipeline:
            node.optimizer.zero_grad()

        outputs = self.prediction_pipeline(inputs, inference_mode=False)

        loss = self.loss_fn(outputs, targets)
        loss.backward()

        for node in self.prediction_pipeline:
            node.optimizer.step()

        return loss, outputs


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
