import torch
from .utils import save_session, switch_to_train_mode, switch_to_evaluation_mode
from .metrics import MovingAverage
from .formatters import Formatter


def train(session, stat_ivl=10):
    data_pipeline = session.data_pipeline
    train_pipeline = session.restore_from_last_checkpoint()
    loss_fn = session.criterion
    metrics = session.metrics
    epochs = session.num_epochs
    start_epoch = session.epochs_trained + 1
    checkpoints_dir = session.checkpoints_dir

    if 'loss' in metrics:
        metrics['loss'] = loss_fn

    train_loader, test_loader = data_pipeline.get_data_loaders()
    formatter = Formatter()

    for epoch in range(start_epoch, start_epoch + epochs):
        trainer = Trainer(train_loader, train_pipeline, loss_fn)
        print_metrics = PrintMetrics(metrics, stat_ivl, epoch, formatter)
        trainer.add_callback(print_metrics)
        trainer.run_epoch()

        switch_to_evaluation_mode(train_pipeline)

        train_metrics, val_metrics = compute_epoch_metrics(train_pipeline, train_loader, test_loader, metrics)
        session.log_metrics(epoch, train_metrics, val_metrics)
        epoch_str = formatter.format_epoch(epoch)
        train_metrics_str = formatter.format_metrics(train_metrics, validation=False)
        val_metrics_str = formatter.format_metrics(val_metrics, validation=True)

        print(f'\r{epoch_str} {train_metrics_str}; {val_metrics_str}')

        if checkpoints_dir:
            session.make_checkpoint(train_pipeline, epoch)


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
