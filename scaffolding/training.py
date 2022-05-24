import torch
from .utils import save_session, switch_to_train_mode, switch_to_evaluation_mode
from .metrics import MovingAverage


def train(data_pipeline, train_pipeline, loss_fn, metrics,
          epochs=2, start_epoch=0, checkpoints_dir=None, stat_ivl=50):
    if 'loss' in metrics:
        metrics['loss'] = loss_fn

    train_loader, test_loader = data_pipeline.get_data_loaders()

    for epoch in range(start_epoch, start_epoch + epochs):
        switch_to_train_mode(train_pipeline)

        running_loss = MovingAverage()
        running_metrics = {name: MovingAverage() for name in metrics}

        for i, batch in enumerate(train_loader):
            loss, outputs = train_on_batch(train_pipeline, batch, loss_fn)

            running_loss.update(loss.item())
            update_running_metrics(running_metrics, metrics, outputs, batch["targets"])

            if i % stat_ivl == stat_ivl - 1:
                print_train_metrics(epoch, i, metrics, running_loss, running_metrics)

                for metric_avg in running_metrics.values():
                    metric_avg.reset()

                running_loss.reset()

        switch_to_evaluation_mode(train_pipeline)

        train_metrics, val_metrics = compute_epoch_metrics(train_pipeline, train_loader, test_loader, metrics)

        print_epoch_metrics(train_metrics, val_metrics, epoch)

        if checkpoints_dir:
            save_session(train_pipeline, epoch, checkpoints_dir)


def print_train_metrics(epoch, i, metrics, running_loss, running_metrics):
    stat_info = f'\rEpoch {epoch + 1}, iteration {i + 1:5d},'
    if 'loss' in metrics:
        stat_info += f' loss: {running_loss.value:.3f}'

    for metric_name, metric_avg in running_metrics.items():
        stat_info += f'; {metric_name}: {metric_avg.value:.3f}'

    print(stat_info, end='')


def compute_epoch_metrics(train_pipeline, train_loader, test_loader, metrics):
    train_metrics = evaluate(train_pipeline, train_loader, metrics, num_batches=32)
    val_metrics = evaluate(train_pipeline, test_loader, metrics, num_batches=32)
    return train_metrics, val_metrics


def print_epoch_metrics(train_metrics, val_metrics, epoch):
    s = f'\rEpoch {epoch + 1}, '
    for name, value in train_metrics.items():
        s += f'{name}: {value}; '

    for name, value in val_metrics.items():
        s += f'val {name}: {value}; '

    print(s)


def train_on_batch(train_pipeline, batch, loss_fn):
    for pipe in train_pipeline:
        pipe.optimizer.zero_grad()

    outputs = do_forward_pass(train_pipeline, batch)

    loss = loss_fn(outputs, batch["targets"])
    loss.backward()

    for pipe in train_pipeline:
        pipe.optimizer.step()

    return loss, outputs


def do_forward_pass(train_pipeline, batch, inference_mode=False):
    inputs = batch["inputs"]

    all_outputs = {}
    for pipe in train_pipeline:
        args = get_dependencies(pipe, inputs, all_outputs)
        if inference_mode:
            outputs = pipe.net.run_inference(*args)
        else:
            outputs = pipe.net(*args)
        all_outputs.update(
            dict(zip(pipe.outputs, outputs))
        )
    return all_outputs


def get_dependencies(pipe, batch_inputs, prev_outputs):
    lookup_table = batch_inputs[pipe.name].copy()
    lookup_table.update(prev_outputs)
    return [lookup_table[var_name] for var_name in pipe.inputs]


def evaluate(val_pipeline, dataloader, metrics, num_batches):
    moving_averages = {metric_name: MovingAverage() for metric_name in metrics}

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            all_outputs = do_forward_pass(val_pipeline, batch)
            update_running_metrics(moving_averages, metrics, all_outputs, batch["targets"])

    return {metric_name: avg.value for metric_name, avg in moving_averages.items()}


def update_running_metrics(moving_averages, metrics, outputs, targets):
    for metric_name, metric in metrics.items():
        moving_averages[metric_name].update(metric(outputs, targets))
