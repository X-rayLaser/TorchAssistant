import torch


def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total


def accuracy_percentage(outputs, labels):
    return accuracy(outputs, labels) * 100


def token_counter(tokens_tensor):
    return len(tokens_tensor)


metric_functions = {
    'loss': None,
    'accuracy': accuracy,
    'accuracy %': accuracy_percentage,
    'token_counter': token_counter
}


class Metric:
    def __init__(self, name, metric_fn, metric_args, transform_fn):
        self.name = name
        self.metric_fn = metric_fn
        self.metric_args = metric_args
        self.transform_fn = transform_fn

    def __call__(self, outputs, targets):
        lookup_table = targets.copy()
        lookup_table.update(outputs)

        arg_values = [lookup_table[arg] for arg in self.metric_args]
        arg_values = self.transform_fn(*arg_values)
        return self.metric_fn(*arg_values)


# todo: support exponentially weighted averages too
class MovingAverage:
    def __init__(self):
        self.x = 0
        self.num_updates = 0

    def reset(self):
        self.x = 0
        self.num_updates = 0

    def update(self, x):
        self.x += x
        self.num_updates += 1

    @property
    def value(self):
        return self.x / self.num_updates
