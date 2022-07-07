import torch


def accuracy(outputs, labels):
    print(outputs.data)
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
    def __init__(self, name, metric_fn, metric_args, transform_fn, device):
        # todo: pass here a shorthand name used in place of full metric name when printing/saving history
        self.name = name
        self.metric_fn = metric_fn
        self.metric_args = metric_args
        self.transform_fn = transform_fn
        self.device = device

    def rename_and_clone(self, new_name):
        return self.__class__(new_name, self.metric_fn, self.metric_args, self.transform_fn, self.device)

    def __call__(self, *args):
        """

        :param outputs: a dictionary of all outputs from prediction pipeline
        :type outputs: Dict[str -> torch.tensor]
        :param targets: a dictionary of all targets
        :type targets: Dict[str -> torch.tensor]
        :return: a metric scalar
        :rtype: degenerate tensor of shape ()
        """
        if len(args) == 2:
            outputs, targets = args
            lookup_table = targets.copy()
            lookup_table.update(outputs)
        else:
            lookup_table = args[0]

        tensors = [lookup_table[arg] for arg in self.metric_args]

        tensors = self.change_device(tensors)
        tensors = self.transform_fn(*tensors)
        # the above operation could change devices
        tensors = self.change_device(tensors)
        return self.metric_fn(*tensors)

    def change_device(self, tensors):
        """Moves all tensors that participate in metric calculation to a given device

        :param tensors: a list of tensors
        :return: a list of tensors
        """

        # some tensor may already be on the right device, if so they kept unchanged
        return [arg if not hasattr(arg, 'device') or arg.device == self.device else arg.to(self.device)
                for arg in tensors]


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
