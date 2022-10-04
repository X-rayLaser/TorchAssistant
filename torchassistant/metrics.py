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
        """

        :param name: A string that identifies the metric
        :param metric_fn: a callable that maps variable number of tensors to a scalar
        :param metric_args: tensor names of batch that are used to compute the metric on
        :type metric_args: iterable of type str
        :param transform_fn: a callable that transforms/reduces # of the metrics arguments
        :param device: compute device to run the metric computation on
        :type device: torch.device
        """
        self.name = name
        self.metric_fn = metric_fn
        self.metric_args = metric_args
        self.transform_fn = transform_fn
        self.device = device

    def rename_and_clone(self, new_name):
        return self.__class__(new_name, self.metric_fn, self.metric_args, self.transform_fn, self.device)

    def __call__(self, batch):
        """Given the batch of predictions and targets, compute the metric.

        The method will first extract all arguments that metric
        function expects from batch. Then it will apply transforms to them.
        Finally, it will place all tensors on a specified device and compute the metric value.

        Note that this method will also compute gradients with respect to tensors that participate in
        the metrics calculation. If this is not required, one should manually detach() tensors or
        make the call under no_grad() context manager.

        :param batch: a dictionary mapping a name of a metric argument to a corresponding tensor
        :type batch: Dict[str -> torch.tensor]
        :return: a metric scalar
        :rtype: degenerate tensor of shape ()
        """

        # todo: get rid of this (make it work with single argument)
        lookup_table = batch

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
