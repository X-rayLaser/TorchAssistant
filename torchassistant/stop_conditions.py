class StopCondition:
    def __call__(self, epoch, history):
        return True


class EpochsCompleted(StopCondition):
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs

    def __call__(self, epoch, history):
        return epoch >= self.num_epochs


class EarlyStopping(StopCondition):
    def __init__(self, monitor, min_delta=0, patience=1):
        """
        :param monitor: name of loss/metric in history that will be monitored
        :param min_delta: smallest change in monitored quantity that counts as an improvement
        :param patience: maximum number of consecutive epochs without improvement
        """

        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

    def __call__(self, epoch, history):
        values = [metrics[self.monitor] for metrics in history]

        changes = [v - w for v, w in zip(values[:-1], values[1:])]

        plateau_len = 0
        for delta in changes:
            if delta < self.min_delta:
                plateau_len += 1
            else:
                plateau_len = 0

            if plateau_len > self.patience:
                return True

        return False
