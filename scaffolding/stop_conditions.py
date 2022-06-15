class StopCondition:
    def __call__(self, *args, **kwargs):
        return True


class EpochsCompleted(StopCondition):
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs

    def __call__(self, epoch, history):
        return epoch >= self.num_epochs
