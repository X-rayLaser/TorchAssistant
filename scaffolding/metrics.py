import torch


def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total


def accuracy_percentage(outputs, labels):
    return accuracy(outputs, labels) * 100


metric_functions = {
    'loss': None,
    'accuracy': accuracy,
    'accuracy %': accuracy_percentage
}
