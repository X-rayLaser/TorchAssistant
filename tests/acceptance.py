import unittest

import torch.optim
from torch.utils.data import DataLoader
from torch import nn

from torchvision.datasets.mnist import MNIST
from torchvision.transforms import transforms

from torchmetrics import Accuracy

from torchassistant.training import train
from torchassistant.session import Session
from torchassistant.session.data_classes import TrainingPipeline, InputLoader
from torchassistant.collators import StackTensors
from torchassistant.data import LoaderFactory
from torchassistant.processing_graph import NeuralBatchProcessor, BatchProcessingGraph, Node
from torchassistant.output_adapters import IdentityAdapter


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(1, 6, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Flatten(),
            nn.Linear(49, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

        self.conv1 = nn

    def forward(self, x):
        return self.sequential(x)


class TrainingOnMnist(unittest.TestCase):
    def setUp(self):
        self.session_dir = "pretrained"

    def test_training_evaluation_and_inference(self):
        session_dir = self.session_dir

        session = Session()

        mnist_training = MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
        mnist_validation = MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
        session.datasets = dict(training_ds=mnist_training, validation_ds=mnist_validation)
        session.collators = dict(simple_collator=StackTensors())

        training_loader = DataLoader(mnist_training, collate_fn=StackTensors(), batch_size=32)
        validation_loader = DataLoader(mnist_validation, collate_fn=StackTensors(), batch_size=32)

        session.data_loaders = dict(training_loader=training_loader, validation_loader=validation_loader)

        training_factory = LoaderFactory(mnist_training, StackTensors(), batch_size=32)
        validation_factory = LoaderFactory(mnist_validation, StackTensors(), batch_size=16)

        model = TestModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        session.models = dict(mnist_model=model)
        session.optimizers = dict(optimizer=optimizer)

        criterion = torch.nn.LogSoftmax()
        session.losses = dict(ce=criterion)
        session.metrics = dict(acc=Accuracy())

        neural_nodes = [Node('mnist_classifier', model, optimizer, inputs=['x'], outputs=['y_hat'])]

        def input_adapter(batch):
            return dict(mnist_classifier={'x': batch['x']})

        output_adapter = IdentityAdapter()

        batch_processor = NeuralBatchProcessor(neural_nodes, input_adapter, output_adapter, device='cpu')
        input_loaders = [InputLoader('input_batch', training_factory, variable_names=['x'])]

        graph = BatchProcessingGraph(['input_batch'], classifier=batch_processor)
        graph.make_edge('input_batch', 'classifier')
        train_pipeline = TrainingPipeline(graph, input_loaders, loss_fns, metric_fns)

        pipelines = dict(train_pipeline=train_pipeline)

        session.pipelines = pipelines

        log = []

        def log_fn(stage_number, epoch, computed_metrics):
            log.append((stage_number, epoch, computed_metrics))

        train(session, log_fn, save_checkpoint=lambda epoch: epoch, stat_ivl=1)
