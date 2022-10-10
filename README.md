# Introduction

TorchAssistant is a deep learning framework on top of PyTorch intended 
to accelerate prototyping (and training) of neural networks as well as testing new ideas. 
It provides a set of tools to automate training and evaluation of models. 
It also reduces the amount of stereotypical code that is typically reinvented 
across many projects.

## Main features:
- scripts for training, evaluation and inference
- training loop
- progress bar
- metrics
- automatic saving and resuming of training session
- saving metrics history to csv files
- configure training session with config file
- highly flexibility and customizable
- support for building complex training pipelines

# Examples

Examples directory contains projects that demonstrate how to use
TorchAssistant to train different kinds of neural networks.

# Quick start

Here we're going to use TorchAssistant to train a LeNet convolutional
neural network on MNIST dataset.

Let's create a new folder my_examples in the project directory.
Now, under my_examples create a directory mnist. Within that
directory create files:
- training.json
- ```__init__.py```

training.json will contain a configuration of the training session.
```__init__.py``` is an ordinary Python module where we can implement functions and classes 
that we can refer to from within a config file.

Open file code.py in the editor (or IDE).

First we import the following modules:
```
import torch
from torch import nn
from torchvision.transforms import ToTensor
from torchassistant.collators import BatchDivide
```

Then we implement a convolutional neural network with LeNet-5 architecture:
```
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, padding="same")
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return [x]
```

Note that we return [x] instead of x from the method.
One quirky detail to remember is that a model's forward method has to return
a list/tuple rather than a single tensor even when there is only one output tensor.


Now let's implement 2 functions:
```
def reverse_onehot(y_hat, ground_true):
    return y_hat.argmax(dim=-1), torch.LongTensor(ground_true)


def convert_labels(y_hat, labels):
    labels = torch.LongTensor(labels)
    return y_hat, labels
```

The purpose of these functions is as follows.
Typically, one iteration of learning consists of forward pass and backward pass.
In forward pass, we use neural network to make a prediction y_hat.
Then, we compute loss by passing predicted and target tensors to a loss function. 
However, it can be useful to transform prediction and/or target tensor before
computing the loss. The same applies to computing metrics.
We'll get back here and explain their implementation later.

Finally, copy the following class definition below:
```
class MyCollator(BatchDivide):
    def __call__(self, batch):
        to_tensor = ToTensor()

        images, labels = super().__call__(batch)

        x = torch.stack([to_tensor(image) for image in images])
        x = x / 255
        return x, labels
```
Let's break this down.
This class collates a bunch of examples. Each example is a pair (image, label),
where image is PIL image depicting a MNIST digit, label is an integer from 0 
to 9. Thus, batch is a list of pairs (image, label). Note that we inherit from
BatchDivide class. Its implementation converts a list of tuples into a tuple of
lists so that ```images``` variable will contain only list of PIL images
and ```lables``` variable will contain a list of integers. 
To invoke this implementation, we use the keyword ```super```.
Then, we convert each PIL image into a tensor and stack tensors along the
batch dimension so that the final tensor x has the shape 
(batch_size, 1, 28, 28).
Finally, we perform very basic normalization by dividing x by 255.

After implementing all the pieces, we can begin writing a specification file which
 is just a normal json file. Here we are not going to discuss all
the syntax details, but rather focus on the essential elements of the file. 
The specification file we are about to create can be adopted 
for many relatively simple training setups.

Open training.json file and copy the following text as shown below
```
{
    "session_dir": "pretrained",
```

Here we tell TorchAssistant to store all session data in the directory 
"pretrained" using relative path. If directory does not exist yet, it will be
created.

Now we are going to fill in the definitions section. First of all, we need to
specify a dataset used to train our model. Add the following code 
under "session_dir":
```
    "initialize": {
        "definitions": [
            {
                "group": "datasets",
                "name": "train_ds",
                "spec": {
                    "class": "torchvision.datasets.MNIST",
                    "args": ["data"],
                    "kwargs": {
                        "train": true,
                        "download": true
                    }
                }
            },
```

This definition says that we define an entity that belongs to the group
"datasets" and we give it a name "train_ds". We also provide a specification
to build a dataset object in the "spec" section. The spec section has a
similar format for all definitions. It expects a fully qualified class name and,
optionally, arguments. We can pass a list of positional arguments together
with key-word arguments.
Behind the scene, the fields under "spec" will be interpreted 
as the following code:
```
torchvision.datasets.MNIST("data", train=True, download=True)
```

Here we could pass a fully qualified path to our
own dataset class, but we are going to use built-in MNIST dataset
from torchvision library instead.

Now, let's define a test dataset: 
```
            {
                "group": "datasets",
                "name": "test_ds",
                "spec": {
                    "class": "torchvision.datasets.MNIST",
                    "args": ["data"],
                    "kwargs": {
                        "train": false,
                        "download": true
                    }
                }
            },
```

This definition looks almost the same, except that now we set 
"train" key-word argument to False.

Next thing to define is a collator. Basically, collator is a callable
that can be passed as an optional key-word argument "collate_fn" to the 
torch DataLoader.
```
            {
                "group": "collators",
                "name": "my_collator",
                "spec": {
                    "class": "torchassistant.collators.BatchDivide"
                }
            },
```

Let's now define a data loader for training dataset
which is essentially the same as a torch DataLoader:
```
            {
                "group": "data_loaders",
                "name": "training_loader",
                "spec": {
                    "dataset": "train_ds",
                    "collator": "my_collator",
                    "kwargs": {
                        "batch_size": 32,
                        "num_workers": 2
                    }
                }
            },
```

Key-word arguments specified in the spec section correspond to key-word
arguments passed to DataLoader upon instantiating it.

A data loader for test dataset looks almost the same:
```
            {
                "group": "data_loaders",
                "name": "test_loader",
                "spec": {
                    "dataset": "test_ds",
                    "collator": "my_collator",
                    "kwargs": {
                        "batch_size": 32,
                        "num_workers": 2
                    }
                }
            },
```

Now let's define a model by plugging it a fully qualified path to our
LeNet5 class that we have defined:
```
            {
                "group": "models",
                "name": "LeNet5",
                "spec": {
                    "class": "my_examples.mnist.LeNet5"
                }
            },
```

Here we specify an optimizer to use during the training:
```
            {
                "group": "optimizers",
                "name": "optimizer",
                "spec": {
                    "class": "Adam",
                    "kwargs": {
                        "lr": 0.01
                    },
                    "model": "LeNet5"
                }
            },
```
This definition says that we want to use Adam optimizer class optimizing
parameters of the model LeNet5 using learning rate 0.01.

Now let's define the loss function:
```
            {
                "group": "losses",
                "name": "cross_entropy",
                "spec": {
                    "class": "CrossEntropyLoss",
                    "inputs": ["y_hat", "input_2"],
                    "transform": "my_examples.mnist.convert_labels"
                }
            },
```
This definition says that we want to use CrossEntropyLoss class as
a loss function. We also specify the names of data pieces that loss
will use for its computation. We will get back to this later, but 
when we do not explicitly give names to the elements of training example, 
TorchAssistant will automatically name them as: 
"input_1", "input_2", ..., "input_n".
Also, by default, the output from the model is called "y_hat".
We want to compute loss between predicted tensors and targets.
Thus, inputs to the loss must be "y_hat" and "input_2".
See the Specification section to learn more.

We also need to fill an extra option "transform" to specify a transform
function we want to apply to the pair (y_hat, targets). The reason for that
is that targets is just a list, whereas CrossEntropyLoss expects targets 
to be a tensor.

Finally, we define an accuracy metric:
```
            {
                "group": "metrics",
                "name": "accuracy",
                "spec": {
                    "class": "Accuracy",
                    "inputs": ["y_hat", "input_2"],
                    "transform": "my_examples.mnist.reverse_onehot"
                }
            }
```
Here, "Accuracy" is a metrics class from torchmetrics package. "inputs"
here is the same as in the previous definition. Now in order to compute
the accuracy, we need 2 tensors of the same lengths: predicted labels and 
ground true labels. To ensure that, we set "transform" to 
"my_examples.mnist.reverse_onehot".

We are done with the definitions section. Now we need to fill in the
pipelines section.



# How it works

Before getting to details of specification format, it is useful to 
understand the inner workings of TorchAssistant and glance
over its major components and concepts. Many of those (such as loss functions,
optimizers, models, etc.) will look familiar, while others will not.

The typical workflow consists of the following steps:
- run init.py to parse a specification file for the training session 
and save the session data in a particular directory
- run train.py to take a given specification and begins/resumes training
- run evaluate.py to validate a trained model performance against a 
given set of metrics
- run infer.py to manually test the model by feeding it raw input data and
observing predicted output

We are going to focus here only on mechanics of train.py script and what it does.

The major components are (neural) batch processor, processing graph,
input injector.

Batch processor can be thought of as a computational unit that converts
a bunch of incoming data. Batch processor can be any object that implements
a certain interface, but typically it contains a number of models/neural nets
connected in some way.

A processing graph is a graph whose nodes are batch processors.
Each leaf node of the graph can participate in metric computation.
In fact, different set of metrics can be applied to different leaf nodes.

Input injector sends prepared pieces of training batch to the right 
input nodes of the processing graph. Essentially, it wraps a bunch of 
torch decorated DataLoader instances to combine their values generated on every 
iteration step.

Input injector does its work by delegating it to the data loaders.
Each of data loaders does its thing, then their results are aggregated in
some way and returned.

Behind the curtain, here are the steps that each of data loaders performs:
- load the next group of raw examples from a dataset, perform necessary data 
transforms
- collate those examples into a batch (usually a tuple of tensors),
perform batch level preprocessing (such as sequence padding) if necessary
- create a data frame by associating each item in a batch tuple with a name
e.g., a batch ```([[1, 0], [0, 0]], [[1], [0]])``` becomes 
```{'input_1': [[1, 0], [0, 0]], 'input_2': [[1], [0]]}```

As soon as all data loaders finish their iteration, input injector aggregates
their outputs into a single dictionary. Concretely, it maps each data frame 
to the right input node of processing graph, e.g.:
```{'graph_input_1': data_frame1, 'graph_input_2: data_frame2}```

With that description in mind, a single iteration of a training loop
involves the following steps:
- prepare the next batch of training data by input injector (as a mapping 
from graph_input_node to a data frame)
- send the batch to the processing graph
- compute predictions for each leaf node
- compute loss and metrics for each node where loss or metrics were specified
- compute the gradient and update parameters for all models

A training epoch consists of running the above for a bunch of times. 
After input injector iterates over all of its training batches, 
the epoch is done.
Whenever another epoch finishes, the framework computes loss and metrics
separately on training and validation datasets. Computed metrics are then
printed to stdout and appended to a history file in csv format.
Also, at this point parameters of all models and their optimizers are 
automatically saved. This allows one to resume training from the last finished 
epoch if the training script was interrupted.

Final important detail about TorchAssistance is that the training process 
is split into multiple stages. Although 1 stage will suffice for most cases,
multiple stages give an extra flexibility. Different stages can use different 
processing graphs, datasets, metrics, etc.
The stage will finish when the stopping condition checked after every epoch
returns True.

Let's discuss those concepts starting from the lowest level first.

## Data related entities
First, let's look at data related entities.
Very briefly, datasets provide individual examples as tuples.
A data split can be defined to split dataset into training, 
validation and test datasets.
Collators turn a collections of examples into batches.
By giving names to columns of batches, the latter become data frames.
Finally, input injector groups data frames coming from different datasets 
and convert them to data frame dicts. We will discuss it later.

Dataset can be any object that implements a sequence protocol. Concretely, 
dataset class needs to implement `__len__` and `__getitem__`. Naturally,
iny built-in dataset class from torchvision package is a valid dataset.
Moreover, datasets can wrap/decorate other datasets and have 
associated data transformations/preprocessors.

Data split is used to randomly split a given dataset into 2 or more slices.
One can control relative size of each slice in the split.
After creation, one can access slices of a given dataset via a dot notation.
For instance, if a dataset called `mnist` was split into 2 parts, we can
refer to a training slice with ```mnist.train```.
Each slice is a special kind of dataset. That means, data slice can be used
anywhere where dataset is needed.

Preprocessor is any object which implements `process(value)` method.
Implementing an optional method `fit(dataset)` makes a given preprocessor
learnable. Whether it's learnable or not, once defined, it can be applied
to a given dataset or a dataset slice.

Collator is a callable object which turns a list of tuples (examples) 
into a tuple of collections (lists, tensors, etc.). It is useful, when
one needs to apply extra preprocessing on a collection of examples rather than
individually 1 example at a time. For instance, in Machine Translation
datasets often contain sentences with variable length. This makes training 
with batch size > 1 problematic. One possible solution is to 
pad sentences of a collection of examples in a collator.

Data frame is dict-like data structure which essentially associates names with
data collections (lists or tensors). Note that it has nothing to do with 
entities called data frames in other libraries such as Pandas.
For example, this is a valid data frame:
```
data_frame = {
  "x1": [1, 2, 3],
  "x2": [10, 20, 30]
}
```

Data loader represents Pytorch DataLoader class.

Input injector is an iterator yielding data frames ready to be injected into 
a processing graph. Data injector is used on every training iteration to
provide training examples to learn from and compute loss/metrics.
It's purpose will become more clear when we look at remaining pieces.

## Entities related for training

Let's go over familiar ones first.

Model is a subclass of torch.nn.Module class.

Optimizer is a built-in optimizer class in torch.optim module.

Loss is a built-in loss function class (e.g. CrossEntropyLoss) in torch.nn
module.

Metric is a built-in class from torchmetrics package.

Now let's look entities specific to TorchAssistant.

Batch processor performs a particular computation on its inputs and produces
some outputs. Typically (but not always), 
it is a graph of nodes where each node is a model (neural network).
This abstraction allows to create quite sophisticated computational graphs
where output from one or more neural nets becomes an input to others.
For example, it is easy to create an encoder-decoder RNN architecture.

Processing graph is a higher-level graph whose nodes are batch processors.
Since each batch processor is itself a graph, processing graph is graph 
of graphs. Processing graph has special input nodes that are called 
input ports. Each port has a name. In order to run a computation on this graph,
one needs to send a dictionary that maps port names to corresponding data frames.
Input injector is the entity that generates inputs and sends them
to the appropriate input ports.
Processing graphs allow to create even complex training pipelines.
But in simple cases, they may be omitted.

Pipeline is a self-contained specification that describes computational
graph, source of training data, which losses and metrics to compute.
In other words, it fully describes a single iteration in a training loop.
It is easy to create multiple pipelines where different pipelines may
use different processing graphs, datasets, losses and metrics.

A (training) stage is a self-contained specification of the entire training
process. Basically, it specifies which pipeline to use for training and 
validation and a stopping condition. The latter serves to determine
when the stage is done. The simplest stopping condition just sets the 
number of epochs for completion.
It is possible to create multiple stages, each of them with different
pipelines and stopping conditions.


# Specification format

Specification (or spec) file is an ordinary json file that is used by the framework
to configure it to perform training, evaluation, inference and other tasks.
For every task there needs to be a separate json file.
You will want to create at least 2 specs: 1 for training and the
other for inference.

## Training spec

The overall structure of training spec file is as follows:
```
{
  "session_dir": "pretrained",
  "initialize": {
    "definitions": [
      {
        "group": "some_group",
        "name": "some_name",
        "spec": {...}
      },
      ...
    ],
    "pipelines": {
      "pipeline1": {
        "graph": "...",
        "input_factories": [
          {
              "input_alias": "....",
              "variable_names": ["...", "..."],
              "loader_factory": "..."
          },
          ...
        ]
      },
      "pipeline2": {...},
      "pipeline10": {...}
    }
  },
  "train": {
    "stages": [
      {
        "mode": "...",
        "training_pipelines": ["..."],
        "validation_pipelines": ["...", "..."],
        "stop_condition": {
            "class": "...",
        }
      },
      {...}
    ]
  }
}
```

Any training spec file has to have the following blocks/options filled in:
"session_dir", "initialize", and "train".

"session_dir" specifies a relative path to a folder used to store session data
(including model checkpoints).

"initialize" block is where the bulk of configuration options is stored.
It consists of 2 sub blocks: "definitions" and "pipelines".
The former defines an individual pieces such as datasets, preprocessing,
models, optimizers and more. The latter, as the name suggests, defines pipelines
by combining and referring to pieces in "definitions" block.
Pipelines are self-contained entities containing all the information necessary
to form a training loop.

Finally, there is "train" block which has only one entry "stages".
"stages" defines a list of stages. Each stage specifies which pipeline 
to use for training and validation and when to stop.

The most important sections of the file are "session_dir", "definitions",
"pipelines" and "stages".


"definitions" represents a list of different entities that can be referred to 
in later sections of the config file (e.g. in "pipelines" or "train" blocks).
Each definition has the following fields:
- name (name or id of the entity)
- group (which group the entity belongs to)
- spec (specification consisting of a few fields used to create entity)

"pipelines" defines concrete pipelines which will be used during training. 
A neat feature of TorchAssistant is that one can construct multiple pipelines 
sharing the same model(s) and interleave those pipelines during training.

Finally, there is "stages" entry.
It allows to create highly flexible multi-stage training setup where
different stages may use different pipelines.
Each stage needs to specify training pipeline, evaluation pipeline and 
stopping condition.

### Batch processor
The next definition looks mysterious.
It defines a so-called batch processor object.
Essentially, it is a network of nodes where each node is a neural net. In other words,
batch processor is a network of neural nets. In our simple example, there is only one neural net.
But in more complex scenarios there may be a graph where each node may depend on a number of
other nodes and/or have many output tensors.

To specify a batch processor, we need to specify "input_adapter" and 
"neural_graph". We may also optionally specify "output_adapter" and 
"device" (CPU or CUDA).

Batch processor is passed inputs as a name->tensor mapping (or a data frame).
Normally, this dataframe will contain all the information needed to extract 
input tensors
expected by corresponding nodes (neural networks). Input adapter does this job.
It takes a dataframe and constructs from it input tensors for every neural network in the batch
processor.

The graph topology is defined by providing an ordered list of nodes such 
that dependent node should be listed after the node(s) it depends on. To 
specify each node, we need to provide:
- a name of a neural network defined earlier,
- a list of names of tensors used as inputs to the network
- a list of names of tensors predicted by the network
- a name of an optimizer defined earlier

Forward pass through the graph respects dependency. That is,
each node runs computation only after all nodes it depends on finished 
computation. Outputs from previous node become inputs to the dependent ones.
When a particular node finishes, it's outputs are added to the
dictionary of results (for example, if node defines its outputs as 
out1, out2, out3, the results dictionary will contain tensors for each 
of these output names).
When leaf nodes finish, we are done and forward pass through the whole batch 
processor object completes. All computation results are aggregated and saved 
in a dictionary (that includes computations on intermediate nodes too).

Here is how the definition of batch processor looks like for our example:
```
            {
                "group": "batch_processors",
                "name": "LeNet_processor",
                "spec": {
                    "input_adapter": {
                        "class": "my_examples.mnist.InputAdapter"
                    },
                    "neural_graph": [
                        {
                            "model_name": "LeNet5",
                            "inputs": ["x"],
                            "outputs": ["y_hat"],
                            "optimizer_name": "optimizer"
                        }
                    ],
                    "device": "cpu"
                }
            },
```

We use InputAdapter class that we implemented earlier as input adapter.
As a reminder, this is how it is implemented:
```
class InputAdapter:
    def __call__(self, dataframe: dict) -> dict:
        images = dataframe["images"]
        to_tensor = ToTensor()
        return {
            "LeNet5": {
                "x": torch.stack([to_tensor(image) for image in images]) / 255.
            }
        }
```

InputAdapter class has to return a nested dictionary. It should be a mapping
from a name of node to a dataframe containing named input tensors
expected by this node. Our network consists of just one node which is neural
network with named "LeNet5". It takes only one tensor named "x" and
outputs a single tensor named "y_hat". Therefore, we need to return a nested
dictionary of the form
```
{
    "LeNet5: {
        "x": ...
    }
}
```

Now we can initiate a new training session by issuing a command

```
python init.py my_examples/mnist/training.json
```

# License

This project is licensed under MIT license.