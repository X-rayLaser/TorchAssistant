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

As the name suggests, this class prepares data to be consumed by
neural pipeline. 


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
similar format for all definitions. It needs a fully qualified class name and,
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
"train" key-word argument False.

Next thing to define is a collator. Basically, collator is a callable
that can be passed as an optional key-word argument "collate_fn" of 
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

Let's now define a loader factory for training dataset
which is essentially a factory object creating instances of torch DataLoader:
```
            {
                "group": "loader_factories",
                "name": "training_factory",
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

A loader factory for test dataset looks almost the same:
```
            {
                "group": "loader_factories",
                "name": "test_factory",
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
LeNet5 class that we defined :
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

# How it works

Before getting to details of specification format, it is useful to 
understand the inner workings of TorchAssistant and glance
over its major components and concepts. Many of those (such as loss functions,
optimizers, models, etc.) will look familiar, while others will not.

Let's discuss those concepts starting from the lowest level first.

First, let's look at data related entities.
Very briefly, datasets provide individual examples as tuples.
A data split can be defined to split dataset into training, 
validation and test datasets.
Collators turn a collection of examples into batches.
By giving names to columns of batches, the latter become data frames.
Finally, data injectors group data frames coming from different datasets 
and convert them to data frame dicts. We will discuss them later.

Dataset is any object that implements a sequence protocol. Concretely, 
dataset class needs to implement `__len__` and `__getitem__`. Naturally,
iny built-in dataset class from torchvision package is a valid dataset.
Moreover, datasets can wrap/decorate other datasets and have 
associated data transformations/preprocessors.

Data split is used to randomly split a given dataset into 2 or more slices.
One can control relative size of each slice in the split.
Upon creation, one can access slices of a given dataset via a dot notation.
For instance, if a dataset called `mnist` was split into 2 parts, we can
refer to training slice with ```mnist.train```.
Each slice is a special kind of dataset. That means, data slice can be used
anywhere where dataset is needed.

Preprocessor is any object which implement `process(value)` method.
Implementing an optional method `fit(dataset)` makes a give preprocessor
learnable. Whether it's learnable or not, once defined, it can be applied
to a given dataset or a dataset slice.

Collator is a callable object which turns a list of tuples (examples) 
into a tuple of collections (lists, tensors, etc.). It is useful, when
one needs to apply extra preprocessing on a collection of examples rather than
individually 1 example at a time. For example, in Machine Translation
datasets often contain sentences with variable length. This makes training 
with batch size > 1 problematic. One possible solution is to 
pad a collection of examples in a collator.

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

Data injector is an iterator yielding data frames ready to be injected into 
a processing graph. Data injector is used on every training iteration to
provide training examples to learn from and compute loss/metrics.
It's purpose will become more clear when we look at remaining pieces.

Batch processor is a graph of nodes where each node is a neural network.
This abstraction allows to create quite sophisticated computational graphs
where output from one or more neural nets becomes an input for others.


For example, it is easy to create an encoder-decoder RNN architecture.

Data injector is glorified data loader




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



Now we can initiate a new training session by issuing a command

```
python init.py my_examples/mnist/training.json
```

# License

This project is licensed under MIT license.