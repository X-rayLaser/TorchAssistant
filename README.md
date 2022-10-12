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

# Usage

The typical workflow consists of the following steps:
- write Python/Pytorch code to implement custom models, datasets, etc.
- write a specification (a spec) file that configures the training process
(by defining optimizers, loss functions, metrics and other parameters)
- run init.py to parse that file and create a training session 
saved in a chosen directory
- run train.py to take a training session directory and begin/resume 
training
- write another specification file that configures evaluation of models
- run evaluate.py to validate performance of a trained model against a 
given set of metrics
- write yet another specification file configuring inference process
- run infer.py to manually test the model by feeding it raw input data and
observing predicted output

Once you have a training spec file, create a training session by 
running a command:
```
python init.py <path_to_specification_file>
```

This should create a training session folder in a location specified in
a spec file.

Start training by running a command (pass a location of the training
session folder as an argument):
```
python train.py <path_to_training_session_folder>
```

States of all models and optimizers are saved automatically whenever
another epoch of training ends.
That means, at any moment you may safely interrupt the script by 
pressing Ctrl+C. You can then resume training picking up from where 
you left off (from the last epoch that was interrupted)
by executing the above command.

To compute metrics on a trained model, run this command:
```
python evaluate.py <path_to_evaluation_specification_file>
```

Finally, to test model predictions on your own data, run the command:
```
python infer.py <path_to_inference_specification_file> input1 input2 ... inputn
```
First argument is the path to specification file for inference, 
the rest are variable number of user inputs. Usually, you have to 
write a converter class for each of those inputs that specifies how to
turn them into a format consumable by the prediction pipeline.

To learn more about the details of the format of different specification
files, see the section on specification format below.

# How it works

Before getting to details of specification format, it is useful to 
understand the inner workings of TorchAssistant and glance
over its major components and concepts. Many of those (such as loss functions,
optimizers, models, etc.) will look familiar, while others will not.

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
decorated torch DataLoader instances to combine their values generated on every 
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

That was a high level intro. Now let's discuss in detail different 
entities participating in the training process.

## Data related entities
First, let's look at data related entities.
Very briefly, datasets provide individual examples as tuples.
A data split can be defined to split dataset into training, 
validation and test datasets.
Collators turn a collections of examples into batches.
By giving names to columns of batches, the latter become data frames.
Finally, input injector groups data frames coming from different datasets 
and convert them to data frame dicts. We will discuss it later.

### Dataset
Dataset can be any object that implements a sequence protocol. Concretely, 
dataset class needs to implement `__len__` and `__getitem__`. Naturally,
any built-in dataset class from torchvision package is a valid dataset.
Moreover, datasets can wrap/decorate other datasets and have 
associated data transformations/preprocessors.


### Data split
Data split is used to randomly split a given dataset into 2 or more slices.
One can control relative size of each slice in the split.
After creation, one can access slices of a given dataset via a dot notation.
For instance, if a dataset called `mnist` was split into 2 parts, we can
refer to a training slice with ```mnist.train```.
Each slice is a special kind of dataset. That means, data slice can be used
anywhere where dataset is needed.

### Preprocessor
Preprocessor is any object which implements `process(value)` method.
Implementing an optional method `fit(dataset)` makes a given preprocessor
learnable. Whether it's learnable or not, once defined, it can be applied
to a given dataset or a dataset slice.


### Collator
Collator is a callable object which turns a list of tuples (examples) 
into a tuple of collections (lists, tensors, etc.). It is useful, when
one needs to apply extra preprocessing on a collection of examples rather than
individually 1 example at a time. For instance, in Machine Translation
datasets often contain sentences with variable length. This makes training 
with batch size > 1 problematic. One possible solution is to 
pad sentences of a collection of examples in a collator.

### Data frame
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

### Data loader
Data loader represents Pytorch DataLoader class.

### Input injector 
Input injector is an iterator yielding data frames ready to be injected into 
a processing graph. Input injector is used on every training iteration to
provide training examples to learn from and compute loss/metrics.
It's purpose will become more clear when we look at remaining pieces.

## Entities related to training

Let's go over familiar ones first and 
discuss entities specific to TorchAssistant later.

### Model
Model is a subclass of torch.nn.Module class.

### Optimizer
Optimizer is a built-in optimizer class in torch.optim module.

### Loss
Loss is a built-in loss function class (e.g. CrossEntropyLoss) in torch.nn
module.

### Metric
Metric is a built-in class from torchmetrics package.

### Batch processor
Batch processor performs a particular computation on its inputs and produces
some outputs. Typically, (but not always), 
it is a graph of nodes where each node is a model (neural network).
This abstraction allows to create quite sophisticated computational graphs
where output from one or more neural nets becomes an input to others.
For example, it makes it easy to create an encoder-decoder RNN architecture.

### Processing graph
Processing graph is a higher-level graph whose nodes are batch processors.
Since each batch processor is itself a graph, processing graph is graph 
of graphs. Processing graph has special input nodes that are called 
input ports. Each port has a name. In order to run a computation on this graph,
one needs to send a dictionary that maps port names to corresponding data frames.
Input injector is the entity that generates inputs and sends them
to the appropriate input ports.
Processing graphs allow to create even complex training pipelines.
But in simple cases, they may be omitted.

### Pipeline
Pipeline is a self-contained entity that describes computational
graph, the source of training data, which losses and metrics to compute.
In other words, it fully describes a single iteration of a training loop.
It is easy to create multiple pipelines where different pipelines may
use different processing graphs, datasets, losses and metrics.

### Training stage
A (training) stage is a self-contained specification of the entire training
process. Basically, it specifies which pipeline to use for training and 
validation as well as a stopping condition. The latter serves to determine
when the stage is done. The simplest stopping condition just sets the 
number of epochs before completion.
It is possible to create multiple stages, each of them having different
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
              "variable_names": [...],
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
        "training_pipelines": [...],
        "validation_pipelines": [...],
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
to run a single training iteration.

Finally, there is "train" block which has only one entry: "stages".
"stages" defines a list of stages. Each stage specifies which pipeline 
to use for training and validation as well as a stopping criterion.

We are going to start with "definitions" section.

### Definitions section

"definitions" represents a list of different entities that can be referred to 
in later sections of the config file (e.g. in "pipelines" or "train" blocks).
Each definition has the following fields:
- name 
- group
- spec

"name" defines a name/id of the entity. It is used as a lookup key
to find and refer to this entity from other sections of the spec file.

"group" defines a group the entity belongs to. The group also determines
the type of the entity.

"spec" stores a set of fields actually configuring 
the entity. In most cases, "spec" describes how to instantiate 
a particular class. The format of the "spec" section is determined by the 
"group" field. Different types of entities may have different format 
of the "spec" section.

The format of "spec" field of every type of entities is shown below.

#### datasets

This group defines and build dataset instances. As a reminder, any
class implementing methods ```__len__``` and ```__getitem__``` is a valid
datasets.

In the simplest case, "spec" only requires to fill a mandatory field 
"class". "class" value has to be a fully-qualified path to
the Python class relative to the current working directory from which
init.py script will be executed. For instance, if the current working
directory contains a Python module ```my_module``` where a dataset class
MyDataset is defined, then the "spec" should look as follows:
```
{
    "class": "my_module.MyDataset"
}
```

If an ```__init__``` method of a
Python class takes positional or keyword arguments, those have to 
specified by the "spec". You can pass positional arguments by providing
them in an array to optional "args" field of "spec":
```
"args": ["arg1", "arg2"]
```

Or you can pass parameters as named keyword arguments 
by filling an optional field "kwargs":
```
"kwargs": {
    "a": 1,
    "b": 2
}
```

Here is a complete example of a dataset definition:
```
{
    "group": "datasets"
    "name": "my_dataset",
    "spec": {
      "class": "my_module.MyDataset",
      "args": ["arg1", "arg2"],
      "kwargs": {
        "a": 1,
        "b": 2
      }
    }
}
```

But that's not all. As was mentioned, we can define a new dataset by
wrapping (decorating) previously defined one.
To do so we can use fields "link" and "preprocessors". 
"link" value should contain a name of the dataset we decorate.
"preprocessors" value is an array of preprocessor names (to that we
have to first define each preprocessor). 
The definition for such a decorated dataset could look like this
(provided that entities named "my_dataset", "my_preprocessor1", 
"my_preprocessor2" were already defined):
```
{
    "group": "datasets",
    "name": "decorated_dataset",
    "spec": {
        "link": "my_dataset",
        "preprocessors": ["my_preprocessor1", "my_preprocessor2"]
    }
}
```

In some cases, parameters that we need to pass to some entity are
unknown because their values are computed dynamically by 
previously instantiated entities. In such a case, we need to 
programmatically configure an entity based on state of 
previously defined one. In order to accomplish that fill "factory_fn" 
field instead of "class" field. "factory_fn" value has to be a fully
qualified path to a factory function that is function that builds and
returns an object. Upon invocation, this function always receives
an instance of Session as a first argument. Session object gives
access to all the entities that were defined before the entity currently
being created. The rest of the signature can be empty, contain 
positional or keyword arguments.

Assuming that you have the factory function, 
```
def some_factory_fn(session, arg1, a=1):
```
the "spec" should look as follows: 
```
"spec": {
    "factory_fn": "my_module.some_factory_fn",
    "args": ["arg1"]
    "kwargs": {"a": 1}
}
```

### splits

As the name suggests, this group is used to define data splits. 

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