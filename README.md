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
- code.py

training.json will contain a configuration of the training session.
code.py is an ordinary Python module where we can implement pieces
that we can refer to from within a config file. 
In particular, we have to implement following classes:
- model (a LeNet in our case)
- preprocessor

A model is a regular subclass of torch.nn.Module class.
One quirky detail to remember is a model's forward method needs to return
a tuple rather than a single tensor even when there is only one output tensor.
A preprocessor must be a subclass of Preprocessor implementing
a single method process. TorchAssistant uses this class to transform the data
before it is input to a neural net.

Let's go ahead and implement them. Open the module code.py.
Add the following code to define a model: 
```

```

Add the code below to define a Preprocessor:
```
```

After implementing all the pieces, we can begin writing a configuration file which
 is just a normal json file. Here we are not going to discuss all
the syntax details, but rather focus on the essential structure of the file.

The general structure of the config file: 
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

"session_dir" specifies a relative path to a folder used to store session data
(including model checkpoints).

"definitions" represents a list of different entities that can be referred to 
in later sections of the config file (e.g. in "pipelines" or "train" blocks).
Each definition has the following fields:
- name (name or id of the entity)
- group (which group the entity belongs to)
- spec (specification, a number of fields used to create entity)

"pipelines" specify how to construct different pipelines. A neat feature
of TorchAssistant is that one can construct multiple pipelines sharing the same
model(s) and interleave those pipelines during training.

Finally, there is "stages" entry.
It allows to create highly flexible multi-stage training setup where
different stages may use different training pipelines.
Each stage needs to specify training pipeline, evaluation pipeline and 
stopping condition.

Open training.json file and copy the following text as shown below
```
{
    "session_dir": "pretrained",
    "initialize": {
        "definitions": [
        ]
    }
}
```

Now we can initiate a new training session by issuing a command

```
python init.py my_examples/mnist/training.json
```

# License

This project is licensed under MIT license.