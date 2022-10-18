# Introduction

TorchAssistant is a deep learning framework built on top of PyTorch intended 
to accelerate prototyping (and training) of neural networks as well as testing new ideas. 
It provides a set of tools to automate training and evaluation of models. 
It also reduces the amount of stereotypical code that is typically reinvented 
across many projects.

Main features:
- scripts for training, evaluation and inference
- training loop
- progress bar
- metrics
- automatic saving and resuming of training session
- saving metrics history to csv files
- highly flexibility and customizable
- support for building complex training pipelines

# Installation
```
pip install torchassistant
```

# Examples

Examples directory contains projects that demonstrate how to use
TorchAssistant to train different kinds of neural networks.

# Documentation

The documentation for the project can be found 
[here](https://github.com/X-rayLaser/TorchAssistant/wiki).

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
files, see the section on specification in documentation.

# License

This project is licensed under MIT license.