# Introduction

**TorchAssistant** is a deep learning framework built on top of **PyTorch**. 
It provides a set of tools to automate the training and evaluation of models. 
It also reduces the amount of trivial code one usually needs to write.

Just create a specification file configuring the training session and let 
the framework do everything else for you.

Main features:
- scripts for training, evaluation, and inference
- automatically calculating metrics
- automatically saving training session
- resuming the interrupted training session
- automatically saving metrics history to CSV files
- nicely formatted information about the training session: epoch, iteration, loss, metrics, etc.
- highly flexible and customizable
- support for building complex training pipelines

# Status: Early development stage

This project is in the early stage of development.
Features and functionality provided here are subject to change.
Furthermore, the code is not yet extensively tested and may contain bugs.

# Prerequisites

This project has dependencies that require separate installation:
- PyTorch (version >= 1.10.1, < 2.0)
- Torchvision (version >= 0.11.2, < 0.12)
- TorchMetrics (version >= 0.7.2 < 0.8)

When possible, try to follow the recommended version range specified in parentheses.

You can install PyTorch and Torchvision together from 
[here](https://pytorch.org/get-started/locally/).
And you install TorchMetrics from 
[here](https://torchmetrics.readthedocs.io/en/stable/pages/quickstart.html).

# Installation

```
pip install torchassistant
```

# Examples

The examples directory contains projects that demonstrate how to use
TorchAssistant to train different kinds of neural networks.

# Documentation

You can find all the documentation for the project 
[here](https://github.com/X-rayLaser/TorchAssistant/wiki).

# Usage

### Create a new training session configured by a training spec file:
```
tainit <path_to_specification_file>
```
It should create a training session folder in a location specified in
a spec file.

### Start training (pass a location of the training session folder as an argument):
```
tatrain <path_to_training_session_folder>
```

The script automatically saves the states of all models and optimizers 
at the end of every training epoch. That means that you may safely 
interrupt this script. You can then resume training picking up from 
where you left off by executing the above command.

### Compute metrics on a trained model (pass a path to a specification file for evaluation):
```
taevaluate <path_to_evaluation_specification_file>
```
The script expects the specification file to contain the location of the 
training session directory.

### Test model predictions on your data:
```
tainfer <path_to_inference_specification_file> input1 input2 ... inputn
```
The first argument is the path to the specification file for inference.
The rest are a variable number of user inputs. Usually, for each of those inputs, 
you have to write a converter class that specifies how to
turn them into a format consumable by the prediction pipeline.

To learn more about the details of the format of different specification
files, see the section on the specification in the documentation.

# License

This project has an MIT license.