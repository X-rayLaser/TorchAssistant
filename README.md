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

# License

This project has an MIT license.