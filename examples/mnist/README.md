Here we're going to use TorchAssistant to train a LeNet convolutional
neural network on MNIST dataset.

Let's create a new folder my_examples in the project directory.
Now, under my_examples create a directory mnist. Within that
directory create files:
- training.json
- ```__init__.py```

training.json will contain a configuration of the training session.
```__init__.py``` is an ordinary Python module where we can implement 
functions and classes that we can refer to from within a config file.

## Writing Python code in ```__init__.py```

Open file ```__init__.py``` in the editor (or IDE).

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
 is just a normal json file.

## Writing a specification file training.json

Here we are not going to discuss all
the syntax details, but rather focus on the essential elements of the file. 
The specification file we are about to create can be adopted 
for many relatively simple training setups.

The specification file has the following general structure:
```
{
    "session_dir": "pretrained",
    "initialize": {
        "definitions": []
        "pipelines": {}
    },
    "train": {
        "stages": []
    }
}
```

"session_dir" is the path to a location of training session data.
"initialize" contains a set of pipelines defined in terms of entities
declared in "definitions" subsection. "stages" contains
a list of training stage objects. Each stage fully specifies a training loop
running for a number of epochs until a stopping condition is met.

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
                        "lr": 0.001
                    },
                    "model": "LeNet5"
                }
            },
```
This definition says that we want to use Adam optimizer class optimizing
parameters of the model LeNet5 using learning rate 0.001. If we had
a multiple models, each of them could have distinct optimizer.
Note this line:
```
"model": "LeNet5"
```
Here we simply refer to the model that we have defined earlier by its name 
(LeNet5). That means that entities may refer to other entities previously defined.

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
        ],
```
Here, "Accuracy" is a metrics class from torchmetrics package. "inputs"
here is the same as in the previous definition. Now in order to compute
the accuracy, we need 2 tensors of the same lengths: predicted labels and 
ground true labels. To ensure that, we set "transform" to 
"my_examples.mnist.reverse_onehot".

We are done with the definitions section. Now we need to fill in the
pipelines section:
```
        "pipelines": {
```

Each pipeline completely defines a single iteration of training.
That is it knows how to learn from one batch. Typically, one would create
2 pipelines: one for training, the other for validation. But that's not a
requirement.
The body of the pipelines section is mapping from pipeline name to its
definition. This allows to refer to a particular pipeline by its name in the
later section of the json file.

Defining a training pipeline:
```
            "training_pipeline": {
                "input_injector": [
                    {
                        "data_loader": "training_loader"
                    }
                ],
                "losses": [
                    {
                        "loss_name": "cross_entropy",
                        "loss_display_name": "loss"
                    }
                ],
                "metrics": [
                    {
                        "metric_name": "accuracy",
                        "display_name": "acc"
                    }
                ]
            },
```
This pipeline specifies an input injector, loss functions and metrics.
Loss function definition expects a mandatory field "loss_name" which is
a reference (by name) to the loss function defined earlier in the definitions
section. "loss_display_name" is an optional fields that defines a displayed
name for the loss (it defaults to the value of "loss_name" field).
Definition for metric looks the same, except the fields are called 
"metric_name" and "display_name".

Input injector is essentially an iterator that wraps one or more
data loaders and combines their output. To configure it, we need to pass 
a reference to training data loader we have already defined ("training_loader").
Input injector spec also accepts optional "variable_names" field.
This field is a list of names of elements in the training example. 
When omitted, TorchAssistant automatically creates names as follows:
first element of the training example is called "input_1", second is called
"input_2", ..., n-th example is called "input_n".

In the definition of loss function above there was this line:
```
"inputs": ["y_hat", "input_2"]
```

Now we know what that mysterious "input_2" stands for. Since our dataset
consists of training examples of the form (image, label), "input_2" must
refer to the batch of labels. Likewise, "input_1" stands for batch of image
tensors.

Now it's time to define a pipeline for the purpose of testing the performance
of the model on test dataset:
```
            "evaluation_pipeline": {
                "input_injector": [
                    {
                        "data_loader": "test_loader"
                    }
                ],
                "losses": [
                    {
                        "loss_name": "cross_entropy",
                        "loss_display_name": "val loss"
                    }
                ],
                "metrics": [
                    {
                        "metric_name": "accuracy",
                        "display_name": "val acc"
                    }
                ]
            }
```
Here we use "test_loader" as a data loader. Loss and metrics definitions
are the same except for their displayed names.

You might have observed that the syntax specifies losses and metrics (plural) 
rather than loss and metric. It's logical that the same model can have 
multiple metrics associated with it, but how can there be multiple loss
functions for the same model?

The answer is the framework allows to define and
work with higher level abstractions than just models. Concretely,
models can be connected to form what's called a batch processor. Moreover,
batch processors can be combined to form a graph. And having done so,
one could apply different loss functions and metrics to the outputs
of different batch processors. But for a single model case, this is not 
necessary since the framework automatically builds these objects so that 
you don't have to worry about them.

With that we are done with pipelines section and with enclosing it initialize
section:
```
        }
    },
```

All that is left to do is to fill in the "stages" subsection of the file
enclosed by "train" section:
```
    "train": {
        "stages": [{
            "training_pipelines": ["training_pipeline"],
            "validation_pipelines": ["training_pipeline", "evaluation_pipeline"],
            "stop_condition": {
                "class": "torchassistant.stop_conditions.EpochsCompleted",
                "kwargs": {
                    "num_epochs": 20
                }
            }
        }]
    }
}
```

Stages allow to achieve higher flexibility. We can use them to 
dynamically vary training regimes by switching from one configuration
to the next. Each stage defines its own list of training and validation 
pipelines as well as stopping condition (a criterion which determines when
a given stage is done).

In our case, a single stage will suffice. The body of the stage requires
2 mandatory fields set: "training_pipelines" and "stop_condition".
The former defines a list of pipelines to train in parallel.
The latter defines a callable class used to tell whether it is time to stop
and mark completion of the stage.
An optional field "validation_pipelines" specifies which pipelines to use
to compute metrics after each training epoch.

For our case, we only have 1 training pipeline called "training_pipeline".
Therefore, we set "training_pipelines" to ```["training_pipeline"]```.
Regarding the validation, we want to compute and show metrics computed on
training and test datasets. Therefore, we set "validation_pipelines" to
```["training_pipeline", "evaluation_pipeline"]```.
Finally, for the field "stop_condition" we use TorchAssistant's built-in
class EpochsCompleted in ```torchassistant.stop_conditions``` module.

Now that the specification file is complete and can start training.

## Creating a training session

To create a fresh training session, run the following command:
```
python init.py my_examples/mnist/training.json
```

You should now see a new directory "pretrained" in the current working directory.
This directory stores all the necessary data. 
You shouldn't worry too much about its internals, but in case
you are curious, here is what it stores:
- copy of the original specification file
- checkpoints folder storing model weights after each epoch
- metrics saved in csv files
- other internal data

## Begin training
To begin training, execute the following command (make sure to run it 
from a directory containing folder "pretrained"):
```
python train.py pretrained
```

After executing it, you should see something like what is shown below:
```
Training stage 0 in progress
Epoch     1 [acc 0.8693, loss 0.4311] ===>................. 366 / 1875
```

You should see the following:
- current epoch number
- a set of computed metrics
- a progress bar, showing # of iterations done vs total # of iterations 

After first epoch completes, you should see a csv file located at 
pretrained/metrics/0.csv. This file stores training and validation metrics 
computed after every epoch in a comma-separated format.

## Evaluate performance on test dataset

Under my_examples/mnist, create and open a file "evaluation.json".
This is another specification file configuring process of evaluating 
performance of trained model/pipeline. Copy the following text inside the file:
```
{
    "session_dir": "pretrained",
    "validation_pipelines": ["training_pipeline", "evaluation_pipeline"]
}
```

We set "session_dir" to "pretrained" which is the folder that contains
the training session data. This field helps
 TorchAssistant access entities that we defined earlier such as models,
losses, pipelines, etc. This field also allows it to find location of the
most recent model checkpoint and load it.

We set "validation_pipelines" to 
```["training_pipeline", "evaluation_pipeline"]```. Thus, the evaluation
script will use these pipelines to compute the metrics defined in the 
training.json file.

Now in order to evaluate the performance of a trained LeNet5 model, 
run this command:
```
python evaluate.py my_examples/mnist/evaluation.json
```

The output should look as follows (but the exact numbers will most likely differ):
```
[acc 0.9438, loss 0.2035, val acc 0.9156, val loss 0.2881]
```

## Run inference using the trained model

In order to make predictions on our own data, we can use infer.py script:
```
python infer.py <inference_spec> <path_to_image_depicting_a_digit> 
```

Here <inference_spec> is the path to the specification file for inference,
<path_to_image_depicting_a_digit> is the path to the actual 28x28
gray-scale image with a handwritten digit.

To make this work we obviously need to somehow create an image file 
and write a specification. Let's start the former.

To create a mnist digit file in the current working directory, 
you can run this trivial code in Python console:
```
from torchvision import datasets
ds = datasets.mnist.MNIST('./data', train=False, download=True)
image, _ = ds[1]
image.save('digit_2.png')
```

Now copy that file "digit_2.png" to my_examples/mnist directory.

Now let's write a specification file.
Under my_examples/mnist, create and open a file "inference.json".
Copy the following text inside the file:
```
{
    "session_dir": "pretrained",
    "inference_pipeline": "evaluation_pipeline",
    "inputs_meta": [
        {
            "input_converter": {
                "class": "my_examples.mnist.InputConverter"
            }
        }
    ]
}
```

The format is similar to the one of training.json. One important detail though,
is that this specification file builds on top of training.json spec file.
That means that all definitions and pipelines defined in training.json are
available to the infer.py script. This is a good thing, since we can reuse the
same pipeline for training and inference purposes. 
Moreover, we can override and even replace
certain definitions and pipelines. Here we are not going to do that, but 
you may want to see a section about specifications format to learn more.

TorchAssistant allows to use previously defined pipeline to
make predictions on raw unprocessed data. It does so with 
help of another entity called an input converter. Input converter essentially
takes a user provided input (a string) from a command line and converts it
to the form matching the one of training examples used for training the model.

To define data converters we need to fill in the "inputs_meta" field.
This field allows to define multiple data converters, one converter per 
command line argument. In this example we only need to provide one argument
which will specify a path to the actual png file depicting a digit. Therefore,
we are going to specify only on data converter called "input_converter".

To specify input converter, it is enough to provide a fully qualified name of the 
converter class. We haven't implemented it yet, let's do just that.
Open a file ```__init__.py``` under my_examples/mnist/ and copy the
following code to the bottom of the file:
```
class InputConverter:
    def __call__(self, image_path):
        from PIL import Image
        with Image.open(image_path) as im:
            return im.copy(), None
```

This class implements just one method: ```__call__```. This method
takes a string which specifies the path to the 28x28 gray-scale image,
opens the image using the Pillow library and returns its copy. 
There is a catch though. 
The pipeline that we use for inference was trained on a dataset whose
entries were pairs (image, label). In order to make predictions using this 
pipeline , we need to ensure that a data converter also returns pairs. 
But since we are not interested in computing metrics, we don't need to know 
the label. We are only interested in making predictions. Therefore, we can 
choose any value in place of label. Here we just set it to None.

We are almost ready to run inference. To complete the specification file,
we need to fill in 2 more fields: "session_dir" and "inference_pipeline".
First field sets the path to training session directory as before, so we 
set its value to "pretrained".
The second field sets a pipeline to use to run inference. In our case,
we could choose "training_pipeline" or "evaluation_pipeline" (both defined
in training.json). It makes no difference which one we will take. Let's
go with "evaluation_pipeline".

That's it. We have completed the specification file for inference. Now
run the following command (again, you should run the command from the
directory containing the folder "pretrained"):
```
python infer.py my_examples/mnist/inference.json 'my_examples/mnist/digit_2.png'
```

After script completes, you should see output similar to the one below:
```
Running inference on:  ['examples/mnist/digit_2.png']
y_hat:tensor([[-0.8075,  5.9970,  9.7451,  2.6474, -6.4378,  0.9854, -0.2322, -3.8444,
          1.8763, -6.0900]])
```

In this output you can see a tensor that represents unnormalized
scores of digits. First score is proportional to the likelihood 
that an image depicts digit 0. Second score is proportional to the a 
likelihood that an image depicts a digit 1.
And so on. Higher the score, higher the probability that an 
image shows a corresponding digit. Here we see that the highest score 
has an index 2 which suggests that the image depicts a digit 2 which 
is indeed the case, neural network made correct prediction!