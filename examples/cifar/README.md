# Running the code

1. Change your current working directory such that it contains the following folders:
   - examples
   - torchassistant

2. Create a fresh training session (note that it will store it in "pretrained" directory,
   it will also create "data" directory for downloading and storing CIFAR10 dataset):
    ```
    python init.py examples/cifar/training.json
    ```
3. Begin training:
    ```
    python train.py pretrained
    ```
4. Evaluate metrics:
   ```
   python evaluate.py examples/cifar/evaluation.json
   ```
5. Use trained model for prediction:
   ```
   python infer.py examples/cifar/inference.json examples/cifar/1.png
   ```