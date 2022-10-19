# Running the code

1. Change your current working directory such that it contains the following folders:
   - examples
   - tests
   - torchassistant

2. Create a fresh training session (note that it will store it in "pretrained" directory,
   it will also create "data" directory for downloading and storing MNIST dataset):
    ```
    python init.py examples/mnist/training.json
    ```
3. Begin training:
    ```
    python train.py pretrained
    ```
4. Evaluate metrics:
   ```
   python evaluate.py examples/mnist/evaluation.json
   ```
5. Use trained model for prediction:
   ```
   python infer.py examples/mnist/inference.json examples/mnist/digit_1.png
   ```