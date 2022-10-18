# Intro

This is a toy example demonstrating how to train an encoder-decoder
architecture to perform machine translation (from French to English).

# Getting the dataset

1. Create a folder "dataset" under "language_translation" directory.
2. Download the dataset file from [here](https://download.pytorch.org/tutorial/data.zip) or
from [here](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#loading-data-files).
3. Extract the zip file, locate the file eng-fra.txt and copy that file to  
"language_translation/dataset/eng-fra.txt".

# Running the code

1. Change your current working directory such that it contains the following folders:
   - examples
   - tests
   - torchassistant

2. Create a fresh training session (note that it will store it in "pretrained" directory):
    ```
    python init.py examples/language_translation/training.json
    ```
3. Begin training:
    ```
    python train.py pretrained
    ```

4. Evaluate metrics:
   ```
   python evaluate.py examples/language_translation/evaluation.json
   ```
5. Use trained model for prediction:
   ```
   python infer.py examples/language_translation/inference.json 'Ça alors !'
   ```