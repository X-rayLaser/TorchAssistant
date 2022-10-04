import os.path
from datetime import datetime
from torchvision.utils import make_grid
import numpy as np
from PIL import Image


class Printer:
    """How to present the result"""
    def __call__(self, result_dict):
        for k, v in result_dict.items():
            print(f'{k}:{v}')


class SaveAsImages:
    def __init__(self, destination_folder, mean=0.5, std=0.5):
        self.folder = destination_folder
        self.mean = mean
        self.std = std

    def __call__(self, result_dict):
        """

        :param result_dict: must contain single key: value entry with value being a tensor of image pixels
        :return:
        :rtype:
        """

        results = dict(result_dict)
        _, tensor = results.popitem()

        t = datetime.now()

        file_name = f'{t.hour}_{t.minute}_{t.second}.png'
        file_path = os.path.join(self.folder, file_name)

        tensor = tensor * self.std + self.mean
        image = make_grid(tensor, nrow=8)

        array = np.transpose(image.cpu().numpy(), (1, 2, 0))
        pillow_image = Image.fromarray(array, mode="RGB")
        pillow_image.save(file_path)
