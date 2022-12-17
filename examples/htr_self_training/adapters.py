from random import randrange, uniform
import math
import torch

from torch.nn.functional import softmax
from PIL import Image
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision import transforms
from torchvision.transforms import Compose, PILToTensor, ToPILImage

from torchassistant.utils import pad_sequences
from torchassistant.collators import one_hot_tensor


class ImagePreprocessor:
    def prepare_images(self, images):
        # apply the same normalization to images that was applied for training VGG19_BN
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        images = self.augment(images)
        tensors = [normalize(self.to_rgb(to_tensor(im))) for im in images]
        return torch.stack(tensors)

    def to_rgb(self, image):
        return image.repeat(3, 1, 1)

    def augment(self, images):
        raise NotImplementedError

    def pad_images(self, images):
        max_height = max([im.height for im in images])
        max_width = max([im.width for im in images])

        padded = []
        for im in images:
            padding_top, padding_bottom = self.calculate_padding(max_height, im.height)
            padding_left, padding_right = self.calculate_padding(max_width, im.width)

            pad = transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=255)
            padded.append(pad(im))

        return padded

    def calculate_padding(self, max_length, length):
        len_diff = max_length - length
        padding1 = len_diff // 2
        padding2 = len_diff - padding1
        return padding1, padding2


class WeakAugmentation(ImagePreprocessor):
    def augment(self, images):
        affine = transforms.RandomAffine(degrees=[-5, 5], scale=[0.9, 1.1], fill=255)
        blur = transforms.GaussianBlur(3, sigma=[2, 2])
        add_noise = gaussian_noise(sigma=20)

        images = [affine(im) for im in images]
        images = self.pad_images(images)
        images = [blur(im) for im in images]
        images = [add_noise(im) for im in images]
        return images


class StrongAugmentation(WeakAugmentation):
    brightness = transforms.ColorJitter(brightness=(0.05, 0.95))
    contrast = transforms.ColorJitter(contrast=(0.05, 0.95))
    equalize = transforms.RandomEqualize(1)
    rotate = transforms.RandomRotation((-30, 30), fill=255)

    degrees_range = (math.degrees(-0.3), math.degrees(0.3))
    shear_x = transforms.RandomAffine(0, shear=degrees_range)
    shear_y = transforms.RandomAffine(0, shear=(0, 0) + degrees_range)
    auto_contrast = transforms.RandomAutocontrast(1)
    translate_x = transforms.RandomAffine(0, translate=(0.3, 0), fill=255)
    translate_y = transforms.RandomAffine(0, translate=(0, 0.3), fill=255)

    transforms_per_image = 2

    def augment(self, images):
        images = [self.transform_image(im) for im in images]
        return self.pad_images(images)

    def transform_image(self, image):
        transformations = self.get_random_transformations(self.transforms_per_image)
        for transform_func in transformations:
            image = transform_func(image)
        return image

    def get_random_transformations(self, n):
        return [self.random_transformation() for _ in range(n)]

    def random_transformation(self):
        all_transforms = [self.auto_contrast, self.brightness, self.contrast,
                          self.equalize, identity, posterize, self.rotate,
                          adjust_sharpness, self.shear_x, self.shear_y,
                          solarize, self.translate_x, self.translate_y]
        idx = randrange(0, len(all_transforms))
        return all_transforms[idx]


class WithoutAugmentation(WeakAugmentation):
    def augment(self, images):
        return self.pad_images(images)


def identity(image): return image


def posterize(image):
    bits = randrange(4, 8 + 1)
    return transforms.RandomPosterize(bits, p=1)(image)


def adjust_sharpness(image):
    factor = uniform(0.05, 0.95)
    return transforms.RandomAdjustSharpness(factor, p=1)(image)


def solarize(image):
    threshold = int(round(uniform(0, 1) * 255))
    return transforms.RandomSolarize(threshold, p=1)(image)


def gaussian_noise(sigma):
    def add_noise(tensor):
        noisy = tensor + sigma * torch.randn_like(tensor.to(torch.float32))
        noisy = torch.clamp(noisy, 0, 255)
        return noisy.to(tensor.dtype)

    return Compose([PILToTensor(), add_noise, ToPILImage()])


class TrainableProcessorInputAdapter:
    image_preprocessor = WithoutAugmentation()

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, batch: dict) -> dict:
        images = self.get_images(batch)

        tensors = self.image_preprocessor.prepare_images(images)

        transcripts_tensor = self.get_teacher_forcing_tensor(batch)

        return {
            "encoder": {
                "x": tensors
            },
            "decoder": {
                "tf_seq": transcripts_tensor
            }
        }

    def get_images(self, batch):
        return batch["input_1"]

    def get_transcripts(self, batch):
        return batch["input_2"]

    def get_teacher_forcing_tensor(self, batch):
        transcripts = self.get_transcripts(batch)

        padded, _ = pad_sequences(transcripts, transcripts[0][-1])

        return one_hot_tensor(padded, self.num_classes)[:, :-1]


class SyntheticInputAdapter(TrainableProcessorInputAdapter):
    image_preprocessor = WeakAugmentation()


class PredictorInputAdapter(TrainableProcessorInputAdapter):
    def get_images(self, batch):
        return batch["input_3"]

    def get_teacher_forcing_tensor(self, batch):
        return None


class BaseOutputAdapter:
    def __call__(self, data_frame):
        transcripts = data_frame["input_2"]
        data_frame["y"] = [t[1:] for t in transcripts]
        return data_frame


class PredictionOutputAdapter:
    def __init__(self, tokenizer, threshold, pseudo_labels_path):
        self.threshold = threshold
        self.tokenizer = tokenizer
        self.pseudo_labels_path = pseudo_labels_path

    def __call__(self, data_frame: dict) -> dict:
        y_hat = data_frame["y_hat"]
        pmf = softmax(y_hat, dim=2)
        values, indices = pmf.max(dim=2)

        end_token = self.tokenizer._encode(self.tokenizer.end)
        for i in range(len(indices)):
            tokens = indices[i].tolist()

            try:
                first_n = tokens.index(end_token)
            except ValueError:
                first_n = len(tokens)

            # todo: usually, first predicted token is <s>,
            #  perhaps it should also be excluded from confidence calculation
            mean_confidence = values[i, :first_n].mean()

            transcript = self.tokenizer.decode_to_string(tokens, clean_output=True)

            if mean_confidence > self.threshold:
                image_path = data_frame["input_1"]
                gray_level = data_frame["input_2"]
                self.save_example(image_path[i], gray_level[i], transcript)

        return data_frame

    def save_example(self, image_path, gray_level, transcript):
        with open(self.pseudo_labels_path, 'a') as f:
            f.write(f'{image_path}, {gray_level}, {transcript}\n')


class AugmentationInputAdapter(TrainableProcessorInputAdapter):
    def __call__(self, data_frame):
        images = self.get_images(data_frame)

        weak_preprocessor = WeakAugmentation()
        strong_preprocessor = StrongAugmentation()
        weak_tensors = weak_preprocessor.prepare_images(images)
        strong_tensors = strong_preprocessor.prepare_images(images)

        transcripts_tensor = self.get_teacher_forcing_tensor(data_frame)

        return {
            'encoder_weak': {
                "x_weak": weak_tensors
            },
            'decoder_weak': {
                "tf_seq": transcripts_tensor
            },
            'encoder_strong': {
                "x_strong": strong_tensors
            },
            'decoder_strong': {
                "tf_seq": transcripts_tensor
            }
        }


class EvaluationProcessorAdapter(TrainableProcessorInputAdapter):
    image_preprocessor = WithoutAugmentation()


def build_synthetic_image_input_adapter(session, **kwargs):
    num_classes = session.preprocessors["tokenize"].charset_size
    return SyntheticInputAdapter(num_classes=num_classes, **kwargs)


def build_predictor_input_adapter(session, **kwargs):
    num_classes = session.preprocessors["tokenize"].charset_size
    return PredictorInputAdapter(num_classes=num_classes, **kwargs)


def build_augmentation_adapter(session, **kwargs):
    num_classes = session.preprocessors["tokenize"].charset_size
    return AugmentationInputAdapter(num_classes=num_classes, **kwargs)


def build_prediction_output_adapter(session, **kwargs):
    tokenizer = session.preprocessors["tokenize"]
    return PredictionOutputAdapter(tokenizer, **kwargs)


def build_evaluation_input_adapter(session, **kwargs):
    num_classes = session.preprocessors["tokenize"].charset_size
    return EvaluationProcessorAdapter(num_classes)


class InputConverter:
    def __call__(self, image_path):
        with Image.open(image_path) as im:
            im = rgb_to_grayscale(im)
            return im.copy(), ''
