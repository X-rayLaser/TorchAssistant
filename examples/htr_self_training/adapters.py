import torch
from torch.nn.functional import softmax

from torchassistant.utils import pad_sequences
from torchassistant.collators import one_hot_tensor
from torchvision import transforms


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
        affine = transforms.RandomAffine(degrees=[-15, 15], scale=[0.75, 1.1], fill=255)
        blur = transforms.GaussianBlur(7, sigma=[2, 2])

        images = [affine(im) for im in images]
        images = self.pad_images(images)
        images = [blur(im) for im in images]
        return images


class StrongAugmentation(WeakAugmentation):
    pass


class TrainableProcessorInputAdapter:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, batch: dict) -> dict:
        images = self.get_images(batch)

        preprocessor = WeakAugmentation()
        tensors = preprocessor.prepare_images(images)

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
    pass


class PredictorInputAdapter(TrainableProcessorInputAdapter):
    def get_images(self, batch):
        return batch["input_3"]

    def get_teacher_forcing_tensor(self, batch):
        return None


class WeaklyAugmentedImageInputAdapter(TrainableProcessorInputAdapter):
    pass


class StronglyAugmentedImageInputAdapter(TrainableProcessorInputAdapter):
    pass


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
        mean_confidence = values.mean(dim=1)

        for i in range(len(indices)):
            transcript = self.tokenizer.decode_to_string(indices[i, :], clean_output=True)

            if mean_confidence[i] > 0:
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


def build_synthetic_image_input_adapter(session, **kwargs):
    num_classes = session.preprocessors["tokenize"].charset_size
    return SyntheticInputAdapter(num_classes=num_classes, **kwargs)


def build_predictor_input_adapter(session, **kwargs):
    num_classes = session.preprocessors["tokenize"].charset_size
    return PredictorInputAdapter(num_classes=num_classes, **kwargs)


def build_weakly_augmented_image_input_adapter(session, **kwargs):
    num_classes = session.preprocessors["tokenize"].charset_size
    return WeaklyAugmentedImageInputAdapter(num_classes=num_classes, **kwargs)


def build_strongly_augmented_image_input_adapter(session, **kwargs):
    num_classes = session.preprocessors["tokenize"].charset_size
    return StronglyAugmentedImageInputAdapter(num_classes=num_classes, **kwargs)


def build_augmentation_adapter(session, **kwargs):
    num_classes = session.preprocessors["tokenize"].charset_size
    return AugmentationInputAdapter(num_classes=num_classes, **kwargs)


def build_prediction_output_adapter(session, **kwargs):
    tokenizer = session.preprocessors["tokenize"]
    return PredictionOutputAdapter(tokenizer, **kwargs)
