from scaffolding.utils import load_pipeline
from scaffolding.training import do_forward_pass
from scaffolding.parse import parse_collator, parse_batch_adapter
from scaffolding.utils import WrappedDataset, AdaptedCollator
import torch


class InputAdapter:
    def __call__(self, x):
        return x


class Net:
    def run_inference(self, x):
        return 0


class PostProcessor:
    def __call__(self, *args, **kwargs):
        return args


class OutputDevice:
    """How to present the result"""
    def __call__(self, x):
        print(x)


def get_data_loader(config_dict, raw_data, input_adapter):
    class MyDataset:
        def __getitem__(self, idx):
            return input_adapter(raw_data)

        def __len__(self):
            return 1

    ds = MyDataset()
    preprocessors = load_preprocessors(config_dict)
    ds = WrappedDataset(ds, preprocessors)

    collate_fn = parse_collator(config_dict)
    batch_adapter = parse_batch_adapter(config_dict)
    final_collate_fn = AdaptedCollator(collate_fn, batch_adapter)

    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                                       collate_fn=final_collate_fn)


def load_preprocessors(config_dict):
    return []


raw_data = "Hello world"
pipeline_dict = {}
data_pipeline, model_pipeline = load_pipeline(pipeline_dict, inference_mode=True)

loader = get_data_loader(InputAdapter, raw_data, InputAdapter())

first_batch = next(loader)

outputs = do_forward_pass(model_pipeline, first_batch)

output_data = PostProcessor()(outputs)

OutputDevice()(output_data)
