import os
import json
from collections import defaultdict, OrderedDict
from random import shuffle
import csv

import torch

from . import parse
from .override_spec import override_spec
from ..utils import get_dataset
from .registry import group_to_loader
from .data_classes import Stage
from .parse.loaders import PipelineLoader, StageLoader, BadSpecificationError
from .. import collators
from ..collators import ColumnWiseCollator
from PIL.Image import Image


def create_session(spec):
    session = Session()
    initializer = SessionInitializer()
    initializer(session, spec)
    return session


def create_and_save_session(spec, session_dir):
    session = create_session(spec)
    saver = SessionSaver(session_dir)
    saver.initial_save(session, spec)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        s = f.read()

    return json.loads(s)


def save_as_json(d, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(d))


class ProgressBar:
    def __init__(self, stage_id, epochs_done, completed):
        self.stage_id = stage_id
        self.epochs_done = epochs_done
        self.completed = completed

    def asdict(self):
        return self.__dict__


class Progress:
    def __init__(self, progress_bars):
        self.progress_bars = progress_bars

    @property
    def epochs_done_total(self):
        return sum(bar.epochs_done for bar in self.progress_bars)

    def increment_progress(self):
        stage_id = self.get_current_stage_id()
        self.progress_bars[stage_id].epochs_done += 1
        self.progress_bars[stage_id].completed = False

    def mark_completed(self):
        stage_id = self.get_current_stage_id()
        self.progress_bars[stage_id].completed = True

    def get_current_stage_id(self):
        ids = [idx for idx, bar in enumerate(self.progress_bars) if not bar.completed]
        if ids:
            return ids[0]

        raise StopTrainingError('All stages are completed')

    def __getitem__(self, idx):
        return self.progress_bars[idx]

    def to_list(self):
        return [bar.asdict() for bar in self.progress_bars]

    def from_list(self, items):
        self.progress_bars = [ProgressBar(**d) for d in items]


class StopTrainingError(Exception):
    pass


# todo: rename to state view
class State:
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers


class Session:
    def __init__(self):
        self.datasets = OrderedDict()
        self.splits = OrderedDict()
        self.preprocessors = OrderedDict()
        self.collators = OrderedDict()
        self.data_loaders = OrderedDict()
        self.loader_factories = {}
        self.models = {}
        self.gradient_clippers = {}
        self.forward_hooks = {}
        self.backward_hooks = {}
        self.optimizers = {}
        self.batch_adapters = {}
        self.batch_processors = {}
        self.batch_pipelines = {}
        self.batch_graphs = {}
        self.losses = {}
        self.metrics = {}

        self.training_pipelines = {}
        self.debug_pipelines = {}
        self.pipelines = OrderedDict()

        self.stages = []

        self.device = torch.device("cpu")

        # tracks progress
        self.progress = Progress([])

    def initialize_state(self):
        return State(models=self.models, optimizers=self.optimizers)


class SessionSaver:
    save_attrs = 'datasets splits preprocessors collators batch_adapters losses metrics'.split(' ')

    def __init__(self, session_dir):
        self.session_dir = session_dir

        self.spec_path = os.path.join(session_dir, 'spec.json')
        self.static_dir = os.path.join(session_dir, 'static')
        self.checkpoints_dir = os.path.join(session_dir, 'checkpoints')
        self.history_dir = os.path.join(session_dir, 'metrics')
        self.extra_path = os.path.join(self.static_dir, 'extra_params.json')

    def initial_save(self, session, spec):
        os.makedirs(self.session_dir)
        os.makedirs(self.static_dir)
        os.makedirs(self.checkpoints_dir)
        os.makedirs(self.history_dir)

        self._save_spec(spec)
        self._save_static(session)
        self.save_checkpoint(session)

    def load_from_latest_checkpoint(self, new_spec=None):
        spec = load_json(self.spec_path)
        if new_spec:
            spec = override_spec(spec, new_spec)

        session = Session()
        restorer = SessionRestorer(self.static_dir)
        restorer(session, spec)

        checkpoint_dirs = os.listdir(self.checkpoints_dir)
        latest_epoch = max(map(int, checkpoint_dirs))
        self.load_checkpoint(session, name=str(latest_epoch))
        return session

    def _save_spec(self, spec):
        save_as_json(spec, self.spec_path)

    def _save_static(self, session):
        object_persistence = ObjectPersistence(self.static_dir)

        for attr in self.save_attrs:
            section = getattr(session, attr)
            for name, instance in section.items():
                object_persistence.save(instance, name)

    def save_checkpoint(self, session):
        state_dir = self.checkpoints_dir
        state = session.initialize_state()

        checkpoint_dir = os.path.join(state_dir, str(session.progress.epochs_done_total))
        os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
        device_path = os.path.join(checkpoint_dir, 'device.txt')

        with open(device_path, 'w', encoding='utf-8') as f:
            f.write(str(session.device))

        models_dict = {name: model.state_dict() for name, model in state.models.items()}

        optimizers_dict = {name: optimizer.state_dict() for name, optimizer in state.optimizers.items()}

        torch.save({
            'models': models_dict,
            'optimizers': optimizers_dict,
            'progress': session.progress.to_list()
        }, checkpoint_path)

    def load_checkpoint(self, session, name):
        state_dir = self.checkpoints_dir
        state = session.initialize_state()

        checkpoint_dir = os.path.join(state_dir, name)
        state_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
        device_path = os.path.join(checkpoint_dir, 'device.txt')

        with open(device_path, encoding='utf-8') as f:
            checkpoint_device = torch.device(f.read())

        if checkpoint_device == session.device:
            checkpoint = torch.load(state_path)
        else:
            checkpoint = torch.load(state_path, map_location=session.device)

        for name, model_state in checkpoint['models'].items():
            state.models[name].load_state_dict(model_state)
            state.models[name].to(session.device)

        for name, optimizer_state in checkpoint['optimizers'].items():
            state.optimizers[name].load_state_dict(optimizer_state)

        session.progress.from_list(checkpoint["progress"])

    def log_metrics(self, stage_number, epoch, train_metrics):
        history_path = os.path.join(self.history_dir, f'{stage_number}.csv')
        history = TrainingHistory(history_path)
        history.add_entry(epoch, train_metrics)


class TrainingHistory:
    def __init__(self, file_path):
        self.file_path = file_path

    def add_entry(self, epoch, train_metrics):
        # todo: make sure the ordering is right

        all_metrics = {}
        all_metrics.update(train_metrics)

        row_dict = {'epoch': epoch}
        row_dict.update({k: self.scalar(v) for k, v in all_metrics.items()})

        field_names = list(row_dict.keys())

        if not os.path.exists(self.file_path):
            self.create(self.file_path, field_names)

        with open(self.file_path, 'a', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writerow(row_dict)

    def scalar(self, t):
        return t.item() if hasattr(t, 'item') else t

    @classmethod
    def create(cls, path, field_names):
        with open(path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(field_names)
        return cls(path)


class ObjectInstaller:
    def setup(self, session, instance, spec=None, **kwargs):
        pass


class PreProcessorInstaller(ObjectInstaller):
    def setup(self, session, instance, spec=None, **kwargs):
        dataset_name = spec.get("fit")
        if dataset_name:
            dataset = get_dataset(session, dataset_name)
            instance.fit(dataset)


class SplitterInstaller(ObjectInstaller):
    def setup(self, session, instance, spec=None, **kwargs):
        ds_name = spec["dataset_name"]
        ds = session.datasets[ds_name]

        shuffled_indices = list(range(len(ds)))
        shuffle(shuffled_indices)
        instance.configure(shuffled_indices)


class CallableInstaller(ObjectInstaller):
    def setup(self, session, instance, spec=None, **kwargs):
        instance()


class SessionInitializer:
    installers = defaultdict(ObjectInstaller, {
        'preprocessors': PreProcessorInstaller(),
        'backward_hooks': CallableInstaller(),
        'splits': SplitterInstaller()
    })

    def get_autocompleter(self):
        return DefinitionAutoCompleter()

    def __call__(self, session, spec):
        self.parse_device(session, spec)

        init_dict = spec["initialize"]

        auto_completer = self.get_autocompleter()
        definitions = init_dict["definitions"]
        data_definitions = definitions["data"]
        for d in data_definitions:
            self.build_object(session, d)

        auto_completer.autocomplete_data_splits(session)
        auto_completer.autocomplete_preprocessors(session)
        auto_completer.autocomplete_datasets(session)
        auto_completer.autocomplete_collator(session)
        auto_completer.autocomplete_data_loaders(session)

        computation_definitions = definitions["computation"]

        for d in computation_definitions:
            self.build_object(session, d)

        auto_completer.autocomplete_loss_and_metrics(session)

        pipelines_dict = init_dict.get("pipelines")
        if pipelines_dict:
            self.prepare_pipelines(
                session, pipelines_dict, parse.loaders.PipelineLoader(), 'pipelines'
            )
        else:
            self.build_pipelines(session)

        self.prepare_pipelines(
            session, init_dict.get("debug_pipelines", {}),
            parse.loaders.DebuggerLoader(), 'debug_pipelines'
        )

        self.load_stages(session, spec)

        self.initialize_progress(session)

    def parse_device(self, session, spec):
        if "device" not in spec:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            device = spec["device"]
        session.device = device

    def prepare_pipelines(self, session, spec, loader, section):
        pipelines = {}
        for name, pipeline_spec in spec.items():
            pipeline = loader.load(session, pipeline_spec)
            pipelines[name] = pipeline

        setattr(session, section, pipelines)

    def build_pipelines(self, session):
        loaders_iterator = iter(session.data_loaders.keys())
        training_data_loader = next(loaders_iterator)
        test_data_loader = next(loaders_iterator)

        def make_pipeline_spec(data_loader, loss, loss_display, metrics):
            return {
                "input_injector": [{"data_loader": data_loader}],
                "losses": [{"loss_name": loss, "loss_display_name": loss_display}],
                "metrics": [dict(metric_name=name, display_name=display_name)
                            for name, display_name in metrics]
            }

        pipelines = {}

        loss_name = next(iter(session.losses.keys()))
        train_loss_display = "loss"
        train_metric_fns = [(name, name.lower()) for name in session.metrics.keys()]

        val_loss_display = "val loss"
        val_metric_fns = [(name, 'val ' + name.lower()) for name in session.metrics.keys()]

        pipelines["training"] = make_pipeline_spec(
            training_data_loader, loss_name, train_loss_display, train_metric_fns
        )

        pipelines["validation"] = make_pipeline_spec(
            test_data_loader, loss_name, val_loss_display, val_metric_fns
        )

        pipelines = {name: PipelineLoader().load(session, spec) for name, spec in pipelines.items()}
        session.pipelines = pipelines

    def load_stages(self, session, spec):
        try:
            stages_spec = spec["train"]["stages"]
        except KeyError:
            self.build_stages(session)
        else:
            stage_loader = parse.loaders.StageLoader()
            session.stages = [stage_loader.load(session, stage) for stage in stages_spec]

    def build_stages(self, session):
        training_pipeline = self.get_pipeline(session, training=True)
        validation_pipeline = self.get_pipeline(session, training=False)

        training_pipelines = [training_pipeline]
        validation_pipelines = [training_pipeline, validation_pipeline]
        debug_pipelines = []
        stop_condition = StageLoader().build_stop_condition(session)
        eval_steps = StageLoader().default_eval_steps(session)

        session.stages = [
            Stage("interleave", training_pipelines, validation_pipelines, debug_pipelines,
                  stop_condition, eval_steps)
        ]

    def get_pipeline(self, session, training=True):
        if len(session.pipelines) > 2:
            raise Exception(
                f'Bad specification. Cannot autocomplete stages section: '
                f'cannot decide which pipelines to use for training and validation'
            )
        for pipeline in session.pipelines.values():
            if len(pipeline.input_loaders) != 1:
                raise Exception(f'Bad specification')

            data_loader = pipeline.input_loaders[0].loader_factory
            for i, k in enumerate(session.data_loaders.values()):
                if k == data_loader:
                    if (training and i == 0) or (not training and i != 0):
                        return pipeline

        pipeline_name = "training" if training else "evaluation"

        raise BadSpecificationError(
            f'Cannot automatically build stages because of missing definition for '
            f'{pipeline_name} pipeline'
        )

    def initialize_progress(self, session):
        bars = [ProgressBar(idx, 0, completed=False) for idx in range(len(session.stages))]
        session.progress = Progress(bars)

    def build_object(self, session, definition):
        group = definition["group"]
        loader = group_to_loader.get(group, parse.loaders.Loader())
        installer = self.installers[group]
        name = definition["name"]
        spec = definition["spec"]

        instance = loader(session, spec, name)
        installer.setup(session, instance, spec)

        section = getattr(session, group)
        section[name] = instance
        return instance


class DefinitionAutoCompleter:
    installers = {
        'preprocessors': PreProcessorInstaller(),
        'splits': SplitterInstaller()
    }

    def __init__(self, **kwargs):
        pass

    def autocomplete(self, session):
        self.autocomplete_data_splits(session)
        self.autocomplete_preprocessors(session)
        self.autocomplete_datasets(session)
        self.autocomplete_collator(session)
        self.autocomplete_data_loaders(session)
        self.autocomplete_loss_and_metrics(session)

    def autocomplete_data_splits(self, session):
        if session.splits:
            return
        if len(session.datasets) > 1:
            return

        ds_name = next(iter(session.datasets.keys()))
        ds = session.datasets[ds_name]

        num_examples = len(ds)
        sizes = self.get_split_sizes(num_examples)

        loader = group_to_loader["splits"]
        split_name = f'autogenerated_split'
        spec = dict(dataset_name=ds_name, ratio=sizes)

        installer = self.installers["splits"]
        instance = loader(session, spec)
        installer.setup(session, instance, spec)

        session.splits[split_name] = instance

    def get_split_sizes(self, num_examples):
        if num_examples > 10 ** 7:
            return [0.98, 0.01, 0.01]
        elif num_examples > 10 ** 6:
            return [0.9, 0.05, 0.05]
        elif num_examples > 10 ** 5:
            return [0.8, 0.1, 0.1]
        elif num_examples > 10 ** 4:
            return [0.6, 0.2, 0.2]
        else:
            return [0.5, 0.25, 0.25]

    def autocomplete_preprocessors(self, session):
        if session.preprocessors:
            return

        if len(session.datasets) > 1:
            dataset_name = next(iter(session.datasets.keys()))
        else:
            data_split = next(iter(session.splits.keys()))
            dataset_name = f'{data_split}.train'

        signature = self.get_consistent_signature(session.datasets.values())
        loader = group_to_loader["preprocessors"]
        installer = self.installers["preprocessors"]

        for i, data_type in enumerate(signature):
            spec = self.preprocessor_spec_for_type(data_type, dataset_name, i)
            name = f'autogenerated_preprocessor_{i + 1}'
            instance = loader(session, spec, name)
            installer.setup(session, instance, spec)
            session.preprocessors[name] = instance

    def preprocessor_spec_for_type(self, data_type, fit_ds, fit_index):
        if data_type is Image:  # PIL image
            return {
                "class": "torchassistant.preprocessors.ImagePreprocessor",
                "args": [fit_index],
                "fit": fit_ds
            }
        elif data_type is int:
            return {
                "class": "torchassistant.preprocessors.NullProcessor"
            }
        elif data_type is str:
            return {
                "class": "torchassistant.preprocessors.TextPreprocessor",
                "args": [fit_index],
                "fit": fit_ds
            }
        else:
            raise BadSpecificationError(
                f'Cannot automatically create a preprocessor for a data type "{data_type}"'
            )

    def autocomplete_datasets(self, session):
        autogen_preprocessors = [name for name in session.preprocessors.keys()
                                 if name.startswith("autogenerated")]
        if not session.preprocessors:
            return

        if not autogen_preprocessors:
            return

        if self.preprocessed_datasets(session):
            return

        if len(session.datasets) > 1:
            dataset_names = session.datasets
        else:
            data_split = next(iter(session.splits.keys()))
            dataset_names = [f'{data_split}.train', f'{data_split}.val']

        loader = group_to_loader["datasets"]
        for name in dataset_names.copy():
            spec = {
                "link": name,
                "preprocessors": list(session.preprocessors.keys())
            }
            autogen_name = f'autogenerated_{name}'
            session.datasets[autogen_name] = loader(session, spec, autogen_name)

    def autocomplete_collator(self, session):
        if session.collators:
            return

        signature = self.get_consistent_signature(self.relevant_datasets(session))

        input_types = signature[:-1]
        column_transforms = [self.type_transform(session, data_index, input_type)
                             for data_index, input_type in enumerate(input_types)]
        column_transforms.append(
            self.type_transform(session, len(signature) - 1, signature[-1], is_target=True)
        )
        session.collators["default_collator"] = ColumnWiseCollator(column_transforms)

    def type_transform(self, session, data_index, data_type, is_target=False):
        if data_type is int:
            return collators.to_long_tensor
        elif data_type is float:
            return collators.to_float_tensor
        elif data_type is list:
            return self.transform_for_lists(session, data_index, is_target)
        elif data_type is Image:
            return collators.images_to_tensor
        elif data_type is torch.Tensor:
            return collators.stack_tensors
        else:
            raise BadSpecificationError(
                f'Cannot automatically build a collator.'
                f'Unsupported data type in example tuple: "{data_type}"'
            )

    def transform_for_lists(self, session, data_index, is_target):
        if is_target:
            return collators.no_transform

        all_attr_values = []
        all_names = []
        for name, ds in session.datasets.items():
            if hasattr(ds, 'preprocessors') and data_index < len(ds.preprocessors):
                p = ds.preprocessors[data_index]
                if hasattr(p, 'num_classes'):
                    all_attr_values.append(p.num_classes)
                    all_names.append(name)

        if len(set(all_attr_values)) > 1:
            raise BadSpecificationError(
                f'Cannot automatically build a collator: found multiple datasets '
                f'associated with preprocessors having conflicting value of "num_classes" '
                f'attribute: datasets {all_names}, attributes {all_attr_values}'
            )

        if all_attr_values:
            return collators.SeqsToOneHot(num_classes=all_attr_values[0])
        else:
            return collators.seqs_to_tensor

    def autocomplete_data_loaders(self, session):
        if session.data_loaders:
            return

        if len(session.collators) > 1:
            raise BadSpecificationError(
                f'Cannot automatically pick collator to use in data loader '
                f'when more than one collator is defined. '
                f'Found {len(session.collators)} collators: {list(session.collators.keys())}'
            )

        try:
            self.get_consistent_signature(self.relevant_datasets(session))
        except BadSpecificationError:
            raise BadSpecificationError(
                'Cannot automatically create a data loader: examples signatures '
                'returned by datasets differ'
            )

        first_one = next(iter(session.collators.keys()))

        loader = group_to_loader["data_loaders"]

        for name in self.relevant_dataset_names(session):
            kwargs = dict(batch_size=32)
            spec = dict(dataset=name, collator=first_one, kwargs=kwargs)
            loader_name = f'autogenerated_data_loader_for_{name}'
            session.data_loaders[loader_name] = loader(session, spec)

    def autocomplete_loss_and_metrics(self, session):
        if session.losses and session.metrics:
            return

        original_datasets = [ds for name, ds in session.datasets.items()
                             if not name.startswith("autogenerated")]
        *head, original_target_type = self.get_consistent_signature(original_datasets)
        target_type = self.get_target_type(session)

        loss_loader = group_to_loader["losses"]
        loss_spec = {"class": "CrossEntropyLoss", "inputs": ["y_hat", "input_2"]}

        metric_loader = group_to_loader["metrics"]
        metrics_spec = {"class": "Accuracy", "inputs": ["y_hat", "input_2"],
                        "transform": "torchassistant.transforms.reverse_onehot"}

        if target_type is int:
            if not session.losses:
                session.losses["cross_entropy"] = loss_loader(session, loss_spec)

            if not session.metrics:
                session.metrics["accuracy"] = metric_loader(
                    session, metrics_spec, object_name="accuracy"
                )
        elif original_target_type is str and target_type is list:
            preprocessed_datasets = self.preprocessed_datasets(session)
            if not preprocessed_datasets:
                raise BadSpecificationError(
                    f'Cannot automatically pick loss/metric function for '
                    f'examples with target type "{target_type}": cannot decode predictions '
                )

            if not session.losses:
                loss_spec = {
                    "class": "torchassistant.loss_functions.MaskedCrossEntropy",
                    "inputs": ["y_hat", "input_2"],
                    "transform": "torchassistant.transforms.pad_targets",
                }
                session.losses["masked_cross_entropy"] = loss_loader(session, loss_spec)

            if not session.metrics:
                metrics_spec = {"class": "CharErrorRate", "inputs": ["y_hat", "input_2"],
                                "transform": "torchassistant.transforms.DecodeText"}
                session.metrics["CER"] = metric_loader(session, metrics_spec, object_name="CER")
        else:
            # todo: extra case when target type is not str and no decoder is available
            raise BadSpecificationError(
                f'Cannot automatically pick loss/metric function for examples '
                f'with target type "{target_type}"'
            )

    def get_target_type(self, session):
        signature = self.get_consistent_signature(self.relevant_datasets(session))
        return signature[-1]

    def preprocessed_datasets(self, session):
        return [ds for name, ds in session.datasets.items()
                if name.startswith("autogenerated")]

    def relevant_datasets(self, session):
        autogen_datasets = self.preprocessed_datasets(session)
        return autogen_datasets if autogen_datasets else session.datasets.values()

    def relevant_dataset_names(self, session):
        autogen_datasets = [name for name, ds in session.datasets.items()
                            if name.startswith("autogenerated")]
        return autogen_datasets if autogen_datasets else session.datasets.keys()

    def get_consistent_signature(self, datasets):
        signatures = set()
        for ds in datasets:
            signatures.add(self.get_example_signature(ds))

        if len(signatures) != 1:
            raise BadSpecificationError(
                f'Cannot automatically pick loss function or metric: '
                f'example signatures returned by datasets differ. '
                f'All signatures: {list(signatures)}'
            )

        return next(iter(signatures))

    def get_example_signature(self, dataset):
        return tuple(type(element) for element in dataset[0])


class RestorationCompleter(DefinitionAutoCompleter):
    installers = defaultdict(ObjectInstaller, {})

    def __init__(self, static_dir=None):
        super().__init__()
        self.static_dir = static_dir

    def autocomplete_data_splits(self, session):
        super().autocomplete_data_splits(session)
        persistence = ObjectPersistence(self.static_dir)

        # todo: load only autogenerated ones
        for name, splitter in session.splits.items():
            persistence.load(splitter, name)

    def autocomplete_preprocessors(self, session):
        super().autocomplete_preprocessors(session)
        persistence = ObjectPersistence(self.static_dir)

        # todo: load only autogenerated ones, because user defined preprocessors were already loaded by this point
        for name, preprocessor in session.preprocessors.items():
            persistence.load(preprocessor, name)


class SessionRestorer(SessionInitializer):
    installers = defaultdict(ObjectInstaller, {
        'backward_hooks': CallableInstaller()
    })

    def __init__(self, static_dir):
        super().__init__()
        self.static_dir = static_dir

    def get_autocompleter(self):
        return RestorationCompleter(self.static_dir)

    def build_object(self, session, definition):
        instance = super().build_object(session, definition)
        name = definition["name"]
        persistence = ObjectPersistence(self.static_dir)
        persistence.load(instance, name)


class ObjectPersistence:
    def __init__(self, static_dir):
        self.static_dir = static_dir

    def save(self, instance, object_name):
        path = self._build_save_path(object_name)
        serialized_dict = instance.state_dict() if hasattr(instance, 'state_dict') else {}
        save_as_json(serialized_dict, path)

    def load(self, instance, object_name):
        path = self._build_save_path(object_name)

        if not os.path.exists(path):
            return

        object_state = load_json(path)

        if hasattr(instance, 'load_state_dict'):
            instance.load_state_dict(object_state)

    def _build_save_path(self, name):
        return os.path.join(self.static_dir, f'{name}.json')
