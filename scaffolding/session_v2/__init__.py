import os
import json
from collections import defaultdict
from random import shuffle
import csv

import torch

from . import parse
from .parse import get_dataset


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
        self.datasets = {}
        self.preprocessors = {}
        self.collators = {}
        self.models = {}
        self.optimizers = {}
        self.batch_adapters = {}
        self.losses = {}
        self.metrics = {}
        self.pipelines = {}

        self.stages = []

        # tracks progress
        # todo: store stage number
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
        self.history_path = os.path.join(session_dir, 'metrics')
        self.extra_path = os.path.join(self.static_dir, 'extra_params.json')

    def initial_save(self, session, spec):
        os.makedirs(self.session_dir)
        os.makedirs(self.static_dir)
        os.makedirs(self.checkpoints_dir)

        self._save_spec(spec)
        self._save_static(session)
        self.save_checkpoint(session)

    def load_from_latest_checkpoint(self):
        spec = load_json(self.spec_path)

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
        section_persistence = SectionPersistence(self.static_dir)
        for attr in self.save_attrs:
            section_persistence.save(session, attr)

    def save_checkpoint(self, session):
        state_dir = self.checkpoints_dir
        # todo: map locations when needed
        state = session.initialize_state()

        checkpoint_dir = os.path.join(state_dir, str(session.progress.epochs_done_total))
        os.makedirs(checkpoint_dir)
        save_path = os.path.join(checkpoint_dir, 'checkpoint.pt')

        models_dict = {name: model.state_dict() for name, model in state.models.items()}

        optimizers_dict = {name: optimizer.state_dict() for name, optimizer in state.optimizers.items()}

        torch.save({
            'models': models_dict,
            'optimizers': optimizers_dict,
            'progress': session.progress.to_list()
        }, save_path)

    def load_checkpoint(self, session, name):
        state_dir = self.checkpoints_dir
        state = session.initialize_state()

        checkpoint_dir = os.path.join(state_dir, name)
        state_path = os.path.join(checkpoint_dir, 'checkpoint.pt')

        checkpoint = torch.load(state_path)

        for name, model_state in checkpoint['models'].items():
            state.models[name].load_state_dict(model_state)

        for name, optimizer_state in checkpoint['optimizers'].items():
            state.optimizers[name].load_state_dict(optimizer_state)

        session.progress.from_list(checkpoint["progress"])

    def log_metrics(self, epoch, train_metrics, val_metrics):
        # todo: log metrics to csv file
        history = TrainingHistory(self.history_path)
        history.add_entry(epoch, train_metrics, val_metrics)


class TrainingHistory:
    def __init__(self, file_path):
        self.file_path = file_path

    def add_entry(self, epoch, train_metrics, val_metrics):
        # todo: make sure the ordering is right
        val_metrics = {f'val {k}': v for k, v in val_metrics.items()}

        all_metrics = {}
        all_metrics.update(train_metrics)
        all_metrics.update(val_metrics)

        row_dict = {'epoch': epoch}
        row_dict.update({k: self.scalar(v) for k, v in all_metrics.items()})

        with open(self.file_path, 'a', encoding='utf-8', newline='') as csvfile:
            fieldnames = list(row_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row_dict)

    def scalar(self, t):
        return t.item() if hasattr(t, 'item') else t

    @classmethod
    def create(cls, path, field_names):
        with open(path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = ['Epoch #'] + field_names
            writer.writerow(row)
        return cls(path)


class ObjectInstaller:
    def setup(self, session, instance, object_name=None, **kwargs):
        pass


class PreProcessorInstaller(ObjectInstaller):
    def setup(self, session, instance, object_name=None, **kwargs):
        dataset_name = instance.spec["fit"]
        dataset = get_dataset(session, dataset_name)
        instance.fit(dataset)


class SplitterInstaller(ObjectInstaller):
    def setup(self, session, instance, object_name=None, **kwargs):
        ds_name = instance.spec["dataset_name"]
        ds = session.datasets[ds_name]

        shuffled_indices = list(range(len(ds)))
        shuffle(shuffled_indices)
        instance.configure(shuffled_indices)


class SessionInitializer:
    installers = defaultdict(
        ObjectInstaller, preprocessors=PreProcessorInstaller(), splits=SplitterInstaller()
    )

    def __call__(self, session, spec):
        sections_with_loaders = [
            ('datasets', parse.Loader()),
            ('splits', parse.SplitLoader()),
            ('preprocessors', parse.PreProcessorLoader()),
            ('collators', parse.Loader()),
            ('models', parse.Loader()),
            ('optimizers', parse.OptimizerLoader()),
            ('batch_adapters', parse.Loader()),
            ('losses', parse.LossLoader()),
            ('metrics', parse.MetricLoader()),
            ('pipelines', parse.PipelineLoader())
        ]

        for section_name, loader in sections_with_loaders:
            installer = self.installers[section_name]
            self.load_section(session, spec, section_name, loader, installer)

        self.load_stages(session, spec)

        self.initialize_progress(session)

    def load_section(self, session, spec, section_name, loader, installer):
        section = self.load_init_section(session, spec, section_name, loader, installer)
        setattr(session, section_name, section)

    def load_stages(self, session, spec):
        stages_spec = spec["train"]["stages"]
        stage_loader = parse.StageLoader()
        session.stages = [stage_loader.load(session, stage) for stage in stages_spec]

    def initialize_progress(self, session):
        bars = [ProgressBar(idx, 0, completed=False) for idx in range(len(session.stages))]
        session.progress = Progress(bars)

    def load_init_section(self, session, spec, section_name, loader, installer):
        init_dict = spec["initialize"]

        section_spec = init_dict[section_name]

        if isinstance(section_spec, dict):
            return {name: self.build_object(session, spec, loader, installer, name)
                    for name, spec in section_spec.items()}

        if isinstance(section_spec, list):
            return [self.build_object(session, d, loader, installer) for d in section_spec]

    def build_object(self, session, spec, loader, installer, name=None):
        instance = loader.load(session, spec, name)
        installer.setup(session, instance, object_name=name)
        return instance


class SessionRestorer(SessionInitializer):
    installers = defaultdict(ObjectInstaller)

    def __init__(self, static_dir):
        super().__init__()
        self.static_dir = static_dir
        self.fit_preprocessors = False

    def load_section(self, session, spec, section_name, loader, installer):
        super().load_section(session, spec, section_name, loader, installer)
        persistence = SectionPersistence(self.static_dir)
        persistence.load(session, section_name)


class SectionPersistence:
    def __init__(self, static_dir):
        self.static_dir = static_dir

    def save(self, session, section_name):
        objects_dict = getattr(session, section_name)

        path = self._build_save_path(section_name)
        serialized_dict = {}
        for name, obj in objects_dict.items():
            if hasattr(obj, 'state_dict'):
                serialized_dict[name] = obj.state_dict()
            else:
                serialized_dict[name] = {}
        save_as_json(serialized_dict, path)

    def load(self, session, section_name):
        objects_dict = getattr(session, section_name)

        path = self._build_save_path(section_name)
        if not os.path.exists(path):
            return

        serialized_dict = load_json(path)

        for name, object_state in serialized_dict.items():
            an_object = objects_dict[name]
            if hasattr(an_object, 'load_state_dict'):
                an_object.load_state_dict(object_state)

    def _build_save_path(self, section_name):
        return os.path.join(self.static_dir, f'{section_name}.json')


# todo: when creating session: 1) parse objects and instantiate them; 2) configure them; 3) use
#       when restoring session: 1) parse objects and instantiate them; 2) load state; 3) use
# 3 sets of subclasses: builders (or parsers), configurators, state loaders
