import os
import json
from collections import defaultdict
from random import shuffle

import torch

from . import parse


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


class State:
    def __init__(self, models, optimizers, metrics=None, epochs_done=0):
        self.models = models
        self.optimizers = optimizers
        self.epochs_done = epochs_done


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
        self.stop_condition = None
        self.device = None

        # tracks progress
        self.epochs_done = 0

    def initialize_state(self):
        return State(models=self.models, optimizers=self.optimizers, epochs_done=self.epochs_done)


class SessionSaver:
    save_attrs = 'datasets splits preprocessors collators batch_adapters losses metrics'.split(' ')

    def __init__(self, session_dir):
        self.session_dir = session_dir

        self.spec_path = os.path.join(session_dir, 'spec.json')
        self.static_dir = os.path.join(session_dir, 'static')
        self.checkpoints_dir = os.path.join(session_dir, 'checkpoints')
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

        extra_params = load_json(self.extra_path)
        if hasattr(session.stop_condition, 'load_state_dict'):
            condition_state = extra_params.get("stop_condition", {})
            session.stop_condition.load_state_dict(condition_state)

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

        extra_params = {}
        if hasattr(session.stop_condition, 'state_dict'):
            extra_params["stop_condition"] = session.stop_condition.state_dict()
        save_as_json(extra_params, self.extra_path)

    def save_checkpoint(self, session):
        state_dir = self.checkpoints_dir
        # todo: map locations when needed
        state = session.initialize_state()

        checkpoint_dir = os.path.join(state_dir, str(state.epochs_done))
        os.makedirs(checkpoint_dir)
        save_path = os.path.join(checkpoint_dir, 'checkpoint.pt')

        models_dict = {name: model.state_dict() for name, model in state.models.items()}

        optimizers_dict = {name: optimizer.state_dict() for name, optimizer in state.optimizers.items()}

        progress_dict = dict(epochs_done=state.epochs_done)

        torch.save({
            'models': models_dict,
            'optimizers': optimizers_dict,
            'progress': progress_dict
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

        state.epochs_done = checkpoint["progress"]["epochs_done"]


class ObjectInstaller:
    def setup(self, session, instance, object_name=None, **kwargs):
        pass


class PreProcessorInstaller(ObjectInstaller):
    def setup(self, session, instance, object_name=None, **kwargs):
        dataset_name = instance.spec["fit"]
        if '.' in dataset_name:
            splitter_name, slice_name = dataset_name.split('.')
            splitter = session.splits[splitter_name]
            split = splitter.split(session.datasets[splitter.spec["dataset_name"]])
            dataset = getattr(split, slice_name)
        else:
            dataset = session.datasets[dataset_name]
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
            ('metrics', parse.MetricLoader())
        ]

        for section_name, loader in sections_with_loaders:
            installer = self.installers[section_name]
            self.load_section(session, spec, section_name, loader, installer)

        session.stop_condition = parse.Loader().load(session, spec["train"]["stop_condition"])
        session.device = torch.device(spec["train"].get("device", "cpu"))

    def load_section(self, session, spec, section_name, loader, installer):
        section = self.load_init_section(session, spec, section_name, loader, installer)
        setattr(session, section_name, section)

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
