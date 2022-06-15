import os
import json
import torch

from scaffolding.session_v2.persistence import CheckpointKeeper
from scaffolding.session_v2.parse import SessionInitializer
from .parse import SessionRestorer


def create_session(spec):
    session = Session()
    initializer = SessionInitializer()
    initializer(session, spec)
    return session


def create_and_save_session(spec, session_dir):
    session = create_session(spec)
    saver = SessionSaver(session_dir)
    saver.save(session, spec)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        s = f.read()

    return json.loads(s)


def save_as_json(d, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(d))


def restore_session(session_dir):
    saver = SessionSaver(session_dir)
    saver.load()


def save_session(state, checkpoints_dir, name):
    keeper = CheckpointKeeper(checkpoints_dir)
    keeper.save(state, name)


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
        self.epoch = 0

    def initialize_state(self):
        return State(models=self.models, optimizers=self.optimizers)
        optimizers = {}
        for pipeline_name, pipeline in self.pipelines.items():
            for model_node in pipeline.neural_graph.models:
                optimizers[pipeline_name][model_node.name] = model_node.optimizer

        return State(models=self.models, optimizers=optimizers)


class SessionSaver:
    save_attrs = 'datasets preprocessors collators batch_adapters losses metrics'.split(' ')

    def __init__(self, session_dir):
        self.session_dir = session_dir

        self.spec_path = os.path.join(session_dir, 'spec.json')
        self.static_dir = os.path.join(session_dir, 'static')
        self.checkpoints_dir = os.path.join(session_dir, 'checkpoints')
        self.extra_path = os.path.join(self.static_dir, 'extra_params.json')

    def save(self, session, spec):
        os.makedirs(self.session_dir)
        os.makedirs(self.static_dir)
        os.makedirs(self.checkpoints_dir)

        self.save_spec(spec)
        self.save_static(session)
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

    def save_spec(self, spec):
        save_as_json(spec, self.spec_path)

    def save_static(self, session):
        static_dir = self.static_dir

        for attr in self.save_attrs:
            objects_dict = getattr(session, attr)

            path = os.path.join(static_dir, f'{attr}.json')
            serialized_dict = {}
            for name, obj in objects_dict.items():
                if hasattr(obj, 'state_dict'):
                    serialized_dict[name] = obj.state_dict()
                else:
                    serialized_dict[name] = {}
            save_as_json(serialized_dict, path)

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
