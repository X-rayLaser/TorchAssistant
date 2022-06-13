import csv
import json
import os

import torch

from scaffolding import parse
from scaffolding.parse import Node, SerializableModel, SerializableOptimizer
from scaffolding.training import PredictionPipeline
from scaffolding.utils import instantiate_class, change_model_device, GenericSerializableInstance


def parse_adapter(batch_adapter_config):
    def instantiate_fn(class_name, *args, **kwargs):
        args = (model,) + args
        return instantiate_class(class_name, *args, *kwargs)

    if batch_adapter_config:
        return parse.build_generic_serializable_instance(batch_adapter_config)
    else:
        # todo: default adapter
        pass


class TrainingSession:
    def __init__(self, path):
        self.path = path

        self.checkpoints_dir = os.path.join(path, 'checkpoints')
        self.history_path = os.path.join(path, 'history.csv')

        self.data_pipeline_path = os.path.join(path, 'data_pipeline.json')
        self.batch_adapter_path = os.path.join(path, 'batch_adapter.json')
        self.extra_params_path = os.path.join(path, 'extra_params.json')

        self.data_pipeline = load_data_pipeline(self.data_pipeline_path)

        d = load_json(self.batch_adapter_path)
        self.batch_adapter = parse.GenericSerializableInstance.from_dict(d)

        self.extra_params = load_json(self.extra_params_path)
        self.device = torch.device(self.extra_params["device"])
        self.num_epochs = self.extra_params["num_epochs"]

        metrics_dict = self.extra_params.get("metrics", {})
        self.metrics = parse.parse_metrics(metrics_dict, self.data_pipeline, self.device)

        loss_config = self.extra_params.get("loss", {})
        self.criterion = parse.parse_loss(loss_config, self.device)

    @property
    def epochs_trained(self):
        _, dir_names, _ = next(os.walk(self.checkpoints_dir))
        # first checkpoint is for untrained model, therefore we subtract it
        return len(dir_names) - 1

    def restore_from_last_checkpoint(self, inference_mode=False):
        model = load_last_checkpoint(self.checkpoints_dir, self.device, inference_mode)
        change_model_device(model, self.data_pipeline.device_str)

        train_pipeline = PredictionPipeline(model, self.device, self.batch_adapter)

        return train_pipeline

    def make_checkpoint(self, train_pipeline, epoch):
        save_model_pipeline(train_pipeline, epoch, self.checkpoints_dir)

    def log_metrics(self, epoch, train_metrics, val_metrics):
        # todo: log metrics to csv file
        history = TrainingHistory(self.history_path)
        history.add_entry(epoch, train_metrics, val_metrics)

    @classmethod
    def create_session(cls, config, save_path):
        SessionBuilder(config).create(save_path)


class SessionBuilder:
    def __init__(self, config):
        self.config = config

    def create2(self):
        datasets = self.config.datasets
        preprocessors = self.config.preprocessors
        models = self.config.models
        preprocessor_learners = self.config.preprocessor_learners

        for learner in preprocessor_learners:
            preprocessor = preprocessors[learner.preprocessor_name]
            ds = datasets[learner.dataset_name]
            preprocessor.fit(ds)

        state = SessionState(datasets=datasets, preprocessors=preprocessors,
                             collators=self.config.collators, models=models,
                             pipelines=self.config.pipelines,
                             stop_condition=self.config.stop_condition,
                             device=self.config.device_str)

        return state

    def save_data(self):
        pass

    def create(self, save_path):
        checkpoints_dir = os.path.join(save_path, 'checkpoints')
        history_path = os.path.join(save_path, 'history.csv')
        data_pipeline_path = os.path.join(save_path, 'data_pipeline.json')
        batch_adapter_path = os.path.join(save_path, 'batch_adapter.json')
        extra_params_path = os.path.join(save_path, 'extra_params.json')

        os.makedirs(checkpoints_dir, exist_ok=True)
        config = self.config

        data_pipeline = config.data_pipeline
        save_data_pipeline(data_pipeline, data_pipeline_path)

        model = config.model
        change_model_device(model, data_pipeline.device_str)
        save_model_pipeline(model, 0, checkpoints_dir)

        batch_adapter = config.batch_adapter
        save_as_json(batch_adapter.to_dict(), batch_adapter_path)

        extra_params = self.get_extra_params()
        save_as_json(extra_params, extra_params_path)

        metrics_dict = extra_params.get("metrics", [])
        field_names = list(metrics_dict.keys()) + [f'val {name}' for name in metrics_dict.keys()]

        history = TrainingHistory.create(history_path, field_names)
        # todo: calculate metrics for 0-th epoch (before any training)

    def get_extra_params(self):
        extra_params = {}

        extra_params["device"] = self.config.device_str

        if self.config.loss:
            extra_params["loss"] = self.config.loss

        if self.config.metrics:
            extra_params["metrics"] = self.config.metrics

        extra_params["num_epochs"] = self.config.num_epochs
        return extra_params


class SessionState:
    def __init__(self, *, datasets, preprocessors,
                 #split_fn,
                 collators, models, pipelines, stop_condition, device, epoch=0):
        self.datasets = datasets
        #self.split_fn = split_fn
        self.preprocessors = preprocessors
        self.collators = collators
        self.models = models
        self.pipelines = pipelines or []
        self.stop_condition = stop_condition
        self.device = device

        # tracks progress
        self.epoch = epoch

    def add_dataset(self, name, dataset):
        self.datasets[name] = dataset

    def add_preprocessor(self, name, preprocessor):
        self.preprocessors[name] = preprocessor

    def add_collator(self, name, collator):
        self.collators[name] = collator

    def add_model(self, name, model):
        self.models[name] = model

    def add_pipeline(self, pipeline):
        # todo: check errors
        self.pipelines.append(pipeline)

    def make_neural_node(self, node_name, model_name, input_vars, output_vars, optimizer):
        return Node(node_name, serializable_model=self.models[model_name],
                    serializable_optimizer=optimizer, inputs=input_vars, outputs=output_vars)

    def make_neural_graph(self, batch_adapter, *nodes):
        return


class Pipeline:
    def __init__(self, *,
                 ds_name,
                 preprocessor_names,
                 collator_name,
                 batch_size,
                 batch_adapter,
                 neural_graph,
                 loss_fn,
                 metrics):
        pass


class CheckpointKeeper:
    def __init__(self, storage_path):
        self.shared_folder = os.path.join(storage_path, 'shared')
        self.checkpoints_folder = os.path.join(storage_path, 'checkpoints')

        self.datasets_path = os.path.join(self.shared_folder, 'datasets.json')
        self.preprocessors_path = os.path.join(self.shared_folder, 'preprocessors.json')
        self.collators_path = os.path.join(self.shared_folder, 'collators.json')
        self.extra_params_path = os.path.join(self.shared_folder, 'extra_params.json')

    def save_checkpoint(self, session, name):
        """

        :param session:
        :type session:
        :param name: checkpoint name
        :return: None
        """
        # todo: implement load function
        # todo: use this in TrainingSession class
        # todo: build state from config
        # todo: interleaving training loops
        # todo: integrate with the rest
        # todo: multiple stages
        # todo: callbacks changing dataset(or use different transform) or model architecture

        checkpoint_folder = os.path.join(self.checkpoints_folder, name)
        self._save_shared_data(session)
        self._save_models(session, checkpoint_folder)
        self._save_pipelines(session, checkpoint_folder)

    def _save_shared_data(self, session):
        if not os.path.exists(self.shared_folder):
            os.makedirs(self.shared_folder)

        name2ds = {name: dataset.to_dict() for name, dataset in session.datasets.items()}
        name2preprocessor = {name: p.to_dict() for name, p in session.preprocessors.items()}
        name2collator = {name: collator.to_dict() for name, collator in session.collators.items()}

        save_as_json(name2ds, self.datasets_path)
        save_as_json(name2preprocessor, self.preprocessors_path)
        save_as_json(name2collator, self.collators_path)

        stop_condition = session.stop_condition.to_dict()

        extra_params = dict(stop_condition=stop_condition, device=str(session.device))
        save_as_json(extra_params, self.extra_params_path)

    def _save_models(self, session, checkpoint_folder):
        os.makedirs(checkpoint_folder, exist_ok=True)

        models_folder = os.path.join(checkpoint_folder, 'models')
        os.makedirs(models_folder, exist_ok=True)

        for model_name, model in session.models.items():
            model_path = os.path.join(models_folder, f'{model_name}.pt')
            torch.save(model.state_dict(), model_path)

    def _save_pipelines(self, session, checkpoint_folder):
        pipelines_folder = os.path.join(checkpoint_folder, 'pipelines')
        os.makedirs(pipelines_folder, exist_ok=True)

        for pipeline in session.pipelines:
            pipeline_name = pipeline.name
            pipeline_dict = dict(
                ds_name=pipeline.ds_name,
                preprocessor_names=pipeline.preprocessor_names,
                collator_name=pipeline.collator_name,
                batch_size=pipeline.batch_size,
                batch_adapter=pipeline.batch_adapter,
                neural_graph=pipeline.neural_graph.to_dict(),
                loss_fn=pipeline.loss_fn.to_dict(),
                metrics=pipeline.metrics.to_dict()
            )
            pipeline_path = os.path.join(pipelines_folder, f'{pipeline_name}.json')
            save_as_json(pipeline_dict, pipeline_path)

    def restore_from_checkpoint(self, name):
        # todo: first, check if it exists
        shared_dict = self._load_shared_data()

    def _load_shared_data(self):
        name2ds = load_json(self.datasets_path)
        name2preprocessor = load_json(self.preprocessors_path)
        name2collator = load_json(self.collators_path)
        extra_params = load_json(self.extra_params_path)

        datasets = {k: GenericSerializableInstance.from_dict(v) for k, v in name2ds.items()}
        preprocessors = {k: GenericSerializableInstance.from_dict(v) for k, v in name2preprocessor.items()}
        collators = {k: GenericSerializableInstance.from_dict(v) for k, v in name2collator.items()}

        stop_condition = GenericSerializableInstance.from_dict(extra_params["stop_condition"])
        device = torch.device(extra_params["device"])

        return dict(datasets=datasets, preprocessors=preprocessors, collators=collators,
                    stop_condition=stop_condition, device=device)

    def _load_models(self, checkpoint_folder, device):
        models_folder = os.path.join(checkpoint_folder, 'models')

        for file_name in os.listdir(models_folder):
            path = os.path.join(models_folder, file_name)
            checkpoint = torch.load(path)

            # this will probably not work
            serializable_model = SerializableModel.from_dict(checkpoint)
            serializable_model.instance.to(device)
            serializable_optimizer = SerializableOptimizer.from_dict(
                checkpoint, serializable_model.instance
            )


class ConfigParser:
    def __init__(self, settings, factory=None):
        self.settings = settings
        self.factory = factory or instantiate_class

    def get_config(self):
        # todo: implement parsing losses and metrics
        #  (implementation should be simple without passing data pipelines or anything extra)
        try:
            init_dict = self.settings["initialize"]
        except KeyError:
            raise BadSpecificationError(f'Missing "initialize" section.')

        try:
            train_dict = self.settings["train"]
        except KeyError:
            raise BadSpecificationError(f'Missing "train" section.')

        datasets_spec = init_dict.get("datasets", {})
        if not datasets_spec:
            raise BadSpecificationError(f'Missing "datasets" definition in "initialize" section.')

        datasets = self.parse_datasets(datasets_spec)

        preprocessors_spec = init_dict.get("preprocessors", {})
        preprocessors = self.parse_preprocessors(preprocessors_spec)

        learners = init_dict.get("preprocessor_learners", {})
        learners = {Learner(proc_name, ds_name) for proc_name, ds_name in learners.items()}

        collators_spec = init_dict.get("collators", {})
        collators = self.parse_collators(collators_spec)

        models_spec = init_dict.get("models", {})
        if not models_spec:
            raise BadSpecificationError()

        models = self.parse_models(models_spec)

        losses_spec = init_dict.get("losses", {})
        if not losses_spec:
            raise BadSpecificationError()

        losses = self.parse_loss_functions(losses_spec)

        metrics_spec = init_dict.get("metrics", {})

        if not metrics_spec:
            raise BadSpecificationError()

        metrics = self.parse_metrics(metrics_spec)

        pipelines_spec = train_dict["pipelines"]
        pipelines = self.parse_pipelines(pipelines_spec)

        stop_condition_spec = train_dict.get("stop_condition")
        if not stop_condition_spec:
            raise BadSpecificationError()

        stop_condition = self.parse_spec(stop_condition_spec)
        device_str = train_dict.get("device", "cpu")

        return Configuration(datasets=datasets,
                             preprocessors=preprocessors,
                             preprocessor_learners=learners,
                             collators=collators,
                             models=models,
                             loss_functions=losses,
                             metrics=metrics,
                             pipelines=pipelines,
                             stop_condition=stop_condition,
                             device_str=device_str)

    def parse_datasets(self, datasets_spec):
        # todo: handle built-in datasets later
        return self.parse_spec(datasets_spec)

    def parse_preprocessors(self, preprocessors_spec):
        return self.parse_spec(preprocessors_spec)

    def parse_collators(self, collators_spec):
        return self.parse_spec(collators_spec)

    def parse_models(self, models_spec):
        return self.parse_spec(models_spec)

    def parse_loss_functions(self, losses_spec):
        pass

    def parse_metrics(self, metrics_spec):
        pass

    def parse_pipelines(self, pipelines_spec):
        pipelines = []
        for spec in pipelines_spec:
            pipeline = Pipeline(ds_name=spec['dataset_name'],
                                preprocessor_names=spec['preprocessor_names'],
                                collator_name=spec['collator_name'],
                                batch_size=spec['batch_size'],
                                batch_adapter=spec['batch_adapter'],
                                neural_graph=spec['neural_graph'],
                                loss_fn=spec['loss_name'],
                                metrics=spec['metric_names'])
            pipelines.append(pipeline)
        return pipelines

    def parse_spec(self, spec):
        res = {}
        for name, spec in spec.items():
            res[name] = SpecParser(factory=self.factory).parse(spec)
        return res


class Learner:
    def __init__(self, preprocessor_name, dataset_name):
        self.preprocessor_name = preprocessor_name
        self.dataset_name = dataset_name


class SpecParser:
    def __init__(self, factory=None):
        self.factory = factory or instantiate_class

    def parse(self, spec_dict):
        if not isinstance(spec_dict, dict):
            raise BadSpecificationError(f'SpecParser: spec_dict should be a dictionary. Got {type(spec_dict)}.')

        try:
            class_name = spec_dict["class"]
        except KeyError:
            raise BadSpecificationError('SpecParser: missing "class" key')

        if not isinstance(class_name, str):
            raise BadSpecificationError(f'SpecParser: class must be a string. Got {type(class_name)}')

        # splitter_spec = spec.get("splitter")
        args = spec_dict.get("args", [])
        kwargs = spec_dict.get("kwargs", {})
        return self.factory(class_name, *args, **kwargs)


class MetricParser(SpecParser):
    def parse(self, spec_dict):
        pass


class NoSplitFactory:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_split(self):
        return [self.dataset]


class DataSplitFactory:
    def __init__(self, dataset, splitter):
        self.dataset = dataset
        self.splitter = splitter
        self.splitter.prepare(self.dataset)

    def get_split(self):
        return self.splitter.parts


class OldConfigParser:
    def __init__(self, settings):
        self.settings = settings
        self.training_config = settings["training"]

    def get_config(self):
        config = OldConfiguration(
            session_dir=self.parse_checkpoint_dir(),
            num_epochs=self.parse_epochs(),
            data_pipeline=self.parse_data_pipeline(),
            model=self.parse_model_pipeline(),
            batch_adapter=self.parse_batch_adapter(),
            device_str=self.parse_device(),
            loss=self.parse_loss_fn(),
            metrics=self.parse_metrics()
        )

        return config

    def parse_checkpoint_dir(self):
        return parse.parse_checkpoint_dir(self.settings)

    def parse_epochs(self):
        return parse.parse_epochs(self.settings)

    def parse_data_pipeline(self):
        return parse.DataPipeline.create(self.settings)

    def parse_model_pipeline(self):
        return parse.parse_model(self.settings)

    def parse_batch_adapter(self):
        return parse_adapter(self.settings["training"].get("batch_adapter"))

    def parse_device(self):
        return self.training_config.get("device", "cpu")

    def parse_loss_fn(self):
        return self.training_config.get("loss")

    def parse_metrics(self):
        return self.training_config.get("metrics")


class OldConfiguration:
    def __init__(self, *, session_dir, num_epochs, data_pipeline, model, batch_adapter,
                 device_str, loss, metrics):
        self.session_dir = session_dir
        self.num_epochs = num_epochs
        self.data_pipeline = data_pipeline
        self.model = model
        self.batch_adapter = batch_adapter
        self.device_str = device_str
        self.loss = loss
        self.metrics = metrics


class Configuration:
    def __init__(self, *, datasets, preprocessors, preprocessor_learners,
                 collators, models, loss_functions, metrics, pipelines,
                 stop_condition, device_str):
        self.datasets = datasets
        self.preprocessors = preprocessors
        self.collators = collators
        self.models = models
        self.loss_functions = loss_functions
        self.metrics = metrics
        self.pipelines = pipelines
        self.stop_condition = stop_condition
        self.device_str = device_str

"""
class SessionConfiguration:
    def __init__(self):
        pass

    def setup_init_stage(self, dataset_configs, transform_learner_configs):
        pass

    def setup_training_stage(self):
        pass


class DatasetConfig:
    def __init__(self, name, kwargs):
        pass


class TransformLearnerConfig:
    def __init__(self, dataset_name, splitter_name, transform_name):
        pass
"""


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


def load_json(path):
    with open(path, encoding='utf-8') as f:
        s = f.read()

    return json.loads(s)


def save_as_json(d, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(d))


def load_checkpoint(checkpoints_dir, epoch, device, inference_mode=False):
    epoch_dir = os.path.join(checkpoints_dir, str(epoch))

    nodes_with_numbers = []
    for file_name in os.listdir(epoch_dir):
        path = os.path.join(epoch_dir, file_name)
        checkpoint = torch.load(path)

        serializable_model = SerializableModel.from_dict(checkpoint)
        serializable_model.instance.to(device)
        serializable_optimizer = SerializableOptimizer.from_dict(
            checkpoint, serializable_model.instance
        )

        # todo: consider doing this outside the function call
        if inference_mode:
            serializable_model.instance.eval()
        else:
            serializable_model.instance.train()

        node = Node(name=checkpoint["name"], serializable_model=serializable_model,
                    serializable_optimizer=serializable_optimizer, inputs=checkpoint["inputs"],
                    outputs=checkpoint["outputs"])
        nodes_with_numbers.append((node, checkpoint["number"]))

    nodes_with_numbers.sort(key=lambda t: t[1])
    return [t[0] for t in nodes_with_numbers]


def load_last_checkpoint(epochs_dir, device, inference_mode=False):
    last_epoch = sorted(os.listdir(epochs_dir), key=lambda d: int(d), reverse=True)[0]
    return load_checkpoint(epochs_dir, int(last_epoch), device, inference_mode)


def save_model_pipeline(train_pipeline, epoch, checkpoints_dir):
    epoch_dir = os.path.join(checkpoints_dir, str(epoch))

    os.makedirs(epoch_dir, exist_ok=True)

    for number, pipe in enumerate(train_pipeline, start=1):
        save_path = os.path.join(epoch_dir, pipe.name)
        d = {
            'name': pipe.name,
            'number': number,
            'inputs': pipe.inputs,
            'outputs': pipe.outputs,
            'epoch': epoch
        }
        d.update(pipe.net.to_dict())
        d.update(pipe.optimizer.to_dict())

        torch.save(d, save_path)


def save_data_pipeline(data_pipeline, path):
    with open(path, 'w', encoding='utf-8') as f:
        state_dict = data_pipeline.to_dict()
        s = json.dumps(state_dict)
        f.write(s)


def load_data_pipeline(path):
    with open(path, encoding='utf-8') as f:
        s = f.read()
        state_dict = json.loads(s)
        return parse.DataPipeline.from_dict(state_dict)


class BadSpecificationError(Exception):
    pass
