import torch
from scaffolding.utils import change_model_device, switch_to_train_mode, switch_to_evaluation_mode


class BatchProcessor:
    def prepare(self):
        pass

    def update(self):
        pass

    def __call__(self, batch):
        return batch

    def train_mode(self):
        pass

    def eval_mode(self):
        pass


class DetachBatch(BatchProcessor):
    def __call__(self, batch):
        return {name: tensor.detach() for name, tensor in batch.items()}


class BatchMerger:
    def __call__(self, batches: list):
        any_batch = batches[0]
        result = {}
        for k in any_batch.keys():
            tensors = [batch[k].to(torch.device("cpu")) for batch in batches]
            concatenation = torch.cat(tensors)
            result[k] = concatenation
        return result


class NeuralBatchProcessor(BatchProcessor):
    def __init__(self, neural_graph, input_adapter, output_adapter, device, inference_mode=False):
        self.neural_graph = neural_graph
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.device = device
        self.inference_mode = inference_mode

    def __call__(self, batch):
        inputs = self.input_adapter(batch)
        change_model_device(self.neural_graph, self.device)

        self.inputs_to(inputs)

        all_outputs = {}
        for node in self.neural_graph:
            outputs = node(inputs, all_outputs, self.inference_mode)
            all_outputs.update(
                dict(zip(node.outputs, outputs))
            )

        result_dict = dict(batch)
        result_dict.update(all_outputs)
        res = self.output_adapter(result_dict)
        return res

    def inputs_to(self, inputs):
        for k, mapping in inputs.items():
            for tensor_name, value in mapping.items():
                if hasattr(value, 'device') and value.device != self.device:
                    mapping[tensor_name] = value.to(self.device)

    def prepare(self):
        for model in self.neural_graph:
            if model.optimizer:
                model.optimizer.zero_grad()

    def update(self):
        for node in self.neural_graph:
            if node.optimizer:
                node.optimizer.step()

    def train_mode(self):
        switch_to_train_mode(self.neural_graph)

    def eval_mode(self):
        switch_to_evaluation_mode(self.neural_graph)


class BatchProcessingGraph:
    # todo: checking arguments, raising exceptions

    def __init__(self, batch_input_names, **nodes):
        self.batch_input_names = set(batch_input_names)
        self.nodes = nodes
        self.ingoing_edges = {}
        self.outgoing_edges = {}
        self.cache = {}

    def train_mode(self):
        for node in self.nodes.values():
            node.train_mode()

    def eval_mode(self):
        for node in self.nodes.values():
            node.eval_mode()

    def prepare(self):
        for _, node in self.nodes.items():
            node.prepare()

    def update(self):
        for _, node in self.nodes.items():
            node.update()

    def make_edge(self, source: str, dest: str):
        self.ingoing_edges.setdefault(dest, []).append(source)
        self.outgoing_edges.setdefault(source, []).append(dest)

    @property
    def leaves(self):
        return [name for name in self.nodes if name not in self.outgoing_edges]

    def __call__(self, batches: dict) -> dict:
        """Propagate given number of batches through the graph and compute results.

        :param batches: mapping from batch name to their value which is itself a mapping variable name -> tensor
        :return: outputs of leaf nodes as a name -> batch mapping
        """
        self.invalidate_cache()
        return {leaf: self.backtrace(leaf, batches) for leaf in self.leaves}

    def invalidate_cache(self):
        self.cache = {}

    def backtrace(self, name, batches):
        if name in self.batch_input_names:
            return batches[name]

        if name in self.cache:
            return self.cache[name]

        node = self.nodes[name]
        ingoing_names = self.ingoing_edges[name]
        ingoing_batches = [self.backtrace(ingoing, batches) for ingoing in ingoing_names]

        if len(ingoing_batches) == 1:
            return node(ingoing_batches[0])

        merge = BatchMerger()
        merged_batch = merge(ingoing_batches)
        result = node(merged_batch)
        self.cache[name] = result
        return result


class BatchLoader:
    def __init__(self, data_loader, var_names):
        self.data_loader = data_loader
        self.var_names = var_names

    def __iter__(self):
        for batch in self.data_loader:
            yield dict(zip(self.var_names, batch))

    def __len__(self):
        return len(self.data_loader)


class Trainer:
    def __init__(self, graph, data_generator, losses: dict, gradient_clippers):
        self.graph = graph
        self.data_generator = data_generator
        self.losses = losses
        self.gradient_clippers = gradient_clippers

    def __iter__(self):
        from .training import IterationLogEntry

        num_iterations = len(self.data_generator)

        inputs = []
        for i, batches in enumerate(self.data_generator):
            losses, results = self.train_one_iteration(batches)
            outputs = results
            targets = results
            # todo: fix this (may get rid of inputs and targets)
            yield IterationLogEntry(i, num_iterations, inputs, outputs, targets, losses)

    def train_one_iteration(self, graph_inputs):
        results = self.graph(graph_inputs)

        losses = {}
        for name, (node_name, loss_fn) in self.losses.items():
            batch = results[node_name]
            loss = loss_fn(batch, batch)

            # invoke zero_grad for each neural network
            self.graph.prepare()
            loss.backward()

            for clip_gradients in self.gradient_clippers.values():
                clip_gradients()

            # invoke optimizer.step() for every neural network if there is one
            self.graph.update()

            losses[node_name] = loss

        return losses, results


class DataBlueprint:
    def __init__(self, input_loaders):
        self.input_loaders = input_loaders

        batch_names = [input_loader.input_alias for input_loader in self.input_loaders]
        self.batch_names = batch_names
        self.batch_loaders = self.refresh_batch_loaders(self.input_loaders)

    def override_datasets(self, new_datasets: dict):
        for input_loader in self.input_loaders:
            dataset = new_datasets[input_loader.input_alias]
            input_loader.loader_factory.swap_dataset(dataset)

        self.batch_loaders = self.refresh_batch_loaders(self.input_loaders)

    def refresh_batch_loaders(self, input_loaders):
        batch_loaders = []
        for input_loader in input_loaders:
            var_names = input_loader.variable_names
            loader_factory = input_loader.loader_factory
            data_loader = loader_factory.build()
            batch_loaders.append(BatchLoader(data_loader, var_names))
        return batch_loaders

    def __len__(self):
        return min(map(len, self.batch_loaders))

    def __iter__(self):
        iterators = [iter(loader) for loader in self.batch_loaders]
        for i, batches in enumerate(zip(*iterators)):
            named_batches = dict(zip(self.batch_names, batches))
            yield named_batches


class LoaderFactory:
    def __init__(self, dataset, collator, **kwargs):
        self.dataset = dataset
        self.collator = collator
        self.kwargs = kwargs

    def build(self):
        return torch.utils.data.DataLoader(
            self.dataset, collate_fn=self.collator, **self.kwargs
        )

    def swap_dataset(self, dataset):
        preprocessors = self.dataset.get_preprocessors()
        self.dataset = dataset
        self.dataset.preprocessors = preprocessors
