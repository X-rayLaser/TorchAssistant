import torch


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
    def __init__(self, neural_nodes, input_adapter, output_adapter, device, inference_mode=False):
        self.neural_nodes = neural_nodes
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.device = device
        self.inference_mode = inference_mode

    def __call__(self, batch):
        inputs = self.input_adapter(batch)
        self.change_model_device()

        self.inputs_to(inputs)

        all_outputs = {}
        for node in self.neural_nodes:
            outputs = node(inputs, all_outputs, self.inference_mode)
            all_outputs.update(
                dict(zip(node.outputs, outputs))
            )

        result_dict = dict(batch)
        result_dict.update(all_outputs)
        res = self.output_adapter(result_dict)
        return res

    def change_model_device(self):
        for model in self.neural_nodes:
            model.net.to(self.device)

    def inputs_to(self, inputs):
        for k, mapping in inputs.items():
            for tensor_name, value in mapping.items():
                if hasattr(value, 'device') and value.device != self.device:
                    mapping[tensor_name] = value.to(self.device)

    def prepare(self):
        for model in self.neural_nodes:
            if model.optimizer:
                model.optimizer.zero_grad()

    def update(self):
        for node in self.neural_nodes:
            if node.optimizer:
                node.optimizer.step()

    def train_mode(self):
        for node in self.neural_nodes:
            node.net.train()

    def eval_mode(self):
        for node in self.neural_nodes:
            node.net.eval()


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

    def __call__(self, data_frames: dict) -> dict:
        """Propagate given number of batches through the graph and compute results.

        :param data_frames: a mapping from name to data_frame object
        :return: outputs of leaf nodes as a data_frame_dict
        """
        self.invalidate_cache()
        return {leaf: self.backtrace(leaf, data_frames) for leaf in self.leaves}

    def invalidate_cache(self):
        self.cache = {}

    def backtrace(self, name, batches):
        # todo: replace batches with data_frames
        # todo: more renaming of similar nature in other modules
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


class Node:
    def __init__(self, name, model, optimizer, inputs, outputs):
        self.name = name
        self.net = model
        self.optimizer = optimizer
        self.inputs = inputs
        self.outputs = outputs

    def get_dependencies(self, batch_inputs, prev_outputs):
        # todo: double check this line
        lookup_table = batch_inputs.get(self.name, {}).copy()
        lookup_table.update(prev_outputs)
        return [lookup_table[var_name] for var_name in self.inputs]

    def predict(self, *args, inference_mode=False):
        # todo: consider to change args device here (need to store device as attribute)
        if inference_mode:
            return self.net.run_inference(*args)
        else:
            return self.net(*args)

    def __call__(self, batch_inputs, prev_outputs, inference_mode=False):
        args = self.get_dependencies(batch_inputs, prev_outputs)
        return self.predict(*args, inference_mode=inference_mode)
