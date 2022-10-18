import torch


class BatchProcessor:
    def prepare(self):
        pass

    def update(self):
        pass

    def __call__(self, data_frame):
        return data_frame

    def train_mode(self):
        pass

    def eval_mode(self):
        pass


class DetachBatch(BatchProcessor):
    def __call__(self, data_frame):
        return {name: tensor.detach() for name, tensor in data_frame.items()}


class BatchMerger:
    def __call__(self, data_frames: list):
        """Concatenates a list of data frames.

        :param data_frames: a list of data frames
        :return: a data frame containing the same column name, with columns concatenated along batch axis
        """

        if not data_frames:
            return {}

        a_frame = max(data_frames, key=len)
        result = {}
        try:
            for k in a_frame.keys():
                tensors = [data_frame[k].to(torch.device("cpu")) for data_frame in data_frames]
                concatenation = torch.cat(tensors)
                result[k] = concatenation
        except KeyError as e:
            raise MergeError(f'Expects all data frames contain the same set of column names. '
                             f'Missing name in one of data frames: {e}')
        return result


class MergeError(Exception):
    pass


class NeuralBatchProcessor(BatchProcessor):
    def __init__(self, neural_nodes, input_adapter, output_adapter, device, inference_mode=False):
        self.neural_nodes = neural_nodes
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.device = device
        self.inference_mode = inference_mode

    def __call__(self, data_frame: dict):
        """

        :param data_frame: named batches to process
        :type data_frame: data_frame
        :return: results of processing
        :rtype: data_frame
        """
        try:
            inputs = self.input_adapter(data_frame)
        except Exception as e:
            raise InputAdapterError(repr(e))

        self.change_model_device()

        self.inputs_to(inputs)

        all_outputs = {}
        for node in self.neural_nodes:
            outputs = node(inputs, all_outputs, self.inference_mode)
            all_outputs.update(
                dict(zip(node.outputs, outputs))
            )

        result_dict = dict(data_frame)
        result_dict.update(all_outputs)
        try:
            res = self.output_adapter(result_dict)
        except Exception as e:
            raise OutputAdapterError(repr(e))

        return res

    def change_model_device(self):
        for model in self.neural_nodes:
            model.net.to(self.device)

    def inputs_to(self, inputs):
        for k, data_frame in inputs.items():
            for tensor_name, value in data_frame.items():
                if hasattr(value, 'device') and value.device != self.device:
                    data_frame[tensor_name] = value.to(self.device)

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


class InputAdapterError(Exception):
    pass


class OutputAdapterError(Exception):
    pass


class BatchProcessingGraph:
    def __init__(self, batch_input_names, **nodes):
        """

        :param batch_input_names:
        :type batch_input_names:
        :param nodes:
        :type nodes:
        """
        self.batch_input_names = set(batch_input_names)
        self.nodes = nodes
        self.ingoing_edges = {}
        self.outgoing_edges = {}
        self.cache = {}

        self._check_names_collision(self.batch_input_names, nodes.keys())

    def _check_names_collision(self, input_names, node_names):
        node_names = set(node_names)
        if node_names.intersection(input_names):
            raise InvalidGraphError(
                f'name collision between input nodes {input_names} and graph nodes {node_names}.'
            )

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
        self._validate_edge(source, dest)
        self.ingoing_edges.setdefault(dest, []).append(source)
        self.outgoing_edges.setdefault(source, []).append(dest)

    def _validate_edge(self, source, dest):
        both_input_nodes = source in self.batch_input_names and dest in self.batch_input_names
        all_names = self.batch_input_names.union(set(self.nodes.keys()))
        missing_nodes = {source, dest} - all_names

        if both_input_nodes:
            raise InvalidEdgeError(
                f'cannot make an edge between input nodes: "{source}"->"{dest}"'
            )

        if dest in self.batch_input_names:
            raise InvalidEdgeError(
                f'cannot make an edge "{source}"->"{dest}". '
                f'Input node "{dest}" is not allowed have ingoing edges'
            )

        if missing_nodes:
            raise InvalidEdgeError(
                f'cannot make an edge "{source}"->"{dest}". Nodes not found in a graph: {missing_nodes}'
            )

        if source == dest:
            raise InvalidEdgeError(
                f'cannot make an edge "{source}"->"{dest}". Self-loops are not allowed'
            )

        self._check_cycles(source, dest)

    def _check_cycles(self, source, dest):
        vertices = set(self.nodes.keys()).union(self.batch_input_names)
        digraph = DiGraph(list(vertices))
        for u in self.outgoing_edges:
            for v in self.outgoing_edges[u]:
                digraph.make_edge(u, v)

        digraph.make_edge(source, dest)
        cycles = digraph.detect_cycles()

        if cycles:
            raise InvalidEdgeError(
                f'cannot make an edge "{source}"->"{dest}". Cycles are not allowed: {cycles}'
            )

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

        if name not in self.ingoing_edges:
            raise DisconnectedGraphError(f'There are no edges pointing to the node "{name}".')

        ingoing_names = self.ingoing_edges[name]
        ingoing_batches = [self.backtrace(ingoing, batches) for ingoing in ingoing_names]

        if len(ingoing_batches) == 1:
            return node(ingoing_batches[0])

        merge = BatchMerger()
        merged_batch = merge(ingoing_batches)
        result = node(merged_batch)
        self.cache[name] = result
        return result


class DiGraph:
    def __init__(self, vertices, edges=None):
        """Simple representation of directed graph

        :param vertices: vertices or nodes comprising this graph
        """

        if len(vertices) != len(set(vertices)):
            raise InvalidGraphError(f'DiGraph not allowed to have duplicate vertices')

        self.vertices = vertices
        self.ingoing_edges = {}
        self.outgoing_edges = {}

        if edges:
            for u, v in edges:
                self.make_edge(u, v)

    def make_edge(self, u, v):
        self._validate_edge(u, v)
        self.outgoing_edges.setdefault(u, []).append(v)
        self.ingoing_edges.setdefault(v, []).append(u)

    def _validate_edge(self, u, v):
        missing_nodes = {u, v} - set(self.vertices)

        if u == v:
            raise InvalidEdgeError(
                f'cannot make an edge "{u}"->"{v}". Self-loops are not allowed'
            )

        if missing_nodes:
            raise InvalidEdgeError(
                f'cannot make an edge "{u}"->"{v}". Nodes not found in a graph: {missing_nodes}'
            )

        if v in self.outgoing_edges.get(u, []):
            raise InvalidEdgeError(
                f'cannot make an edge "{u}"->"{v}". Edge already exists'
            )

    def detect_cycles(self):
        components = self.strong_components()
        cycles = []
        for component in components.values():
            if len(component) > 1:
                cycles.append(component)
        return cycles

    def is_acyclic(self):
        return not bool(self.detect_cycles())

    def strong_components(self):
        """Computes strongly connected components in a directed graph.

        Implementation of Kosaraju's algorithm.

        :return: a sequence of strongly connected components
        :rtype: a sequence of sets
        """
        scc = SCCs(self)
        return scc()


class SCCs:
    def __init__(self, digraph):
        self.digraph = digraph

        self.visited = set()

        self.lifo_vertices = []  # vertices in the increasing order of finishing times
        self.leaders = {}
        self.leader = 0

    def __call__(self):
        self.run_dfs(self.digraph.vertices, reverse=True)
        self.run_dfs(reversed(self.lifo_vertices), reverse=False)
        return self.leaders

    def run_dfs(self, vertices, reverse):
        self.leaders.clear()
        self.visited.clear()

        for s in vertices:
            if s not in self.visited:
                self.leader = s
                self.dfs(s, reverse=reverse)

    def dfs(self, source, reverse):
        self.visited.add(source)
        edges = self.digraph.ingoing_edges if reverse else self.digraph.outgoing_edges
        for v in edges.get(source, []):
            if v not in self.visited:
                self.dfs(v, reverse)

        self.lifo_vertices.append(source)
        self.leaders.setdefault(self.leader, set()).add(source)


class InvalidGraphError(Exception):
    pass


class InvalidEdgeError(Exception):
    pass


class DisconnectedGraphError(Exception):
    pass


class Node:
    def __init__(self, name, model, optimizer, inputs, outputs):
        self.name = name
        self.net = model
        self.optimizer = optimizer
        self.inputs = inputs
        self.outputs = outputs

    def get_dependencies(self, batch_inputs: dict, prev_outputs: dict):
        """Retrieve quantities needed to process inputs."""
        # todo: pick a better name for arguments
        # todo: double check this line
        data_frame = batch_inputs.get(self.name, {})
        if not isinstance(data_frame, dict):
            fmt = """
            {
                "model_name1": {
                    "var_name1": "value1",
                    "var_name2": "value2",
                    ...
                },
                "model_name2": {
                    "var_name1": "value1",
                    "var_name1": "value1",
                    ...
                }
                ...
            }
            """
            raise InvalidFormatOfInputsError(
                f'"batch_inputs" should be a nested dict of shape: {fmt}\nGot {batch_inputs}'
            )
        lookup_table = data_frame.copy()
        lookup_table.update(prev_outputs)

        try:
            return [lookup_table[var_name] for var_name in self.inputs]
        except KeyError as e:
            name = e.args[0]
            raise DependencyNotFoundError(
                f'Node dependency is missing: "{name}". All node dependencies: {self.inputs}.'
            )

    def predict(self, *args, inference_mode=False):
        # todo: consider to change args device here (need to store device as attribute)
        if inference_mode and hasattr(self.net, 'run_inference'):
            return self.net.run_inference(*args)
        else:
            return self.net(*args)

    def __call__(self, batch_inputs, prev_outputs, inference_mode=False):
        args = self.get_dependencies(batch_inputs, prev_outputs)
        return self.predict(*args, inference_mode=inference_mode)


class DependencyNotFoundError(Exception):
    pass


class InvalidFormatOfInputsError(Exception):
    pass
