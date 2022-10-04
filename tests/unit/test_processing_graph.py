import unittest
from torchassistant import processing_graph
import torch
from operator import mul


def square(x): return x**2
def do_nothing(): return 42


class InferenceModel:
    def run_inference(self, x):
        return x ** 2


class SquareModel(torch.nn.Module):
    def forward(self, x):
        return x**2,


class AdditionModel(torch.nn.Module):
    def forward(self, *terms):
        return sum(terms),


def void_adapter(data_frame):
    return {}


def identity_adapter(data_frame):
    return data_frame


def raising_adapter(data_frame):
    data_frame["some_missing_key"]
    return {}


class TrainableLinearModel(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, 1)

    def forward(self, *x):
        if len(x) > 1:
            x = torch.cat(x, dim=1)
        else:
            x = x[0]
        return self.linear(x),


class TrainableNodeTests(unittest.TestCase):
    def test_error_raised_when_predicting_on_wrong_input_dict(self):
        inputs = ["x"]
        outputs = []
        node = processing_graph.Node('my model', square, optimizer=None, inputs=inputs, outputs=outputs)
        d = {"my model": {}}
        self.assertRaises(processing_graph.DependencyNotFoundError, node, d, {})
        self.assertRaises(processing_graph.DependencyNotFoundError, node, {}, {})

        d = {"my model": 43}
        self.assertRaises(processing_graph.InvalidFormatOfInputsError, node, d, {})

    def test_predicting_should_not_modify_passed_arguments(self):
        inputs = ["x"]
        node = processing_graph.Node('my model', square, optimizer=None, inputs=inputs, outputs=[])
        d1 = {
            "my model": {
                "y": 20
            }
        }

        d2 = {'two': 2, "x": torch.tensor(10)}

        node(d1, d2)
        self.assertEqual({
            "my model": {
                "y": 20
            }
        }, d1)
        self.assertEqual({'two': 2, "x": torch.tensor(10)}, d2)

    def test_make_prediction_using_no_arguments(self):
        node = processing_graph.Node('my model', do_nothing, optimizer=None, inputs=[], outputs=[])
        d = {"my model": {}}
        self.assertEqual(42, node(d, {}))
        self.assertEqual(42, node({}, {}))

    def test_make_prediction_using_1_argument(self):
        inputs = ["x"]
        node = processing_graph.Node('my model', square, optimizer=None, inputs=inputs, outputs=[])
        d = {
            "my model": {
                "x": torch.tensor(10),
                "y": 20
            },
            "something_else": {
                "x": torch.tensor(14)
            }
        }
        y_hat = node(d, {})
        self.assertEqual(100, y_hat.item())

    def test_make_prediction_using_2_arguments(self):
        inputs = ["x1", "x2"]
        node = processing_graph.Node('my model', mul, optimizer=None, inputs=inputs, outputs=[])
        d = {
            "my model": {
                "x1": torch.tensor(2),
                "x2": torch.tensor(3),
                "x3": 10
            }
        }
        y_hat = node(d, {})
        self.assertEqual(6, y_hat.item())

    def test_make_prediction_by_passing_dependency_via_previous_predictions(self):
        inputs = ["x1", "x2"]
        node = processing_graph.Node('my model', mul, optimizer=None, inputs=inputs, outputs=[])
        d = {
            "my model": {
                "x1": torch.tensor(2)
            }
        }
        y_hat = node(d, {
            'x2': torch.tensor(3)
        })
        self.assertEqual(6, y_hat.item())

        y_hat = node({}, {
            'x1': torch.tensor(2),
            'x2': torch.tensor(3)
        })
        self.assertEqual(6, y_hat.item())

    def test_make_predictions_with_inference_mode_turned_on(self):
        model = InferenceModel()
        inputs = ["x"]
        node = processing_graph.Node('my model', model, optimizer=None, inputs=inputs, outputs=[])
        d = {
            "my model": {
                "x": torch.tensor(10)
            }
        }
        y_hat = node(d, {}, inference_mode=True)
        self.assertEqual(100, y_hat.item())

        node = processing_graph.Node('my model', square, optimizer=None, inputs=inputs, outputs=[])
        y_hat = node(d, {}, inference_mode=True)
        self.assertEqual(100, y_hat.item())


class NeuralBatchProcessor(unittest.TestCase):
    def test_processor_that_does_nothing(self):
        device = torch.device('cpu')
        processor = processing_graph.NeuralBatchProcessor([], void_adapter, identity_adapter, device)
        res = processor({'a': [1, 2], 'b': [3, 4]})
        self.assertEqual(dict(a=[1, 2], b=[3, 4]), res)

    def test_output_adapter_determines_output(self):
        def output_adapter(data_frame): return {'a': data_frame['x'] + 50, 'b': 1}

        device = torch.device('cpu')
        processor = processing_graph.NeuralBatchProcessor([], void_adapter, output_adapter, device)
        self.assertEqual({'a': 100, 'b': 1}, processor(dict(x=50, y=20)))

    def test_input_adapter_raises_an_error(self):
        device = torch.device('cpu')
        processor = processing_graph.NeuralBatchProcessor([], raising_adapter, identity_adapter, device)
        self.assertRaises(processing_graph.InputAdapterError, processor, dict(x=50, y=20))

    def test_output_adapter_raises_an_error(self):
        def input_adapter(data_frame):
            return {
                "square": {"x": data_frame["x"]}
            }

        node1 = processing_graph.Node('square', SquareModel(), None, inputs=["x"], outputs=["y_hat"])

        device = torch.device('cpu')
        processor = processing_graph.NeuralBatchProcessor([node1], input_adapter, raising_adapter, device)
        self.assertRaises(processing_graph.OutputAdapterError, processor, dict(x=50, y=20))

    def test_missing_dependency_cases(self):
        def input_adapter1(data_frame): return {"add": {"x2": data_frame["x2"]}}

        def input_adapter2(data_frame): return {"square1": {"x1": data_frame["x1"]}}

        def input_adapter3(data_frame):
            return {"square1": {"x1": data_frame["x1"]}, "add": {"x2": data_frame["x2"]}}

        def output_adapter(data_frame): return {'res': data_frame['y_hat']}

        do_square = processing_graph.Node('square1', SquareModel(), None, inputs=["x1"], outputs=["s1"])
        add = processing_graph.Node('add', AdditionModel(), None, inputs=["s1", "x2"], outputs=["y_hat"])

        device = torch.device('cpu')
        for adapter in [input_adapter1, input_adapter2]:
            processor = processing_graph.NeuralBatchProcessor(
                [do_square, add], adapter, output_adapter, device
            )

            self.assertRaises(processing_graph.DependencyNotFoundError, processor, dict(x1=4, x2=3))

        add = processing_graph.Node('add', AdditionModel(), None, inputs=["x", "x2"], outputs=["y_hat"])
        processor = processing_graph.NeuralBatchProcessor(
            [do_square, add], input_adapter3, output_adapter, device
        )

        self.assertRaises(processing_graph.DependencyNotFoundError, processor, dict(x1=4, x2=3))

    def test_successful_processing(self):
        def input_adapter(data_frame):
            return {
                "square1": {"x1": data_frame["x1"]},
                "square2": {"x2": data_frame["x2"]},
                "add": {"x3": data_frame["x3"]}
            }

        square1 = processing_graph.Node('square1', SquareModel(), None, inputs=["x1"], outputs=["s1"])
        square2 = processing_graph.Node('square2', SquareModel(), None, inputs=["x2"], outputs=["s2"])
        add = processing_graph.Node('add', AdditionModel(), None, inputs=["s1", "s2", "x3"], outputs=["y_hat"])

        device = torch.device('cpu')
        processor = processing_graph.NeuralBatchProcessor(
            [square1, square2, add], input_adapter, identity_adapter, device
        )

        res = processor(dict(x1=4, x2=3, x3=25))
        self.assertEqual({'x1': 4, 'x2': 3, 'x3': 25, 's1': 16, 's2': 9, 'y_hat': 50}, res)

    def test_simple_regression(self):
        def input_adapter(data_frame):
            return {
                "model1": {"x1": data_frame["x"]},
                "model2": {"x2": data_frame["x"]},
            }

        model1 = TrainableLinearModel(num_features=1)
        model2 = TrainableLinearModel(num_features=1)
        model3 = TrainableLinearModel(num_features=2)

        sgd = torch.optim.SGD(model1.parameters(), lr=0.1)
        node1 = processing_graph.Node('model1', model1, sgd, inputs=["x1"], outputs=["t1"])
        node2 = processing_graph.Node('model2', model2, sgd, inputs=["x2"], outputs=["t2"])
        node3 = processing_graph.Node('model3', model3, sgd, inputs=["t1", "t2"], outputs=["y_hat"])

        device = torch.device('cpu')
        processor = processing_graph.NeuralBatchProcessor(
            [node1, node2, node3], input_adapter, identity_adapter, device
        )

        ds_x = [[1], [2], [3], [4]]
        ds_y = [[3 * v[0] + 1] for v in ds_x]

        criterion = torch.nn.MSELoss()

        y = torch.tensor(ds_y, dtype=torch.float32)

        for epoch in range(200):
            data_frame = dict(x=torch.tensor(ds_x, dtype=torch.float32))
            res = processor(data_frame)
            loss = criterion(res["y_hat"], y)
            processor.prepare()
            loss.backward()

            processor.update()

        inputs = torch.tensor([[0], [-1], [-2], [6]], dtype=torch.float32)
        res = processor(dict(x=inputs))
        expected = torch.tensor([[1], [-2], [-5], [19]], dtype=torch.float32)

        self.assertTrue(torch.allclose(expected, res["y_hat"], rtol=3))

    def test_input_device_change(self):
        # todo: find a way to test this without having access to nvidia graphic card
        pass


class DetachBatchTests(unittest.TestCase):
    def test(self):
        detach = processing_graph.DetachBatch()

        data_frame = {
            'x': torch.tensor([1, 2, 3, 4], dtype=torch.float32, requires_grad=True),
            'y': torch.tensor([1, 1, 1, 1], dtype=torch.float32, requires_grad=False)
        }
        data_frame = detach(data_frame)

        self.assertFalse(data_frame['x'].requires_grad)
        self.assertFalse(data_frame['y'].requires_grad)


class BatchMergerTests(unittest.TestCase):
    def test_merge_empty_data_frames(self):
        merger = processing_graph.BatchMerger()
        self.assertEqual({}, merger([]))

        self.assertEqual({}, merger([{}]))
        self.assertEqual({}, merger([{}, {}]))
        self.assertEqual({}, merger([{}, {}, {}, {}, {}, {}]))

    def test_merge_with_empty_data_frame(self):
        merger = processing_graph.BatchMerger()
        df = {'a': torch.tensor([1, 2, 3])}
        self.assertRaises(processing_graph.MergeError, merger, [df, {}])
        self.assertRaises(processing_graph.MergeError, merger, [df, df, {}, df])
        self.assertRaises(processing_graph.MergeError, merger, [{}, df, df, {}])

    def test_merge_with_itself(self):
        merger = processing_graph.BatchMerger()
        df = {
            'a': torch.tensor([[1, 10], [2, 20], [3, 30]]),
            'b': torch.tensor([5, 6])
        }
        expected = {
            'a': torch.tensor([[1, 10], [2, 20], [3, 30], [1, 10], [2, 20], [3, 30]]),
            'b': torch.tensor([5, 6, 5, 6])
        }
        res = merger([df, df])
        self.assertEqual({'a', 'b'}, set(res.keys()))
        self.assertTrue(torch.allclose(expected['a'], res['a']))
        self.assertTrue(torch.allclose(expected['b'], res['b']))

    def test_merge_3_data_frames(self):
        merger = processing_graph.BatchMerger()
        df1 = {
            'a': torch.tensor([[1, 10], [2, 20], [3, 30]], dtype=torch.float32),
            'b': torch.tensor([5, 6], dtype=torch.float32)
        }
        df2 = {
            'a': torch.tensor([[4, 40], [5, 50]], dtype=torch.float32),
            'b': torch.tensor([7, 8, 9, 10], dtype=torch.float32)
        }

        df3 = {
            'a': torch.tensor([[6, 60]], dtype=torch.float32),
            'b': torch.tensor([], dtype=torch.float32)
        }

        res = merger([df1, df2, df3])
        expected = {
            'a': torch.tensor([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60]], dtype=torch.float32),
            'b': torch.tensor([5, 6, 7, 8, 9, 10], dtype=torch.float32)
        }
        self.assertEqual({'a', 'b'}, set(res.keys()))
        self.assertTrue(torch.allclose(expected['a'], res['a']))
        self.assertTrue(torch.allclose(expected['b'], res['b']))

        res = merger([df3, df2, df1])
        expected = {
            'a': torch.tensor([[6, 60], [4, 40], [5, 50], [1, 10], [2, 20], [3, 30]], dtype=torch.float32),
            'b': torch.tensor([7, 8, 9, 10, 5, 6], dtype=torch.float32)
        }

        self.assertTrue(torch.allclose(expected['a'], res['a']))
        self.assertTrue(torch.allclose(expected['b'], res['b']))


def square_input(data_frame):
    df = data_frame.copy()
    df["x"] = data_frame["x"] ** 2
    return df


def add_1(data_frame):
    df = data_frame.copy()
    df["x"] = data_frame["x"] + 1
    return df


class ProcessingGraphTests(unittest.TestCase):
    def test_computing_on_empty_graph(self):
        graph = processing_graph.BatchProcessingGraph([])
        res = graph({'x': 32, 'y': 0})
        self.assertEqual({}, res)

        graph = processing_graph.BatchProcessingGraph(["x1", "x2"])
        self.assertEqual({}, graph({'a': 1, 'b': 2, 'x': 32, 'y': 0}))

    def test_cannot_have_name_collision_between_input_nodes_and_graph_nodes(self):
        self.assertRaises(processing_graph.InvalidGraphError, processing_graph.BatchProcessingGraph,
                          ["x", "x", "y"], a=lambda d: d, x=lambda d: d)

    def test_cannot_make_edges_between_input_nodes_and_missing_nodes(self):
        graph = processing_graph.BatchProcessingGraph(["x1", "x2"], a=lambda d: d)
        self.assertRaises(processing_graph.InvalidEdgeError, graph.make_edge, 'foo', 'bar')
        self.assertRaises(processing_graph.InvalidEdgeError, graph.make_edge, 'a', 'bar')
        self.assertRaises(processing_graph.InvalidEdgeError, graph.make_edge, 'bar', 'a')
        self.assertRaises(processing_graph.InvalidEdgeError, graph.make_edge, 'bar', 'x1')
        self.assertRaises(processing_graph.InvalidEdgeError, graph.make_edge, 'x1', 'x2')
        self.assertRaises(processing_graph.InvalidEdgeError, graph.make_edge, 'x1', 'x1')
        self.assertRaises(processing_graph.InvalidEdgeError, graph.make_edge, 'x2', 'x2')

    def test_input_nodes_cannot_have_ingoing_edges(self):
        graph = processing_graph.BatchProcessingGraph(["x1", "x2"], x_squared=square_input)
        self.assertRaises(processing_graph.InvalidEdgeError, graph.make_edge, "x_squared", "x1")
        self.assertRaises(processing_graph.InvalidEdgeError, graph.make_edge, "x_squared", "x2")

    def test_cannot_run_computation_on_disconnected_graph(self):
        graph = processing_graph.BatchProcessingGraph(["x1", "x2"], x_squared=square_input)

        frames = dict(input1={"x": torch.tensor([2, 3])})
        self.assertRaises(processing_graph.DisconnectedGraphError, graph, frames)

    def test_graph_cannot_have_cycles(self):
        graph = processing_graph.BatchProcessingGraph(
            ["inp"], a=square_input, b=add_1, c=lambda x: x
        )
        self.assertRaises(processing_graph.InvalidEdgeError, graph.make_edge, "a", "a")

        graph.make_edge("a", "b")

        self.assertRaises(processing_graph.InvalidEdgeError, graph.make_edge, "b", "a")

        graph = processing_graph.BatchProcessingGraph(
            ["inp"], a=square_input, b=add_1, c=lambda x: x
        )

        graph.make_edge("a", "b")
        graph.make_edge("b", "c")
        self.assertRaises(processing_graph.InvalidEdgeError, graph.make_edge, "c", "a")

    def test_feeding_simplest_graph(self):
        graph = processing_graph.BatchProcessingGraph(["input1"], output=square_input)
        graph.make_edge("input1", "output")
        frames = dict(input1={"x": torch.tensor([2, 3])})

        res = graph(frames)
        self.assertEqual({'output'}, set(res.keys()))
        self.assertEqual({'x'}, set(res['output'].keys()))
        self.assertTrue(torch.allclose(torch.tensor([4, 9]), res['output']['x']))

    def test_feeding_a_sequential_graph(self):
        graph = processing_graph.BatchProcessingGraph(["input1"], x_squared=square_input, output=add_1)

        graph.make_edge("input1", "x_squared")
        graph.make_edge("x_squared", "output")

        frames = dict(input1={"x": torch.tensor([2, 3])})
        res = graph(frames)
        self.assertEqual({'output'}, set(res.keys()))
        self.assertEqual({'x'}, set(res['output'].keys()))
        self.assertTrue(torch.allclose(torch.tensor([5, 10]), res['output']['x']))

        # todo: more complex computation graph


class DiGraphTests(unittest.TestCase):
    def test_cannot_have_duplicate_vertices(self):
        self.assertRaises(processing_graph.InvalidGraphError,
                          processing_graph.DiGraph, [1, 4, 2, 1])

        self.assertRaises(processing_graph.InvalidGraphError,
                          processing_graph.DiGraph, [1, 2, 2, 2])

        self.assertRaises(processing_graph.InvalidGraphError,
                          processing_graph.DiGraph, [3, 3])

    def test_cannot_create_self_loops(self):
        digraph = processing_graph.DiGraph([1, 4])
        self.assertRaises(processing_graph.InvalidEdgeError, digraph.make_edge, 4, 4)
        self.assertRaises(processing_graph.InvalidEdgeError, digraph.make_edge, 1, 1)

        self.assertRaises(processing_graph.InvalidEdgeError,
                          processing_graph.DiGraph, [1, 4], [(1, 1)])

    def test_cannot_create_edge_when_vertex_missing(self):
        digraph = processing_graph.DiGraph([1, 4, 3])

        self.assertRaises(processing_graph.InvalidEdgeError, digraph.make_edge, 10, 41)
        self.assertRaises(processing_graph.InvalidEdgeError, digraph.make_edge, 1, 41)
        self.assertRaises(processing_graph.InvalidEdgeError, digraph.make_edge, 41, 1)

        self.assertRaises(processing_graph.InvalidEdgeError,
                          processing_graph.DiGraph, [1, 4, 3], [(41, 1)])

    def test_cannot_have_duplicate_edge(self):
        digraph = processing_graph.DiGraph([1, 4], [(1, 4), (4, 1)])

        self.assertRaises(processing_graph.InvalidEdgeError, digraph.make_edge, 1, 4)
        self.assertRaises(processing_graph.InvalidEdgeError, digraph.make_edge, 4, 1)

    def test_graph_without_edges(self):
        digraph = processing_graph.DiGraph([10])
        self.assertEqual({10: {10}}, digraph.strong_components())

        digraph = processing_graph.DiGraph([10, 20])
        self.assertEqual({10: {10}, 20: {20}}, digraph.strong_components())
        self.assertEqual([], digraph.detect_cycles())

        digraph = processing_graph.DiGraph([3, 1, 2])
        self.assertEqual({1: {1}, 2: {2}, 3: {3}}, digraph.strong_components())
        self.assertEqual([], digraph.detect_cycles())

    def test_graph_with_bi_directional_link(self):
        digraph = processing_graph.DiGraph([1, 2], [(1, 2), (2, 1)])
        components = digraph.strong_components()
        self.assertEqual({1: {1, 2}}, components)
        self.assertEqual([{1, 2}], digraph.detect_cycles())

        digraph = processing_graph.DiGraph([1, 2, 3], [(1, 2), (2, 1)])
        components = digraph.strong_components()
        self.assertEqual({1: {1, 2}, 3: {3}}, components)
        self.assertEqual([{1, 2}], digraph.detect_cycles())

    def test_two_strongly_connected_pairs(self):
        digraph = processing_graph.DiGraph([1, 2, 3, 4], [(1, 2), (2, 1), (3, 4), (4, 3), (2, 3)])
        components = digraph.strong_components()
        self.assertEqual({1: {1, 2}, 3: {3, 4}}, components)
        self.assertEqual([{3, 4}, {1, 2}], digraph.detect_cycles())

        digraph = processing_graph.DiGraph(
            [1, 2, 3, 4], [(1, 2), (2, 1), (3, 4), (4, 3), (2, 3), (1, 4)]
        )
        components = digraph.strong_components()
        self.assertEqual({1: {1, 2}, 3: {3, 4}}, components)
        self.assertEqual([{3, 4}, {1, 2}], digraph.detect_cycles())

    def test_two_strongly_connected_triangles(self):
        vertices = [1, 2, 3, 4, 5, 6]
        edges = [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4), (4, 2)]
        digraph = processing_graph.DiGraph(vertices, edges)
        self.assertEqual({1: {1, 2, 3}, 4: {4, 5, 6}}, digraph.strong_components())
        self.assertEqual([{1, 2, 3}, {4, 5, 6}], digraph.detect_cycles())

    def test_two_strongly_connected_triangles_and_2_isolated_vertices(self):
        vertices = [1, 2, 3, 4, 5, 6, 7, 8]
        edges = [(1, 5), (5, 6), (6, 1), (1, 6), (1, 8), (6, 4), (4, 2), (2, 3), (3, 4), (7, 2)]

        digraph = processing_graph.DiGraph(vertices, edges)
        self.assertEqual({1: {1, 5, 6}, 2: {2, 3, 4}, 7: {7}, 8: {8}}, digraph.strong_components())
        self.assertEqual([{2, 3, 4}, {1, 5, 6}], digraph.detect_cycles())
