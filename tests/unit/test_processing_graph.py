import unittest
from scaffolding import processing_graph
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
        def input_adapter(data_frame): return {}

        def output_adapter(data_frame): return data_frame

        device = torch.device('cpu')
        processor = processing_graph.NeuralBatchProcessor([], input_adapter, output_adapter, device)
        res = processor({'a': [1, 2], 'b': [3, 4]})
        self.assertEqual(dict(a=[1, 2], b=[3, 4]), res)

    def test_output_adapter_determines_output(self):
        def input_adapter(data_frame): return {}

        def output_adapter(data_frame): return {'a': data_frame['x'] + 50, 'b': 1}

        device = torch.device('cpu')
        processor = processing_graph.NeuralBatchProcessor([], input_adapter, output_adapter, device)
        self.assertEqual({'a': 100, 'b': 1}, processor(dict(x=50, y=20)))

    def test_input_adapter_raises_an_error(self):
        def input_adapter(data_frame):
            d = data_frame['cube']
            return {}

        def output_adapter(data_frame): return {'a': data_frame['x'] + 50, 'b': 1}

        node1 = processing_graph.Node('square', square, None, inputs=["x"], outputs=["t"])
        node2 = processing_graph.Node('42', do_nothing(), None, inputs=["t"], outputs=["y_hat"])

        device = torch.device('cpu')
        processor = processing_graph.NeuralBatchProcessor([], input_adapter, output_adapter, device)
        self.assertRaises(processing_graph.InputAdapterError, processor, dict(x=50, y=20))

    def test_output_adapter_raises_an_error(self):
        def input_adapter(data_frame):
            return {
                "square": {"x": data_frame["x"]}
            }

        def output_adapter(data_frame):
            t = data_frame['t']
            return {'a': data_frame['x'] + 50, 'b': 1}

        class MyModel(torch.nn.Module):
            def forward(self, x):
                # todo: consider to support returning single variable rather than iterable
                return x ** 2,

        node1 = processing_graph.Node('square', MyModel(), None, inputs=["x"], outputs=["y_hat"])

        device = torch.device('cpu')
        processor = processing_graph.NeuralBatchProcessor([node1], input_adapter, output_adapter, device)
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

        def output_adapter(data_frame): return {'res': data_frame['y_hat']}

        square1 = processing_graph.Node('square1', SquareModel(), None, inputs=["x1"], outputs=["s1"])
        square2 = processing_graph.Node('square2', SquareModel(), None, inputs=["x2"], outputs=["s2"])
        add = processing_graph.Node('add', AdditionModel(), None, inputs=["s1", "s2", "x3"], outputs=["y_hat"])

        device = torch.device('cpu')
        processor = processing_graph.NeuralBatchProcessor(
            [square1, square2, add], input_adapter, output_adapter, device
        )
        self.assertEqual({'res': 50}, processor(dict(x1=4, x2=3, x3=25)))

    def test_simple_regression(self):
        def input_adapter(data_frame):
            return {
                "model1": {"x1": data_frame["x"]},
                "model2": {"x2": data_frame["x"]},
            }

        def output_adapter(data_frame): return data_frame

        model1 = TrainableLinearModel(num_features=1)
        model2 = TrainableLinearModel(num_features=1)
        model3 = TrainableLinearModel(num_features=2)

        sgd = torch.optim.SGD(model1.parameters(), lr=0.1)
        node1 = processing_graph.Node('model1', model1, sgd, inputs=["x1"], outputs=["t1"])
        node2 = processing_graph.Node('model2', model2, sgd, inputs=["x2"], outputs=["t2"])
        node3 = processing_graph.Node('model3', model3, sgd, inputs=["t1", "t2"], outputs=["y_hat"])

        device = torch.device('cpu')
        processor = processing_graph.NeuralBatchProcessor(
            [node1, node2, node3], input_adapter, output_adapter, device
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
        print(res["y_hat"])
        self.assertTrue(torch.allclose(expected, res["y_hat"], rtol=1))
