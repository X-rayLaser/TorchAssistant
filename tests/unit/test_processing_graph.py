import unittest
from scaffolding import processing_graph
import torch
from operator import mul


def square(x): return x**2
def do_nothing(): return 42


class InferenceModel:
    def run_inference(self, x):
        return x ** 2


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
