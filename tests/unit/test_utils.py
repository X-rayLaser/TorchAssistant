import unittest
from torchassistant import utils
from torchassistant.data import MultiSplitter, DatasetSlice
from torchassistant.session import Session


class InstantiateClassTests(unittest.TestCase):
    def test_with_wrong_path(self):
        self.assertRaisesRegex(utils.ClassImportError, 'Invalid import path: ""', utils.instantiate_class, '')

        self.assertRaisesRegex(utils.ClassImportError,
                               'Invalid import path: "  "', utils.instantiate_class, '  ')

        self.assertRaisesRegex(utils.ClassImportError, 'Invalid import path: "missing_module"',
                               utils.instantiate_class, 'missing_module')

        self.assertRaisesRegex(utils.ClassImportError,
                               'Invalid import path: "  contains spaces    and tabs"', utils.instantiate_class,
                               '  contains spaces    and tabs')

        msg = 'Failed to import and instantiate a class "utils" from "torchassistant": \'module\' object is not callable'
        self.assertRaisesRegex(utils.ClassImportError, msg, utils.instantiate_class, 'torchassistant.utils')

        msg = 'Failed to import and instantiate a class "" from ' \
              '"torchassistant.utils": module \'torchassistant.utils\' has no attribute \'\''
        self.assertRaisesRegex(utils.ClassImportError, msg, utils.instantiate_class, 'torchassistant.utils.')

        msg = 'Failed to import and instantiate a class "Foo" from ' \
              '"torchassistant.utils": module \'torchassistant.utils\' has no attribute \'Foo\''

        self.assertRaisesRegex(utils.ClassImportError, msg, utils.instantiate_class, 'torchassistant.utils.Foo')

        msg = 'Failed to import and instantiate a class "Foo" from "torchassistant.missing": ' \
              'No module named \'torchassistant.missing\''
        self.assertRaisesRegex(utils.ClassImportError, msg, utils.instantiate_class, 'torchassistant.missing.Foo')

    def test_with_wrong_arguments(self):
        self.assertRaises(utils.ClassImportError,
                          lambda: utils.instantiate_class('scaffolding.session.Session', kwarg='kwarg'))

        self.assertRaises(utils.ClassImportError,
                          lambda: utils.instantiate_class('scaffolding.session.Session', 1, 2))

    def test_correct_instantiation(self):
        session = utils.instantiate_class('torchassistant.session.Session')
        self.assertTrue(hasattr(session, 'datasets'))

        metric = utils.instantiate_class(
            'torchassistant.metrics.Metric', 'name', 'metric_fn', metric_args=[], transform_fn=''
        )

        self.assertEqual('name', metric.name)


class TestImportFunction(unittest.TestCase):
    def setUp(self) -> None:
        self.exception_class = utils.FunctionImportError
        self.function_to_test = utils.import_function

    def test_with_wrong_path(self):
        self.assertRaises(self.exception_class, self.function_to_test, '')
        self.assertRaises(self.exception_class, self.function_to_test, '  ')
        self.assertRaises(self.exception_class, self.function_to_test, 'missing_module')
        self.assertRaises(self.exception_class, self.function_to_test, '  contains spaces    and tabs')
        self.assertRaises(self.exception_class, self.function_to_test, 'torchassistant.utils.')
        self.assertRaises(self.exception_class, self.function_to_test, 'torchassistant.foo')
        self.assertRaises(self.exception_class, self.function_to_test, 'torchassistant.missing.import_function')

    def test_correct_import(self):
        fn = self.function_to_test('torchassistant.utils.import_function')
        self.assertTrue(callable(fn))


class TestImportEntity(TestImportFunction):
    def setUp(self):
        self.exception_class = utils.EntityImportError
        self.function_to_test = utils.import_entity


class TestGetDataset(unittest.TestCase):
    def test_dataset_not_found(self):
        session = Session()

        self.assertRaises(utils.DatasetNotFoundError, utils.get_dataset, session, 'other dataset')

        session.datasets = dict(dataset=[0, 1])
        session.splits['my_split'] = MultiSplitter('other', [0.5, 0.5])
        self.assertRaises(utils.DatasetNotFoundError, utils.get_dataset, session, 'other dataset')
        self.assertRaises(utils.DatasetNotFoundError, utils.get_dataset, session, 'my_split.train')

    def test_data_splitter_not_found(self):
        session = Session()
        session.datasets = dict(dataset=[0, 1])
        session.splits['my_split'] = MultiSplitter('dataset', [0.5, 0.5])
        self.assertRaises(utils.SplitterNotFoundError, utils.get_dataset, session, 'other_split.train')

    def test_get_normal_dataset(self):
        session = Session()
        session.datasets = dict(ds_a=[0, 1], ds_b=[2, 3])
        ds = utils.get_dataset(session, 'ds_a')
        self.assertEqual([0, 1], ds)

    def test_get_data_split_part(self):
        ds = [(4, 16), (3, 9), (2, 4), (1, 1)]
        session = Session()
        session.datasets['ds_a'] = ds
        session.datasets['other_ds'] = [(10, 100), (9, 81), (8, 64), (7, 49), (6, 36)]

        session.splits['my_split'] = MultiSplitter('ds_a', [0.5, 0.5])
        session.splits['other_split'] = MultiSplitter('other_ds', [0.4, 0.4, 0.2])

        train_slice = utils.get_dataset(session, 'my_split.train')
        val_slice = utils.get_dataset(session, 'my_split.val')
        self.assertIsInstance(train_slice, DatasetSlice)
        self.assertIsInstance(val_slice, DatasetSlice)

        self.assertEqual([(4, 16), (3, 9)], list(train_slice))
        self.assertEqual([(2, 4), (1, 1)], list(val_slice))
        self.assertRaises(utils.BadDatasetSliceError, utils.get_dataset, session, 'my_split.test')

        train_slice = utils.get_dataset(session, 'other_split.train')
        val_slice = utils.get_dataset(session, 'other_split.val')
        test_slice = utils.get_dataset(session, 'other_split.test')
        self.assertIsInstance(train_slice, DatasetSlice)
        self.assertIsInstance(val_slice, DatasetSlice)
        self.assertIsInstance(test_slice, DatasetSlice)

        self.assertEqual([(10, 100), (9, 81)], list(train_slice))
        self.assertEqual([(8, 64), (7, 49)], list(val_slice))
        self.assertEqual([(6, 36)], list(test_slice))

    def test_get_data_split_part_by_numeric_index(self):
        session = Session()
        session.datasets['other_ds'] = [(10, 100), (9, 81), (8, 64), (7, 49), (6, 36)]

        session.splits['my_split'] = MultiSplitter('other_ds', [0.4, 0.4, 0.2])

        train_slice = utils.get_dataset(session, 'my_split[0]')
        val_slice = utils.get_dataset(session, 'my_split[1]')
        test_slice = utils.get_dataset(session, 'my_split[2]')

        self.assertEqual([(10, 100), (9, 81)], list(train_slice))
        self.assertEqual([(8, 64), (7, 49)], list(val_slice))
        self.assertEqual([(6, 36)], list(test_slice))

        self.assertRaises(utils.BadDatasetSliceError, utils.get_dataset, session, 'my_split.haha')
