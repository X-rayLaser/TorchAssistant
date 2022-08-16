import unittest
from scaffolding import utils


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

        msg = 'Failed to import and instantiate a class "utils" from "scaffolding": \'module\' object is not callable'
        self.assertRaisesRegex(utils.ClassImportError, msg, utils.instantiate_class, 'scaffolding.utils')

        msg = 'Failed to import and instantiate a class "" from ' \
              '"scaffolding.utils": module \'scaffolding.utils\' has no attribute \'\''
        self.assertRaisesRegex(utils.ClassImportError, msg, utils.instantiate_class, 'scaffolding.utils.')

        msg = 'Failed to import and instantiate a class "Foo" from ' \
              '"scaffolding.utils": module \'scaffolding.utils\' has no attribute \'Foo\''

        self.assertRaisesRegex(utils.ClassImportError, msg, utils.instantiate_class, 'scaffolding.utils.Foo')

        msg = 'Failed to import and instantiate a class "Foo" from "scaffolding.missing": ' \
              'No module named \'scaffolding.missing\''
        self.assertRaisesRegex(utils.ClassImportError, msg, utils.instantiate_class, 'scaffolding.missing.Foo')

    def test_with_wrong_arguments(self):
        self.assertRaises(utils.ClassImportError,
                          lambda: utils.instantiate_class('scaffolding.session.Session', kwarg='kwarg'))

        self.assertRaises(utils.ClassImportError,
                          lambda: utils.instantiate_class('scaffolding.session.Session', 1, 2))

    def test_correct_instantiation(self):
        session = utils.instantiate_class('scaffolding.session.Session')
        self.assertTrue(hasattr(session, 'datasets'))

        metric = utils.instantiate_class(
            'scaffolding.metrics.Metric', 'name', 'metric_fn', metric_args=[], transform_fn='', device=''
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
        self.assertRaises(self.exception_class, self.function_to_test, 'scaffolding.utils.')
        self.assertRaises(self.exception_class, self.function_to_test, 'scaffolding.foo')
        self.assertRaises(self.exception_class, self.function_to_test, 'scaffolding.missing.import_function')

    def test_correct_import(self):
        fn = self.function_to_test('scaffolding.utils.import_function')
        self.assertTrue(callable(fn))


class TestImportEntity(TestImportFunction):
    def setUp(self):
        self.exception_class = utils.EntityImportError
        self.function_to_test = utils.import_entity
