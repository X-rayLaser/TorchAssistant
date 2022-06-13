import unittest
from scaffolding import session
from scaffolding.utils import GenericSerializableInstance


class DummyFactory:
    def __call__(self, class_name, *args, **kwargs):
        class Dummy: pass
        return GenericSerializableInstance(Dummy(), class_name, args, kwargs)


class SpecParserTests(unittest.TestCase):
    def test_raises_error_for_invalid_spec(self):
        self.assert_raises_exception({})

        settings = dict({"class": 0})
        self.assert_raises_exception(settings)

        settings = dict({"class": "SomeClass"})
        instance = session.SpecParser(factory=DummyFactory()).parse(settings)

        settings = dict({"class": "SomeClass", "args": [1, 2], "kwargs": dict(x='a', y='b')})
        instance = session.SpecParser(factory=DummyFactory()).parse(settings)
        self.assertEqual('GenericSerializableInstance', instance.__class__.__name__)
        self.assertEqual('Dummy', instance.instance.__class__.__name__)
        self.assertEqual((1, 2), instance.args)
        self.assertIn('x', instance.kwargs)
        self.assertIn('y', instance.kwargs)
        self.assertEqual(2, len(instance.kwargs))
        self.assertEqual('a', instance.kwargs['x'])
        self.assertEqual('b', instance.kwargs['y'])

    def assert_raises_exception(self, settings):
        self.assertRaises(session.BadSpecificationError,
                          lambda: session.SpecParser().parse(settings))


class ConfigParserTests(unittest.TestCase):
    def test_raises_error_for_incomplete_spec(self):
        # todo: pass factory to parser
        self.assert_raises_exception({})

        settings = dict({'x': 0})
        self.assert_raises_exception(settings)

        settings = dict(initialize={'x': 0})
        self.assert_raises_exception(settings)

        settings = dict(initialize={'x': 0}, train={'x': 0})
        self.assert_raises_exception(settings)

        settings = dict(initialize={'datasets': {}}, train={'x': 0})
        self.assert_raises_exception(settings)

    def test_parse_datasets(self):
        datasets_spec = {
            'a': {
                'class': 'FirstDataset',
                'args': [13, 34]
            },
            'b': {
                'class': 'SecondDataset',
                'kwargs': dict(x=99, y=999)
            }
        }
        settings = dict(initialize={'datasets': datasets_spec}, train={'x': 0})

        datasets = session.ConfigParser(settings, factory=DummyFactory()).parse_datasets(datasets_spec)
        self.assertEqual(2, len(datasets))
        self.assertIn('a', datasets)
        self.assertIn('b', datasets)
        self.assertEqual((13, 34), datasets['a'].args)
        self.assertEqual(99, datasets['b'].kwargs['x'])
        self.assertEqual(999, datasets['b'].kwargs['y'])

    def assert_raises_exception(self, settings):
        self.assertRaises(session.BadSpecificationError,
                          lambda: session.ConfigParser(settings).get_config())


class SessionTests(unittest.TestCase):
    def test(self):
        pass
