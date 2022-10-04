import unittest
from torchassistant.data import MultiSplitter, BadSplitError
from torchassistant.session import override_spec
from torchassistant.session.override_spec import MetaDict, MetaList, parse_object, override_dict, override_list, \
    InvalidNumberOfMatchesError


class ParseObjectTests(unittest.TestCase):
    def test_on_primitives(self):
        self.assertEqual(4, parse_object(4))
        self.assertEqual('x', parse_object('x'))
        self.assertEqual(False, parse_object(False))
        self.assertEqual(True, parse_object(True))

    def test_on_list(self):
        meta_list = parse_object([])
        self.assertEqual([], meta_list)
        self.assertEqual("replace", meta_list.replace_strategy)

        meta_list = parse_object([1, 2])
        self.assertEqual([1, 2], meta_list)
        self.assertEqual("replace", meta_list.replace_strategy)

    def test_on_annotated_list(self):
        meta_list = parse_object(dict(override_key='key', options=[]))
        self.assertEqual([], meta_list)
        self.assertEqual("replace", meta_list.replace_strategy)
        self.assertEqual("key", meta_list.override_key)

        meta_list = parse_object(dict(override_key='key', replace_strategy='override', options=[]))
        self.assertEqual([], meta_list)
        self.assertEqual("override", meta_list.replace_strategy)
        self.assertEqual("key", meta_list.override_key)

        meta_list = parse_object(dict(override_key=['key'], replace_strategy='override',
                                      options=[1, 2, 3]))
        self.assertEqual([1, 2, 3], meta_list)
        self.assertEqual("override", meta_list.replace_strategy)
        self.assertEqual(["key"], meta_list.override_key)

    def test_on_unannotated_dict(self):
        meta_dict = parse_object({})
        self.assertEqual({}, meta_dict)
        self.assertEqual("override", meta_dict.replace_strategy)

        meta_dict = parse_object(dict(a=1, b=2))
        self.assertEqual(dict(a=1, b=2), meta_dict)
        self.assertEqual("override", meta_dict.replace_strategy)

    def test_on_annotated_dict(self):
        meta_dict = parse_object(dict(options={}))
        self.assertEqual({}, meta_dict)
        self.assertEqual("replace", meta_dict.replace_strategy)

        meta_dict = parse_object(dict(replace_strategy="override", options={}))
        self.assertEqual({}, meta_dict)
        self.assertEqual("override", meta_dict.replace_strategy)

        meta_dict = parse_object(dict(replace_strategy="override", options={'a': 1, 'b': 2}))
        self.assertEqual(dict(a=1, b=2), meta_dict)
        self.assertEqual("override", meta_dict.replace_strategy)

    def test_on_nested_structure(self):
        structure = {
            'k1': 'v1',
            'k2': [1, 2, {'a': 1, 'b': 2}],
            'l1': {
                'replace_strategy': 'override',
                'override_key': ['id'],
                'options': [{'id': 10, 'name': 'first'}]
            }
        }

        expected = {
            'k1': 'v1',
            'k2': [1, 2, {'a': 1, 'b': 2}],
            'l1': [{'id': 10, 'name': 'first'}]
        }
        result = parse_object(structure)
        self.assertEqual(expected, result)
        self.assertEqual("replace", result['k2'].replace_strategy)
        self.assertEqual("override", result['l1'].replace_strategy)
        self.assertEqual(["id"], result['l1'].override_key)


class OverrideListTests(unittest.TestCase):
    def test_override_empty_list(self):
        l1 = []

        l2 = MetaList([{'id': 15, 'y': 34}])
        l2.replace_strategy = 'override'
        l2.override_key = ['id']

        expected = [{'id': 15, 'y': 34}]
        self.assertEqual(expected, override_list(l1, l2))

    def test_override_with_empty_list(self):
        l1 = [{'id': 12, 'x': 0}, {'id': 15, 'y': 10}]

        l2 = MetaList([])
        l2.replace_strategy = 'override'
        l2.override_key = ['id']

        expected = list(l1)
        self.assertEqual(expected, override_list(l1, l2))

    def test_override_empty_list_with_empty_list(self):
        l1 = []

        l2 = MetaList([])
        l2.replace_strategy = 'override'
        l2.override_key = ['id']

        self.assertEqual([], override_list(l1, l2))

    def test_override_list_item_and_add_new(self):
        l1 = [{'id': 12, 'x': 0}, {'id': 15, 'y': 10}]

        item1 = MetaDict({'id': 15, 'y': 34})
        item2 = MetaDict({'id': 20, 'y': 34, 'c': 100})
        item1.replace_strategy = 'override'
        item2.replace_strategy = 'override'

        l2 = MetaList([item1, item2])
        l2.replace_strategy = 'override'
        l2.override_key = ['id']

        expected = [{'id': 12, 'x': 0}, {'id': 15, 'y': 34}, {'id': 20, 'y': 34, 'c': 100}]

        self.assertEqual(expected, override_list(l1, l2))

    def test_override_using_tuple_as_key(self):
        l1 = [{'id1': 12, 'id2': 10, 'x': 0, 'y': 34}, {'id1': 10, 'id2': 12, 'x': 100, 'y': 10}]

        item = MetaDict({'id1': 10, 'id2': 12, 'x': 42})
        item.replace_strategy = 'override'
        l2 = MetaList([item])
        l2.replace_strategy = 'override'
        l2.override_key = ['id1', 'id2']

        expected = [{'id1': 12, 'id2': 10, 'x': 0, 'y': 34}, {'id1': 10, 'id2': 12, 'x': 42, 'y': 10}]
        self.assertEqual(expected, override_list(l1, l2))

    def test_override_dicts_with_list_of_strings(self):
        l1 = [{'id': 12, 'x': 0}, {'id': 15, 'y': 10}]
        l2 = MetaList([1, 2, 3])
        l2.replace_strategy = 'replace'
        self.assertEqual([1, 2, 3], override_list(l1, l2))

    def test_cannot_have_more_than_one_match(self):
        l1 = [{'id': 12, 'x': 0}, {'id': 15, 'y': 10}, {'id': 12, 'c': 250}]

        l2 = MetaList([{'id': 12, 'x': 34}, {'id': 20, 'y': 34, 'c': 100}])
        l2.replace_strategy = 'override'
        l2.override_key = ['id']

        self.assertRaises(InvalidNumberOfMatchesError, lambda: override_list(l1, l2))

    def test_replace_empty_list_with_empty_list(self):
        l1 = []
        l2 = MetaList([])
        l2.replace_strategy = 'replace'
        self.assertEqual([], override_list(l1, l2))

    def test_replace_empty_list(self):
        l1 = []
        l2 = MetaList(['a', 'b', 'c'])
        l2.replace_strategy = 'replace'
        self.assertEqual(['a', 'b', 'c'], override_list(l1, l2))

    def test_replace_with_empty_list(self):
        l1 = [{'id': 12, 'x': 0}, {'id': 15, 'y': 10}]
        l2 = MetaList([])
        l2.replace_strategy = 'replace'
        self.assertEqual([], override_list(l1, l2))


class OverrideDictTests(unittest.TestCase):
    def test_override_empty_dict_with_empty_dict(self):
        d2 = MetaDict({})
        d2.replace_strategy = 'override'
        self.assertEqual({}, override_dict({}, d2))

    def test_override_empty_dict(self):
        d2 = MetaDict({'key1': 'value1', 'key2': 2})
        d2.replace_strategy = 'override'
        self.assertEqual({'key1': 'value1', 'key2': 2}, override_dict({}, d2))

    def test_override_dict(self):
        d1 = {'key1': 'value1', 'key2': 2}
        d2 = MetaDict({'key1': 'new value', 'new_key': 1234})
        d2.replace_strategy = 'override'
        self.assertEqual({'key1': 'new value', 'key2': 2, 'new_key': 1234}, override_dict(d1, d2))

    def test_override_nested_dict_with_dict(self):
        d1 = {
            'nested': {
                'a': 1,
                'b': 2
            }
        }

        nested_dict = MetaDict({
            'b': 24,
            'c': 90
        })
        nested_dict.replace_strategy = 'override'

        d2 = MetaDict({
            'nested': nested_dict
        })
        d2.replace_strategy = 'override'

        expected = {
            'nested': {
                'a': 1,
                'b': 24,
                'c': 90
            }
        }
        self.assertEqual(expected, override_dict(d1, d2))

    def test_override_nested_dict_with_number(self):
        d1 = {
            'key1': ['value1'],
            'key_nested': {
                'a': 1,
                'b': 2
            },
            'key2': 2
        }
        d2 = MetaDict({
            'key1': 'new value', 'key_nested': 1234, 'new_key': 42
        })
        d2.replace_strategy = 'override'

        expected = {
            'key1': 'new value',
            'key_nested': 1234,
            'new_key': 42,
            'key2': 2
        }
        self.assertEqual(expected, override_dict(d1, d2))

    def test_override_list_with_list(self):
        d1 = {
            'alist': [{
                'id': 1,
                'value': 10
            }, {
                'id': 2,
                'value': 90
            }]
        }

        list_item = MetaDict({
            'id': 2,
            'value': 1120,
            'extra_option': 111
        })
        list_item.replace_strategy = 'override'
        alist = MetaList([list_item])

        alist.replace_strategy = 'override'
        alist.override_key = ['id']

        d2 = MetaDict({
            'alist': alist
        })

        d2.replace_strategy = 'override'

        expected = {
            'alist': [{
                'id': 1,
                'value': 10
            }, {
                'id': 2,
                'value': 1120,
                'extra_option': 111
            }]
        }
        self.assertEqual(expected, override_dict(d1, d2))

    def test_override_list_with_primitive(self):
        d1 = {
            'alist': [{
                'id': 1,
                'value': 10
            }, {
                'id': 2,
                'value': 90
            }]
        }

        d2 = MetaDict({
            'alist': 'string'
        })
        d2.replace_strategy = 'override'

        self.assertEqual({'alist': 'string'}, override_dict(d1, d2))

    def test_override_primitive_with_dict(self):
        d1 = {
            'a': 1,
            'b': 2
        }

        nested_dict = MetaDict({
            'b': 24,
            'c': 90
        })
        nested_dict.replace_strategy = 'override'

        d2 = MetaDict({
            'a': nested_dict
        })
        d2.replace_strategy = 'override'

        expected = {
            'a': {
                'b': 24,
                'c': 90
            },
            'b': 2
        }
        self.assertEqual(expected, override_dict(d1, d2))

    def test_override_primitive_with_list(self):
        d1 = {
            'a': 1,
            'b': 2
        }

        nested_list = MetaList([1, 2, 3, 4, 5])
        nested_list.replace_strategy = 'override'

        d2 = MetaDict({
            'a': nested_list
        })
        d2.replace_strategy = 'override'

        expected = {
            'a': [1, 2, 3, 4, 5],
            'b': 2
        }

        self.assertEqual(expected, override_dict(d1, d2))

    def test_replace_empty_dict_with_empty_dict(self):
        d1 = {}
        d2 = MetaDict({})
        d2.replace_strategy = 'replace'
        self.assertEqual({}, override_dict(d1, d2))

    def test_replace_empty_dict(self):
        d1 = {}
        d2 = MetaDict({
            'a': 1,
            'b': 2
        })
        d2.replace_strategy = 'replace'
        self.assertEqual(dict(a=1, b=2), override_dict(d1, d2))

    def test_replace_dict_with_another_one(self):
        d1 = dict(x=10, y=20)
        d2 = MetaDict({
            'a': 1,
            'b': 2,
            'c': 3
        })
        d2.replace_strategy = 'replace'
        self.assertEqual(dict(a=1, b=2, c=3), override_dict(d1, d2))

    def test_replace_dict_with_primitive(self):
        d1 = dict(x=10, y=20, z={'a': 1, 'b': 2})
        item = MetaDict({'d': 4})
        item.replace_strategy = 'replace'
        d2 = MetaDict({
            'x': 100,
            'z': item
        })
        d2.replace_strategy = 'override'
        self.assertEqual(dict(x=100, y=20, z={'d': 4}), override_dict(d1, d2))


class OverridingSpecTests(unittest.TestCase):
    def test_1_level_depth(self):
        self.assertEqual({}, override_spec({}, {}))
        d = {'a': 1, 'b': 2}
        self.assertEqual(dict(d), override_spec(d, {}))
        self.assertEqual(dict(d), override_spec({}, d))

        d1 = {'a': 1, 'b': 2}
        d2 = {'a': 10, 'c': 15}
        self.assertEqual(dict(a=10, b=2, c=15), override_spec(d1, d2))

    def test_nested_dict(self):
        d1 = {'a': 1, 'b': 2, 'nested_dict': {'x': 0, 'y': 1}}
        d2 = {'a': 10, 'nested_dict': {'replace_strategy': 'override', 'options': {'x': 100, 't': 50}}}
        expected = dict(a=10, b=2, nested_dict={'x': 100, 'y': 1, 't': 50})
        self.assertEqual(expected, override_spec(d1, d2))

    def test_override_list_item(self):
        d1 = dict(alist=[{'id': 12, 'x': 0}, {'id': 15, 'y': 10}])
        d2 = dict(alist={'replace_strategy': 'override', 'override_key': ['id'], 'options': [{'id': 15, 'y': 34}]})
        expected = dict(alist=[{'id': 12, 'x': 0}, {'id': 15, 'y': 34}])
        self.assertEqual(expected, override_spec(d1, d2))

    def test_when_dicts_contain_lists(self):
        d1 = {'a': 1, 'b': 2, 'alist': [{'id': 12, 'x': 0}, {'id': 15, 'y': 10}]}
        alist = {'replace_strategy': 'override', 'override_key': ['id'],
                 'options': [{'id': 12, 'x': 40}, {'id': 100, 'c': 128}]}
        d2 = {'a': 10, 'alist': alist}
        expected = dict(
            a=10, b=2, alist=[{'id': 12, 'x': 40}, {'id': 15, 'y': 10}, {'id': 100, 'c': 128}]
        )
        self.assertEqual(expected, override_spec(d1, d2))

    def test_with_deep_nesting(self):
        d1 = {
            'initialize': {
                'definitions': [
                    {
                        "group": "batch_processors",
                        "name": "neural_translator",
                        "spec": {
                            "input_adapter": {
                                "class": "examples.language_translation.adapters.BatchAdapter",
                                "kwargs": {
                                    "hidden_size": 32
                                }
                            },
                            "neural_graph": [
                                {
                                    "model_name": "encoder_model",
                                    "inputs": ["x", "h"],
                                    "outputs": ["outputs", "h_e"],
                                    "optimizer_name": "encoder_optimizer"
                                },
                                {
                                    "model_name": "decoder_model",
                                    "inputs": ["y_shifted", "h_e"],
                                    "outputs": ["y_hat", "h_d"],
                                    "optimizer_name": "decoder_optimizer"
                                }
                            ],
                            "output_adapter": {
                                "class": "examples.language_translation.adapters.OutputAdapter"
                            },
                            "device": "cpu"
                        }
                    },
                    {
                        "name": "second definition"
                    }
                ]
            }
        }

        d2 = {
            "initialize": {
                "definitions": {
                    "replace_strategy": "override",
                    "override_key": ["name"],
                    "options": [
                        {
                            "name": "neural_translator",
                            "spec": {
                                "input_adapter": {
                                    "class": "examples.language_translation.adapters.BatchInferenceAdapter"
                                },
                                "neural_graph": {
                                    "replace_strategy": "override",
                                    "override_key": ["model_name"],
                                    "options": [{
                                        "model_name": "decoder_model",
                                        "inputs": ["sos", "h_e"],
                                        "outputs": ["y_hat"]
                                    }]
                                },
                                "output_adapter": {
                                    "class": "examples.language_translation.adapters.NullAdapter"
                                }
                            }
                        }
                    ]
                }
            }
        }

        expected = {
            'initialize': {
                'definitions': [
                    {
                        "group": "batch_processors",
                        "name": "neural_translator",
                        "spec": {
                            "input_adapter": {
                                "class": "examples.language_translation.adapters.BatchInferenceAdapter",
                                "kwargs": {
                                    "hidden_size": 32
                                }
                            },
                            "neural_graph": [
                                {
                                    "model_name": "encoder_model",
                                    "inputs": ["x", "h"],
                                    "outputs": ["outputs", "h_e"],
                                    "optimizer_name": "encoder_optimizer"
                                },
                                {
                                    "model_name": "decoder_model",
                                    "inputs": ["sos", "h_e"],
                                    "outputs": ["y_hat"],
                                    "optimizer_name": "decoder_optimizer"
                                }
                            ],
                            "output_adapter": {
                                "class": "examples.language_translation.adapters.NullAdapter"
                            },
                            "device": "cpu"
                        }
                    },
                    {
                        "name": "second definition"
                    }
                ]
            }
        }

        self.assertEqual(expected, override_spec(d1, d2))


class MultiSplitterTests(unittest.TestCase):
    def test_cannot_create_instance_with_empty_ratio(self):
        self.assertRaises(BadSplitError, lambda: MultiSplitter('ds', ratio=[]))

    def test_cannot_create_instance_with_wrong_ratio(self):
        self.assertRaises(BadSplitError, lambda: MultiSplitter('ds', ratio=[0.5]))

        # do not add to 1
        self.assertRaises(BadSplitError, lambda: MultiSplitter('ds', ratio=[0.5, 0.25]))
        self.assertRaises(BadSplitError, lambda: MultiSplitter('ds', ratio=[0.5, 0.3, 0.1]))
        self.assertRaises(BadSplitError, lambda: MultiSplitter('ds', ratio=[0.1] * 6))

    def test_cannot_split_an_empty_dataset(self):
        self.assertRaises(BadSplitError, lambda: MultiSplitter('ds', ratio=[0.5, 0.5]).split([]))

    def test_cannot_split_when_dataset_size_is_smaller_than_number_of_parts(self):
        splitter = MultiSplitter('ds', ratio=[0.7, 0.3])
        self.assertRaises(BadSplitError, lambda: splitter.split([1]))

        splitter = MultiSplitter('ds', ratio=[0.5, 0.4, 0.1])
        self.assertRaises(BadSplitError, lambda: splitter.split([1, 2]))

    def test_cannot_have_split_with_empty_slices(self):
        splitter = MultiSplitter('ds', ratio=[0.1, 0.9])
        dataset = [1, 2, 3, 4]
        self.assertRaises(BadSplitError, lambda: splitter.split(dataset))

        splitter = MultiSplitter('ds', ratio=[0.5, 0.1, 0.4])
        dataset = list(range(4))
        self.assertRaises(BadSplitError, lambda: splitter.split(dataset))

        splitter = MultiSplitter('ds', ratio=[0.2] * 5)
        dataset = list(range(3))
        self.assertRaises(BadSplitError, lambda: splitter.split(dataset))

    def test_sum_of_slice_sizes_equals_the_size_of_original_dataset(self):
        splitter = MultiSplitter('ds', ratio=[1])
        split = splitter.split([1, 2, 3])
        self.assertEqual(3, sum(map(len, split)))

        splitter = MultiSplitter('ds', ratio=[0.4, 0.3, 0.1, 0.2])
        split = splitter.split(list(range(12)))
        self.assertEqual(12, sum(map(len, split)))

        splitter = MultiSplitter('ds', ratio=[0.1] * 10)
        split = splitter.split(list(range(16)))
        self.assertEqual(16, sum(map(len, split)))

    def test_slices(self):
        splitter = MultiSplitter('ds', ratio=[1])
        split = splitter.split([1, 2, 3])
        self.assertEqual([1, 2, 3], list(split[0]))
        self.assertEqual([1, 2, 3], list(split.train))

        splitter = MultiSplitter('ds', ratio=[0.5, 0.5])
        split = splitter.split([1, 2, 3])
        self.assertEqual([1], list(split[0]))
        self.assertEqual([1], list(split.train))

        self.assertEqual([2, 3], list(split[1]))
        self.assertEqual([2, 3], list(split.val))

        splitter = MultiSplitter('ds', ratio=[0.4, 0.5, 0.1])
        split = splitter.split([1, 2, 3, 4, 5, 6])
        self.assertEqual([1, 2], list(split[0]))
        self.assertEqual([3, 4, 5], list(split[1]))
        self.assertEqual([6], list(split[2]))

        self.assertEqual([1, 2], list(split.train))
        self.assertEqual([3, 4, 5], list(split.val))
        self.assertEqual([6], list(split.test))

    def test_with_bad_shuffling_indices(self):
        indices = [0, 2, 1]
        splitter = MultiSplitter('ds', ratio=[1])
        splitter.configure(shuffled_indices=indices)
        self.assertRaises(BadSplitError, lambda: splitter.split([1]))
        self.assertRaises(BadSplitError, lambda: splitter.split([1, 2]))
        self.assertRaises(BadSplitError, lambda: splitter.split([1, 2, 3, 4]))
        self.assertRaises(BadSplitError, lambda: splitter.split([1, 2, 3, 4, 5]))

    def test_shuffling(self):
        indices = [0, 2, 1, 3]
        splitter = MultiSplitter('ds', ratio=[0.6, 0.4])
        splitter.configure(shuffled_indices=indices)
        split = splitter.split([6, 7, 8, 9])
        self.assertEqual([6, 8], list(split.train))
        self.assertEqual([7, 9], list(split.val))
