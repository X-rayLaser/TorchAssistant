import unittest
from scaffolding.utils import MultiSplitter, BadSplitError


class MultiSplitterTests(unittest.TestCase):
    def test_cannot_create_instance_with_empty_ratio(self):
        self.assertRaises(BadSplitError, lambda: MultiSplitter(ratio=[]))

    def test_cannot_create_instance_with_wrong_ratio(self):
        self.assertRaises(BadSplitError, lambda: MultiSplitter(ratio=[0.5]))

        # do not add to 1
        self.assertRaises(BadSplitError, lambda: MultiSplitter(ratio=[0.5, 0.25]))
        self.assertRaises(BadSplitError, lambda: MultiSplitter(ratio=[0.5, 0.3, 0.1]))
        self.assertRaises(BadSplitError, lambda: MultiSplitter(ratio=[0.1] * 6))

    def test_cannot_split_an_empty_dataset(self):
        self.assertRaises(BadSplitError, lambda: MultiSplitter(ratio=[0.5, 0.5]).split([]))

    def test_cannot_split_when_dataset_size_is_smaller_than_number_of_parts(self):
        splitter = MultiSplitter(ratio=[0.7, 0.3])
        self.assertRaises(BadSplitError, lambda: splitter.split([1]))

        splitter = MultiSplitter(ratio=[0.5, 0.4, 0.1])
        self.assertRaises(BadSplitError, lambda: splitter.split([1, 2]))

    def test_cannot_have_split_with_empty_slices(self):
        splitter = MultiSplitter(ratio=[0.1, 0.9])
        dataset = [1, 2, 3, 4]
        self.assertRaises(BadSplitError, lambda: splitter.split(dataset))

        splitter = MultiSplitter(ratio=[0.5, 0.1, 0.4])
        dataset = list(range(4))
        self.assertRaises(BadSplitError, lambda: splitter.split(dataset))

        splitter = MultiSplitter(ratio=[0.2] * 5)
        dataset = list(range(3))
        self.assertRaises(BadSplitError, lambda: splitter.split(dataset))

    def test_sum_of_slice_sizes_equals_the_size_of_original_dataset(self):
        splitter = MultiSplitter(ratio=[1])
        split = splitter.split([1, 2, 3])
        self.assertEqual(3, sum(map(len, split)))

        splitter = MultiSplitter(ratio=[0.4, 0.3, 0.1, 0.2])
        split = splitter.split(list(range(12)))
        self.assertEqual(12, sum(map(len, split)))

        splitter = MultiSplitter(ratio=[0.1] * 10)
        split = splitter.split(list(range(16)))
        self.assertEqual(16, sum(map(len, split)))

    def test_slices(self):
        splitter = MultiSplitter(ratio=[1])
        split = splitter.split([1, 2, 3])
        self.assertEqual([1, 2, 3], list(split[0]))
        self.assertEqual([1, 2, 3], list(split.train))

        splitter = MultiSplitter(ratio=[0.5, 0.5])
        split = splitter.split([1, 2, 3])
        self.assertEqual([1], list(split[0]))
        self.assertEqual([1], list(split.train))

        self.assertEqual([2, 3], list(split[1]))
        self.assertEqual([2, 3], list(split.val))

        splitter = MultiSplitter(ratio=[0.4, 0.5, 0.1])
        split = splitter.split([1, 2, 3, 4, 5, 6])
        self.assertEqual([1, 2], list(split[0]))
        self.assertEqual([3, 4, 5], list(split[1]))
        self.assertEqual([6], list(split[2]))

        self.assertEqual([1, 2], list(split.train))
        self.assertEqual([3, 4, 5], list(split.val))
        self.assertEqual([6], list(split.test))

    def test_with_bad_shuffling_indices(self):
        indices = [0, 2, 1]
        splitter = MultiSplitter(ratio=[1], shuffled_indices=indices)
        self.assertRaises(BadSplitError, lambda: splitter.split([1]))
        self.assertRaises(BadSplitError, lambda: splitter.split([1, 2]))
        self.assertRaises(BadSplitError, lambda: splitter.split([1, 2, 3, 4]))
        self.assertRaises(BadSplitError, lambda: splitter.split([1, 2, 3, 4, 5]))

    def test_shuffling(self):
        indices = [0, 2, 1, 3]
        splitter = MultiSplitter(ratio=[0.6, 0.4], shuffled_indices=indices)
        split = splitter.split([6, 7, 8, 9])
        self.assertEqual([6, 8], list(split.train))
        self.assertEqual([7, 9], list(split.val))
