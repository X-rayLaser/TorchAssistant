import unittest
from scaffolding import data


class MergedDatasetTests(unittest.TestCase):
    def test_merge_without_datasets(self):
        ds = data.MergedDataset()
        self.assertEqual(0, len(ds))
        self.assertEqual([], list(ds))
        self.assertRaises(IndexError, lambda: ds[0])

    def test_single_dataset_merge(self):
        ds = data.MergedDataset([10, 11, 12])
        self.assertEqual(3, len(ds))
        self.assertEqual(10, ds[0])
        self.assertEqual(11, ds[1])
        self.assertEqual(12, ds[2])
        self.assertEqual([10, 11, 12], list(ds))

        self.assertRaises(IndexError, lambda: ds[3])
        self.assertRaises(IndexError, lambda: ds[12])

    def test_2_dataset_merge(self):
        ds1 = [3, 2, 1]
        ds2 = [0, 1, 2, 3]
        ds = data.MergedDataset(ds1, ds2)
        self.assertEqual(7, len(ds))

        self.assertEqual(3, ds[0])
        self.assertEqual(2, ds[1])
        self.assertEqual(1, ds[2])
        self.assertEqual(0, ds[3])
        self.assertEqual(1, ds[4])
        self.assertEqual(2, ds[5])
        self.assertEqual(3, ds[6])

        self.assertEqual([3, 2, 1, 0, 1, 2, 3], list(ds))

        self.assertRaises(IndexError, lambda: ds[7])
        self.assertRaises(IndexError, lambda: ds[12])

    def test_merge_empty_datasets(self):
        ds = data.MergedDataset([], [], [])
        self.assertEqual(0, len(ds))
        self.assertEqual([], list(ds))

        ds = data.MergedDataset([], [12])
        self.assertEqual(1, len(ds))
        self.assertEqual([12], list(ds))

        ds = data.MergedDataset([], [12], [])
        self.assertEqual(1, len(ds))
        self.assertEqual([12], list(ds))

    def test_merging_5_datasets(self):
        ds1 = [3, 2, 1]
        ds2 = [0, 1, 2, 3]
        ds3 = [15, 22, 0]
        ds4 = [99]
        ds5 = ['ds5']
        all_5_combined = ds1 + ds2 + ds3 + ds4 + ds5

        ds = data.MergedDataset(ds1, ds2, ds3, ds4, ds5)
        self.assertEqual(12, len(ds))

        self.assertEqual('ds5', ds[11])
        self.assertEqual(15, ds[7])
        self.assertEqual(all_5_combined, list(ds))

        self.assertRaises(IndexError, lambda: ds[12])


class MultiSplitterTests(unittest.TestCase):
    def test_cannot_create_instance_with_wrong_ratio(self):
        self.assertRaises(data.BadSplitError, data.MultiSplitter, 'dataset', [0])
        self.assertRaises(data.BadSplitError, data.MultiSplitter, 'dataset', [0.5])
        self.assertRaises(data.BadSplitError, data.MultiSplitter, 'dataset', [2])
        self.assertRaises(data.BadSplitError, data.MultiSplitter, 'dataset', [0.1, 0.2])
        self.assertRaises(data.BadSplitError, data.MultiSplitter, 'dataset', [1, 1])
        self.assertRaises(data.BadSplitError, data.MultiSplitter, 'dataset', [0.5, 0.6])
        self.assertRaises(data.BadSplitError, data.MultiSplitter, 'dataset', [])

    def test_degenerate_split(self):
        splitter = data.MultiSplitter('dataset', [1])
        data_split = splitter.split([5, 3, 4])
        data_slice = data_split[0]
        self.assertEqual([5, 3, 4], list(data_slice))
        self.assertIsInstance(data_slice, data.DatasetSlice)
        self.assertEqual([5, 3, 4], list(data_split.train))

        self.assertRaises(IndexError, lambda: data_split[1])
        self.assertRaises(IndexError, lambda: data_split[12])
        self.assertRaises(AttributeError, lambda: data_split.val)
        self.assertRaises(AttributeError, lambda: data_split.test)

    def test_splitting_into_2_sets(self):
        ds = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        splitter = data.MultiSplitter('dataset', [0.8, 0.2])
        shuffled_indices = [0, 3, 2, 1, 4, 5, 6, 9, 8, 7]
        splitter.configure(shuffled_indices)
        data_split = splitter.split(ds)
        expected_list1 = [10, 13, 12, 11, 14, 15, 16, 19]
        expected_list2 = [18, 17]

        split_list1 = list(data_split[0])
        split_list2 = list(data_split[1])
        self.assertEqual(expected_list1, split_list1)
        self.assertEqual(expected_list2, split_list2)

        self.assertEqual(expected_list1, list(data_split.train))
        self.assertEqual(expected_list2, list(data_split.val))

    def test_splitting_into_3_sets(self):
        ds = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        splitter = data.MultiSplitter('dataset', [0.4, 0.2, 0.4])

        shuffled_indices = [0, 3, 2, 1, 4, 5, 6, 9, 8, 7]
        splitter.configure(shuffled_indices)
        data_split = splitter.split(ds)
        self.assertEqual([10, 13, 12, 11], list(data_split[0]))
        self.assertEqual([14, 15], list(data_split[1]))
        self.assertEqual([16, 19, 18, 17], list(data_split[2]))

    def test_corner_cases(self):
        ds = [10]
        splitter = data.MultiSplitter('dataset', [0.5, 0.5])
        self.assertRaises(data.BadSplitError, splitter.split, ds)

        splitter = data.MultiSplitter('dataset', [0, 1])
        self.assertRaises(data.BadSplitError, splitter.split, ds)

        splitter = data.MultiSplitter('dataset', [1, 0])
        self.assertRaises(data.BadSplitError, splitter.split, ds)

        splitter = data.MultiSplitter('dataset', [1, 0, 0])
        self.assertRaises(data.BadSplitError, splitter.split, ds)

        splitter = data.MultiSplitter('dataset', [0, 1, 0])
        self.assertRaises(data.BadSplitError, splitter.split, ds)

        splitter = data.MultiSplitter('dataset', [0, 0, 1])
        self.assertRaises(data.BadSplitError, splitter.split, ds)

        ds = [1, 2]
        splitter = data.MultiSplitter('dataset', [1/3., 2/3.])
        self.assertRaises(data.BadSplitError, splitter.split, ds)

        ds = [1, 2, 3]
        splitter = data.MultiSplitter('dataset', [0.5, 0.5])
        splitter.configure([1, 0, 2])
        data_split = splitter.split(ds)
        self.assertEqual([2], list(data_split[0]))
        self.assertEqual([1, 3], list(data_split[1]))


class DatasetSliceTests(unittest.TestCase):
    def test_slice_with_invalid_end_points(self):
        self.assertRaises(data.BadSplitError, data.DatasetSlice, [0, 1, 2], 2, 2)
        self.assertRaises(data.BadSplitError, data.DatasetSlice, [0, 1, 2], 1, 1)
        self.assertRaises(data.BadSplitError, data.DatasetSlice, [0, 1, 2], 0, 0)
        self.assertRaises(data.BadSplitError, data.DatasetSlice, [0, 1, 2], 2, 1)
        self.assertRaises(data.BadSplitError, data.DatasetSlice, [0, 1, 2], 2, 0)
        self.assertRaises(data.BadSplitError, data.DatasetSlice, [0, 1, 2], -3, 2)
        self.assertRaises(data.BadSplitError, data.DatasetSlice, [0, 1, 2], 0, -1)

        self.assertRaises(data.BadSplitError, data.DatasetSlice, [1], 0, 2)
        self.assertRaises(data.BadSplitError, data.DatasetSlice, [1], 3, 4)

        self.assertRaises(data.BadSplitError, data.DatasetSlice, [], 0, 0)
        self.assertRaises(data.BadSplitError, data.DatasetSlice, [], 0, 1)

    def test_single_item_slice(self):
        data_slice = data.DatasetSlice([10], 0, 1)
        self.assertEqual(1, len(data_slice))
        self.assertEqual([10], list(data_slice))
        self.assertEqual(10, data_slice[0])
        self.assertRaises(IndexError, lambda: data_slice[1])
        self.assertRaises(IndexError, lambda: data_slice[10])
        self.assertRaises(IndexError, lambda: data_slice[-1])

    def test_many_items_slice(self):
        data_slice = data.DatasetSlice([10, 11, 12], 0, 2)
        self.assertEqual(2, len(data_slice))
        self.assertEqual([10, 11], list(data_slice))
        self.assertEqual(10, data_slice[0])
        self.assertEqual(11, data_slice[1])

        data_slice = data.DatasetSlice([10, 11, 12], 1, 3)
        self.assertEqual(2, len(data_slice))
        self.assertEqual([11, 12], list(data_slice))
        self.assertEqual(11, data_slice[0])
        self.assertEqual(12, data_slice[1])

        data_slice = data.DatasetSlice([10, 11, 12], 0, 3)
        self.assertEqual(3, len(data_slice))
        self.assertEqual([10, 11, 12], list(data_slice))
        self.assertEqual(10, data_slice[0])
        self.assertEqual(11, data_slice[1])
        self.assertEqual(12, data_slice[2])

        data_slice = data.DatasetSlice([10, 11, 12], 1, 2)
        self.assertEqual(1, len(data_slice))
        self.assertEqual([11], list(data_slice))
        self.assertEqual(11, data_slice[0])


class DataSplitTests(unittest.TestCase):
    def test_indexing(self):
        split = data.DataSplit([[0, 2]])
        self.assertEqual([0, 2], split[0])
        self.assertEqual([0, 2], split.train)
        self.assertRaises(IndexError, lambda: split[1])
        self.assertRaises(AttributeError, lambda: split.val)

        split = data.DataSplit([[1, 2], [5]])
        self.assertEqual([1, 2], split[0])
        self.assertEqual([5], split[1])
        self.assertEqual([5], split.val)

        self.assertRaises(IndexError, lambda: split[2])
        self.assertRaises(AttributeError, lambda: split.test)

        split = data.DataSplit([[1, 2], [5], [3]])
        self.assertEqual([1, 2], split[0])
        self.assertEqual([5], split[1])
        self.assertEqual([3], split[2])
        self.assertEqual([3], split.test)

        self.assertRaises(IndexError, lambda: split[3])


class WrappedDatasetTests(unittest.TestCase):
    def test_without_preprocessors(self):
        ds = data.WrappedDataset([10, 11, 12], [])
        self.assertEqual(3, len(ds))
        self.assertEqual([[10], [11], [12]], list(ds))
        self.assertEqual([10], ds[0])
        self.assertEqual([11], ds[1])
        self.assertEqual([12], ds[2])

        ds = data.WrappedDataset([(1, 1), (2, 4), (3, 9)], [])
        self.assertEqual(3, len(ds))
        self.assertEqual([[1, 1], [2, 4], [3, 9]], list(ds))

        ds = data.WrappedDataset([[1, 1], [2, 4], [3, 9]], [])
        self.assertEqual(3, len(ds))
        self.assertEqual([[1, 1], [2, 4], [3, 9]], list(ds))

    def test_with_one_preprocessor_and_one_element_examples(self):
        def square(x): return x**2

        ds = data.WrappedDataset([1, 2, 3], [square])
        self.assertEqual(3, len(ds))
        self.assertEqual([[1], [4], [9]], list(ds))

    def test_when_example_size_is_greater_than_number_of_preprocessors(self):
        def square(x): return x**2

        ds = data.WrappedDataset([(1, 10, 100), (2, 20, 200), (3, 30, 300)], [square])
        self.assertEqual(3, len(ds))
        self.assertEqual([[1, 10, 100], [4, 20, 200], [9, 30, 300]], list(ds))

    def test_when_example_size_is_smaller_than_number_of_preprocessor(self):
        def square(x): return x**2
        def cube(x): return x**3

        ds = data.WrappedDataset([1, 2, 3], [square, cube])
        self.assertEqual(3, len(ds))
        self.assertEqual([[1], [4], [9]], list(ds))

    def test_when_example_size_equals_number_of_preprocessor(self):
        def square(x): return x**2
        def cube(x): return x**3

        ds = data.WrappedDataset([(10, 1), (20, 2), (30, 3)], [square, cube])
        self.assertEqual(3, len(ds))
        self.assertEqual([[10**2, 1**3], [20**2, 2**3], [30**2, 3**3]], list(ds))

# todo: write tests for get_preprocessors method
