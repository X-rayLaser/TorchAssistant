import unittest
from torchassistant import data
from torchassistant.preprocessors import ValuePreprocessor
from torchassistant.collators import BatchDivide


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


def square(x): return x**2
def cube(x): return x**3
def increment(x): return x + 1


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
        ds = data.WrappedDataset([1, 2, 3], [square])
        self.assertEqual(3, len(ds))
        self.assertEqual([[1], [4], [9]], list(ds))

    def test_when_example_size_is_greater_than_number_of_preprocessors(self):
        ds = data.WrappedDataset([(1, 10, 100), (2, 20, 200), (3, 30, 300)], [square])
        self.assertEqual(3, len(ds))
        self.assertEqual([[1, 10, 100], [4, 20, 200], [9, 30, 300]], list(ds))

    def test_when_example_size_is_smaller_than_number_of_preprocessor(self):
        ds = data.WrappedDataset([1, 2, 3], [square, cube])
        self.assertEqual(3, len(ds))
        self.assertEqual([[1], [4], [9]], list(ds))

    def test_when_example_size_equals_number_of_preprocessor(self):
        ds = data.WrappedDataset([(10, 1), (20, 2), (30, 3)], [square, cube])
        self.assertEqual(3, len(ds))
        self.assertEqual([[10**2, 1**3], [20**2, 2**3], [30**2, 3**3]], list(ds))


class SimpleDataset(list, data.BaseDataset):
    def get_preprocessors(self):
        return []


class FunctionalPreprocessor(ValuePreprocessor):
    def __init__(self, f):
        self.func = f

    def process(self, x):
        return self.func(x)


class GetPreprocessorsTests(unittest.TestCase):
    def test_get_preprocessors_from_wrapped_dataset(self):
        examples = [(1, 10), (2, 20), (3, 30)]
        ds = data.WrappedDataset(SimpleDataset(examples), [cube, square])
        self.assertEqual([cube, square], ds.get_preprocessors())

        ds = data.WrappedDataset(examples, [cube, square])
        self.assertEqual([cube, square], ds.get_preprocessors())

        ds = data.WrappedDataset(examples, [])
        self.assertEqual([], ds.get_preprocessors())

    def test_get_preprocessors_from_merged_dataset(self):
        ds1 = data.WrappedDataset(SimpleDataset([(1, 10), (2, 20), (3, 30)]), [cube, square])
        ds2 = data.WrappedDataset(SimpleDataset([(3, 30), (4, 40)]), [square])

        ds = data.MergedDataset(ds1, ds2)
        self.assertEqual([cube, square], ds.get_preprocessors())

        ds1 = data.WrappedDataset([(1, 10), (2, 20), (3, 30)], [cube])
        ds2 = data.WrappedDataset([(3, 30), (4, 40)], [square])

        ds = data.MergedDataset(ds1, ds2)
        self.assertEqual([cube], ds.get_preprocessors())

        ds = data.MergedDataset()
        self.assertEqual([], ds.get_preprocessors())

    def test_get_preprocessors_from_dataset_slice(self):
        ds = data.WrappedDataset(SimpleDataset([(1, 10), (2, 20), (3, 30)]), [cube, square])
        dataset_slice = data.DatasetSlice(ds, 0, 2)
        self.assertEqual([cube, square], dataset_slice.get_preprocessors())

        ds = data.WrappedDataset([(1, 10), (2, 20), (3, 30)], [cube])
        dataset_slice = data.DatasetSlice(ds, 0, 2)
        self.assertEqual([cube], dataset_slice.get_preprocessors())

        ds = [(1, 10), (2, 20), (3, 30)]
        dataset_slice = data.DatasetSlice(ds, 0, 2)
        self.assertEqual([], dataset_slice.get_preprocessors())

    def test_preprocessing_for_dataset_with_many_layers_of_wrappers(self):
        ds1 = data.WrappedDataset([0, 1, 2], [increment])
        self.assertEqual([[1], [2], [3]], list(ds1))

        ds2 = data.WrappedDataset(ds1, [cube])
        self.assertEqual([[1], [8], [27]], list(ds2))

        ds3 = data.WrappedDataset(ds2, [square])
        self.assertEqual([[1], [8**2], [27**2]], list(ds3))

        ds1 = data.WrappedDataset([0, 1, 2], [cube])
        self.assertEqual([[0], [1], [8]], list(ds1))

        ds2 = data.WrappedDataset(ds1, [increment])
        self.assertEqual([[1], [2], [9]], list(ds2))

        ds3 = data.WrappedDataset(ds2, [square])
        self.assertEqual([[1], [4], [81]], list(ds3))

        ds1 = data.WrappedDataset([(0, 0), (1, 1), (2, 2)], [increment, cube])
        self.assertEqual([[1, 0], [2, 1], [3, 8]], list(ds1))

        ds2 = data.WrappedDataset(ds1, [cube])
        self.assertEqual([[1, 0], [8, 1], [27, 8]], list(ds2))

        ds3 = data.WrappedDataset(ds2, [square, square])
        self.assertEqual([[1, 0], [8**2, 1], [27**2, 8**2]], list(ds3))

    def test_get_preprocessors_from_nested_wrapped_dataset(self):
        raw_data = [0, 1, 2]
        p1 = FunctionalPreprocessor(increment)
        ds1 = data.WrappedDataset(raw_data, [p1])
        self.assertEqual([p1], ds1.get_preprocessors())

        p2 = FunctionalPreprocessor(cube)
        ds2 = data.WrappedDataset(ds1, [p2])

        returned_preprocessors = ds2.get_preprocessors()
        self.assertEqual(1, len(returned_preprocessors))
        self.assertEqual([1, 8, 27], [returned_preprocessors[0](x) for x in raw_data])

        p3 = FunctionalPreprocessor(square)
        ds3 = data.WrappedDataset(ds2, [p3])

        returned_preprocessors = ds3.get_preprocessors()
        self.assertEqual(1, len(returned_preprocessors))
        self.assertEqual([1, 8**2, 27**2], [returned_preprocessors[0](x) for x in raw_data])

    def test_get_preprocessors_from_nested_wrapped_dataset_with_2_preprocessors_per_example(self):
        incrementer = FunctionalPreprocessor(increment)
        take_square = FunctionalPreprocessor(square)
        take_cube = FunctionalPreprocessor(cube)

        raw_data = [(0, 0), (1, 1), (2, 2)]
        ds1 = data.WrappedDataset(raw_data, [incrementer, take_cube])
        preprocessors = ds1.get_preprocessors()
        p1, p2 = preprocessors
        self.assertEqual([(1, 0), (2, 1), (3, 8)], [(p1(x), p2(y)) for x, y in raw_data])

        ds2 = data.WrappedDataset(ds1, [take_cube])
        preprocessors = ds2.get_preprocessors()
        p1, p2 = preprocessors
        self.assertEqual([(1, 0), (8, 1), (27, 8)], [(p1(x), p2(y)) for x, y in raw_data])

        ds3 = data.WrappedDataset(ds2, [take_square, take_square])
        preprocessors = ds3.get_preprocessors()
        p1, p2 = preprocessors
        self.assertEqual([(1, 0), (8 ** 2, 1), (27 ** 2, 8 ** 2)], [(p1(x), p2(y)) for x, y in raw_data])

    def test_can_have_different_number_of_preprocessors_on_different_levels(self):
        incrementer = FunctionalPreprocessor(increment)
        take_square = FunctionalPreprocessor(square)
        take_cube = FunctionalPreprocessor(cube)

        raw_data = [(0, 0), (1, 1), (2, 2)]
        ds1 = data.WrappedDataset(raw_data, [incrementer])
        ds2 = data.WrappedDataset(ds1, [take_cube, take_square])
        ds3 = data.WrappedDataset(ds2, [FunctionalPreprocessor(increment)])

        preprocessors = ds3.get_preprocessors()

        p1, p2 = preprocessors
        self.assertEqual([(1 + 1, 0), (8 + 1, 1), (27 + 1, 4)], [(p1(x), p2(y)) for x, y in raw_data])


class BatchLoaderTests(unittest.TestCase):
    def test(self):
        ints1 = [0, 1, 2, 3]
        squares1 = [0, 1, 4, 9]
        cubes1 = [0, 1, 8, 27]

        ints2 = [0, -1, -2, -3]
        squares2 = [0, 1, 4, 9]
        cubes2 = [0, -1, -8, -27]

        first_batch = [ints1, squares1, cubes1]
        second_batch = [ints2, squares2, cubes2]
        batches = [first_batch, second_batch]

        loader = data.BatchLoader(batches, ['x', 'square', 'cube'])
        self.assertEqual(2, len(loader))
        it = iter(loader)
        b1 = next(it)
        self.assertEqual(dict(x=list(ints1), square=list(squares1), cube=list(cubes1)), b1)
        b2 = next(it)

        self.assertEqual(dict(x=list(ints2), square=list(squares2), cube=list(cubes2)), b2)


class LoaderFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        ints = [0, 1, 2, 3]
        self.ints = ints
        raw_data = list(zip(ints, ints))
        self.raw_data = raw_data

        ds1 = data.WrappedDataset(raw_data, [FunctionalPreprocessor(square),
                                             FunctionalPreprocessor(cube)])
        ds2 = data.WrappedDataset(ds1, [FunctionalPreprocessor(increment)])

        self.factory = data.LoaderFactory(ds2, BatchDivide(), batch_size=4, shuffle=False)

    def test_without_transformations(self):
        factory = data.LoaderFactory(self.raw_data, BatchDivide(), batch_size=4, shuffle=False)
        loader = factory.build()
        self.assertEqual(1, len(loader))
        x, y = next(iter(loader))
        self.assertEqual(self.ints, x)
        self.assertEqual(self.ints, y)

        ints = [0, -1, -2, -3]
        raw_data = list(zip(ints, ints))
        factory.swap_dataset(raw_data)
        loader = factory.build()
        self.assertEqual(1, len(loader))
        x, y = next(iter(loader))
        self.assertEqual(ints, x)
        self.assertEqual(ints, y)

    def test_without_swapping_dataset(self):
        factory = self.factory
        loader = factory.build()
        self.assertEqual(1, len(loader))
        x, y = next(iter(loader))
        self.assertEqual(4, len(x))
        self.assertEqual(4, len(y))

        self.assertEqual(self.expected_x(self.ints), x)
        self.assertEqual(self.expected_y(self.ints), y)

    def test_swapping_preserves_associated_data_transformations(self):
        factory = self.factory
        ints = [0, -1, -2, -3]
        ds = list(zip(ints, ints))
        factory.swap_dataset(ds)
        loader = factory.build()

        self.assertEqual(1, len(loader))
        x, y = next(iter(loader))
        self.assertEqual(4, len(x))
        self.assertEqual(4, len(y))

        self.assertEqual(self.expected_x(ints), x)
        self.assertEqual(self.expected_y(ints), y)

    def expected_x(self, ints):
        return [elem**2 + 1 for elem in ints]

    def expected_y(self, ints):
        return [elem**3 for elem in ints]
