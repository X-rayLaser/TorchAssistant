import math
from itertools import zip_longest
from scaffolding.preprocessors import NullProcessor

import torch


class BaseDataset:
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_preprocessors(self):
        raise NotImplementedError


class WrappedDataset(BaseDataset):
    def __init__(self, dataset, preprocessors):
        self.dataset = dataset
        self.preprocessors = preprocessors

    def __getitem__(self, idx):
        example = self.dataset[idx]
        if not (isinstance(example, list) or isinstance(example, tuple)):
            example = [example]

        if len(example) > len(self.preprocessors):
            # when number of inputs > number of preprocessors, leave redundant ones as is
            pairs = zip_longest(example, self.preprocessors)
            return [p(v) if p else v for v, p in pairs]
        else:
            # when number of inputs <= number of preprocessors, ignore redundant preprocessors
            return [preprocessor(v) for v, preprocessor in zip(example, self.preprocessors)]

    def __len__(self):
        return len(self.dataset)

    def get_preprocessors(self):
        dataset = self.dataset
        old_ones = dataset.get_preprocessors() if isinstance(dataset, BaseDataset) else []
        old_ones = list(old_ones)
        new_ones = list(self.preprocessors)

        if old_ones and new_ones:
            self.pad_smaller_list(old_ones, new_ones)
            res = [new_p.wrap_preprocessor(old_p) for old_p, new_p in zip(old_ones, new_ones)]
        else:
            res = old_ones or new_ones
        return res

    def pad_smaller_list(self, old_preprocessors: list, new_preprocessors: list):
        """Make both lists have equal length by padding a smaller one with instances of NullProcessor class."""

        len_diff = len(old_preprocessors) - len(new_preprocessors)
        smaller_list = new_preprocessors if len_diff >= 0 else old_preprocessors

        for i in range(abs(len_diff)):
            smaller_list.append(NullProcessor())


class MergedDataset(BaseDataset):
    class SemiInterval:
        def __init__(self, a, b):
            self.a = a
            self.b = b

        def __contains__(self, n):
            return self.a <= n < self.b

    def __init__(self, *datasets):
        """
        All datasets are assumed to have identical shape of examples and preprocessors.
        """
        self.datasets = datasets
        sizes = [len(ds) for ds in self.datasets]

        s = 0
        self.intervals = []
        for size in sizes:
            self.intervals.append(self.SemiInterval(s, s + size))
            s += size

    def __getitem__(self, idx):
        datasets_with_intervals = zip(self.intervals, self.datasets)
        for ivl, dataset in datasets_with_intervals:
            if idx in ivl:
                offset = idx - ivl.a
                return dataset[offset]

        raise IndexError('dataset index out of range')

    def __len__(self):
        return sum([len(ds) for ds in self.datasets])

    def get_preprocessors(self):
        # since all datasets are assumed to have the same preprocessors,
        # we can return preprocessors from any of them
        return self.datasets[0].get_preprocessors() if self.datasets else []


class MultiSplitter:
    def __init__(self, dataset_name, ratio):
        if not math.isclose(sum(ratio), 1., rel_tol=1e-5):
            raise BadSplitError(f'Values must add to 1, but they add to {sum(ratio)} instead.')

        self.dataset_name = dataset_name
        self.ratio = ratio
        self.shuffled_indices = None

    def configure(self, shuffled_indices):
        self.shuffled_indices = shuffled_indices

    def state_dict(self):
        return dict(shuffled_indices=self.shuffled_indices)

    def load_state_dict(self, state_dict):
        self.shuffled_indices = state_dict["shuffled_indices"]

    def split(self, dataset):
        if not dataset:
            raise BadSplitError('Cannot split an empty dataset')

        shuffled_indices = self.shuffled_indices or list(range(len(dataset)))

        shuffled_size = len(shuffled_indices)
        ds_size = len(dataset)
        if shuffled_size != ds_size:
            raise BadSplitError(
                f'Shuffled_indices size mismatch: {shuffled_size} for a dataset of size {ds_size}'
            )

        dataset = self.shuffled_dataset(dataset, shuffled_indices)

        sizes = [int((x * ds_size)) for x in self.ratio[:-1]]
        last_slice_size = ds_size - sum(sizes)
        sizes.append(last_slice_size)

        slices = []
        for i in range(len(sizes)):
            from_index = sum(sizes[:i])
            to_index = from_index + sizes[i]
            slices.append(DatasetSlice(dataset, from_index, to_index))

        return DataSplit(slices)

    def shuffled_dataset(self, ds: BaseDataset, shuffled_indices):
        class ShuffledDataset(BaseDataset):
            def __getitem__(self, idx):
                new_index = shuffled_indices[idx]
                return ds[new_index]

            def __len__(self):
                return len(ds)

            def get_preprocessors(self):
                return ds.get_preprocessors()

        return ShuffledDataset()

    @property
    def num_parts(self):
        return len(self.ratio)


class DataSplit:
    def __init__(self, slices):
        self.slices = slices

    def __getitem__(self, idx):
        return self.slices[idx]

    def __getattr__(self, attr):
        try:
            if attr == 'train':
                return self[0]
            elif attr == 'val':
                return self[1]
            elif attr == 'test':
                return self[2]
            else:
                raise IndexError
        except IndexError:
            raise AttributeError(f'MultiSplitter instance has no "{attr}" attribute')


class BadSplitError(Exception):
    pass


class DatasetSlice(BaseDataset):
    def __init__(self, ds, index_from, index_to):
        """Create a dataset slice

        :param ds: original dataset
        :type ds: BaseDataset or Sequence
        :param index_from: start index of the slice (included in a slice)
        :param index_to: last index of the slice (excluded from a slice)
        """

        if index_to <= index_from:
            raise BadSplitError(
                f'index_to must be greater than index_from: Got {index_to}, {index_from}'
            )

        if index_to < 0 or index_from < 0:
            raise BadSplitError(
                f'Negative indices are not allowed: Got index_to={index_to}, index_from={index_from}'
            )

        if index_to > len(ds) or index_from > len(ds):
            raise BadSplitError(
                f'At least one of indices is out of range: Got index_to={index_to}, index_from={index_from}'
            )

        self.ds = ds
        self.index_from = index_from
        self.index_to = index_to

    def __getitem__(self, idx):
        if not (0 <= idx < len(self)):
            raise IndexError(f'DatasetSlice: Index out of bounds: {idx}')
        return self.ds[idx + self.index_from]

    def __len__(self):
        return self.index_to - self.index_from

    def get_preprocessors(self):
        return self.ds.get_preprocessors() if isinstance(self.ds, BaseDataset) else []


# todo: consider renaming it (e.g. DataFactory, BatchFactory, TrainingBatchGenerator)
class DataBlueprint:
    def __init__(self, input_loaders):
        self.input_loaders = input_loaders

        batch_names = [input_loader.input_alias for input_loader in self.input_loaders]
        self.batch_names = batch_names
        self.batch_loaders = self.refresh_batch_loaders(self.input_loaders)

    def override_datasets(self, new_datasets: dict):
        for input_loader in self.input_loaders:
            dataset = new_datasets[input_loader.input_alias]
            input_loader.loader_factory.swap_dataset(dataset)

        self.batch_loaders = self.refresh_batch_loaders(self.input_loaders)

    def refresh_batch_loaders(self, input_loaders):
        batch_loaders = []
        for input_loader in input_loaders:
            var_names = input_loader.variable_names
            loader_factory = input_loader.loader_factory
            data_loader = loader_factory.build()
            batch_loaders.append(BatchLoader(data_loader, var_names))
        return batch_loaders

    def __len__(self):
        return min(map(len, self.batch_loaders))

    def __iter__(self):
        iterators = [iter(loader) for loader in self.batch_loaders]
        for i, batches in enumerate(zip(*iterators)):
            named_batches = dict(zip(self.batch_names, batches))
            yield named_batches


class BatchLoader:
    """An iterable of named batches.

    It is a wrapper around a torch DataLoader class.
    An item is yielded as a dictionary with "name" -> "batch" mapping.
    """
    def __init__(self, data_loader, var_names):
        self.data_loader = data_loader
        self.var_names = var_names

    def __iter__(self):
        for batch in self.data_loader:
            yield dict(zip(self.var_names, batch))

    def __len__(self):
        return len(self.data_loader)


class LoaderFactory:
    """Factory producing instances of torch.utils.data.DataLoader."""
    def __init__(self, dataset, collator, **kwargs):
        self.dataset = dataset
        self.collator = collator
        self.kwargs = kwargs

    def build(self):
        return torch.utils.data.DataLoader(
            self.dataset, collate_fn=self.collator, **self.kwargs
        )

    def swap_dataset(self, dataset):
        if isinstance(self.dataset, BaseDataset):
            preprocessors = self.dataset.get_preprocessors()
        else:
            preprocessors = []

        self.dataset = WrappedDataset(dataset, preprocessors)
