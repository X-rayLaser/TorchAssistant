import math

from torch.utils.data import Dataset

from scaffolding.utils import Serializable


class DataSplitter(Serializable):
    def __init__(self, train_fraction=0.8):
        self.dataset = None
        self.train_fraction = train_fraction

        self.num_train = None
        self.num_val = None

    def prepare(self, dataset):
        self.dataset = dataset
        self.num_train = int(self.train_fraction * len(dataset))
        self.num_val = len(dataset) - self.num_train

    @property
    def train_ds(self):
        return []

    @property
    def val_ds(self):
        return []

    def state_dict(self):
        return {}

    def load(self, state):
        pass


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
                f'Shuffled_indices size mismatch: {shuffled_size} for a dataset of size {ds_size}')

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

    def shuffled_dataset(self, ds, shuffled_indices):
        class ShuffledDataset:
            def __getitem__(self, idx):
                new_index = shuffled_indices[idx]
                return ds[new_index]

            def __len__(self):
                return len(ds)

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


class SimpleSplitter(DataSplitter):
    @property
    def train_ds(self):
        return DatasetSlice(self.dataset, 0, self.num_train)

    @property
    def val_ds(self):
        offset = self.num_train
        return DatasetSlice(self.dataset, offset, offset + self.num_val)


class DatasetSlice(Dataset):
    def __init__(self, ds, index_from, index_to):
        """Create a dataset slice

        :param ds: original dataset
        :type ds: type of sequence
        :param index_from: start index of the slice (included in a slice)
        :param index_to: last index of the slice (excluded from a slice)
        """

        if index_to <= index_from:
            raise BadSplitError(
                f'index_to must be greater than index_from: Got {index_to}, {index_from}'
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
