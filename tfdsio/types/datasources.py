# Data Sources
import abc
import inspect

from typing import Any, Callable, Iterable, Mapping, List, Optional, Sequence, Union
from tensorflow.python.data import Dataset


from .registries import DatasetProviderBase
from .base import Feature, ShardInfo
from .loaders import LazyTFDSLoader, LazyTFDSIOLoader, LazyHFTFDSLoader


class DataSource(DatasetProviderBase):
    """A `DatasetProvider` that provides raw data from an input source.
    Inherits all abstract methods and properties of `DatasetProviderBase` except
    those overidden below.
    """

    def __init__(self, splits: Iterable[str], num_input_examples: Optional[Mapping[str, int]] = None):
        self._splits = tuple(splits)
        self._num_input_examples = (dict(num_input_examples) if num_input_examples is not None else None)

    @property
    def splits(self) -> Sequence[str]:
        return self._splits

    @property
    def output_features(self) -> Mapping[str, Feature]:
        """Override unused property of `DatasetProviderBase`."""
        raise NotImplementedError

    @abc.abstractmethod
    def list_shards(self, split: str) -> Sequence[str]:
        """Returns string identifiers of input shards."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_dataset(self, split: str, shuffle: bool = True, seed: Optional[int] = None, shard_info: Optional[ShardInfo] = None) -> Dataset:
        """Overrides base class to add shard identifier and remove use_cached.
        Args:
        split: string, the split to return.
        shuffle: bool, whether to shuffle the input source.
        seed: tf.int64 scalar tf.Tensor (or None) for shuffling input source.
        shard_info: optional specification for loading a shard of the split.
        """
        raise NotImplementedError

    def num_input_examples(self, split: str) -> Optional[int]:
        if self._num_input_examples is None:
            return None
        return self._num_input_examples[split]


    def _validate_args(fn, expected_pos_args):
        """Ensure function has exactly expected positional args."""
        argspec = inspect.getfullargspec(fn)
        expected_pos_args = tuple(expected_pos_args)
        actual_args = tuple(argspec.args)
        if actual_args[:len(expected_pos_args)] != expected_pos_args:
            raise ValueError("'%s' must have positional args %s, got: %s" % (fn.__name__, expected_pos_args, actual_args))
        actual_pos_args = tuple(argspec.args[:-len(argspec.defaults)] if argspec.defaults else argspec.args)
        if actual_pos_args != expected_pos_args[:len(actual_pos_args)]:
            raise ValueError("'%s' may only have positional args %s, got: %s" % (fn.__name__, expected_pos_args, actual_pos_args))


class TFDSDataSource(DataSource):
    """A `DataSource` that uses TensorFlow Datasets to provide the input data."""

    def __init__(
        self, 
        tfds_name: str, 
        tfds_data_dir: Optional[str] = None, 
        splits: Optional[Union[Iterable[str], Mapping[str, str]]] = None
    ):
        """TfdsTask constructor.
        Args:
        tfds_name: string, the name and version number of a TFDS dataset,
            optionally with a config.
        tfds_data_dir: string, an optional path to a specific TFDS data directory
            to use.
        splits: an iterable of allowable string split names, a dict mapping
            allowable canonical splits (e.g., 'validation') to TFDS splits or slices
            (e.g., 'train[':1%']), or None. The default, None, uses all available
            splits from the TFDS dataset info.
        """
        if ":" not in tfds_name:
            raise ValueError(f"TFDS name must contain a version number, got: {tfds_name}")

        if splits and not isinstance(splits, dict):
            splits = {k: k for k in splits}

        self._tfds_dataset = LazyTFDSLoader(
            tfds_name, 
            data_dir = tfds_data_dir, 
            split_map = splits if isinstance(splits, dict) else None
        )

        # If splits are not provided, we pass an empty tuple and use the lazy
        # lookup in the `splits` property.
        super().__init__(splits=splits or ())

    @property
    def splits(self):
        """Overrides since we can't call `info.splits` until after init."""
        return self._splits or self._tfds_dataset.info.splits

    @property
    def tfds_dataset(self):
        return self._tfds_dataset

    def get_dataset(self, split: str, shuffle: bool = True, seed: Optional[int] = None, shard_info: Optional[ShardInfo] = None) -> Dataset:
        return self.tfds_dataset.load(split, shuffle_files=shuffle, seed=seed, shard_info=shard_info)

    def num_input_examples(self, split: str) -> int:
        """Overrides since we can't call `info.splits` until after init."""
        return self.tfds_dataset.size(split)

    def list_shards(self, split: str) -> Sequence[str]:
        return self.tfds_dataset.files(split)

class TFDSIODataSource(DataSource):
    def __init__(
        self, 
        config_or_file: Any, 
        preprocessors: Optional[Union[Callable, List[Callable]]] = None,
        splits: Optional[Union[Iterable[str], Mapping[str, str]]] = None,
    ):
        self._tfds_dataset = LazyTFDSIOLoader(
            config_or_file = config_or_file, 
            preprocessors = preprocessors, 
            split_map = splits if isinstance(splits, dict) else None
        )
        super().__init__(splits = splits or ())
    
    @property
    def splits(self):
        """Overrides since we can't call `info.splits` until after init."""
        return self._tfds_dataset.info.splits

    @property
    def tfds_dataset(self):
        return self._tfds_dataset

    def get_dataset(self, split: str, shuffle: bool = True, seed: Optional[int] = None, shard_info: Optional[ShardInfo] = None) -> Dataset:
        return self.tfds_dataset.load(split, shuffle_files = shuffle, seed = seed, shard_info = shard_info)

    def num_input_examples(self, split: str) -> int:
        """Overrides since we can't call `info.splits` until after init."""
        return self.tfds_dataset.size(split)

    def list_shards(self, split: str) -> Sequence[str]:
        return self.tfds_dataset.files(split)


class HFDataSource(DataSource):
    def __init__(
        self, 
        config_or_file: Any, 
        dataset: Any,
        splits: Optional[Union[Iterable[str], Mapping[str, str]]] = None,
    ):
        self._tfds_dataset = LazyHFTFDSLoader(
            config_or_file = config_or_file, 
            dataset = dataset, 
            split_map = splits if isinstance(splits, dict) else None
        )
        super().__init__(splits = splits or ())
    
    @property
    def splits(self):
        """Overrides since we can't call `info.splits` until after init."""
        return self._tfds_dataset.info.splits

    @property
    def tfds_dataset(self):
        return self._tfds_dataset

    def get_dataset(self, split: str, shuffle: bool = True, seed: Optional[int] = None, shard_info: Optional[ShardInfo] = None) -> Dataset:
        return self.tfds_dataset.load(split, shuffle_files = shuffle, seed = seed, shard_info = shard_info)

    def num_input_examples(self, split: str) -> int:
        """Overrides since we can't call `info.splits` until after init."""
        return self.tfds_dataset.size(split)

    def list_shards(self, split: str) -> Sequence[str]:
        return self.tfds_dataset.files(split)

# class T5DataSource(DataSource):
#     def __init__(self, config_or_file: Any, preprocessor: Optional[Any] = None, splits: Optional[Union[Iterable[str], Mapping[str, str]]] = None):
#         self._tfds_dataset = LazyTFDSIOLoader(config_or_file=config_or_file, preprocessor=preprocessor, split_map=splits if isinstance(splits, dict) else None)
#         super().__init__(splits=splits or ())

#     @property
#     def splits(self):
#         """Overrides since we can't call `info.splits` until after init."""
#         return self._tfds_dataset.info.splits

#     @property
#     def tfds_dataset(self):
#         return self._tfds_dataset

#     def get_dataset(self, split: str, shuffle: bool = True, seed: Optional[int] = None, shard_info: Optional[ShardInfo] = None) -> tf.data.Dataset:
#         return self.tfds_dataset.load(split, shuffle_files=shuffle, seed=seed, shard_info=shard_info)

#     def num_input_examples(self, split: str) -> int:
#         """Overrides since we can't call `info.splits` until after init."""
#         return self.tfds_dataset.size(split)

#     def list_shards(self, split: str) -> Sequence[str]:
#         return self.tfds_dataset.files(split)
