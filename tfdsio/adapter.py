
# adapter for T5 from https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/seqio/utils.py

import abc
import collections
import inspect
import json
import os
import re

from absl import logging
import dataclasses
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, Union
from tfdsio.dataset import TFDSIOConfig, TFDSIOCorpus

try:
    import seqio
    _TFDS_DATA_DIR_OVERRIDE = seqio.utils._TFDS_DATA_DIR_OVERRIDE
    _GLOBAL_CACHE_DIRECTORIES = seqio.utils._GLOBAL_CACHE_DIRECTORIES
    Vocabulary = seqio.vocabularies.Vocabulary
except ImportError:
    _TFDS_DATA_DIR_OVERRIDE = None
    _GLOBAL_CACHE_DIRECTORIES = []
    Vocabulary = Any

@dataclasses.dataclass(frozen=True)
class Feature:
    """A container for attributes of output features of data providers."""
    vocabulary: Vocabulary
    add_eos: bool = True
    required: bool = True
    dtype: tf.DType = tf.int32


@dataclasses.dataclass(frozen=True)
class ShardInfo:
    """A container for specifying sharding info."""
    index: int
    num_shards: int


class DatasetProviderBase(metaclass=abc.ABCMeta):
    """Abstract base for classes that provide a tf.data.Dataset."""

    @abc.abstractproperty
    def output_features(self) -> Mapping[str, Feature]:
        raise NotImplementedError

    @abc.abstractproperty
    def splits(self) -> Sequence[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_dataset(self, sequence_length: int, split: str, use_cached: bool = False, shuffle: bool = True,
        seed: Optional[int] = None, shard_info: Optional[ShardInfo] = None, num_epochs: int = 1) -> tf.data.Dataset:
        """Returns the requested tf.data.Dataset."""
        raise NotImplementedError

    @abc.abstractmethod
    def num_input_examples(self, split: str) -> int:
        raise NotImplementedError


class DatasetProviderRegistry(object):
    _REGISTRY: MutableMapping[str, DatasetProviderBase]
    _PROVIDER_TYPE: Type[DatasetProviderBase]

    @classmethod
    def add_provider(cls, name: str, provider):
        """Adds a data provider instance to the registry."""
        if name in cls._REGISTRY:
            raise ValueError("Attempting to register duplicate provider: %s" % name)
        if not isinstance(provider, cls._PROVIDER_TYPE):
            raise ValueError("Attempting to register a class not of an invalid type. Expecting instance of %s, got %s" % (cls._PROVIDER_TYPE, type(provider).__name__))
        cls._REGISTRY[name] = provider

    @classmethod
    def add(cls, name: str, provider_cls, *provider_args, **provider_kwargs):
        """Instantiates and adds provider to the registry."""
        if not issubclass(provider_cls, cls._PROVIDER_TYPE):
            raise ValueError("Attempting to register a class not of an invalid type. Expecting instance of %s, got %s" %(cls._PROVIDER_TYPE, provider_cls))
        provider = provider_cls(*provider_args, **provider_kwargs)
        cls.add_provider(name, provider)
        return provider

    @classmethod
    def remove(cls, name):
        """Remove provider from the registry, if it exists."""
        if name in cls._REGISTRY:
            del cls._REGISTRY[name]

    @classmethod
    def get(cls, name):
        """Returns provider from the registry."""
        if name not in cls._REGISTRY:
            raise ValueError("Provider name not registered: %s" % name)
        return cls._REGISTRY[name]

    @classmethod
    def names(cls):
        """Returns all provider names in registry."""
        return cls._REGISTRY.keys()

    @classmethod
    def reset(cls):
        """Removes all of the registered tasks."""
        cls._REGISTRY = {}

    @classmethod
    def get_dataset(cls, name, sequence_length, split, use_cached=False, shuffle=True, seed=None, shard_info=None, num_epochs=1):
        """Returns the requested tf.data.Dataset."""
        return cls.get(name).get_dataset(sequence_length=sequence_length, split=split, use_cached=use_cached, shuffle=shuffle, seed=seed, shard_info=shard_info, num_epochs=num_epochs)



class LazyTFDSIOLoader(object):
    """Wrapper for TFDS datasets with memoization and additional functionality.
    Lazily loads info from TFDS and provides memoization to avoid expensive hidden
    file operations. Also provides additional utility methods.
    """

    _MEMOIZED_BUILDERS = {}

    def __init__(self, config_or_file, preprocessor=None, split_map=None):
        """LazyTfdsLoader constructor.
        Args:
        name: str, the name of the TFDS dataset.
        data_dir: str (optional), directory to read/write TFDS data.
        split_map: dict (optional), mapping from canonical splits
            (e.g., 'validation') to TFDS splits or slices
            (e.g., 'train[':1%']).
        """
        self._config = TFDSIOConfig()
        self._config.from_auto(config_or_file)
        self._preprocessor = preprocessor

        self._name = self._config.dataset_name
        self._data_dir = self._config.data_dir
        self._split_map = split_map

    @property
    def name(self):
        return self._name

    @property
    def data_dir(self):
        if _TFDS_DATA_DIR_OVERRIDE:
            if self._data_dir:
                logging.warning("Overriding TFDS data directory '%s' with '%s' for dataset '%s'.", self._data_dir, _TFDS_DATA_DIR_OVERRIDE, self.name)
            return _TFDS_DATA_DIR_OVERRIDE
        return self._data_dir

    @property
    def builder(self):
        builder_key = (self.name, self.data_dir)
        if builder_key not in LazyTFDSIOLoader._MEMOIZED_BUILDERS:
            LazyTFDSIOLoader._MEMOIZED_BUILDERS[builder_key] = TFDSIOCorpus(config=self._config)
        return LazyTFDSIOLoader._MEMOIZED_BUILDERS[builder_key]

    @property
    def info(self):
        return self.builder.info

    def _map_split(self, split):
        return self._split_map[split] if self._split_map else split

    def files(self, split):
        """Returns set of instructions for reading TFDS files for the dataset."""
        split = self._map_split(split)

        if "/" not in self.name and self.builder.BUILDER_CONFIGS:
            # If builder has multiple configs, and no particular config was
            # requested, raise an error.
            raise ValueError("Dataset '%s' has multiple configs." % self.name)

        split_info = self.builder.info.splits[split]
        files = split_info.file_instructions

        if not files:
            logging.fatal("No TFRecord files found for dataset: %s", self.name)
        return files

    def load(self, split, shuffle_files, seed=None, shard_info=None):
        """Returns a tf.data.Dataset for the given split."""
        split = self._map_split(split)
        input_context = (tf.distribute.InputContext(num_input_pipelines=shard_info.num_shards, input_pipeline_id=shard_info.index) if shard_info else None)
        builder = TFDSIOCorpus(config=self._config)
        if self._preprocessor:
            builder.set_preprocessor(self._preprocessor)
        builder.download_and_prepare()
        ds = builder.as_dataset(split=split, shuffle_files=shuffle_files, read_config=tfds.ReadConfig(shuffle_seed=seed, skip_prefetch=True, input_context=input_context))
        return ds

    def load_shard(self, file_instruction, shuffle_files=False, seed=None):
        """Returns a dataset for a single shard of the TFDS TFRecord files."""
        ds = self.builder._tfrecords_reader.read_files([file_instruction], read_config=tfds.ReadConfig(shuffle_seed=seed), shuffle_files=shuffle_files)
        return ds

    def size(self, split):
        """Returns the number of examples in the split."""
        split = self._map_split(split)
        ds_splits = self.info.splits
        dataset_size = ds_splits[split].num_examples
        # Very large datasets have num_examples = 0; default instead to np.inf
        dataset_size = dataset_size if dataset_size > 0 else np.inf
        return dataset_size


class LazyTfdsLoader(object):
    """Wrapper for TFDS datasets with memoization and additional functionality.
    Lazily loads info from TFDS and provides memoization to avoid expensive hidden
    file operations. Also provides additional utility methods.
    """

    _MEMOIZED_BUILDERS = {}

    def __init__(self, name, data_dir=None, split_map=None):
        """LazyTfdsLoader constructor.
        Args:
        name: str, the name of the TFDS dataset.
        data_dir: str (optional), directory to read/write TFDS data.
        split_map: dict (optional), mapping from canonical splits
            (e.g., 'validation') to TFDS splits or slices
            (e.g., 'train[':1%']).
        """
        self._name = name
        self._data_dir = data_dir
        self._split_map = split_map

    @property
    def name(self):
        return self._name

    @property
    def data_dir(self):
        if _TFDS_DATA_DIR_OVERRIDE:
            if self._data_dir:
                logging.warning("Overriding TFDS data directory '%s' with '%s' for dataset '%s'.", self._data_dir, _TFDS_DATA_DIR_OVERRIDE, self.name)
            return _TFDS_DATA_DIR_OVERRIDE
        return self._data_dir

    @property
    def builder(self):
        builder_key = (self.name, self.data_dir)
        if builder_key not in LazyTfdsLoader._MEMOIZED_BUILDERS:
            LazyTfdsLoader._MEMOIZED_BUILDERS[builder_key] = tfds.builder(self.name, data_dir=self.data_dir)
        return LazyTfdsLoader._MEMOIZED_BUILDERS[builder_key]

    @property
    def info(self):
        return self.builder.info

    def _map_split(self, split):
        return self._split_map[split] if self._split_map else split

    def files(self, split):
        """Returns set of instructions for reading TFDS files for the dataset."""
        split = self._map_split(split)

        if "/" not in self.name and self.builder.BUILDER_CONFIGS:
            # If builder has multiple configs, and no particular config was
            # requested, raise an error.
            raise ValueError("Dataset '%s' has multiple configs." % self.name)

        split_info = self.builder.info.splits[split]
        files = split_info.file_instructions

        if not files:
            logging.fatal("No TFRecord files found for dataset: %s", self.name)
        return files

    def load(self, split, shuffle_files, seed=None, shard_info=None):
        """Returns a tf.data.Dataset for the given split."""
        split = self._map_split(split)
        input_context = (tf.distribute.InputContext(num_input_pipelines=shard_info.num_shards, input_pipeline_id=shard_info.index) if shard_info else None)
        return tfds.load(self._name,
            split=split,
            data_dir=self.data_dir,
            shuffle_files=shuffle_files,
            download=True,
            try_gcs=True,
            read_config=tfds.ReadConfig(shuffle_seed=seed, skip_prefetch=True, input_context=input_context)
        )

    def load_shard(self, file_instruction, shuffle_files=False, seed=None):
        """Returns a dataset for a single shard of the TFDS TFRecord files."""
        ds = self.builder._tfrecords_reader.read_files([file_instruction], read_config=tfds.ReadConfig(shuffle_seed=seed), shuffle_files=shuffle_files)
        return ds

    def size(self, split):
        """Returns the number of examples in the split."""
        split = self._map_split(split)
        ds_splits = self.info.splits
        dataset_size = ds_splits[split].num_examples
        # Very large datasets have num_examples = 0; default instead to np.inf
        dataset_size = dataset_size if dataset_size > 0 else np.inf
        return dataset_size

def dict_to_tfexample(ex):
    """Convert example dictionary to tf.train.Example proto."""
    feature_dict = {}
    for k, v in ex.items():
        t = tf.constant(v)
        if len(t.shape) == 0:  # pylint:disable=g-explicit-length-test
            v = [v]
        elif len(t.shape) == 1:
            v = list(v)
        else:
            raise ValueError("Unsupported shape (%s) for '%s' value: %s" % (t.shape, k, v))

    if t.dtype == tf.string and len(t.shape) <= 1:
        feature_dict[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(t) for t in v]))
    elif t.dtype in (tf.bool, tf.int32, tf.int64) and len(t.shape) <= 1:
        feature_dict[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    elif t.dtype in (tf.float32, tf.float64) and len(t.shape) <= 1:
        feature_dict[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
    else:
        raise ValueError("Unsupported type (%s) and shape (%s) for '%s' value: %s" % (t.dtype, t.shape, k, v))

    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


# =============================== DataSources ==================================


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
    def get_dataset(self, split: str, shuffle: bool = True, seed: Optional[int] = None, shard_info: Optional[ShardInfo] = None) -> tf.data.Dataset:
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


class TfdsDataSource(DataSource):
    """A `DataSource` that uses TensorFlow Datasets to provide the input data."""

    def __init__(self, tfds_name: str, tfds_data_dir: Optional[str] = None, splits: Optional[Union[Iterable[str], Mapping[str, str]]] = None):
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
            raise ValueError("TFDS name must contain a version number, got: %s" % tfds_name)

        if splits and not isinstance(splits, dict):
            splits = {k: k for k in splits}

        self._tfds_dataset = LazyTfdsLoader(tfds_name, data_dir=tfds_data_dir, split_map=splits if isinstance(splits, dict) else None)

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

    def get_dataset(self, split: str, shuffle: bool = True, seed: Optional[int] = None, shard_info: Optional[ShardInfo] = None) -> tf.data.Dataset:
        return self.tfds_dataset.load(split, shuffle_files=shuffle, seed=seed, shard_info=shard_info)

    def num_input_examples(self, split: str) -> int:
        """Overrides since we can't call `info.splits` until after init."""
        return self.tfds_dataset.size(split)

    def list_shards(self, split: str) -> Sequence[str]:
        return self.tfds_dataset.files(split)


class T5DataSource(DataSource):
    def __init__(self, config_or_file: Any, preprocessor: Optional[Any] = None, splits: Optional[Union[Iterable[str], Mapping[str, str]]] = None):
        self._tfds_dataset = LazyTFDSIOLoader(config_or_file=config_or_file, preprocessor=preprocessor, split_map=splits if isinstance(splits, dict) else None)
        super().__init__(splits=splits or ())

    @property
    def splits(self):
        """Overrides since we can't call `info.splits` until after init."""
        return self._tfds_dataset.info.splits

    @property
    def tfds_dataset(self):
        return self._tfds_dataset

    def get_dataset(self, split: str, shuffle: bool = True, seed: Optional[int] = None, shard_info: Optional[ShardInfo] = None) -> tf.data.Dataset:
        return self.tfds_dataset.load(split, shuffle_files=shuffle, seed=seed, shard_info=shard_info)

    def num_input_examples(self, split: str) -> int:
        """Overrides since we can't call `info.splits` until after init."""
        return self.tfds_dataset.size(split)

    def list_shards(self, split: str) -> Sequence[str]:
        return self.tfds_dataset.files(split)

