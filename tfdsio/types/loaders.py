import numpy as np

from .base import BuilderConfig, TFDSDatasetBuilder, ShardInfo
from typing import List, Callable, Optional, Union, Dict
from tfdsio.utils import logger
from tfdsio.config import TFDSIOConfig

from tensorflow_datasets.core.load import load as tfds_load
from tensorflow_datasets.core.load import builder as tfds_builder

from tensorflow.python.distribute.distribute_lib import InputContext
from tensorflow_datasets.core.utils.read_config import ReadConfig


class LazyTFDSLoader(object):
    """Wrapper for TFDS datasets with memoization and additional functionality.
    Lazily loads info from TFDS and provides memoization to avoid expensive hidden
    file operations. Also provides additional utility methods.
    """

    _MEMOIZED_BUILDERS = {}

    def __init__(
        self, 
        name, 
        data_dir=None, 
        split_map=None
    ):
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
        if TFDSIOConfig.tfds_data_dir_override:
            if self._data_dir:
                logger.warning(f"Overriding TFDS data directory '{self._data_dir}' with '{TFDSIOConfig.tfds_data_dir_override}' for dataset '{self.name}'.")
            return TFDSIOConfig.tfds_data_dir_override
        return self._data_dir

    @property
    def builder(self):
        builder_key = (self.name, self.data_dir)
        if builder_key not in LazyTFDSLoader._MEMOIZED_BUILDERS:
            LazyTFDSLoader._MEMOIZED_BUILDERS[builder_key] = tfds_builder(self.name, data_dir=self.data_dir)
        return LazyTFDSLoader._MEMOIZED_BUILDERS[builder_key]

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
            logger.error("No TFRecord files found for dataset: %s", self.name)
        return files

    def load(self, split, shuffle_files: bool = True, seed: int = None, shard_info: ShardInfo = None, download: bool = True, try_gcs: bool = True):
        """Returns a tf.data.Dataset for the given split."""
        split = self._map_split(split)
        input_context = InputContext(
            num_input_pipelines = shard_info.num_shards, 
            input_pipeline_id = shard_info.index
        ) if shard_info else None
        
        return tfds_load(self._name,
            split = split,
            data_dir = self.data_dir,
            shuffle_files = shuffle_files,
            download = download,
            try_gcs = try_gcs,
            read_config = ReadConfig(
                shuffle_seed = seed, 
                skip_prefetch = True, 
                input_context = input_context
            )
        )

    def load_shard(self, file_instruction, shuffle_files: bool = False, seed: int = None):
        """Returns a dataset for a single shard of the TFDS TFRecord files."""
        return self.builder._tfrecords_reader.read_files(
            [file_instruction],
            read_config = ReadConfig(shuffle_seed = seed),
            shuffle_files = shuffle_files
        )

    def size(self, split):
        """Returns the number of examples in the split."""
        split = self._map_split(split)
        ds_splits = self.info.splits
        dataset_size = ds_splits[split].num_examples
        # Very large datasets have num_examples = 0; default instead to np.inf
        dataset_size = dataset_size if dataset_size > 0 else np.inf
        return dataset_size

class LazyTFDSIOLoader(object):
    """Wrapper for TFDS datasets with memoization and additional functionality.
    Lazily loads info from TFDS and provides memoization to avoid expensive hidden
    file operations. Also provides additional utility methods.
    """

    _MEMOIZED_BUILDERS = {}

    def __init__(
        self, 
        config_or_file, 
        preprocessors: Optional[Union[Callable, List[Callable]]] = None, 
        split_map: Optional[Dict[str, str]] = None
    ):
        """LazyTfdsLoader constructor.
        Args:
        name: str, the name of the TFDS dataset.
        data_dir: str (optional), directory to read/write TFDS data.
        split_map: dict (optional), mapping from canonical splits
            (e.g., 'validation') to TFDS splits or slices
            (e.g., 'train[':1%']).
        """
        self._config = BuilderConfig()
        self._config.from_auto(config_or_file)
        self._preprocessors = preprocessors

        self._name = self._config.dataset_name
        self._data_dir = self._config.data_dir
        self._split_map = split_map

    @property
    def name(self):
        return self._name

    @property
    def data_dir(self):
        if TFDSIOConfig.tfds_data_dir_override:
            if self._data_dir:
                logger.warning(f"Overriding TFDS data directory '{self._data_dir}' with '{TFDSIOConfig.tfds_data_dir_override}' for dataset '{self.name}'.")
            return TFDSIOConfig.tfds_data_dir_override
        return self._data_dir

    @property
    def builder(self):
        builder_key = (self.name, self.data_dir)
        if builder_key not in LazyTFDSIOLoader._MEMOIZED_BUILDERS:
            LazyTFDSIOLoader._MEMOIZED_BUILDERS[builder_key] = TFDSDatasetBuilder(config=self._config)
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
            raise ValueError(f"Dataset '{self.name}' has multiple configs.")

        split_info = self.builder.info.splits[split]
        files = split_info.file_instructions
        if not files:
            logger.error(f"No TFRecord files found for dataset: {self.name}")
        return files

    def load(self, split, shuffle_files: bool, seed: int = None, shard_info: ShardInfo = None):
        """Returns a tf.data.Dataset for the given split."""
        split = self._map_split(split)
        input_context = InputContext(
            num_input_pipelines = shard_info.num_shards, 
            input_pipeline_id = shard_info.index
            ) if shard_info else None
        
        builder = TFDSDatasetBuilder(config=self._config)
        if self._preprocessors:
            builder.set_preprocessors(self._preprocessors)
        builder.download_and_prepare()
        return builder.as_dataset(
            split = split, 
            shuffle_files = shuffle_files, 
            read_config = ReadConfig(
                shuffle_seed = seed, 
                skip_prefetch = True, 
                input_context = input_context
            )
        )

    def load_shard(self, file_instruction, shuffle_files: bool = False, seed: int = None):
        """Returns a dataset for a single shard of the TFDS TFRecord files."""
        return self.builder._tfrecords_reader.read_files(
            [file_instruction],
            read_config = ReadConfig(shuffle_seed = seed),
            shuffle_files = shuffle_files
        )


    def size(self, split):
        """Returns the number of examples in the split."""
        split = self._map_split(split)
        ds_splits = self.info.splits
        dataset_size = ds_splits[split].num_examples
        # Very large datasets have num_examples = 0; default instead to np.inf
        dataset_size = dataset_size if dataset_size > 0 else np.inf
        return dataset_size


class LazyTFDSIOLoader(object):
    """Wrapper for TFDS datasets with memoization and additional functionality.
    Lazily loads info from TFDS and provides memoization to avoid expensive hidden
    file operations. Also provides additional utility methods.
    """

    _MEMOIZED_BUILDERS = {}

    def __init__(
        self, 
        config_or_file, 
        preprocessors: Optional[Union[Callable, List[Callable]]] = None, 
        split_map: Optional[Dict[str, str]] = None
    ):
        """LazyTfdsLoader constructor.
        Args:
        name: str, the name of the TFDS dataset.
        data_dir: str (optional), directory to read/write TFDS data.
        split_map: dict (optional), mapping from canonical splits
            (e.g., 'validation') to TFDS splits or slices
            (e.g., 'train[':1%']).
        """
        self._config = BuilderConfig()
        self._config.from_auto(config_or_file)
        self._preprocessors = preprocessors

        self._name = self._config.dataset_name
        self._data_dir = self._config.data_dir
        self._split_map = split_map

    @property
    def name(self):
        return self._name

    @property
    def data_dir(self):
        if TFDSIOConfig.tfds_data_dir_override:
            if self._data_dir:
                logger.warning(f"Overriding TFDS data directory '{self._data_dir}' with '{TFDSIOConfig.tfds_data_dir_override}' for dataset '{self.name}'.")
            return TFDSIOConfig.tfds_data_dir_override
        return self._data_dir

    @property
    def builder(self):
        builder_key = (self.name, self.data_dir)
        if builder_key not in LazyTFDSIOLoader._MEMOIZED_BUILDERS:
            LazyTFDSIOLoader._MEMOIZED_BUILDERS[builder_key] = TFDSDatasetBuilder(config=self._config)
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
            raise ValueError(f"Dataset '{self.name}' has multiple configs.")

        split_info = self.builder.info.splits[split]
        files = split_info.file_instructions
        if not files:
            logger.error(f"No TFRecord files found for dataset: {self.name}")
        return files

    def load(self, split, shuffle_files, seed=None, shard_info=None):
        """Returns a tf.data.Dataset for the given split."""
        split = self._map_split(split)
        input_context = InputContext(
            num_input_pipelines = shard_info.num_shards, 
            input_pipeline_id = shard_info.index
            ) if shard_info else None
        
        builder = TFDSDatasetBuilder(config=self._config)
        if self._preprocessors:
            builder.set_preprocessors(self._preprocessors)
        builder.download_and_prepare()
        return builder.as_dataset(
            split = split, 
            shuffle_files = shuffle_files, 
            read_config = ReadConfig(
                shuffle_seed = seed, 
                skip_prefetch = True, 
                input_context = input_context
            )
        )

    def load_shard(self, file_instruction, shuffle_files: bool = False, seed: int = None):
        """Returns a dataset for a single shard of the TFDS TFRecord files."""
        return self.builder._tfrecords_reader.read_files(
            [file_instruction],
            read_config = ReadConfig(shuffle_seed = seed),
            shuffle_files = shuffle_files
        )


    def size(self, split):
        """Returns the number of examples in the split."""
        split = self._map_split(split)
        ds_splits = self.info.splits
        dataset_size = ds_splits[split].num_examples
        # Very large datasets have num_examples = 0; default instead to np.inf
        dataset_size = dataset_size if dataset_size > 0 else np.inf
        return dataset_size
