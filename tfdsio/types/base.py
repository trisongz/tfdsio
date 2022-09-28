import os
import csv
import json
import termcolor
import dataclasses
import posixpath
import numpy as np
from seqio.vocabularies import Vocabulary
from tensorflow.python.framework.dtypes import (
    int32,
    int64,
    DType
)
#import multiprocessing as mp
import tensorflow_datasets as tfds
from tensorflow_datasets.core import constants
from tensorflow_datasets.core import file_adapters
from tensorflow_datasets.core.utils import file_utils
from tensorflow_datasets.core.utils import Version, version
from tensorflow_datasets.core.download import DownloadManager

from tensorflow.python.platform import gfile
from tensorflow.python.platform.gfile import GFile
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.python.data.ops.readers import TextLineDataset
from tensorflow.python.data.ops.readers import TFRecordDataset
from tensorflow.python.framework.ops import enable_eager_execution, disable_eager_execution

from typing import Union, Dict, Any, Optional, List, ClassVar, Callable
from tfdsio.utils import logger
from tfdsio.io import (
    PipelineIO
)
try:
    from datasets import Dataset, DatasetDict
except ImportError:
    Dataset = object
    DatasetDict = object


_has_logged_ds_info = set()


def log_dataset_info(name, msg):
    global _has_logged_ds_info
    if name in _has_logged_ds_info:
        return
    termcolor.cprint(msg, attrs=["bold"])
    _has_logged_ds_info.add(name)

@dataclasses.dataclass(frozen=True)
class Feature:
    """A container for attributes of output features of data providers."""
    vocabulary: Vocabulary
    add_eos: bool = True
    required: bool = True
    dtype: DType = int32


@dataclasses.dataclass(frozen=True)
class ShardInfo:
    """A container for specifying sharding info."""
    index: int
    num_shards: int

_DataFeatures: Dict[str, Any] = {
    'text': tfds.features.Text(),
    'image': tfds.features.Image(),
    'label': tfds.features.ClassLabel(),
    'audio': tfds.features.Audio(),
    #'sequence': tfds.features.Sequence(tfds.features.Text()),
}

_FilePipelines: Dict[str, Callable] = {
    'text': PipelineIO.get_text_reader,
    'textlines': PipelineIO.get_textline_reader,
    'csv': PipelineIO.get_csv_reader,
    'json': PipelineIO.get_json_reader,
    'jsonlines': PipelineIO.get_jsonlines_reader,
}

_DataSplits: Dict[str, tfds.Split] = {
    'train': tfds.Split.TRAIN,
    'validation': tfds.Split.VALIDATION,
    'test': tfds.Split.TEST,
}

class DataFeatures:

    @classmethod
    def get_feature(cls, name: str):
        return _DataFeatures.get(name, _DataFeatures['text'])

    @classmethod
    def to_features(
        cls,
        features: Dict[str, Any]
    ) -> tfds.features.FeaturesDict:
        """
        Convert a Dictionary of Features to a FeaturesDict
        """
        return tfds.features.FeaturesDict({
            k: cls.get_feature(v) for k, v in features.items()
        })


# class DataSplits:
#     train = tfds.Split.TRAIN
#     validation = tfds.Split.VALIDATION
#     test = tfds.Split.TEST


VersionOrStr = Union[Version, str]


@dataclasses.dataclass
class BuilderConfig:
    name: Optional[str] = None
    classifier: Optional[str] = None
    version: Optional[VersionOrStr] = '1.0.0'
    release_notes: Optional[Dict[str, str]] = None
    supported_versions: List[str] = None
    description: Optional[str] = ''
    
    dataset_urls: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]]] = None
    file_format: Optional[str] = None
    dataset_format: Optional[str] = 'jsonlines'
    features: Optional[Any] = None
    datamap: Optional[Any] = None
    supervised_keys: Optional[Any] = None
    homepage: Optional[str] = ''
    citation: Optional[Any] = None
    metadata: Optional[Any] = None
    redistribution_info: Optional[Any] = None
    
    data_dir: Optional[str] = None
    stream_data: bool = True # Stream data if from GCS / FileLike without Downloading
    preprocessors: Optional[List[Callable]] = None
    preprocessor_kwargs: Optional[Dict[str, Any]] = None
    
    # Will re-process the dataset
    rewrite_existing: bool = False

    def get_datasources(
        self, 
        dataset_urls: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]]]
    ) -> Dict[str, Union[str, List[str]]]:

        if isinstance(dataset_urls, str):
            return {'train': [dataset_urls]}
        if isinstance(dataset_urls, list):
            return {'train': dataset_urls}
        
        dataset_src = {'extra': []}
        for k, v in dataset_urls.items():
            if k in {'train', 'training'}:
                dataset_src['train'] = v
            elif k in {'dev', 'develop', 'development', 'val', 'validation', 'eval', 'evaluation'}:
                dataset_src['validation'] = v
            elif k in {'test', 'gold'}:
                dataset_src['test'] = v
            elif isinstance(v, str):
                dataset_src['extra'].append(v)
            elif isinstance(v, list):
                dataset_src['extra'].extend(v)
            else:
                dataset_src['extra'] = v
        
        for k, v in dataset_src.items():
            if isinstance(v, str):
                dataset_src[k] = [v]
        return dataset_src

    def parse_config(self):
        if self.features: self.features = DataFeatures.to_features(self.features)
        if self.supported_versions is None: self.supported_versions = []
        if self.dataset_urls: self.dataset_urls = self.get_datasources(self.dataset_urls)
        if self.dataset_format: assert self.dataset_format in _FilePipelines, f"Invalid dataset format: {self.dataset_format}. Supported formats: {_FilePipelines.keys()}"

    def from_config(self, config: Dict[str, Any]):
        keys = list(self.__dict__.keys())
        for k,v in config.items():
            if k in keys: self.__dict__[k] = v
        self.parse_config()

    def from_file(self, filename):
        assert gfile.Exists(filename)
        with GFile(filename, 'r') as f:
            config = json.load(f)
        self.from_config(config)
    
    def from_auto(self, config_or_file: Union[Dict[str, Any], str]):
        if isinstance(config_or_file, dict):
            return self.from_config(config_or_file)
        return self.from_file(config_or_file)
    
    @property
    def dataset_info(self):
        keys = ['description', 'features', 'supervised_keys', 'homepage', 'citation', 'metadata', 'redistribution_info']
        return {k: self.__dict__[k] for k in keys}
    
    @property
    def dataset_name(self):
        return f'{self.name}/{self.classifier}:{self.version}'


class DatasetInfo(tfds.core.DatasetInfo):
    @property
    def full_name(self):
        """Full canonical name: (<dataset_name>/<dataset_classifier>/<version>)."""
        names = []
        if self._builder.builder_config:
            names.extend(
                (
                    self._builder.builder_config.name, 
                    self._builder.builder_config.classifier, 
                    str(self.version)
                )
            )
        return posixpath.join(*names)


class TFDSDatasetBuilder(tfds.core.GeneratorBasedBuilder):

    VERSION = None
    RELEASE_NOTES: ClassVar[Dict[str, str]] = {}
    SUPPORTED_VERSIONS = []
    BUILDER_CONFIGS = []
    MANUAL_DOWNLOAD_INSTRUCTIONS = None

    def __init__(
        self, 
        *, 
        config: Union[None, str, BuilderConfig] = None, 
        **kwargs
    ):
        if config.data_dir:
            config.data_dir = os.fspath(config.data_dir)  # Pathlib -> str
        elif os.environ.get('TFDS_DIR', None):
            config.data_dir = os.fspath(os.environ['TFDS_DIR'])
        elif os.environ.get('GCS_DIR', None):
            config.data_dir = os.fspath(os.environ['GCS_DIR'])
        
        file_format = config.file_format or file_adapters.DEFAULT_FILE_FORMAT
        self._original_state = dict(data_dir = config.data_dir, config = config, version = config.version)
        self._builder_config: BuilderConfig = self._create_builder_config(config)
        self.name = self._builder_config.name
        self._version = self._pick_version(config.version)
        self._data_dir_root, self._data_dir = self._build_data_dir(config)
        if config.rewrite_existing and gfile.Exists(self._data_dir):
            self._remove_existing_data_dir()
        
        if gfile.Exists(self._data_dir):
            self.info.read_from_directory(self._data_dir)
            self._log_dataset_info()
        else:
            self.info.initialize_from_bucket()
        
        try:
            self._file_format = file_adapters.FileFormat(file_format)
            self.info.set_file_format(self._file_format)

        except ValueError as e:
            all_values = [f.value for f in file_adapters.FileFormat]
            raise ValueError(f"{file_format} is not a valid format. Valid file formats: {all_values}") from e

    def _remove_existing_data_dir(self):
        """
        Removes a previously created dataset
        if config.rewrite_existing is True
        """
        logger.warning(f"Removing and overwriting dataset {self.name} at {self._data_dir}")
        gfile.DeleteRecursively(self._data_dir)
    
    def _log_dataset_info(self):
        msg = "--------------------------------------------\n"
        msg += f"TFDSIO Dataset: {self.info.full_name}\n"
        msg += "Splits:\n"
        for split, details in self.info.splits.items():
            msg += f" - {split}: {details.num_examples} examples, {details.num_shards} shards\n"
        msg += f"Features: {str(self.info.features)}\n"
        msg += f"Download Size: {self.info.download_size}, Dataset Size: {self.info.dataset_size}\n"
        msg += "--------------------------------------------"
        log_dataset_info(self.info.full_name, msg)
    
    def _log_download_done(self):
        msg = f"Dataset {self.info.full_name} downloaded and prepared to {self._data_dir}.\n"
        termcolor.cprint(msg, attrs=["bold"])
        self._log_dataset_info()
    
    def _log_download_bytes(self):
        termcolor.cprint(
            f"Downloading and Preparing dataset {self.info.full_name} - {self.info.download_size} "
            f"(download: {self.info.download_size}, "
            f"generated: {self.info.dataset_size}, "
            f"total: {self.info.download_size + self.info.dataset_size}) "
            f"\nto {self._data_dir}...",
            attrs=["bold"],
        )
    
    def _split_generators(
        self, 
        dl_manager: DownloadManager
    ):
        dataset_urls = self.builder_config.dataset_urls.copy()
        _extra = dataset_urls.pop('extra', None)
        # sort through the dataset_urls for web urls and local/gcs files 
        # if streaming is enabled
        excluded_paths = {}
        if self.builder_config.stream_data:
            excluded_paths = {
                k: [i for i in v if not i.startswith('http')] for k,v in dataset_urls.items()
            }
            logger.info(f'Excluded paths: {excluded_paths}')
            dataset_urls = {
                k: [i for i in v if i.startswith('http')] for k,v in dataset_urls.items()
            }
            logger.info(f'Included paths to download: {dataset_urls}')
            if _extra:
                excluded_paths['extra'] = [i for i in _extra if not i.startswith('http')]
                _extra = [i for i in _extra if i.startswith('http')]
            
        
        dl_paths = dl_manager.download(dataset_urls)
        msg = f"Downloading Temp Files to:\n"
        for fs, fn in dl_paths.items():
            msg += f" - {fs}: {str(fn)}\n"
        termcolor.cprint(msg)
        if _extra:
            _extra = dl_manager.download(_extra)
        
        dataset_splits = list(dataset_urls.keys())
        # if streamign is enabled, re-merge the excluded paths
        if self.builder_config.stream_data:
            if _extra:
                _extra.extend(excluded_paths.pop('extra', []))
            for k,v in excluded_paths.items():
                dl_paths[k].extend(v)
        
        return [
            tfds.core.SplitGenerator(
                name = _DataSplits[split], 
                gen_kwargs = {
                    "paths": dl_paths[split], 
                    "extra": _extra
                }
            ) for split in dataset_splits
        ]
    
    def map_to_features(self, ex, idx: int, has_preprocessor: bool = False):
        """
        Map the resulting example to the features specified in the dataset_info.json
        """
        if isinstance(ex, dict):
            if self.builder_config.datamap:
                ex = {v: ex.get(k, '') for k,v in self.builder_config.datamap.items()}
            yield idx, ex
        
        elif isinstance(ex, (list, np.array)) and has_preprocessor:
            # The preprocessor could return a list of permutated examples
            for i in ex:
                if self.builder_config.datamap:
                    i = {v: i.get(k, '') for k,v in self.builder_config.datamap.items()}
                yield idx, i
                idx += 1
        
        elif isinstance(ex, (list, np.array)):
            ex = {self.builder_config.features[x]: ex[x] for x in enumerate(ex)}
            yield idx, ex
        
        elif isinstance(ex, (int, float, str, bytes)):
            ex = {self.builder_config.features[0]: ex}
            yield idx, ex


    def _generate_examples(
        self, 
        paths: Union[str, List[str]],
        extra = None, 
        **kwargs
    ):
        idx = 0
        pipeline = _FilePipelines[self.builder_config.dataset_format](paths)
        for ex in pipeline:
            if not self.builder_config.preprocessors:
                for (n, i) in self.map_to_features(ex, idx, has_preprocessor = False):
                    yield n, i
                    idx += 1
            else:
                for preprocessor in self.builder_config.preprocessors:
                    ex = preprocessor(idx = idx, data = ex, extra = extra, **self.get_preprocessor_kwargs(**kwargs))
                    for (n, i) in self.map_to_features(ex, idx, has_preprocessor = True):
                        yield n, i
                        idx += 1

    def _relative_data_dir(self, with_version=True):
        """Relative path of this dataset in data_dir."""
        builder_data_dir = self.name
        builder_config = self._builder_config
        if builder_config:
            if builder_config.classifier:
                builder_data_dir = os.path.join(builder_data_dir, builder_config.classifier)
            else:
                builder_data_dir = os.path.join(builder_config.name)

        return os.path.join(builder_data_dir, str(self._version)) if with_version else builder_data_dir

    def _build_data_dir(self, config):
        """Return the data directory for the current version.
        Args:
        given_data_dir: `Optional[str]`, root `data_dir` passed as
            `__init__` argument.
        Returns:
        data_dir_root: `str`, The root dir containing all datasets, downloads,...
        data_dir: `str`, The version data_dir
            (e.g. `<data_dir_root>/<ds_name>/<config>/<version>`)
        """
        given_data_dir = config.data_dir
        builder_dir = self._relative_data_dir(with_version=False)
        version_dir = self._relative_data_dir(with_version=True)
        default_data_dir = file_utils.get_default_data_dir(given_data_dir=given_data_dir)
        all_data_dirs = file_utils.list_data_dirs(given_data_dir=given_data_dir)
        all_versions = set()
        requested_version_dirs = {}
        for data_dir_root in all_data_dirs:
            full_builder_dir = os.path.join(data_dir_root, builder_dir)
            data_dir_versions = set(version.list_all_versions(full_builder_dir))
            if self.version in data_dir_versions:
                requested_version_dirs[data_dir_root] = os.path.join(data_dir_root, version_dir)
            all_versions.update(data_dir_versions)

        if len(requested_version_dirs) > 1:
            raise ValueError(f"Dataset was found in more than one directory: {requested_version_dirs.values()}. Please resolve the ambiguity by explicitly specifying `data_dir=`.")

        elif len(requested_version_dirs) == 1:  # The dataset is found once
            return next(iter(requested_version_dirs.items()))

        # No dataset found, use default directory
        data_dir = os.path.join(default_data_dir, version_dir)
        if all_versions:
            logger.warning(
                "Found a different version of the requested dataset:\n"
                "%s\n"
                "Using %s instead.",
                "\n".join(str(v) for v in sorted(all_versions)),
                data_dir
            )
        return default_data_dir, data_dir

    #def set_preprocessor(self, function):
    #    self._proc = function
    
    def set_preprocessors(self, functions: Union[Callable, List[Callable]]):
        if not self.builder_config.preprocessors:
            self.builder_config.preprocessors = []
        if isinstance(functions, list):
            self.builder_config.preprocessors.extend(functions)
        else:
            self.builder_config.preprocessors.append(functions)


    def get_preprocessor_kwargs(self, **kwargs):
        if not kwargs: kwargs = {}
        if self.builder_config.preprocessor_kwargs:
            kwargs.update(self.builder_config.preprocessor_kwargs)
        return kwargs
    
    # @property
    # def has_preprocessor(self):
    #     return bool('_proc' in self.__dict__)

    # def preprocessor(self, *args, **kwargs):
    #     # sourcery skip: dict-assign-update-to-union
    #     if self.has_preprocessor:
    #         if self.builder_config.process_args:
    #             kwargs.update(self.builder_config.process_args)
    #         return self._proc(*args, **kwargs)
    #     return None

    @property
    def libname(self):
        return 'tfdsio'

    def _info(self) -> DatasetInfo:
        return DatasetInfo(builder = self, **self.builder_config.dataset_info)


class HFBuilderConfig(BuilderConfig):
    """
    Adds on additional params for HF Datasets
    """
    # For Loading the Dataset
    ds_name: Optional[str] = None
    ds_load_kwargs: Optional[Dict[str, Any]] = None

    # For Preprocessing the Dataset
    preprocessors: Optional[List[Dict[str, Union[Callable, Any]]]] = None



class HFDatasetBuilder(TFDSDatasetBuilder):

    VERSION = None
    RELEASE_NOTES: ClassVar[Dict[str, str]] = {}
    SUPPORTED_VERSIONS = []
    BUILDER_CONFIGS = []
    MANUAL_DOWNLOAD_INSTRUCTIONS = None

    def __init__(
        self, 
        *, 
        dataset: Union[Dataset, DatasetDict] = None,
        config: Union[None, str, HFBuilderConfig] = None, 
        **kwargs
    ):
        if config.data_dir:
            config.data_dir = os.fspath(config.data_dir)  # Pathlib -> str
        elif os.environ.get('TFDS_DIR', None):
            config.data_dir = os.fspath(os.environ['TFDS_DIR'])
        elif os.environ.get('GCS_DIR', None):
            config.data_dir = os.fspath(os.environ['GCS_DIR'])
        
        self.dataset = dataset
        file_format = config.file_format or file_adapters.DEFAULT_FILE_FORMAT
        self._original_state = dict(data_dir = config.data_dir, config = config, version = config.version)
        self._builder_config: HFBuilderConfig = self._create_builder_config(config)
        self.__dict__['name'] = self._builder_config.name
        self._version = self._pick_version(config.version)
        self._data_dir_root, self._data_dir = self._build_data_dir(config)
        
        if config.rewrite_existing and gfile.Exists(self._data_dir):
            self._remove_existing_data_dir()
        
        if gfile.Exists(self._data_dir):
            self.info.read_from_directory(self._data_dir)
            self._log_dataset_info()
        else:
            self.info.initialize_from_bucket()
        
        try:
            self._file_format = file_adapters.FileFormat(file_format)
            self.info.set_file_format(self._file_format)

        except ValueError as e:
            all_values = [f.value for f in file_adapters.FileFormat]
            raise ValueError(f"{file_format} is not a valid format. Valid file formats: {all_values}") from e

    def _load_dataset(self, **kwargs):
        if not self.dataset:
            if not self._builder_config.ds_load_kwargs:
                self._builder_config.ds_load_kwargs = {}
            if kwargs:
                self._builder_config.ds_load_kwargs.update(kwargs)
            ds_name = self._builder_config.ds_name or self.name
            logger.info(f'Loading Dataset: {ds_name} with params {self._builder_config.ds_load_kwargs}')
            import datasets
            self.dataset = datasets.load_dataset(ds_name, **self._builder_config.ds_load_kwargs)
        return self.dataset

    def map_split_name(self, split: str):
        """
        Remaps Split Names to the ones used in the dataset
        """
        #if 'train' in split:
        #    return _DataSplits['train']
        if split in {'dev', 'develop', 'development', 'val', 'validation', 'eval', 'evaluation'}:
            return _DataSplits['validation']
        if split in {'test', 'testing'}:
            return _DataSplits['test']
        # assume train
        return _DataSplits['train']


    def _split_generators(
        self, 
        dl_manager: DownloadManager
    ):
        dataset = self._load_dataset()
        if isinstance(dataset, Dataset):
            dataset = DatasetDict({'train': dataset})
        return [
            tfds.core.SplitGenerator(
                name = self.map_split_name(split), 
                gen_kwargs = {
                    'dataset': dataset[split],
                }
            ) for split in dataset
        ]
    

    def set_preprocessors(self, functions: Union[Callable, List[Dict[str, Union[Callable, Any]]]]):
        if not self.builder_config.preprocessors:
            self.builder_config.preprocessors = []
        if isinstance(functions, list):
            self.builder_config.preprocessors.extend(functions)
        else:
            self.builder_config.preprocessors.append(functions)


    def _generate_examples(
        self, 
        dataset: Dataset,
        **kwargs
    ):
        idx = 0
        if self.builder_config.preprocessors:
            # Map the Preprocessors to the Dataset
            for preprocessor in self.builder_config.preprocessors:
                if isinstance(preprocessor, dict):
                    dataset = dataset.map(**preprocessor)
                else:
                    dataset = dataset.map(preprocessor)

        for ex in dataset:
            for (n, i) in self.map_to_features(ex, idx, has_preprocessor = False):
                yield n, i
                idx += 1



__all__ = [
    'Feature',
    'ShardInfo',
    'Vocabulary',
    'GFile',
    'int32',
    'int64',
    'DType',
    'logger',
    'tfds',
    'DataFeatures',
    #'DataSplits',
    'VersionOrStr',
    'BuilderConfig',
    'HFBuilderConfig',
    'TFDSDatasetBuilder',
    'HFDatasetBuilder',
]