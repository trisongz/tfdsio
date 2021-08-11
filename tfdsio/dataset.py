import os
import sys
import numpy as np
import tensorflow as tf
import posixpath
import termcolor
from dataclasses import dataclass, asdict


from absl import logging
import tensorflow_datasets as tfds
from tensorflow_datasets.core import constants
from tensorflow_datasets.core import file_adapters
from tensorflow_datasets.core.utils import Version, PathLike, version


from tfdsio.utils import File, _tfdsio_dir, _tfdsio_imports
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Union

_feats = {
    'text': tfds.features.Text(),
    'audio': tfds.features.Audio(),
    'image': tfds.features.Image(),
}

_splits = {
    'train': tfds.Split.TRAIN,
    'validation': tfds.Split.VALIDATION,
    'test': tfds.Split.TEST
}

_parsers = {
    'jsonlines': File.tfjl,
    'text': File.tftl,
    'csv': File.tfcsv,
}


VersionOrStr = Union[Version, str]
_has_logged_ds_info = set()

def log_dataset_info(name, msg):
    global _has_logged_ds_info
    if name in _has_logged_ds_info:
        return
    termcolor.cprint(msg, attrs=["bold"])
    _has_logged_ds_info.add(name)
    

def create_features(features):
    return tfds.features.FeaturesDict({key: _feats[val] for key, val in features.items()})


@dataclass
class BuilderConfig:
    name: Optional[str] = None
    classifier: Optional[str] = None
    version: Optional[VersionOrStr] = '1.0.0'
    release_notes: Optional[Dict[str, str]] = None
    supported_versions: List[str] = None
    description: Optional[str] = ''
    dataset_urls: Optional[Any] = None
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
    process_args: Optional[Any] = None

class TFDSIOConfig(BuilderConfig):
    def parse_config(self):
        for key, val in self.__dict__.items():
            if key == 'features' and val:
                self.__dict__[key] = create_features(val)
            elif key == 'supported_versions' and not val:
                self.__dict__[key] = []
            elif key == 'dataset_format':
                assert val in list(_parsers.keys())
            elif key == 'dataset_urls' and val:
                _dsrc = {'extra': []}
                if isinstance(val, str):
                    _dsrc['train'] = val
                    
                elif isinstance(val, dict):
                    for k,v in val.items():
                        if k in ['train', 'training']:
                            _dsrc['train'] = v
                        elif k in ['dev', 'develop', 'development', 'val', 'validation']:
                            _dsrc['validation'] = v
                        elif k in ['test', 'gold']:
                            _dsrc['test'] = v
                        else:
                            if isinstance(v, str):
                                _dsrc['extra'].append(v)
                            elif isinstance(v, list):
                                _dsrc['extra'].extend(v)
                            else:
                                _dsrc['extra'] = v

                self.__dict__[key] = _dsrc.copy()
            
            else:
                self.__dict__[key] = val


    def from_config(self, config):
        for k,v in config.items():
            if k in self.__dict__:
                self.__dict__[k] = v
        self.parse_config()

    def from_file(self, filename):
        assert File.exists(filename)
        config = File.jsonload(filename)
        self.from_config(config)
    
    def from_auto(self, config_or_file):
        if isinstance(config_or_file, dict):
            return self.from_config(config_or_file)
        return self.from_file(config_or_file)
    
    @property
    def dataset_info(self):
        keys = ['description', 'features', 'supervised_keys', 'homepage', 'citation', 'metadata', 'redistribution_info']
        _info = {k: self.__dict__[k] for k in keys}
        return _info
    
    @property
    def dataset_name(self):
        return f'{self.name}/{self.classifier}:{self.version}'

class TFDSIOInfo(tfds.core.DatasetInfo):
    @property
    def full_name(self):
        """Full canonical name: (<dataset_name>/<dataset_classifier>/<version>)."""
        names = []
        if self._builder.builder_config:
            names.append(self._builder.builder_config.name)
            names.append(self._builder.builder_config.classifier)
            names.append(str(self.version))
        return posixpath.join(*names)

class TFDSIOCorpus(tfds.core.GeneratorBasedBuilder):
    VERSION = None
    RELEASE_NOTES: ClassVar[Dict[str, str]] = {}
    SUPPORTED_VERSIONS = []
    BUILDER_CONFIGS = []
    MANUAL_DOWNLOAD_INSTRUCTIONS = None
    def __init__(self, *, config: Union[None, str, BuilderConfig] = None, **kwargs: Any):
        if config.data_dir:
            config.data_dir = os.fspath(config.data_dir)  # Pathlib -> str
        elif os.environ.get('TFDS_DIR', None):
            config.data_dir = os.fspath(os.environ['TFDS_DIR'])
        elif os.environ.get('GCS_DIR', None):
            config.data_dir = os.fspath(os.environ['GCS_DIR'])

        file_format = config.file_format or file_adapters.DEFAULT_FILE_FORMAT
        #kwargs.get('file_format', file_adapters.DEFAULT_FILE_FORMAT)
        self._original_state = dict(data_dir=config.data_dir, config=config, version=config.version)
        self._builder_config = self._create_builder_config(config)
        self.__dict__['name'] = self._builder_config.name
        self._version = self._pick_version(config.version)
        self._data_dir_root, self._data_dir = self._build_data_dir(config)
        if tf.io.gfile.exists(self._data_dir):
            self.info.read_from_directory(self._data_dir)
            self._log_dataset_info()
        else:
            self.info.initialize_from_bucket()
        
        try:
            self._file_format = file_adapters.FileFormat(file_format)
            self.info.set_file_format(self._file_format)

        except ValueError:
            all_values = [f.value for f in file_adapters.FileFormat]
            raise ValueError(f"{file_format} is not a valid format. Valid file formats: {all_values}")

    
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
    
    def _split_generators(self, dl_manager):
        dataset_urls = self.builder_config.dataset_urls.copy()
        _extra = dataset_urls.pop('extra', None)
        dl_paths = dl_manager.download(dataset_urls)
        msg = f"Downloading Temp Files to:\n"
        for fs, fn in dl_paths.items():
            msg += f" - {fs}: {str(fn)}\n"
        termcolor.cprint(msg)
        if _extra:
            dl_manager.download(_extra)
        dataset_splits = list(dataset_urls.keys())
        file_splits = [tfds.core.SplitGenerator(name=_splits[split], gen_kwargs={"paths": dl_paths[split], "extra": _extra}) for split in dataset_splits]
        return file_splits
    
    def _generate_examples(self, paths, extra=None, **kwargs):
        pipeline = _parsers[self.builder_config.dataset_format](paths)
        _do_preprocess = self.has_preprocessor
        idx = 0
        for n, data in enumerate(pipeline):
            if _do_preprocess:
                data = self.preprocessor(idx=idx, data=data, extra=extra, **kwargs)
            if data:
                if isinstance(data, dict):
                    if self.builder_config.datamap:
                        data = {v: data.get(k, '') for k,v in self.builder_config.datamap.items()}
                    yield idx, data
                    idx += 1
                
                elif isinstance(data, (list, np.array)) and _do_preprocess:
                    for ex in data:
                        if self.builder_config.datamap:
                            ex = {v: ex.get(k, '') for k,v in self.builder_config.datamap.items()}
                        yield idx, ex
                        idx += 1

                elif isinstance(data, (list, np.array)):
                    data = {self.builder_config.features[x]: data[x] for x in enumerate(data)}
                    yield idx, data
                    idx += 1
                
                elif isinstance(data, (int, float, str, bytes)):
                    data = {self.builder_config.features[0]: data}
                    yield idx, data
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
        
        if not with_version:
            return builder_data_dir
        version_data_dir = os.path.join(builder_data_dir, str(self._version))
        return version_data_dir

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
        default_data_dir = constants.get_default_data_dir(given_data_dir=given_data_dir)
        all_data_dirs = constants.list_data_dirs(given_data_dir=given_data_dir)
        all_versions = set()
        requested_version_dirs = {}
        for data_dir_root in all_data_dirs:
            full_builder_dir = os.path.join(data_dir_root, builder_dir)
            data_dir_versions = set(version.list_all_versions(full_builder_dir))
            if self.version in data_dir_versions:
                requested_version_dirs[data_dir_root] = os.path.join(data_dir_root, version_dir)
            all_versions.update(data_dir_versions)

        if len(requested_version_dirs) > 1:
            raise ValueError("Dataset was found in more than one directory: {}. Please resolve the ambiguity by explicitly specifying `data_dir=`.".format(requested_version_dirs.values()))
        elif len(requested_version_dirs) == 1:  # The dataset is found once
            return next(iter(requested_version_dirs.items()))

        # No dataset found, use default directory
        data_dir = os.path.join(default_data_dir, version_dir)
        if all_versions:
            logging.warning(
                "Found a different version of the requested dataset:\n"
                "%s\n"
                "Using %s instead.",
                "\n".join(str(v) for v in sorted(all_versions)),
                data_dir
            )
        return default_data_dir, data_dir

    def set_preprocessor(self, function):
        self._proc = function
    
    @property
    def has_preprocessor(self):
        return bool('_proc' in self.__dict__)

    def preprocessor(self, *args, **kwargs):
        if self.has_preprocessor:
            if self.builder_config.process_args:
                kwargs.update(self.builder_config.process_args)
            return self._proc(*args, **kwargs)
        return None

    @property
    def libname(self):
        return 'tfdsio'

    def _info(self) -> tfds.core.DatasetInfo:
        return TFDSIOInfo(builder=self, **self.builder_config.dataset_info)


def tfds_dataset(config_or_file, preprocessor=None, build=True, as_tfdataset=True, as_numpy=False, as_df=False, **kwargs):
    config = TFDSIOConfig()
    config.from_auto(config_or_file)
    builder = TFDSIOCorpus(config=config, **kwargs)
    #builder = TFDSIOCorpus(data_dir=config.data_dir, config=config, version=config.version, **kwargs)
    if preprocessor:
        builder.set_preprocessor(preprocessor)
    if build:
        build_args = kwargs.get('build_args', {})
        builder.download_and_prepare(**build_args)
    if as_tfdataset or as_numpy:
        dataset_args = kwargs.get('dataset_args', {})
        ds = builder.as_dataset(**dataset_args)
        if as_numpy:
            return tfds.as_numpy(ds)
        if as_df:
            return tfds.as_dataframe(ds, builder.info)
        return ds
    return builder

def tfds_remove(config_or_file, prompt=True):
    config = TFDSIOConfig()
    config.from_auto(config_or_file)
    builder = TFDSIOCorpus(config=config)
    del_dir = os.fspath(builder.__cached_info.data_path)
    full_name = builder.__cached_info.full_name
    if prompt:
        del_conf = input(f'Deleting TFDSIO {full_name} at {del_dir} - Confirm [y/n]')
        if del_conf.lower() != 'y':
            logging.info('Exiting')
            return
    File.rmdir(del_dir)
    logging.info(f'Removed: {del_dir}')


def tfds_sample(dataset, num_samples=5, return_samples=True):
    samples = []
    for ex in tfds.as_numpy(dataset.take(num_samples)):
        logging.info(ex)
        samples.append(ex)
    if return_samples:
        return samples
