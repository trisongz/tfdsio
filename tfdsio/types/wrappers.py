# Wrapper for Funcs
from typing import Union, Dict, Optional, List, Callable, Any, Union
from .base import BuilderConfig, HFBuilderConfig, TFDSDatasetBuilder, HFDatasetBuilder
from tensorflow_datasets.core.as_dataframe import as_dataframe
from tensorflow_datasets.core.dataset_utils import as_numpy
# from .loaders import LazyTFDSIOLoader
from tfdsio.utils import logger


def tfds_dataset(
    config_or_file: Union[BuilderConfig, HFBuilderConfig, Dict[str, Any], str], 
    dataset: Optional[Any] = None,
    preprocessors: Optional[Union[List[Callable], List[Dict[str, Union[Callable, Any]]]]] = None, 
    build: bool = True, 
    from_datasets: bool = False,
    to_tfdataset: bool = True, 
    to_numpy: bool = False, 
    to_df: bool = False, 
    **kwargs
):
    """
    Wrapper Function that quickly creates a TFDSIO Dataset
    """
    if isinstance(config_or_file, (str, dict)):
        config = HFBuilderConfig() if from_datasets else BuilderConfig()
        config.from_auto(config_or_file)
    else: config = config_or_file
    builder = HFDatasetBuilder(dataset = dataset, config = config, **kwargs) if dataset or from_datasets  else TFDSDatasetBuilder(config = config, **kwargs)
    if preprocessors:
        builder.set_preprocessors(preprocessors)
    if build:
        build_args = kwargs.get('build_args', {})
        builder.download_and_prepare(**build_args)
    if to_tfdataset or to_numpy:
        dataset_args = kwargs.get('dataset_args', {})
        ds = builder.as_dataset(**dataset_args)
        if to_numpy:
            return as_numpy(ds)
        if to_df:
            return as_dataframe(ds, builder.info)
        return ds
    return builder


def tfds_sample(dataset, num_samples: int = 5, return_samples: bool = True):
    samples = []
    for ex in as_numpy(dataset.take(num_samples)):
        logger.info(ex)
        samples.append(ex)
    if return_samples:
        return samples


def hf_tfds_dataset(
    config_or_file: Union[HFBuilderConfig, Dict[str, Any], str], 
    dataset: Any,
    preprocessors: Optional[Union[List[Callable], List[Dict[str, Union[Callable, Any]]]]] = None,
    build: bool = True, 
    to_tfdataset: bool = True, 
    to_numpy: bool = False, 
    to_df: bool = False, 
    **kwargs
):
    """
    Wrapper Function that quickly creates a TFDSIO Dataset
    """
    if isinstance(config_or_file, (str, dict)):
        config = HFBuilderConfig()
        config.from_auto(config_or_file)
    else: config = config_or_file
    builder = HFDatasetBuilder(dataset = dataset, config = config, **kwargs)
    if preprocessors:
        builder.set_preprocessors(preprocessors)
    
    if build:
        build_args = kwargs.get('build_args', {})
        builder.download_and_prepare(**build_args)
    if to_tfdataset or to_numpy:
        dataset_args = kwargs.get('dataset_args', {})
        ds = builder.as_dataset(**dataset_args)
        if to_numpy:
            return as_numpy(ds)
        if to_df:
            return as_dataframe(ds, builder.info)
        return ds
    return builder

