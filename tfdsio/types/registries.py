import abc
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, Union
from tensorflow.python.data import Dataset
from .base import Feature, ShardInfo

class DatasetProviderBase(metaclass=abc.ABCMeta):
    """Abstract base for classes that provide a tf.data.Dataset."""

    @abc.abstractproperty
    def output_features(self) -> Mapping[str, Feature]:
        raise NotImplementedError

    @abc.abstractproperty
    def splits(self) -> Sequence[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_dataset(
        self, 
        sequence_length: int, 
        split: str, 
        use_cached: bool = False, 
        shuffle: bool = True,
        seed: Optional[int] = None, 
        shard_info: Optional[ShardInfo] = None, 
        num_epochs: int = 1
    ) -> Dataset:
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
            raise ValueError(f"Attempting to register duplicate provider: {name}")
        if not isinstance(provider, cls._PROVIDER_TYPE):
            raise ValueError(f"Attempting to register a class not of an invalid type. Expecting instance of {cls._PROVIDER_TYPE}, got {type(provider).__name__}")
        cls._REGISTRY[name] = provider

    @classmethod
    def add(cls, name: str, provider_cls, *provider_args, **provider_kwargs):
        """Instantiates and adds provider to the registry."""
        if not issubclass(provider_cls, cls._PROVIDER_TYPE):
            raise ValueError(f"Attempting to register a class not of an invalid type. Expecting instance of {cls._PROVIDER_TYPE}, got {provider_cls}")

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
            raise ValueError(f"Provider name not registered: {name}")
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
