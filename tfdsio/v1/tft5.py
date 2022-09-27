from .adapter import (
    T5DataSource,
    load_sp_model,
    create_sp_model,
    set_vocab,
    default_output_features,
    default_preprocessor,
    default_metrics,
    add_to_registry,
    remove_from_registry,
    add_to_mixture,
    remove_from_mixture,
    get_registry,
    get_mixtures,
    sample_registry,
    sample_mixture
)

from .adapter import t5_preprocessors as preprocessors
from fileio import TFDSIODataset as DatasetConfig
