import seqio
import functools
import seqio.utils
import t5.data.preprocessors
import t5.evaluation.metrics
from typing import List, Dict, Any, Optional

# Global Configuration Options


DEFAULT_PREPROCESSORS = [
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    seqio.preprocessors.append_eos_after_trim,
]

DEFAULT_METRICS = [t5.evaluation.metrics.bleu, t5.evaluation.metrics.rouge]

PREPROCESSOR_FUNCS = {
    'unsupervised': t5.data.preprocessors.unsupervised,
    'random_spans_noise_mask': t5.data.preprocessors.random_spans_noise_mask,
    'span_corruption': t5.data.preprocessors.span_corruption,
    'iid_denoising': t5.data.preprocessors.iid_denoising,
    'lm': t5.data.preprocessors.lm,
    'prefix_lm': t5.data.preprocessors.prefix_lm,
    'random_split_text': t5.data.preprocessors.random_split_text,
}

LM_PREPROCESSORS = {
    'unsupervised_lm': [
        functools.partial(
            t5.data.preprocessors.rekey, 
            key_map={"inputs": None, "targets": "targets"}
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.unsupervised,
        seqio.preprocessors.append_eos_after_trim
    ],
    'span_corruption_lm': [
        functools.partial(
            t5.data.preprocessors.rekey, 
            key_map={"inputs": None, "targets": "targets"}
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim
    ],
    'iid_denoising_lm': [
        functools.partial(
            t5.data.preprocessors.rekey, 
            key_map={"inputs": None, "targets": "targets"}
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.iid_denoising,
        seqio.preprocessors.append_eos_after_trim
    ],
    'prefix_lm': [
        functools.partial(
            t5.data.preprocessors.rekey, 
            key_map={"inputs": None, "targets": "targets"}
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.prefix_lm,
        seqio.preprocessors.append_eos_after_trim
    ]
}


PRETRAIN_PREPROCESSORS = {
    'unsupervised': [
        functools.partial(
            t5.data.preprocessors.rekey, 
            key_map={"inputs": "inputs", "targets": "targets"}
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.unsupervised,
        seqio.preprocessors.append_eos_after_trim
    ],
    'span_corruption': [
        functools.partial(
            t5.data.preprocessors.rekey, 
            key_map={"inputs": "inputs", "targets": "targets"}
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim
    ],
    'iid_denoising': [
        functools.partial(
            t5.data.preprocessors.rekey, 
            key_map={"inputs": "inputs", "targets": "targets"}
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.iid_denoising,
        seqio.preprocessors.append_eos_after_trim
    ]
}


class TFDSIOConfig:
    tfds_data_dir_override: str = seqio.utils._TFDS_DATA_DIR_OVERRIDE
    global_cache_directories: List[str] = seqio.utils._GLOBAL_CACHE_DIRECTORIES

    default_preprocessors: List[Any] = DEFAULT_PREPROCESSORS
    lm_preprocessors: Dict[str, List[Any]] = LM_PREPROCESSORS
    pretrain_preprocessors: Dict[str, List[Any]] = PRETRAIN_PREPROCESSORS


    # Override TFDS Configuration
    @classmethod
    def set_tfds_data_dir_override(cls, tfds_data_dir_override: str):
        seqio.utils.set_tfds_data_dir_override(tfds_data_dir_override)

    @classmethod
    def set_global_cache_dirs(cls, global_cache_directories: List[str]):
        seqio.utils.set_global_cache_dirs(global_cache_directories)

    @classmethod
    def add_global_cache_dirs(cls, global_cache_directories: List[str]):
        seqio.utils.add_global_cache_dirs(global_cache_directories)
    
    @classmethod
    def create_lm_preprocessors(
        cls, 
        rekey_map: Dict[str, str] = {'inputs': None, 'targets': 'targets'}, 
        exclude_funcs: List[str] = None,
        include_funcs: List[str] = None,
    ) -> List[Any]:  # sourcery skip: default-mutable-arg
        """
        Create a list of preprocessors for language modeling tasks.
        If no exclude_funcs or include_funcs are provided, the default tasks are used.        
        """
        tasks = []
        for task_name in {'unsupervised', 'span_corruption', 'iid_denoising', 'prefix_lm'}:
            if include_funcs is not None and task_name not in include_funcs:
                continue
            if exclude_funcs is not None and task_name in exclude_funcs:
                continue
            task = []
            if rekey_map is not None:
                task.append(functools.partial(t5.data.preprocessors.rekey, key_map = rekey_map))
            task.extend(
                (
                    seqio.preprocessors.tokenize, 
                    seqio.CacheDatasetPlaceholder(), 
                    cls.lm_preprocessors[task_name], 
                    seqio.preprocessors.append_eos_after_trim
                )
            )
            tasks.append(task)
        return tasks




