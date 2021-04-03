
# Adapted from https://github.com/rwightman/pytorch-image-models/blob/benchmark-fixes-vit_hybrids/timm/data/parsers/parser_tfds.py

import os
import io
import math
import numpy as np
import tensorflow as tf
from tfdsio.dataset import tfds
from tfdsio.dataset import TFDSIOConfig, TFDSIOCorpus

try:
    import torch
    import torch.distributed as dist
    _torch_installed = True
except ImportError as e:
    _torch_installed = False

try:
    import transformers
    from transformers import AutoTokenizer
    _transformers_installed = True
except ImportError as e:
    _transformers_installed = False

try:
    from PIL import Image
    _pil_installed = True
except ImportError as e:
    _pil_installed = False


def assert_torch():
    if _torch_installed:
        return
    print("Please install torch package `pip install torch`.")
    exit(1)

def assert_transformers():
    if _transformers_installed:
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        return
    print("Please install transformers package `pip install transformers`.")
    exit(1)


def assert_pil():
    if _pil_installed:
        return
    print("Please install PIL package `pip install PIL`.")
    exit(1)


def even_split_indices(split, n, num_samples):
    partitions = [round(i * num_samples / n) for i in range(n + 1)]
    return [f"{split}[{partitions[i]}:{partitions[i+1]}]" for i in range(n)]


class TorchTFDSIO(object):
    """ Wrap Tensorflow Datasets for use in PyTorch
    There several things to be aware of:
      * To prevent excessive samples being dropped per epoch w/ distributed training or multiplicity of
         dataloader workers, the train iterator wraps to avoid returning partial batches that trigger drop_last
         https://github.com/pytorch/pytorch/issues/33413
      * With PyTorch IterableDatasets, each worker in each replica operates in isolation, the final batch
        from each worker could be a different size. For training this is worked around by option above, for
        validation extra samples are inserted iff distributed mode is enabled so that the batches being reduced
        across replicas are of same size. This will slightly alter the results, distributed validation will not be
        100% correct. This is similar to common handling in DistributedSampler for normal Datasets but a bit worse
        since there are up to N * J extra samples with IterableDatasets.
      * The sharding (splitting of dataset into TFRecord) files imposes limitations on the number of
        replicas and dataloader workers you can use. For really small datasets that only contain a few shards
        you may have to train non-distributed w/ 1-2 dataloader workers. This is likely not a huge concern as the
        benefit of distributed training or fast dataloading should be much less for small datasets.
      * This wrapper is currently configured to return individual, decompressed image samples from the TFDS
        dataset. The augmentation (transforms) and batching is still done in PyTorch. It would be possible
        to specify TF augmentation fn and return augmented batches w/ some modifications to other downstream
        components.
    """
    def __init__(self, config_or_file, preprocessor=None, tokenizer=None, split='train', data_type='text', shuffle=False, is_training=False, batch_size=None, repeats=0, seed=42, shuffle_size=16834, prefetch_size=4096, max_tf_threadpool=8, **kwargs):
        super().__init__()
        assert_torch()
        self.tokenizer = None
        if tokenizer:
            assert_transformers()
            if isinstance(tokenizer, str):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            else:
                self.tokenizer = tokenizer
            self.tokenizer_args = kwargs.get('tokenizer_args', {})
        self.config = TFDSIOConfig()
        self.config.from_auto(config_or_file)
        self.datatype = data_type
        self.split = split
        self.seed, self.shuffle_size, self.prefetch_size, self.max_tf_threadpool = seed, shuffle_size, prefetch_size, max_tf_threadpool
        self.shuffle = shuffle
        self.is_training = is_training
        if self.is_training:
            assert batch_size is not None,\
                "Must specify batch_size in training mode for reasonable behaviour w/ TFDS wrapper"
        self.batch_size = batch_size
        self.repeats = repeats
        self.subsplit = None
        self.builder = TFDSIOCorpus(data_dir=self.config.data_dir, config=self.config, version=self.config.version)
        if self.preprocessor:
            self.builder.set_preprocessor(preprocessor)
        
        self.builder.download_and_prepare()
        self.num_samples = self.builder.info.splits[split].num_examples
        self.ds = None
        self.worker_info = None
        self.dist_rank = 0
        self.dist_num_replicas = 1
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            self.dist_rank = dist.get_rank()
            self.dist_num_replicas = dist.get_world_size()

    def _lazy_init(self):
        """ Lazily initialize the dataset.
        This is necessary to init the Tensorflow dataset pipeline in the (dataloader) process that
        will be using the dataset instance. The __init__ method is called on the main process,
        this will be called in a dataloader worker process.
        NOTE: There will be problems if you try to re-use this dataset across different loader/worker
        instances once it has been initialized. Do not call any dataset methods that can call _lazy_init
        before it is passed to dataloader.
        """
        worker_info = torch.utils.data.get_worker_info()
        split = self.split
        num_workers = 1
        if worker_info is not None:
            self.worker_info = worker_info
            num_workers = worker_info.num_workers
            global_num_workers = self.dist_num_replicas * num_workers
            worker_id = worker_info.id
            if not self.is_training and '[' not in self.split:
                self.subsplit = even_split_indices(self.split, global_num_workers, self.num_samples)[self.dist_rank * num_workers + worker_id]

        if self.subsplit is None:
            input_context = tf.distribute.InputContext(num_input_pipelines=self.dist_num_replicas * num_workers, input_pipeline_id=self.dist_rank * num_workers + worker_id,
                num_replicas_in_sync=self.dist_num_replicas
            )
        else:
            input_context = None

        read_config = tfds.ReadConfig(shuffle_seed=self.seed, shuffle_reshuffle_each_iteration=True, input_context=input_context)
        ds = self.builder.as_dataset(split=self.subsplit or self.split, shuffle_files=self.shuffle, read_config=read_config)
        ds.options().experimental_threading.private_threadpool_size = max(1, self.max_tf_threadpool // num_workers)
        ds.options().experimental_threading.max_intra_op_parallelism = 1
        if self.is_training or self.repeats > 1:
            ds = ds.repeat()
        if self.shuffle:
            ds = ds.shuffle(min(self.num_samples // self._num_pipelines, self.shuffle_size), seed=0)
        ds = ds.prefetch(min(self.num_samples // self._num_pipelines, self.prefetch_size))
        self.ds = tfds.as_numpy(ds)

    def __iter__(self):
        if self.ds is None:
            self._lazy_init()
        if self.datatype in ['image', 'img']:
            assert_pil()
            return self._img_iter()
        return self._text_iter()


    def _text_iter(self):
        target_sample_count = math.ceil(max(1, self.repeats) * self.num_samples / self._num_pipelines)
        if self.is_training:
            target_sample_count = math.ceil(target_sample_count / self.batch_size) * self.batch_size
        idx = 0
        for example in self.ds:
            for k,v in example.items():
                if isinstance(v, np.ndarray):
                    pass
                
                else:
                    v_text = tf.compat.as_text(v)
                    if self.tokenizer:
                        example[k] = {'text': v_text, 'tokens': self.tokenizer(v_text, **self.tokenizer_args)}
                    else:
                        example[k] = v_text
            yield example
            idx += 1
            if self.is_training and idx >= target_sample_count:
                break
        if not self.is_training and self.dist_num_replicas and 0 < idx < target_sample_count:
            yield example
            idx += 1


    def _img_iter(self):
        # compute a rounded up sample count that is used to:
        #   1. make batches even cross workers & replicas in distributed validation.
        #     This adds extra samples and will slightly alter validation results.
        #   2. determine loop ending condition in training w/ repeat enabled so that only full batch_size
        #     batches are produced (underlying tfds iter wraps around)
        target_sample_count = math.ceil(max(1, self.repeats) * self.num_samples / self._num_pipelines)
        if self.is_training:
            # round up to nearest batch_size per worker-replica
            target_sample_count = math.ceil(target_sample_count / self.batch_size) * self.batch_size
        sample_count = 0
        for example in self.ds:
            for k,v in example.items():
                example[k] = tf.compat.as_text(v)
            img = Image.fromarray(sample['image'], mode='RGB')
            yield img, sample['label']
            sample_count += 1
            if self.is_training and sample_count >= target_sample_count:
                # Need to break out of loop when repeat() is enabled for training w/ oversampling
                # this results in extra samples per epoch but seems more desirable than dropping
                # up to N*J batches per epoch (where N = num distributed processes, and J = num worker processes)
                break
        if not self.is_training and self.dist_num_replicas and 0 < sample_count < target_sample_count:
            # Validation batch padding only done for distributed training where results are reduced across nodes.
            # For single process case, it won't matter if workers return different batch sizes.
            # FIXME if using input_context or % based subsplits, sample count can vary by more than +/- 1 and this
            # approach is not optimal
            yield img, sample['label']  # yield prev sample again
            sample_count += 1


    @property
    def _num_workers(self):
        return 1 if self.worker_info is None else self.worker_info.num_workers

    @property
    def _num_pipelines(self):
        return self._num_workers * self.dist_num_replicas

    def __len__(self):
        # this is just an estimate and does not factor in extra samples added to pad batches based on
        # complete worker & replica info (not available until init in dataloader).
        return math.ceil(max(1, self.repeats) * self.num_samples / self.dist_num_replicas)

    def _filename(self, index, basename=False, absolute=False):
        assert False, "Not supported" # no random access to samples

    def filenames(self, basename=False, absolute=False):
        """ Return all filenames in dataset, overrides base"""
        if self.ds is None:
            self._lazy_init()
        names = []
        for sample in self.ds:
            if len(names) > self.num_samples:
                break  # safety for ds.repeat() case
            if 'file_name' in sample:
                name = sample['file_name']
            elif 'filename' in sample:
                name = sample['filename']
            elif 'id' in sample:
                name = sample['id']
            else:
                assert False, "No supported name field present"
            names.append(name)
        return names

