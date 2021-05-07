# TensorFlow Datasets IO (tfdsio)
 Dynamic TensorFlow Datasets with Pytorch Support + More

## Features
`tfdsio` allows for creation of tensorflow_datasets dynamically defined by a config (json/dict) 
without requiring writing custom python classes, which can lead to library bloat. Additionally, 
you can also define a preprocessor to handle the data transformation prior to being written as a tfrecord. 

`tfdsio` enables you to create multiple version-controlled dataset variations, so that the final tfrecords 
contain only what you need to train your models, ensuring a more efficient data pipeline.

* Single function call to return a custom tensorflow_dataset/tf.data.Dataset object (or as numpy iterator/pandas dataframe)
* Efficiently read `text`, `csv`, `jsonl` files.
* Enables efficient chained functional operators by leveraging a modified [PyFunctional](https://github.com/trisongz/PyFunctional) for preprocessing
    * Chained operators: `seq(1, 2, 3).map(lambda x: x * 2).reduce(lambda x, y: x + y)`
    * Expressive and feature complete API
    * Read and write `text`, `csv`, `json`, `jsonl`, `sqlite`, `gzip`, `bz2`, and `lzma/xz` files
    * Parallelize "embarrassingly parallel" operations like `map` easily
    * More examples available in [PyFunctional](https://github.com/trisongz/PyFunctional) repo
* Support for loading `tfdsio` datasets in other formats through Custom Adapters
    *  `text-to-text-transformers`
    *  `torch` (WIP)



## Installation
`tfdsio` is available on [pypi](https://pypi.python.org/pypi/tfdsio) and can be
installed by running:

```bash
# Install from command line
$ pip install tfdsio

# Install from source
$ pip install --upgrade https://github.com/trisongz/tfdsio
```

## TFDSIO Dataset Configuration

The full config spec is shown below. The values below will be overwritten after loading 
the dataset configuration passed.

```python
name: Optional[str] = None # Dataset Name i.e. 'my_dataset'
classifier: Optional[str] = None # Dataset variation, such as 'lm' or 'qa' 
version: Optional[VersionOrStr] = '1.0.0' # Dataset version, i.e. '0.0.1'
release_notes: Optional[Dict[str, str]] = None # Release notes included in the dataset metadata
supported_versions: List[str] = None # Optional list of versions ['0.0.1', '0.0.2']
description: Optional[str] = None # Description of dataset included in dataset metadata
dataset_urls: Optional[Any] = None # Defines your dataset urls, expected to be a dict
dataset_format: Optional[str] = 'jsonlines' # ['jsonlines', 'text', 'csv']
features: Optional[Any] = None # Maps the dataset output dict keys to tf.features, supports ['text', 'audio', 'image']
datamap: Optional[Any] = None # Maps your dataset input dict keys to the dataset output keys
supervised_keys: Optional[Any] = None # Optional
homepage: Optional[str] = '' # homepage for dataset 
citation: Optional[Any] = None # citation for dataset
metadata: Optional[Any] = None # metadata for dataset
redistribution_info: Optional[Any] = None # redistribution info for dataset
data_dir: Optional[str] = None # [IMPORTANT]: This should be your GCS Directory or local drive that stores your dataset.
process_args: Optional[Any] = None # Args passed to your preprocessor function
```

## Examples
`tfdsio` aims to make it simple to turn your custom dataset into a training ready data pipeline. 
Built into `tfdsio` include useful utilities to allow working with GCS storage/Files much easier 
along with `PyFunctional` to enable efficient data processing pipelines.

### Minimal Example
```python

from tfdsio import tfds_dataset, tfds_sample, set_adc
from tfdsio import tfds # import tensorflow_datasets already initialized

# Remember if you are reading from a private bucket to ensure ADC is set
set_adc('/path/to/adc.json')
dataset_config = {
    'name': 'my_dataset',
    'classifier': 'qa',
    'version': '1.0.0',
    'features': {
        'input_text': 'text',
        'target_text': 'text'
    },
    'datamap': {
        'question': 'input_text',
        'text': 'target_text',
    },
    'dataset_urls': {
        'train': 'gs://your-storage-bucket/datasets/custom_dataset.jsonl'
    },
    'dataset_format': 'jsonlines',
    'homepage': 'https://growthengineai.com',
    'data_dir': 'gs://your-storage-bucket/datasets/cached',
    'description': 'My Custom Question Answering Dataset'
}
# As long as the above configuration matches, the next time it's called, it will load from pre-built dataset
dataset = tfds_dataset(dataset_config, preprocessor=None, build=True, as_tfdataset=True, as_numpy=False, as_df=False)

# If it wasn't already created with as_numpy=True
samples = tfds_sample(dataset, num_samples=5, return_samples=True)

# or the standard method
for ex in tfds.as_numpy(dataset.take(5)):
    print(ex)
```

### Create Dataset Variations Easily
Using the same config as above, you can modify your config and define a different feature/datamap
and have a different dataset variation.

A dataset's identifier is comprised of `(<dataset_name>/<dataset_classifier>/<dataset_version>)`, 
correlating to `name`, `classifier`, and `version`. 

```python
dataset_config_2 = dataset_config.copy()
dataset_config_2['classifier'] = 'lm'
dataset_config_2['datamap'] = {
    'context': 'input_text',
    'answer': 'target_text'
}

dataset2 = tfds_dataset(dataset_config_2, preprocessor=None, build=True, as_tfdataset=True, as_numpy=False, as_df=False)


# Your new dataset variation
samples = tfds_sample(dataset2, num_samples=5, return_samples=True)
```

### Remove a Dataset
If you something messed up during the dataset creation process, you can use `tfds_remove` to remove the directory

```python
from tfdsio import tfds_remove

tfds_remove(dataset_config, prompt=True) # prompt = False will delete without asking
```

### Using Preprocessors
Preprocessors can be any function, but should expect the minimum of the following args
kwargs will contain any args that were passed from config.process_args set in the original
config

```python
def preprocessor(idx: int, data: dict, extra: Optional[Filepath] = None, **kwargs):
    # if return_data is a list, will create an example per item in list
    return_data = []
    # do stuff
    return return_data

    # if return_data is a dict, only one example will be created
    return_data = {}
    # do stuff
    return return_data

dataset = tfds_dataset(dataset_config, preprocessor=preprocessor)
```

### Using Adapters
Currently, tfdsio has support for `text-to-text-transformers` or `t5`.

```python
import tensorflow as tf
import t5
from t5 import seqio
from tfdsio import tfds, tfds_sample
from tfdsio.adapter import T5DataSource

vocab = '/path/to/vocab/sentencepiece.model'
seqio.TaskRegistry.add(
    "my_dataset",
    source=T5DataSource(
      config_or_file=dataset_config,
      splits={
          "train": "train[:90%]",
          "validation": "train[90%:]",
          "test": "validation"
      }),
    preprocessors=[
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos,
    ],
    output_features={
        "inputs": seqio.Feature(
           seqio.SentencePieceVocabulary(vocab),
           add_eos=False, dtype=tf.int32
        ),
        "targets": seqio.Feature(
            seqio.SentencePieceVocabulary(vocab),
           add_eos=True, dtype=tf.int32
        ),
    },
    metric_fns=[]
)

my_task = t5.data.TaskRegistry.get("my_dataset")
ds = my_task.get_dataset(split="validation", sequence_length={"inputs": 128, "targets": 128})
print("A few preprocessed validation examples...")
samples = tfds_sample(ds, num_samples=5, return_samples=True)

```

## Limitations
While `tfdsio` has many useful utilities that allows extending `tensorflow_datasets` beyond the
base library, there are still challenges, including not being able to simply use `tfds.load()`
for any custom dataset, which can be limiting when working with many datasets in a single run.
However, taking from `t5`, you can develop your own adapters to get around this limitation.

Contributions are welcome!

## Motivation & About
I have worked extensively with all dataset formats and came to really like tfds when working with TPUs
in TensorFlow. However, I found that continually expanding the dataset came with lots of bloat and time
to dig into each dataset to fix a bug, rather than having a change that affected all the datasets to be
updated.

The long term roadmap is to expand `tfdsio` to enable cross-compatibility with any framework, supporting all
major data types, backed with the high-performant `tf.data.Dataset` backend.

I lead the ML Team at [Growth Engine AI](https://growthengineai.com) working with large scale NLP models in EdTech.
If you find working on NLP and MLOps challenges exciting, and would want to join our team, shoot me an 
email: ts at growthengineai.com

## Acknowledgements
Development of `tfdsio` relied on contributions made from the following projects, 
and I recommend checking them out as well!

* [tensorflow-datasets](https://github.com/tensorflow/datasets)
* [text-to-text-transformers](https://github.com/google-research/text-to-text-transfer-transformer/)
* [EntilZha: PyFunctional](https://github.com/EntilZha/PyFunctional/)
* [rwightman](https://github.com/rwightman/pytorch-image-models/)
