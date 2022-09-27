#import tensorflow as tf
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.util.compat import as_bytes
from tensorflow.python.framework.dtypes import (
    int32,
    int64,
    string,
    float32,
    float64,
    DType
)
from tensorflow.python.framework.dtypes import bool as tf_bool

from tensorflow.python.training.training import (
    BytesList,
    Feature,
    Int64List,
    Example,
    FloatList,
    Features,
)


def dict_to_tfexample(ex):
    """Convert example dictionary to tf.train.Example proto."""
    feature_dict = {}
    for k, v in ex.items():
        t = constant(v)
        if len(t.shape) == 0:  # pylint:disable=g-explicit-length-test
            v = [v]
        elif len(t.shape) == 1: v = list(v)
        else: raise ValueError("Unsupported shape (%s) for '%s' value: %s" % (t.shape, k, v))

    if t.dtype == string and len(t.shape) <= 1:
        feature_dict[k] = Feature(bytes_list = BytesList(value = [as_bytes(t) for t in v]))
    elif t.dtype in (tf_bool, int32, int64) and len(t.shape) <= 1:
        feature_dict[k] = Feature(int64_list = Int64List(value=v))
    elif t.dtype in (float32, float64) and len(t.shape) <= 1:
        feature_dict[k] = Feature(float_list = FloatList(value=v))
    else:
        raise ValueError("Unsupported type (%s) and shape (%s) for '%s' value: %s" % (t.dtype, t.shape, k, v))

    return Example(features = Features(feature=feature_dict))
