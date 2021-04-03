import os
try:
    import functional
except ImportError:
    os.system('pip install PyFunctional @ git+https://github.com/trisongz/PyFunctional')

from . import utils
from . import dataset
from . import adapter
from .dataset import tfds, tfds_dataset, tfds_remove, tfds_sample
from .utils import set_adc