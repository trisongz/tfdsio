import os
import time

#try:
#    from functional.io import File
#except ImportError:
#    os.system('pip install PyFunctional @ git+https://github.com/trisongz/PyFunctional')

from functional.io import File


_tfdsio_dir = os.path.abspath(os.path.dirname(__file__))
_tfdsio_imports = os.path.join(_tfdsio_dir, 'imported')

def set_adc(path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path

def timer(start_time=None):
    if not start_time:
        return time.time()
    return time.time() - start_time
