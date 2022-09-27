import os
import time

from fileio import File


_tfdsio_dir = os.path.abspath(os.path.dirname(__file__))
_tfdsio_imports = os.path.join(_tfdsio_dir, 'imported')

def set_adc(path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path

def timer(start_time=None):
    if not start_time:
        return time.time()
    return time.time() - start_time
