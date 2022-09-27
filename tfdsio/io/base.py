import csv
import json
from tfdsio.utils import logger
from typing import Union, List
from tensorflow.python.platform import gfile
from tensorflow.python.platform.gfile import GFile

from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.python.data.ops.readers import TextLineDataset
from tensorflow.python.data.ops.readers import TFRecordDataset
from tensorflow.python.framework.ops import enable_eager_execution, disable_eager_execution


"""
Core IO Methods
"""

def parse_filenames(
    filenames: Union[str, List[str]]
) -> List[str]:
    """
    Parses a list of filenames, expanding wildcards and directories.
    """
    _results = []
    if isinstance(filenames, str) or not isinstance(filenames, list):
        filenames = [filenames]
    for fn in filenames:
        if not isinstance(fn, str): fn = str(fn)
        if fn.endswith('*'):
            _newfns = [f for f in gfile.Glob(fn) if not gfile.IsDirectory(f) and gfile.Exists(f)]
            _results.extend(_newfns)
        elif not gfile.IsDirectory(fn) and gfile.Exists(fn):
            _results.append(fn)
    return _results

def read_files(
    filenames: Union[str, List[str]], 
    mode: str = 'r+'
):
    """
    [Generator] Iterates through a list of filenames, yielding a file handle for each.
    """
    filenames = parse_filenames(filenames)
    for fn in filenames:
        with GFile(fn, mode) as f:
            yield f

def read_file_lines(
    filenames: Union[str, List[str]], 
    mode: str = 'r+'
):
    """
    [Generator] Iterates through a list of filenames, yielding a line for each.
    """
    for f in read_files(filenames, mode):
        yield from f

def read_json_lines(
    filenames: Union[str, List[str]], 
    mode: str = 'r+', 
    raise_errors: bool = False,
    verbose: bool = False,
    **kwargs
):
    """
    [Generator] Iterates through a list of filenames, yielding a JSON object for each.
    """
    for line in read_file_lines(filenames, mode):
        try:
            yield json.loads(line, **kwargs)
        except Exception as e:
            if raise_errors: raise e
            elif verbose: logger.error(f'Error reading line: {line}: {e}')

def read_text(
    filenames: Union[str, List[str]], 
    mode: str = 'r+'
):
    """
    [Generator] Iterates through a list of filenames, yielding a string for each.
    """
    for f in read_files(filenames, mode):
        yield f.read()

def read_text_lines(
    filenames: Union[str, List[str]],
    mode: str = 'r+'
):
    """
    [Generator] Iterates through a list of filenames, yielding a line for each.
    """
    for f in read_file_lines(filenames, mode):
        yield f.strip()

def read_json(
    filenames: Union[str, List[str]], 
    mode: str = 'r+', 
    raise_errors: bool = False,
    verbose: bool = False,
    **kwargs
):
    """
    [Generator] Iterates through a list of filenames, yielding a JSON object for each.
    """
    for f in read_files(filenames, mode):
        try:
            yield json.load(f, **kwargs)
        except Exception as e:
            if raise_errors: raise e
            elif verbose: logger.error(f'Error reading file: {f}: {e}')

def read_csv(
    filenames: Union[str, List[str]], 
    mode: str = 'r+', 
    delimiter: str = ',',
    raise_errors: bool = False,
    verbose: bool = False,
    **kwargs
):
    """
    [Generator] Iterates through a list of filenames, yielding a CSV object for each.
    """
    for f in read_files(filenames, mode):
        reader = csv.DictReader(f, delimiter = delimiter, **kwargs)
        for row in reader:
            try:
                yield row
            except Exception as e:
                if raise_errors: raise e
                elif verbose: logger.error(f'Error reading row: {row}: {e}')

class PipelineIO:
    """
    A pipeline for reading files.
    """

    #eager_enabled: bool = False

    @staticmethod
    def enable_eager():
        #if cls.eager_enabled: return
        enable_eager_execution()
        logger.info('Enabled eager execution')
        #cls.eager_enabled = True
    
    @staticmethod
    def disable_eager():
        #if not cls.eager_enabled: 
        disable_eager_execution()
        logger.info('Disabled eager execution')
        #cls.eager_enabled = False

    @staticmethod
    def parse_files(filenames: Union[str, List[str]]):
        """
        Parses a list of filenames, expanding wildcards and directories.
        """
        return parse_filenames(filenames)

    @staticmethod
    def read_files(filenames: Union[str, List[str]], mode: str = 'r+'):
        """
        [Generator] Iterates through a list of filenames, yielding a file handle for each.
        """
        return read_files(filenames, mode)
    
    @staticmethod
    def read_file_lines(filenames: Union[str, List[str]], mode: str = 'r+'):
        """
        [Generator] Iterates through a list of filenames, yielding a line for each.
        """
        return read_file_lines(filenames, mode)

    @staticmethod
    def get_json_reader(
        filenames: Union[str, List[str]], 
        mode: str = 'r+', 
        raise_errors: bool = False,
        verbose: bool = False,
        **kwargs
    ):
        """
        Returns a generator for reading JSON lines.
        """
        return read_json(filenames, mode, raise_errors, verbose, **kwargs)

    @staticmethod
    def get_jsonlines_reader(
        filenames: Union[str, List[str]], 
        mode: str = 'r+', 
        raise_errors: bool = False, 
        **kwargs
    ):
        """
        Returns a pipeline for reading JSON lines.
        """
        return read_json_lines(filenames, mode, raise_errors, **kwargs)

    @staticmethod
    def get_csv_reader(
        filenames: Union[str, List[str]], 
        mode: str = 'r+', 
        delimiter: str = ',', 
        raise_errors: bool = False, 
        **kwargs
    ):
        """
        Returns a pipeline for reading CSV lines.
        """
        return read_csv(filenames, mode, delimiter, raise_errors, **kwargs)

    @staticmethod
    def get_text_reader(
        filenames: Union[str, List[str]], 
        mode: str = 'r+'
    ):
        """
        Returns a pipeline for reading text.
        """
        return read_text(filenames, mode)
    
    @staticmethod
    def get_textline_reader(
        filenames: Union[str, List[str]], 
        mode: str = 'r+'
    ):
        """
        Returns a pipeline for reading text lines.
        """
        return read_text_lines(filenames, mode)
    
    @staticmethod
    def get_tfrecord_dataset(
        filenames: Union[str, List[str]],
        num_parallel_reads: int = AUTOTUNE,
        **kwargs
    ) -> TFRecordDataset:

        """
        Returns a pipeline for reading TFRecord files.
        """
        return TFRecordDataset(parse_filenames(filenames), num_parallel_reads = num_parallel_reads, **kwargs)
    
    @staticmethod
    def get_tftextline_dataset(
        filenames: Union[str, List[str]],
        num_parallel_reads: int = AUTOTUNE,
        **kwargs
    ) -> TextLineDataset:

        """
        Returns a pipeline for reading text lines.
        """
        return TextLineDataset(parse_filenames(filenames), num_parallel_reads = num_parallel_reads, **kwargs)

__all__ = [
    'PipelineIO',
    'gfile',
    'GFile',
    'TextLineDataset',
    'TFRecordDataset',
]