import os
import sys
import logging
import warnings
import atexit as _atexit

from loguru import _defaults
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger
from typing import Any

warnings.filterwarnings('ignore', message='Unverified HTTPS request')
warnings.filterwarnings('ignore', message='aliases are no longer used by BaseSettings')
warnings.filterwarnings('ignore', message='DEPRECATED: please do not use _builder as this may change')
# warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='tensorflow_datasets')  # noqa                      

LEVEL_COLOR_MAP = {
    "TRACE": "<cyan>",
    "DEBUG": "<blue>",
    "INFO": "<green>",
    "SUCCESS": "<green>",
    "WARNING": "<yellow>",
    "ERROR": "<red>",
    "CRITICAL": "<red,bg:white>",
}

# Setup Default Logger
class Logger(_Logger):
    def __call__(self, message: Any, *args, level: str = 'info', **kwargs):
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'INFO'``."""
        if isinstance(message, list):
            __message = "".join(f'- {item}\n' for item in message)
        elif isinstance(message, dict):
            __message = "".join(f'- {key}: {value}\n' for key, value in message.items())
        else:
            __message = str(message)
        self._log(level.upper(), None, False, self._options, __message.strip(), args, kwargs)


logger = Logger(
    core=_Core(),
    exception=None,
    depth=0,
    record=False,
    lazy=False,
    colors=False,
    raw=False,
    capture=True,
    patcher=None,
    extra={},
)

if _defaults.LOGURU_AUTOINIT and sys.stderr:
    logger.add(sys.stderr)

_atexit.register(logger.remove)

class InterceptHandler(logging.Handler):
    loglevel_mapping = {
        50: 'CRITICAL',
        40: 'ERROR',
        30: 'WARNING',
        20: 'INFO',
        10: 'DEBUG',
        5: 'CRITICAL',
        4: 'ERROR',
        3: 'WARNING',
        2: 'INFO',
        1: 'DEBUG',
        0: 'NOTSET',
    }

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = self.loglevel_mapping[record.levelno]
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        log = logger.bind(request_id='app')
        log.opt(
            depth=depth,
            exception=record.exc_info
        ).log(level, record.getMessage())


class CustomizeLogger:

    @classmethod
    def make_default_logger(cls, level: str = "INFO"):
        # todo adjust this later to use a ConfigModel
        logger.remove()
        logger.add(
            sys.stdout,
            enqueue=True,
            backtrace=True,
            colorize=True,
            level=level.upper(),
            format=cls.logger_formatter,
        )
        logging.basicConfig(handlers=[InterceptHandler()], level=0)
        *options, extra = logger._options
        return Logger(logger._core, *options, {**extra})

    @staticmethod
    def logger_formatter(record: dict) -> str:
        """
        To add a custom format for a module, add another `elif` clause with code to determine `extra` and `level`.

        From that module and all submodules, call logger with `logger.bind(foo='bar').info(msg)`.
        Then you can access it with `record['extra'].get('foo')`.
        """
        #module = record['name']
        ts_color = LEVEL_COLOR_MAP.get(record['level'].name, "<green>")
        extra = '<cyan>{name}</>:<cyan>{function}</>: '
        return "<level>{level: <8}</> " + ts_color + "{time:YYYY-MM-DD HH:mm:ss.SSS}</>: "\
            + extra + "<level>{message}</level>\n"


get_logger = CustomizeLogger.make_default_logger
logger = CustomizeLogger.make_default_logger()
