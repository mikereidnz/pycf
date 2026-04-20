from datetime import datetime
import sys

from pycf.__version__ import __version__
try:
    from pycf.__version__ import __build_timestamp__, __build_comment__
except ImportError:
    __build_timestamp__ = 'unknown'
    __build_comment__ = ''


def _fmt_pycf_time(value=None):
    if value is None:
        value = datetime.now()
    if isinstance(value, str):
        return value
    return value.strftime('%Y-%m-%d %H:%M:%S')


def pycf_info(current_time=None, stream=None):
    r"""
    Print and return a short pycf metadata block for scripts and notebooks.
    """
    if stream is None:
        stream = sys.stdout

    info = (
        "----------------------------------------------------------\n"
        "pycf details\n"
        "============\n\n"
        "pycf revision: {}  built at {}\n"
        "Build comment: {}\n"
        "Current time: {}\n\n"
        "----------------------------------------------------------"
    ).format(__version__, __build_timestamp__, __build_comment__, _fmt_pycf_time(current_time))

    print(info, file=stream)
    return info


__all__ = ['__version__', '__build_timestamp__', '__build_comment__', 'pycf_info']
