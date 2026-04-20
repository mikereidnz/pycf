import os
import subprocess

__version__ = 'unknown'
__build_timestamp__ = 'unknown'
__build_comment__ = ''

try:
    from pycf._build_info import __version__, __build_timestamp__, __build_comment__
except ImportError:
    try:
        from _build_info import __version__, __build_timestamp__, __build_comment__
    except ImportError:
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            __version__ = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=repo_root,
                universal_newlines=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
