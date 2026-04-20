# pycf installation proposal

For this repo, the lowest-risk PyPA migration is: keep `setuptools`, add `pyproject.toml`, keep a custom `setup.py`, and add a `MANIFEST.in` so sdists contain the C/Cython sources.

Below is a concrete version of the files I would use.

## `pyproject.toml`

```toml
[build-system]
requires = [
  "setuptools>=68",
  "wheel",
  "Cython>=3",
  "numpy>=1.26",
]
build-backend = "setuptools.build_meta"
```

## `setup.py`

```python
#!/usr/bin/env python3

from pathlib import Path
from shutil import which, rmtree
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools import Command

import os
import shlex
import subprocess
import sys

import numpy as np
from Cython.Build import cythonize


ROOT = Path(__file__).resolve().parent
CFL_DIR = ROOT / "cfl"
PYCF_DIR = ROOT / "pycf"
PYX_FILE = PYCF_DIR / "cfl.pyx"
VERSION_FILE = PYCF_DIR / "__version__.py"


def get_git_revision() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return "unknown"
    rev = proc.stdout.strip()
    return rev or "unknown"


def write_version_file() -> str:
    git_revision = get_git_revision()
    VERSION_FILE.write_text(f'\n__version__ = "{git_revision}"\n\n', encoding="utf-8")
    return git_revision


def run_make(target: str | None = None) -> str:
    cmd = ["make"]
    if target:
        cmd.append(target)

    proc = subprocess.Popen(
        cmd,
        cwd=CFL_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    output = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        output.append(line)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed in cfl/.")

    return "".join(output)


def build_cfl() -> None:
    output = run_make()

    # Preserve the current behavior: if the C archive changed, force Cython
    # to consider the extension stale.
    if "Nothing to be done for 'all'." not in output and "Nothing to be done" not in output:
        PYX_FILE.touch()


def clean_cfl() -> None:
    try:
        run_make("clean")
    except RuntimeError:
        # Keep clean resilient if cfl/ was only partially built.
        pass


def split_flags(value: str | None) -> list[str]:
    if not value:
        return []
    return shlex.split(value)


def compute_build_flags() -> tuple[list[str], list[str]]:
    compile_args = split_flags(os.environ.get("CFL_CFLAGS"))
    link_args = split_flags(os.environ.get("CFL_LDLIBS"))

    link_args += [str(CFL_DIR / "libcfl.a"), "-lgsl", "-lnlopt", "-lm"]

    if os.environ.get("CFL_CC") == "icc":
        intel_path = os.environ.get("INTEL_PATH")
        if not intel_path:
            icc = which("icc")
            if icc is None:
                raise RuntimeError(
                    "CFL_CC=icc was requested but icc could not be found and "
                    "INTEL_PATH was not provided."
                )
            intel_path = icc[: -len("/bin/icc")]

        compile_args += [f"-I{intel_path}/include", "-openmp"]
        link_args += [
            "-mkl",
            "-lmkl_def",
            f"-L{intel_path}/lib/intel64/",
            f"-L{intel_path}/mkl/lib/intel64/",
            f"-Wl,-rpath,{intel_path}/lib/intel64/",
            f"-Wl,-rpath,{intel_path}/mkl/lib/intel64/",
        ]
    else:
        link_args += ["-llapacke", "-llapack", "-lblas", "-lgfortran", "-lgslcblas"]

    return compile_args, link_args


class BuildExtCommand(build_ext):
    def run(self) -> None:
        build_cfl()
        super().run()


class CleanCommand(Command):
    description = "Clean Python and cfl build artifacts"
    user_options: list[tuple[str, str | None, str]] = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        clean_cfl()

        for path in [
            ROOT / "build",
            ROOT / "dist",
            ROOT / "pycf.egg-info",
        ]:
            if path.exists():
                rmtree(path)

        for path in PYCF_DIR.glob("cfl*.so"):
            path.unlink()

        c_file = PYCF_DIR / "cfl.c"
        if c_file.exists():
            c_file.unlink()


git_revision = write_version_file()
version = f"0+{git_revision}"

compile_args, link_args = compute_build_flags()

ext_modules = cythonize(
    [
        Extension(
            "pycf.cfl",
            sources=[str(PYCF_DIR / "cfl.pyx")],
            include_dirs=[
                str(CFL_DIR / "include"),
                np.get_include(),
                "/usr/include/lapacke",
            ],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    ]
)

setup(
    name="pycf",
    version=version,
    description="Python crystal field theory modules",
    long_description=(ROOT / "README.rst").read_text(encoding="utf-8"),
    author="Sebastian Horvath",
    author_email="sebastian.horvath@gmail.com",
    url="https://bitbucket.org/sebastianhorvath/pycf/",
    packages=["pycf"],
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtCommand,
        "clean": CleanCommand,
    },
    zip_safe=False,
)
```

## `MANIFEST.in`

```text
include README.rst
include TODO.rst
include pyproject.toml
include setup.py

include cfl/makefile
include cfl/cfl_testing.sh
recursive-include cfl/include *.h
recursive-include cfl/src *.c
recursive-include cfl/tests *.c

recursive-include pycf *.py
recursive-include pycf *.pyx
recursive-include pycf *.pxd

recursive-include tests *.py
recursive-include tests *.txt
recursive-include tests *.mi_
recursive-include tests *.st_

recursive-include examples *.py
recursive-include examples *.txt
recursive-include examples *.mi_
recursive-include examples *.st_

recursive-include doc *.rst
```

## What changes for users

After this, the normal workflows become:

### Build sdist + wheel

```bash
python -m pip install build
python -m build
```

### Install from source

```bash
python -m pip install .
```

### Editable install for development

```bash
python -m pip install -e .
```

### Clean

```bash
python setup.py clean
```

## Notes specific to this repo

1. You still keep `setup.py`. The modernization is about how builds are invoked, not necessarily deleting `setup.py`.
2. `Cython` and `numpy` move into build requirements. That means `pip install .` and `python -m build` can create an isolated build environment and install them automatically.
3. The `cfl/` make step still runs. That behavior is preserved in `BuildExtCommand`.
4. `pycf/__version__.py` is still generated from git. That matches the current repo behavior.
5. Intel/MKL builds should still be driven by environment variables, for example:

```bash
CFL_CC=icc INTEL_PATH=/path/to/intel python -m pip install .
```

## Recommendation

If you want to modernize packaging without rewriting the native build system, this is the safest first step. It keeps the existing C and Cython build architecture, but moves user-facing installs and builds onto the standard PyPA workflow.
