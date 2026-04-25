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
    from datetime import datetime
    
    git_revision = get_git_revision()
    build_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    build_comment = os.environ.get('PYCF_BUILD_COMMENT', "Build via setup.py")
    
    version_text = f'''
__version__ = "{git_revision}"
__build_timestamp__ = "{build_timestamp}"
__build_comment__ = "{build_comment}"

'''
    VERSION_FILE.write_text(version_text, encoding="utf-8")
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

    link_args += ["cfl/libcfl.a", "-lgsl", "-lnlopt", "-lm"]

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
            sources=["pycf/cfl.pyx"],
            include_dirs=[
                "cfl/include",
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
    url="https://github.com/mikereidnz/pycf",
    packages=["pycf"],
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtCommand,
        "clean": CleanCommand,
    },
    zip_safe=False,
)
