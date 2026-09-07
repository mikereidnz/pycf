#!/usr/bin/env python3

import os
import shlex
import subprocess
import sys
from pathlib import Path
from shutil import rmtree, which
from typing import List, Optional, Tuple

import numpy as np
from Cython.Build import cythonize
from setuptools import Command, Extension, setup
from setuptools.command.build_ext import build_ext

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


def is_release_tag() -> bool:
    """Check if current HEAD is tagged with a release tag (v*.*.*)"""
    proc = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False
    tag = proc.stdout.strip()
    return tag.startswith("v") and tag[1].isdigit()


def write_version_file() -> str:
    from datetime import datetime

    git_revision = get_git_revision()
    build_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    build_comment = os.environ.get("PYCF_BUILD_COMMENT", "Build via setup.py")

    # Use a valid PEP 440 version format
    # For release tags (v0.2.0), use just the base version
    # For dev builds, add .dev0+git_hash
    #
    # NOTE: `base_version` is the single source of truth for the package
    # version. pyproject.toml declares `dynamic = ["version"]` with
    # `version = {attr = "pycf.__version__"}`, so the value written here flows
    # through to the installed package metadata. When bumping the version,
    # update this constant and keep CHANGELOG.md in sync.
    base_version = "0.2.0"
    if is_release_tag():
        version_str = base_version
    elif git_revision != "unknown":
        version_str = f"{base_version}.dev0+{git_revision}"
    else:
        version_str = base_version

    version_text = (
        f'__version__ = "{version_str}"\n'
        f'__build_timestamp__ = "{build_timestamp}"\n'
        f'__build_comment__ = "{build_comment}"\n'
    )
    VERSION_FILE.write_text(version_text, encoding="utf-8")
    return version_str


def run_make(target: Optional[str] = None, env: Optional[dict] = None) -> str:
    cmd = ["make"]
    if target:
        cmd.append(target)

    # Prepare environment with LAPACKE_INCLUDE if not already set
    make_env = os.environ.copy()
    if env:
        make_env.update(env)

    proc = subprocess.Popen(
        cmd,
        cwd=CFL_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=make_env,
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
    # Set up environment with CFL_CFLAGS defaults if not already provided
    make_env = {}
    if "CFL_CFLAGS" not in os.environ:
        # Add default include paths for lapacke only on Linux
        # On macOS, /usr/include doesn't exist; the makefile fallback will handle it
        if sys.platform.startswith("linux"):
            make_env["CFL_CFLAGS"] = "-I/usr/include -I/usr/include/lapacke"
        elif sys.platform == "darwin":
            prefix = find_homebrew_prefix() or "/opt/homebrew"
            make_env["CFL_CFLAGS"] = (
                f"-I{prefix}/opt/lapack/include "
                f"-I{prefix}/opt/gsl/include "
                f"-I{prefix}/opt/nlopt/include "
            )

    output = run_make(env=make_env)

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


def split_flags(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return shlex.split(value)


def find_lapacke_include() -> str:
    """Find LAPACKE include directory.

    Returns the directory containing lapacke.h. Users can override via
    LAPACKE_INCLUDE_DIR environment variable.
    """
    # Check environment variable first
    if lapacke_env := os.environ.get("LAPACKE_INCLUDE_DIR"):
        return lapacke_env

    # Use sensible defaults - the compiler will report an error if the
    # header is not found
    return "/usr/include"


def find_homebrew_prefix() -> Optional[str]:
    """Return the active Homebrew prefix for the current machine.

    On Apple Silicon Homebrew usually lives in /opt/homebrew.
    On Intel macOS it is commonly /usr/local.
    """
    for candidate in ("/opt/homebrew", "/usr/local"):
        if Path(candidate).exists():
            return candidate
    return None


def compute_build_flags() -> Tuple[List[str], List[str]]:
    compile_args = split_flags(os.environ.get("CFL_CFLAGS"))
    link_args = split_flags(os.environ.get("CFL_LDLIBS"))

    link_args += ["cfl/libcfl.a", "-lgsl", "-lnlopt", "-lm"]

    # GCC OpenMP: enable on Linux unless CFL_NO_OPENMP=1 is set or we are
    # building via icc (which has its own OpenMP runtime, libiomp5, wired up
    # in the icc branch below).
    #
    # NOTE: the linker chooses libgomp from whichever directory appears first
    # in LIBRARY_PATH / -L paths.  In a conda/Anaconda environment that is
    # usually $CONDA_PREFIX/lib/libgomp.so.1, not the gcc-shipped libgomp from
    # /usr/lib/x86_64-linux-gnu.  This is normally fine -- conda's libgomp is
    # ABI-compatible -- but if you observe odd OMP behaviour (deadlocks,
    # missing parallelism, mismatched thread counts), it may be worth forcing
    # the system libgomp by prepending /usr/lib/x86_64-linux-gnu to LDFLAGS.
    if (
        sys.platform.startswith("linux")
        and os.environ.get("CFL_CC") != "icc"
        and not os.environ.get("CFL_NO_OPENMP")
    ):
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")
        link_args.append("-lgomp")

    if os.environ.get("CFL_CC") == "icc":
        intel_path = os.environ.get("INTEL_PATH")
        if not intel_path:
            icc = which("icc")
            if icc is None:
                raise RuntimeError(
                    "CFL_CC=icc was requested but icc could not be found and "
                    "INTEL_PATH was not provided."
                )
            # Use Path to safely extract parent directory
            intel_path = str(Path(icc).parent.parent)

        # Validate the path exists
        if not Path(intel_path).is_dir():
            raise RuntimeError(f"INTEL_PATH={intel_path} does not exist or is not a directory")

        compile_args += [
            f"-I{intel_path}/include",
            "-openmp",
        ]  # NOTE: -openmp is the legacy icc flag; modern Intel oneAPI/icx requires -qopenmp.
        link_args += [
            "-mkl",
            "-lmkl_def",
            f"-L{intel_path}/lib/intel64/",
            f"-L{intel_path}/mkl/lib/intel64/",
            f"-Wl,-rpath,{intel_path}/lib/intel64/",
            f"-Wl,-rpath,{intel_path}/mkl/lib/intel64/",
        ]
    elif sys.platform == "darwin":
        homebrew_prefix = find_homebrew_prefix()

        if not homebrew_prefix:
            raise RuntimeError(
                "Could not find a Homebrew prefix on macOS. "
                "Set CFL_CFLAGS/CFL_LDLIBS manually or install Homebrew."
            )

        link_args += [
            f"-L{homebrew_prefix}/opt/lapack/lib",
            f"-L{homebrew_prefix}/opt/gsl/lib",
            f"-L{homebrew_prefix}/opt/nlopt/lib",
            "-llapacke",
            "-llapack",
            "-lblas",
            "-lgslcblas",
        ]
    else:
        link_args += ["-llapacke", "-llapack", "-lblas", "-lgslcblas"]

        # Only add GNU Fortran runtime on Linux
        if sys.platform.startswith("linux"):
            link_args.append("-lgfortran")

    return compile_args, link_args


class BuildExtCommand(build_ext):
    def run(self) -> None:
        build_cfl()
        super().run()


class CleanCommand(Command):
    description = "Clean Python and cfl build artifacts"
    user_options: List[Tuple[str, Optional[str], str]] = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        clean_cfl()

        for path in [
            ROOT / "build",
            ROOT / "dist",
            ROOT / "pycf_crystalfield.egg-info",
        ]:
            if path.exists():
                rmtree(path)

        for path in PYCF_DIR.glob("cfl*.so"):
            path.unlink()

        c_file = PYCF_DIR / "cfl.c"
        if c_file.exists():
            c_file.unlink()


git_revision = write_version_file()

compile_args, link_args = compute_build_flags()
lapacke_include = find_lapacke_include()
homebrew_prefix = find_homebrew_prefix()

include_dirs = [
    "cfl/include",
    np.get_include(),
    lapacke_include,
]

if homebrew_prefix:
    # extra directories to include for macOS
    include_dirs += [
        f"{homebrew_prefix}/opt/lapack/include",
        f"{homebrew_prefix}/opt/gsl/include",
        f"{homebrew_prefix}/opt/nlopt/include",
    ]

ext_modules = cythonize(
    [
        Extension(
            "pycf.cfl",
            sources=["pycf/cfl.pyx"],
            include_dirs=include_dirs,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    ]
)

setup(
    name="pycf-crystalfield",
    description="Python crystal field theory modules",
    long_description=(ROOT / "README.rst").read_text(encoding="utf-8"),
    long_description_content_type="text/x-rst",
    author="Mike Reid",
    author_email="mike.reid@canterbury.ac.nz",
    url="https://github.com/mikereidnz/pycf",
    packages=["pycf"],
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtCommand,
        "clean": CleanCommand,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: C",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    zip_safe=False,
)
