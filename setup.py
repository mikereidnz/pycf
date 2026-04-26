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
    
    # Use a valid PEP 440 version format
    # Base version + dev suffix with git hash for development builds
    base_version = "0.1.0"
    if git_revision != "unknown":
        # Use PEP 440 local version identifier
        version_str = f"{base_version}.dev0+{git_revision}"
    else:
        version_str = base_version
    
    version_text = f'''
__version__ = "{version_str}"
__build_timestamp__ = "{build_timestamp}"
__build_comment__ = "{build_comment}"

'''
    VERSION_FILE.write_text(version_text, encoding="utf-8")
    return version_str


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


def find_lapacke_include() -> str:
    """Find LAPACKE include directory with fallbacks for different systems."""
    # Check environment variable first
    if lapacke_env := os.environ.get("LAPACKE_INCLUDE_DIR"):
        if Path(lapacke_env).is_dir():
            return lapacke_env
        print(f"Warning: LAPACKE_INCLUDE_DIR={lapacke_env} does not exist", file=sys.stderr)
    
    # Try common system paths
    candidates = [
        "/usr/include/lapacke",           # Linux (Debian/Ubuntu/RHEL)
        "/usr/local/opt/lapack/include",  # macOS Homebrew (Intel)
        "/opt/homebrew/opt/lapack/include",  # macOS Homebrew (Apple Silicon)
        "/opt/local/include",             # MacPorts
        "/usr/local/include",             # Generic custom installs
    ]
    
    for path in candidates:
        if Path(path).is_dir():
            return path
    
    # No LAPACKE headers found - raise error
    raise RuntimeError(
        "LAPACKE headers not found. Please either:\n"
        "1. Install LAPACK development files (e.g., liblapack-dev on Debian/Ubuntu)\n"
        "2. Set LAPACKE_INCLUDE_DIR environment variable to the directory containing lapacke.h\n"
        "3. Install via Homebrew: brew install lapack"
    )


def compute_build_flags() -> tuple[list[str], list[str]]:
    compile_args = split_flags(os.environ.get("CFL_CFLAGS"))
    link_args = split_flags(os.environ.get("CFL_LDLIBS"))

    link_args += ["cfl/libcfl.a", "-lgsl", "-lnlopt", "-lm", "-lgomp"]

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
            raise RuntimeError(
                f"INTEL_PATH={intel_path} does not exist or is not a directory"
            )

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

compile_args, link_args = compute_build_flags()
lapacke_include = find_lapacke_include()

ext_modules = cythonize(
    [
        Extension(
            "pycf.cfl",
            sources=["pycf/cfl.pyx"],
            include_dirs=[
                "cfl/include",
                np.get_include(),
                lapacke_include,
            ],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    ]
)

setup(
    name="pycf",
    description="Python crystal field theory modules",
    long_description=(ROOT / "README.rst").read_text(encoding="utf-8"),
    long_description_content_type="text/x-rst",
    author="Mike Reid",
    author_email="mike.reid@canterbury.ac.nz",
    url="https://github.com/mikereidnz/pycf",
    license="GPL-3.0",
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    extras_require={
        "test": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-benchmark>=4.0",
            "hypothesis>=6.0",
            "coverage>=7.0",
        ],
        "examples": [
            "pymatgen>=2022.0",
            "matplotlib>=3.5",
            "scipy>=1.10",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.20",
            "sphinx-copy-button>=0.5",
            "sphinxcontrib-napoleon>=0.7",
            "myst-parser>=0.18",
        ],
        "dev": [
            "black>=23.0",
            "isort>=5.13",
            "flake8>=6.0",
            "mypy>=1.7",
            "bandit>=1.7",
            "semgrep>=1.45",
            "pre-commit>=3.0",
        ],
    },
    zip_safe=False,
)
