#!/usr/bin/env python

from setuptools import setup, Extension
from shutil import which

import subprocess 
import os
import sys
from datetime import datetime
import numpy as np
try:
    import numpy.distutils.intelccompiler
except ImportError:
    pass
from Cython.Distutils import build_ext

DEFAULT_BUILD_COMMENT = "\
    MFR: Updated to python 3.13. \n\
    Added conjugation before lapack call.\n\
    Changed MAGX and MAGY to standard signs.\n\
    C memory fixes and other minor changes may change behaviour of some calculations."

try:
    compile_args = [os.environ['CFL_CFLAGS']]
except KeyError:
    compile_args = []

try:
    link_args=[os.environ['CFL_LDLIBS']]
except KeyError:
    link_args=[]
link_args += ['cfl/libcfl.a', '-lgsl', '-lnlopt', '-lm']

if '--compiler=intel' in sys.argv:
    icc = which('icc')
    if icc == None:
        raise RuntimeError("Cannot locate the icc compiler.")
    else:
        intelpath = icc[:-len('/bin/icc')]
    
    os.environ['CFL_CC'] = 'icc'
    os.environ['INTEL_PATH'] = intelpath
    compile_args += ['-openmp -I%s/include' % intelpath]
    link_args += ['-mkl', '-lmkl_def', 
            '-L%s/lib/intel64/' % intelpath, 
            '-L%s/mkl/lib/intel64/' % intelpath, 
            '-Wl,-rpath,%s/lib/intel64/' % intelpath, 
            '-Wl,-rpath,%s/mkl/lib/intel64/' % intelpath]
else:
    link_args += ['-llapacke', '-llapack', '-lblas', '-lgfortran', '-lgslcblas']


if 'clean' in sys.argv:
    ret = subprocess.call(['make', 'clean'], cwd='./cfl')
    if ret != 0:
        raise RuntimeError("Clean failed for cfl.")
else:
    popen = subprocess.Popen(['make'], cwd='./cfl', stdout=subprocess.PIPE, universal_newlines=True)
    
    output = ""
    for line in iter(popen.stdout.readline, ''):

        sys.stdout.write(line)
        output += line
    
    popen.wait()
    if popen.returncode != 0:
        raise RuntimeError("Building cfl failed.")

    if not "make: Nothing to be done" in output:
        subprocess.call(['touch', 'pycf/cfl.pyx'])

popen = subprocess.Popen(
    ['git', 'rev-parse', '--short', 'HEAD'],
    stdout=subprocess.PIPE,
    universal_newlines=True,
)
git_revision = popen.communicate()[0].strip()
if popen.returncode != 0 or not git_revision:
    git_revision = 'unknown'

build_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
build_comment = os.environ.get('PYCF_BUILD_COMMENT', DEFAULT_BUILD_COMMENT)

with open('pycf/_build_info.py', 'w') as f:
    f.write('\n__version__ = %r\n' % git_revision)
    f.write('__build_timestamp__ = %r\n' % build_timestamp)
    f.write('__build_comment__ = %r\n\n' % build_comment)

version = '0+%s' % git_revision

pycfl_ext = Extension('pycf.cfl', 
        sources=['pycf/cfl.pyx'],
        extra_compile_args = compile_args,
        extra_link_args=link_args,
        include_dirs=['cfl/include', np.get_include(), '/usr/include/lapacke'])

setup(name='pycf',
      version=version,
      description='Python crystal field theory modules',
      author='Sebastian Horvath',
      author_email='sebastian.horvath@gmail.com',
      url='https://bitbucket.org/sebastianhorvath/pycf/',
      packages=['pycf'],
      cmdclass = {'build_ext': build_ext},
      ext_modules = [pycfl_ext],
      )
