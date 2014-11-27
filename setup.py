#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from distutils.spawn import find_executable

import subprocess 
import os
import sys
import numpy as np
import numpy.distutils.intelccompiler
from Cython.Distutils import build_ext

try:
    compile_args = [os.environ['CFL_CFLAGS']]
except KeyError:
    compile_args = []

try:
    link_args=['-lgsl', '-lnlopt', '-lm', 'cfl/libcfl.a', os.environ['CFL_LDLIBS']]
except KeyError:
    link_args=['-lgsl', '-lnlopt', '-lm', 'cfl/libcfl.a']

if '--compiler=intel' in sys.argv:
    if find_executable('icc') == None:
        raise RuntimeError("Cannot locate the icc compiler.")
    else:
        intelpath = find_executable('icc')[:-len('/bin/icc')]
    
    os.environ['CFL_CC'] = 'icc'
    os.environ['INTEL_PATH'] = intelpath
    compile_args += ['-openmp -I%s/include' % intelpath]
    link_args += ['-mkl', '-lmkl_mc', 
            '-L%s/lib/intel64/' % intelpath, 
            '-L%s/mkl/lib/intel64/' % intelpath, 
            '-Wl,-rpath,%s/lib/intel64/' % intelpath, 
            '-Wl,-rpath,%s/mkl/lib/intel64/' % intelpath]
else:
    link_args += ['-llapacke', '-llapack', '-lblas', '-lgfortran', '-lgslcblas']


if 'clean' in sys.argv:
    ret = subprocess.call(['make', 'clean'], cwd='./cfl')
    if ret != 0:
        raise RuntimeError("Clean failed for cfl")
else:
    ret = subprocess.call(['make'], cwd='./cfl')
    if ret != 0:
        raise RuntimeError("Building cfl failed")


pycfl_ext = Extension('pycf.cfl', 
        sources=['pycf/cfl.pyx'],
        extra_compile_args = compile_args,
        extra_link_args=link_args,
        include_dirs=['cfl/include', np.get_include(), '/usr/include/lapacke'])

setup(name='pycf',
      version='1.1',
      description='Python crystal field theory modules',
      author='Sebastian Horvath',
      author_email='sebastian.horvath@gmail.com',
      url='https://bitbucket.org/sebastianhorvath/pycf/',
      packages=['pycf'],
      cmdclass = {'build_ext': build_ext},
      ext_modules = [pycfl_ext],
      )
