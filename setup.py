#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess 
import os
import numpy as np

try:
    compile_args = [os.environ['CFL_CFLAGS']]
except KeyError:
    compile_args = []

try:
    link_args=['-llapacke', '-llapack', '-lblas', '-lgfortran', '-lgslcblas',
            '-lgsl', '-lnlopt', '-lm', 'cfl/libcfl.a', os.environ['CFL_LDLIBS']]
except KeyError:
    link_args=['-llapacke', '-llapack', '-lblas', '-lgfortran', '-lgslcblas',
            '-lgsl', '-lnlopt', '-lm', 'cfl/libcfl.a']
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
