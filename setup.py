#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from subprocess import call
import os
import numpy as np

# extra_compile_args and extra_link_args options to the Extension builder can be
# used to set compiler and linker arguments explicitly.

pycfl_ext = Extension('pycf.cfl', 
        sources=['pycf/cfl.pyx'],
        extra_link_args=['-llapacke', '-llapack', '-lblas', '-lgsl', '-lnlopt',
            '-lgfortran', 'cfl/libcfl.a'],
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
