#!/usr/bin/env python

from distutils.core import setup
from Cython.Build import cythonize

setup(name='pycf',
      version='1.0',
      description='Python crystal field theory modules',
      author='Sebastian Horvath',
      author_email='sebastian.horvath@gmail.com',
      url='https://bitbucket.org/sebastianhorvath/pycf/',
      packages=['pycf'],
      ext_modules = cythonize('pycf/spinh.pyx'),
      )
