#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext = Extension('pycf.spinh_c', ['pycf/spinh_c.pyx'], 
        libraries=['lapacke'], include_dirs=['pycf'])

setup(name='pycf',
      version='1.0',
      description='Python crystal field theory modules',
      author='Sebastian Horvath',
      author_email='sebastian.horvath@gmail.com',
      url='https://bitbucket.org/sebastianhorvath/pycf/',
      packages=['pycf'],
      cmdclass = {'build_ext': build_ext},
      ext_modules = [ext],
      )
