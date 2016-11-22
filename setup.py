#!/usr/bin/env python

from distutils.core import setup

#To prepare a new release
#python setup.py sdist upload

setup(name='georaster',
      version='1.0',
      description='Libraries and command-line utilities for raster data processing/analysis',
      author='Andrew Tedstone - Amaury Dehecq',
      author_email='',
      license='MIT',
      url='https://github.com/atedstone/georaster',
      packages=['georaster'])

