#!/usr/bin/env python

from distutils.core import setup

#To prepare a new release
#python setup.py sdist upload

setup(name='georaster',
      version='1.0',
      description='Libraries and command-line utilities for raster data processing/analysis',
      author='GeoHackWeek',
      author_email='',
      license='MIT',
      url='https://github.com/dlilien/georaster',
      packages=['georaster'],
      scripts=['georaster/georaster.py'])

