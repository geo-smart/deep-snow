#!/usr/bin/env python

from distutils.core import setup
import setuptools

setup(name='deep_snow',
      version='0.1',
      description='Neural networks for snow depth prediction',
      packages=setuptools.find_packages(),
      package_data={"deep_snow": ["data/*.pkl"]},
      include_package_data=True,
     )