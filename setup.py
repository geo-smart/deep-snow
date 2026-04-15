#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="deep_snow",
    version="0.1",
    description="Neural networks for snow depth prediction",
    packages=find_packages(),
    package_data={
        "deep_snow": [
            "data/*.pkl",
            "resources/weights/*",
            "resources/polygons/ne_50m_land.*",
        ]
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "deep-snow=deep_snow.cli:main",
        ]
    },
)
