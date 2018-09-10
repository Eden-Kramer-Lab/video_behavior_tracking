#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['numpy >= 1.11', 'scipy', 'pandas', 'opencv', 'tqdm']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='video_behavior_tracking',
    version='0.1.1.dev0',
    license='MIT',
    description=('Extract behavior from video'),
    author='Eric Denovellis',
    author_email='edeno@bu.edu',
    url='https://github.com/Eden-Kramer-Lab/video_behavior_tracking',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
