#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['numpy >= 1.11', 'scipy', 'pandas', 'opencv-python',
                    'tqdm']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='video_behavior_tracking',
    version='0.1.8.dev0',
    license='MIT',
    description=('Extract behavior from video'),
    author='Eric Denovellis',
    author_email='edeno@bu.edu',
    url='https://github.com/Eden-Kramer-Lab/video_behavior_tracking',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    entry_points={
        'console_scripts': [
            'track_behavior = video_behavior_tracking.track_behavior:main',
        ],
    },
)
