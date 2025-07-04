# -*- coding: UTF-8 -*-

__author__ = "Anthony Baptista"
__copyright__ = "Copyright 2023, Anthony Baptista"
__email__ = "abaptista@turing.ac.uk"
__license__ = "MIT"

import codecs
import configparser
from setuptools import setup
from setuptools import find_packages
import os
import sys

config = configparser.RawConfigParser()
config.read(os.path.join('.', 'setup.cfg'))
author = config['metadata']['author']
email = config['metadata']['email']
license = config['metadata']['license']

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

if sys.version_info < (3, 6):
    print("At least Python 3.6 is required.\n", file=sys.stderr)
    exit(1)

try:
    from setuptools import setup, find_packages
except ImportError:
    print("Please install setuptools before installing trim.",
          file=sys.stderr)
    exit(1)

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as fin:
    long_description = fin.read()

CLASSIFIERS = """\
Development Status :: 4 - Beta
Environment :: Console
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Operating System :: OS Independent
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3 :: Only
Topic :: Software Development :: Libraries :: Python Modules
Topic :: Scientific/Engineering :: Mathematics
Topic :: Scientific/Engineering :: Physics
Topic :: Scientific/Engineering :: Network theory
"""

# Create list of package data files
def data_files_to_list(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

data_example_file_list = data_files_to_list('trim/data')

setup(
    name='trim',
    version=get_version("trim/__init__.py"),
    description="",
    author=author,
    author_email=email,
    license=license,
    long_description=long_description,
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    packages=find_packages(),
    package_dir={'trim': 'trim'},
    package_data={'trim': data_example_file_list},
    install_requires=['networkx', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'scikit-learn'],
    entry_points={},
)
