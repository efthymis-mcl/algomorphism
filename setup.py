from codecs import open
from os.path import abspath, dirname, join

from distutils.core import setup
from setuptools import find_packages

from algomorphism import __version__

this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.md'), encoding='utf-8') as file:
    long_description = file.read()

setup(
    name='algomorphism',
    version=__version__,
    description='General driven framework for object-oriented programming on Back-Propagation Algorithm',
    long_description=long_description,
    url='https://github.com/efth-mcl/algomorphism',
    author='Efthymis Michalis',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    packages=find_packages(),
    package_dir={'algomorphism': 'algomorphism'},
    install_requires=[
        'tensorflow==2.15.0.post1',
        'scikit-learn',
        'numpy',
        'networkx',
        'matplotlib',
    ],
)
