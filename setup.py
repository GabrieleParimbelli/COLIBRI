#from distutils.core import setup
#import setuptools
from setuptools import setup, Extension, find_packages

setup(
    name='colibri-cosmology',
    version='0.1.0',
    author='Gabriele Parimbelli',
    author_email='g.parimbelli90@gmail.com',
    packages=find_packages(),
    scripts=[],
    url='https://github.com/GabrieleParimbelli/COLIBRI',
    license='doc/source/license/license.rst',
    description='Python libraries for cosmology.',
    long_description=open('README.rst').read(),
    install_requires=[
        "numpy >= 1.14",
        "scipy >= 0.16",
    ],
)
