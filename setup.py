#from distutils.core import setup
from setuptools import setup, Extension, find_packages

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='colibri-cosmology',
    version='0.2',
    author='Gabriele Parimbelli',
    author_email='g.parimbelli90@gmail.com',
    packages=find_packages(),
    scripts=[],
    project_urls={
        'Documentation': 'https://colibri-cosmology.readthedocs.io/en/latest/',
        'Source': 'https://github.com/GabrieleParimbelli/COLIBRI',
    },
    license='LICENSE.txt',
    description='Python libraries for cosmology.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    py_modules=['six', 'numpy', 'scipy', 'matplotlib'],
	python_requires='>=2.7',
    install_requires=[
        "numpy >= 1.14",
        "scipy >= 0.16",
    ],
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    options = {'bdist_wheel':{'universal':'1'}},	# Generate wheel for both Python 2 and 3
)
