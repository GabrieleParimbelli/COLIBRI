#from distutils.core import setup
from setuptools import setup, Extension, find_packages

# To create build/ and dist/ files, do:
# python setup.py sdist
# python setup.py bdist_wheel

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='colibri-cosmology',
    version='1.0',
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
    #py_modules=['six', 'numpy', 'scipy', 'matplotlib'],
    py_modules=['numpy', 'scipy', 'matplotlib'],
	python_requires='>=3.6',
    install_requires=[
        "numpy >= 1.14",
        "scipy >= 0.16",
    ],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #options = {'bdist_wheel':{'universal':'1'}},	# Generate wheel for both Python 2 and 3
)
