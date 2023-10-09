# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import subprocess
import glob
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Configuration for ReadTheDocs setup -------------------------------------

import subprocess
 
# Check if we're running on Read the Docs' servers
read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'
 
# Sort functions as they are, not in alphabetical order
autodoc_member_order = 'bysource'

breathe_projects = {}


# -- Project information -----------------------------------------------------

project = 'COLIBRI'
copyright = '2023, Gabriele Parimbelli'
author = 'Gabriele Parimbelli'

# Master document is `index.rst`
master_doc = 'index'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [ 'sphinx.ext.autodoc',
               'sphinx.ext.coverage',
               'sphinx.ext.napoleon',
               'sphinx.ext.todo',
               'sphinx.ext.intersphinx',
               'sphinx.ext.viewcode',
               'sphinx.ext.autosectionlabel',
               #'sphinx_rtd_theme',
               'breathe']

# Add any Sphinx extension module names here, as strings. They can be                                           
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom                                           
# ones.                                                                                                         
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://docs.scipy.org/doc/numpy/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
                       'matplotlib': ('https://matplotlib.org/', None)}

# Generate summary of functions
autosummary_generate = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Avoid documentation from these files
autodoc_mock_imports = ['numpy', 'scipy', 'math', 'classy', 'camb', 'cython', 'fftlog']

# Import Mock classes to avoid import errors on libraries that depend on C modules
# (this basically avoids that when other files import e.g. `cosmology` the compilation
#  does not get stuck in some `import numpy` or similar command)
#from unittest.mock import MagicMock
#class Mock(MagicMock):
#    @classmethod
#    def __getattr__(cls, name):
#        return MagicMock()
#MOCK_MODULES = ['numpy', 'scipy', 'pandas']
#sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
from unittest import mock
MOCK_MODULES = ['numpy', 'scipy', 'pandas']
for mod_name in MOCK_MODULES:
    #sys.modules[mod_name] = mock.Mock()
    sys.modules[mod_name] = mock.MagicMock()


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'sphinx_rtd_theme'
html_theme = 'classic'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

