COLIBRI: cosmological libraries in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Presentation
==================


**COLIBRÌ** is a set of Python files containing useful routines to compute cosmological quantities such as ages, distances, power spectra, correlation functions. It supports Lambda-CDM cosmologies plus extensions including massive neutrinos, non-flat geometries and evolving dark energy (w0-wa) models.
These files are built especially for large-scale structure purposes.

COLIBRÌ is compatible with both Python 2 and 3.
Furthermore, it can interact with the Boltzmann solvers `CAMB <https://camb.info/>`__ (by Antony Lewis and Anthony Challinor) and `Class <http://class-code.net/>`__ (by Julien Lesgourgues).


FYI, "colibrì" is the Italian for `"hummingbird" <https://en.wikipedia.org/wiki/Hummingbird>`__.

Prerequisites and installation
==============================

COLIBRÌ work properly provided that the usual Python packages such as NumPy and SciPy are installed.
However, some of the routines require some external libraries and Boltzmann solvers to be installed.
The list that follows itemizes all the packages needed to let all the routines work.

* **FFTLog** by Dieter Werthmüller. This package is a Python wrapper for the FFTLog by Andrew Hamilton and allows to compute Fourier transforms of a log-spaced array. Download `FFTLog <https://github.com/prisae/fftlog>`__ and unzip the folder. When in the folder compile with ``python setup.py install --user`` (the system may not recognize the `ü` on the name of the author, if so just remove it).

* **Cython**: C compiler for Python. Its presence is needed for the compilation of CLASS (see below). The installation can be done both using ``pip install Cython`` or from source, downloading it from `here <https://cython.org/>`__, unzipping the folder, go into it and then running ``python setup.py install --user``. Important: if one follows the latter way, the folder `cython-master` contains another folder named in the same way. This must not happen, since the paths may not be recognized. Just move everything in the outer of the two folders and delete the inner one. 

* the Python wrapper of the Cosmic Linear Anisotropy Solving System (**Class**) by Julien Lesgourgues. Download `Class <http://class-code.net/>`__, untar and compile with ``make`` only (not ``make class`` otherwise the Python wrapper will not work). Go now to the *python* folder inside `Class` and run ``python setup.py install --user``. This will build the `classy` file: check that everything has gone fine by typing in terminal ``python -c "from classy import Class"``. If nothing is returned then everything is ok.

* the Python wrapper of the Code for Anisotropies in the Microwave Background (**CAMB**) by Antony Lewis and Anthony Challinor. Clone `CAMB <https://github.com/cmbant/CAMB>`__ or install it using `pip <https://camb.readthedocs.io/en/latest/>`__. In order for COLIBRÌ to work, a version > 1.0 of CAMB is required. Notice that to install this, a gfortran compiler > 6 is needed.

The importation of these packages is always conditional in COLIBRÌ: if they are not found they are not imported. Thus, no error will be returned until a function that requires some of them is called.
However, we strongly recommend to install all these 4 packages (for convenience and future employment of the user).

Here we report briefly what the various packages are needed for.

* FFTLog is needed to use the Fourier and Hankel transforms for log-spaced arrays in ``fourier.py``.

* CAMB and/or Class are needed to compute power spectra given a cosmological model. These packages are required in many routines across all the COLIBRÌ files. If you wish, you can install only one of the two, although it is a good thing to have them both also for comparisons.

* Cython is just needed to compile the Python wrapper of Class: thus if you do not want to install Class, Cython is not necessary as well.

Installation
=============

The best way to install **COLIBRÌ** is to clone the GitHub repository `<https://github.com/GabrieleParimbelli/COLIBRI>`__ , enter in the ``COLIBRI`` directory and run::

    python setup.py install --user

A second way consists of cloning the repository and add the libraries to your ``PYTHONPATH`` editing your ``.bashrc`` file.::

    export PYTHONPATH="${PYTHONPATH}:/path/to/COLIBRI/"

Finally, **COLIBRÌ** is also available on `pip <https://pypi.org/project/colibri-cosmology/>`__, but it is less frequently updated.
To install it is sufficient to run::

    pip install colibri-cosmology


So, set your preferences and get started!


Overview of the modules
==============================

COLIBRÌ consists of simple Python files: one, named ``constants.py`` is just a collection of physical constants, useful numbers and conversion factors between different units of measurement. A second one, ``useful_functions.py`` contains some routines of everyday life for a cosmologist. The remaining ones each contain one or more Python classes. Moreover also a folder ``Tests`` is provided, which contains test functions for the libraries and providing some example of how to use them.

In the following sections we briefly present the codes.
The documentation for both classes and built-in methods is inside the code itself and can be found using the following Python commands.

For any COLIBRÌ file and function you can type the following commands.::

    import file_name
    help(file_name)                       # Info on classes, functions and routines in 'file_name'
    help(file_name.myfunction)            # Info on specific function in 'file_name'
    help(file_name.class_name.myfunction) # Info on specific function of 'class_name' in 'file_name'

to report useful information about what the file itself contains and its functions do. Type `q` for exiting.

Also, from a Python program:

* ``print file_name.class_name.__doc__`` returns the documentation on initialization of a class
* ``print file_name.class_name.__init__`` prints what the class returns
* ``print file_name.class_name.__dict__.keys()`` lists the methods of the class
* ``print file_name.class_name.method_name.__doc__`` prints the documentation of a given method of a class

.. _cosmology_overview:

`cosmology` module
^^^^^^^^^^^^^^^^^^^^^^^^

The main file of COLIBRÌ is without doubt the ``cosmology.py`` file.
This file contains one Python class, named ``cosmology.cosmo()``, in whose initialization the cosmological parameters must be provided (otherwise the default values will be loaded).
This class has all the routines necessary to compute distances, ages, density parameters, halo mass functions, void size functions and power spectra.
For the latter, both Boltzmann solvers **CAMB** and **Class** can be used, provided their Python wrapper is correctly compiled, as well as the Eisenstein-Hu formula (see `arXiv:9710252 <https://arxiv.org/abs/astro-ph/9710252>`__).

`nonlinear` module
^^^^^^^^^^^^^^^^^^

There are two classes, one called ``halofit_operator`` , which transforms a linear input power spectrum to its non-linear counterpart, and the other called ``nonlinear_pk`` , that directly computes the nonlinear total matter power spectrum given a set of cosmological parameters.
Both these classes use the Halofit model by Mead `et al.` (see `arXiv:1505.07833 <https://arxiv.org/abs/1505.07833>`__ and `arXiv:1602.02154 <https://arxiv.org/abs/1602.02154>`__), taking as inputs redshift, scales, power spectrum and a cosmology.
The latter class uses the so-called `cold dark matter prescription` (see e.g. `arXiv:1311.0866 <https://arxiv.org/abs/1311.0866>`__, `arXiv:1311.1212 <https://arxiv.org/abs/1311.1212>`__, `arXiv:1311.1514 <https://arxiv.org/abs/1311.1514>`__ ), where the Halofit operator is applied only to the cold dark matter+baryons linear component, while the neutrino and the cross part are added linearly afterwards.

`angular_spectra` module
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``angular_spectra`` class inside this file is finalized to compute the weak lensing and galaxy clustering power spectrum and correlation functions in the flat sky and Limber's approximations. In this file are provided the routines to compute non-linear power spectra, example functions for galaxy distributions (although different ones can be defined by the user outside the class) and window functions. The possibility to add the intrinsic alignment effect (see `arXiv:0406275 <https://arxiv.org/abs/astro-ph/0406275>`__) is included.

`halo`, `galaxy`, `RSD`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This three files are linked with each other. The basis is ``halo.py`` : it contains the class ``halo`` which computes the non-linear matter power spectrum according to the pure halo model (see for instance `Cooray & Sheth (2001) <https://arxiv.org/abs/astro-ph/0206508>`__).
While this is known to return a poor description of the matter clustering, the class has routines able to compute properly halo mass functions and halo biases.
In the file ``galaxy.py`` the class ``galaxy`` is implemented, which uses the Halo Occupation Distribution (see e.g. `arXiv:0408564 <https://arxiv.org/pdf/astro-ph/0408564.pdf>`__ prescription to predict the galaxy power spectrum in real space.
Conversely, the redshift-space power spectrum is provided by the class ``RSD`` in the file ``RSD.py``: currently the dispersion model is implemented (with both Gaussian and Lorentzian dampings) as well as a halo model based prescription.

`fourier` module
^^^^^^^^^^^^^^^^

This file contains routines to compute Fourier and Hankel Transforms. They employ the NumPy FFT libraries as well as FFTlog in some cases. They return sorted frequencies for an immediate interpretation of the outcomes.
In particular, these routines can be useful to compute two-point correlation functions starting from a power spectrum.


`constants` module
^^^^^^^^^^^^^^^^^^

This file is just a compilation of physical constants and does not contain any class or method. While typing ``help(constants)`` will provide the list of quantities, it will not be documented. To obtain a full description of the quantities, type in a Python session or program::


    import constants
    constants.explanatory()


`useful_functions` module
^^^^^^^^^^^^^^^^^^^^^^^^^

The file contains (as is obvious) useful functions such as extrapolation of arrays and top-hat window functions.


Tests
^^^^^

Together with the files, a folder named ``tests`` containing some useful and explanatory tests is provided. Each of them is adequately commented, so check them out and run them!


