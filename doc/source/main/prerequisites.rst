.. _prerequisites:

Prerequisites
=============

The only mandatory requirements needed for COLIBRÌ are the usual Python packages such as NumPy and SciPy (better if updated versions).


However, some of the routines require some external libraries and Boltzmann solvers to be installed.
The list that follows itemizes all the packages needed to let all the routines work.

* **FFTLog** by Dieter Werthmüller. This package is a Python wrapper for the FFTLog by Andrew Hamilton and allows to compute Fourier transforms of a log-spaced array. Download `FFTLog <https://github.com/prisae/fftlog>`_ and unzip the folder. When in the folder compile with ``python setup.py install --user`` (the system may not recognize the `ü` on the name of the author, if so just remove it).

* **Cython**: C compiler for Python. Its presence is needed for the compilation of Class (see below). The installation can be done both using ``pip install Cython`` or from source, downloading it from `here <https://cython.org/>`_, unzipping the folder, go into it and then running ``python setup.py install --user``. Important: if one follows the latter way, the folder `cython-master` contains another folder named in the same way. This must not happen, since the paths may not be recognized. Just move everything in the outer of the two folders and delete the inner one. 

* the Python wrapper of the Cosmic Linear Anisotropy Solving System (**Class**) by Julien Lesgourgues. Download `Class <http://class-code.net/>`_, untar and compile with ``make`` only (not ``make class`` otherwise the Python wrapper will not work). Go now to the *python* folder inside `Class` and run ``python setup.py install --user``. This will build the `classy` file: check that everything has gone fine by typing in terminal ``python -c "from classy import Class"``. If nothing is returned then everything is ok.

* the Python wrapper of the Code for Anisotropies in the Microwave Background (**CAMB**) by Antony Lewis and Anthony Challinor. Clone `CAMB <https://github.com/cmbant/CAMB>`_ or install it using `pip <https://camb.readthedocs.io/en/latest/>`_. In order for COLIBRÌ to work, a version > 1.0 of CAMB is required. Notice that to install this, a gfortran compiler > 6 is needed.

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

So, set your preferences and get started!

