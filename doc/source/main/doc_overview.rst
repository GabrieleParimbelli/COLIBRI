.. _doc_overview:

Overview of the modules
==============================

COLIBRÌ consists of simple Python files. One, named ``constants.py`` is just a collection of physical constants, useful numbers and conversion factors between different units of measurement. A second one, ``useful_functions.py`` contains some routines of everyday life for a cosmologist. The file ``fourier.py`` contains routines to perform Fourier and Hankel transforms. The remaining ones each contain one or more Python classes. Moreover also a folder ``tests`` is provided, which contains test functions for the libraries and providing some example of how to use them.

In the following sections we briefly present the codes.
The documentation for both classes and built-in methods is also inside the code itself and can be found using the following Python commands.

For any COLIBRÌ file and function you can type the following commands.::

    import file_name
    help(file_name)                       # Info on classes, functions and routines in 'file_name'
    help(file_name.myfunction)            # Info on specific function in 'file_name'
    help(file_name.class_name.myfunction) # Info on specific function of 'class_name' in 'file_name'

to report useful information about what the file itself contains and its functions do. Type `q` for exiting.

Also, from a Python (2.x in this case) program:

* ``print file_name.class_name.__doc__`` returns the documentation on initialization of a class
* ``print file_name.class_name.__init__`` prints what the class returns
* ``print file_name.class_name.__dict__.keys()`` lists the methods of the class
* ``print file_name.class_name.method_name.__doc__`` prints the documentation of a given method of a class

.. _cosmology_overview:

`cosmology` module
^^^^^^^^^^^^^^^^^^^^^^^^

The main file of COLIBRÌ is without doubt the ``colibri.cosmology.py`` file.
This file contains one Python class, named :func:`~colibri.cosmology.cosmo()`, in whose initialization the cosmological parameters must be provided (otherwise the default values will be loaded).
This class has all the routines necessary to compute distances, ages, density parameters, halo mass functions, void size functions and power spectra.
For the latter, both Boltzmann solvers **CAMB** and **Class** can be used, provided their Python wrapper is correclty compiled (see :ref:`prerequisites`), as well as the Eisenstein-Hu formula (see `arXiv:9710252 <https://arxiv.org/abs/astro-ph/9710252>`_).

`nonlinear` module
^^^^^^^^^^^^^^^^^^

There are two classes, one called :func:`~colibri.nonlinear.halofit_operator` , which transforms a linear input power spectrum to its non-linear counterpart, and the other called :func:`~colibri.nonlinear.nonlinear_pk` , that directly computes the nonlinear total matter power spectrum given a set of cosmological parameters.
Both these classes use the Halofit model by Mead `et al.` (see `arXiv:1505.07833 <https://arxiv.org/abs/1505.07833>`_ and `arXiv:1602.02154 <https://arxiv.org/abs/1602.02154>`_), taking as inputs redshift, scales, power spectrum and a cosmology.
The latter class uses the so-called `cold dark matter prescription` (see e.g. `arXiv:1311.0866 <https://arxiv.org/abs/1311.0866>`_, `arXiv:1311.1212 <https://arxiv.org/abs/1311.1212>`_, `arXiv:1311.1514 <https://arxiv.org/abs/1311.1514>`_ ), where the Halofit operator is applied only to the cold dark matter+baryons linear component, while the neutrino and the cross part are added linearly afterwards.

`weak_lensing` module
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~colibri.weak_lensing.weak_lensing` class inside this file is finalized to compute the shear power spectrum and correlation functions in the flat sky and Limber's approximations. In this file are provided the routines to compute non-linear power spectra, example functions for galaxy distributions (although different ones can be defined by the user outside the class) and window functions. The possibility to add the intrinsic alignment effect (see `arXiv:0406275 <https://arxiv.org/abs/astro-ph/0406275>`_) is included.

`halo`, `galaxy`, `RSD`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This three files are linked with each other. The basis is ``halo.py`` : it contains the class :func:`~colibri.halo.halo` which computes the non-linear matter power spectrum according to the pure halo model (see for instance `arXiv:0206508 <https://arxiv.org/abs/astro-ph/0206508>`_).
While this is known to return a poor description of the matter clustering, the class has routines able to compute properly halo mass functions and halo biases.
In the file ``galaxy.py`` the class :func:`~colibri.galaxy.galaxy` is implemented, which uses the Halo Occupation Distribution (see e.g. `arXiv:0408564 <https://arxiv.org/pdf/astro-ph/0408564.pdf>`_) prescription to predict the galaxy power spectrum in real space.
Conversely, the redshift-space power spectrum is provided by the class :func:`~colibri.RSD.RSD` in the file ``RSD.py``: currently the dispersion model is implemented (with both Gaussian and Lorentzian dampings) as well as a halo model based prescription.

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

If the code has been cloned from GitHub, together with the files, a folder containing some useful and explanatory tests is provided. Each of them is adequately commented, so check them out and run them!

Otherwise, you may want to go the :ref:`cosmology_test` section.


