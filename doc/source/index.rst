COLIBRI: cosmological libraries in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Presentation
==================


**COLIBRÌ** (which roughly stands for "Cosmological Libraries") is a set of Python files containing useful routines to compute cosmological quantities such as ages, distances, power spectra, correlation functions. It supports Lambda-CDM cosmologies plus extensions including massive neutrinos, non-flat geometries, evolving dark energy (w0-wa) models and numerical recipes for f(R) gravity.
These files are built especially for large-scale structure purposes.

It is compatible with Python 3.
Furthermore, it can interact with the Boltzmann solvers `CAMB <https://camb.info/>`_ (by Antony Lewis and Anthony Challinor) and `Class <http://class-code.net/>`_ (by Julien Lesgourgues).

Please, cite the GitHub repository ( `<https://github.com/GabrieleParimbelli/COLIBRI>`_ ) if you use this library.

FYI, "colibrì" is the Italian for `"hummingbird" <https://en.wikipedia.org/wiki/Hummingbird>`_.

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   main/prerequisites
   main/doc_overview


.. toctree::
   :caption: Quickstart
   :maxdepth: 1

   quickstart/test_cosmology
   quickstart/test_mass_function
   quickstart/test_void_size_function
   quickstart/test_nonlinear
   quickstart/test_halo_model
   quickstart/test_limber
   quickstart/test_fourier

.. toctree::
   :caption: Code documentation
   :maxdepth: 1

   modules/cosmology_doc
   modules/nonlinear_doc
   modules/halo_model_doc
   modules/limber_doc
   modules/fourier_doc
   modules/useful_functions_doc
   modules/constants_doc

.. toctree::
   :caption: Extras
   :maxdepth: 1

   license/license


.. toctree::

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


