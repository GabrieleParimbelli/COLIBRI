.. _mass_function_test:

Halo mass function
======================================

In this section we show how to compute the halo mass functions using the class :func:`~colibri.cosmology.cosmo`
We also provide alternatives to the test code.
The text below refers to the file named``test_mass_function.py`` provided in the ``tests`` directory in the GitHub repository.


Initialization
-------------------------------

First of all, let us define scales, redshifts, masses and a :func:`~colibri.cosmology.cosmo` instance.
Also, we load the linear power spectra, which are necessary to compute mass functions and related quantities.

.. code-block:: python

 import colibri.cosmology as cc
 import matplotlib.pyplot as plt
 import numpy as np

 C=cc.cosmo()

 zz   = np.linspace(0., 5., 6)
 kk   = np.logspace(-4.,2.,1001)
 logM = np.linspace(5.,16.,111)
 nz, nk, nm = len(np.atleast_1d(zz)), len(np.atleast_1d(kk)), len(np.atleast_1d(logM))

 # Load linear power spectra
 k,pk=C.camb_Pk(z=zz)

Mass variances, peak height and mass functions
-----------------------------------------------

The :func:`~colibri.cosmology.cosmo` class has routines able to compute many interesting quantities for galaxy clustering.
In particular, the mass variance in spheres, the peak height for the computation of peak-background split and the halo mass function itself.

.. code-block:: python

 # Compute...
 sigma2_interp      = C.sigma2(k = k, pk = pk)                                        # mass variance in spheres
 peak_height_interp = C.peak_height(k = k, pk = pk)                                   # peak height
 mass_functions     = C.halo_mass_function(k = k, pk = pk, mass_fun = 'ShethTormen')  # halo mass function

 # The ones abouve are interpolated quantities.
 # Let us fill the tables for the values
 sigma_squared = np.zeros((nz,nm))
 nu            = np.zeros((nz,nm))
 HMF           = np.zeros((nz,nm))
 for iz in range(nz):
     sigma_squared[iz] = sigma2_interp[iz](logM)
     nu[iz]            = peak_height_interp[iz](logM)
     HMF[iz]           = mass_functions [iz](logM)

One can also retrieve the universal mass function. Here is the Sheth-Tormen case (see :func:`~colibri.cosmology.ShethTormen_mass_function`), but there are also :func:`~colibri.cosmology.PressSchechter_mass_function`, :func:`~colibri.cosmology.Tinker_mass_function` and :func:`~colibri.cosmology.MICE_mass_function` .

.. code-block:: python

 ShethTormen = C.ShethTormen_mass_function(sigma = sigma_squared**0.5, a = 0.707, p = 0.3)

If one wants to plot all the previous quantities, the following picture is the result.

.. image:: ../_static/mass_function.png
   :width: 700


