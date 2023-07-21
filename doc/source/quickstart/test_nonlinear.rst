.. _nonlinear_test:

The Halofit operator
======================================

There are several ways to compute the non-linear power spectrum.
On one hand there is non-linear perturbation theory with its well-known limits; on the other hand the application of an operator which transform a linear spectrum into its non-linear counterpart.
This operator is commonly known as Halofit, which consists of functions built to mimick N-body simulations, and it is typically included in Boltzmann solvers.
In time, there have been different versions, from the first one of `arXiv:0207664 <https://arxiv.org/abs/astro-ph/0207664>`_ to the more recent  .

We provide a file in ``tests`` named ``test_nonlinear.py`` that computes the non-linear matter power spectra starting from a linear one.
The same thing is done in the ``test_nonlinear.ipynb`` in the ``notebooks`` directory.
The file features the classes implemented in the ``nonlinear.py`` file, i.e. different methods to compute the non-linear power spectrum that should return exaclty what Boltzmann solvers do:
* the class :func:`colibri.nonlinear.HMcode2016` uses Mead's HMcode version of 2015 (`arxiv:1505.07833 <https://arxiv.org/abs/1505.07833>`_);
* the class :func:`colibri.nonlinear.HMcode2020` uses Mead's HMcode version of 2020 (`arxiv:2009.01858 <https://arxiv.org/abs/2009.01858>`_);
* the class :func:`colibri.nonlinear.Takahashi` employs the version by Takahashi (`arxiv:1208.2701 <https://arxiv.org/abs/1208.2701>`_ );
* the class :func:`colibri.nonlinear.Bird` is the same of the latter but it includes the corrections for the presence of massive neutrinos by Bird et al. (`arXiv:1109.4416 <https://arxiv.org/abs/1109.4416>`_ );
* the class :func:`colibri.nonlinear.halomodel` computes the total matter power spectrum assuming the halo model by Mead et al., 2015 (`arxiv:1505.07833 <https://arxiv.org/abs/1505.07833>`_ ).
* the class :func:`colibri.nonlinear.classic_halomodel` computes the total matter power spectrum assuming the classic halo model (e.g. `arxiv:0206508 <https://arxiv.org/abs/astro-ph/0206508>`_ ).

We first define a ``cosmo`` instance, without massive neutrinos to have a direct comparison with the Halofit implemented in CAMB:

.. code-block:: python

 import colibri.cosmology as cc
 import colibri.nonlinear as NL


 C  = cc.cosmo(Omega_m = 0.32,
               Omega_b = 0.05,
               As      = 2.12605e-9,
               ns      = 0.96,
               h       = 0.67,
               M_nu    = 0.06)
 zz = np.linspace(0., 5., 6)
 kk = np.logspace(-4., 2., 201)

Let us set then a method with which to compute the spectra. We will use the HMcode2020 method.
We compute at this point the non-linear power spectrum with CAMB

.. code-block:: python

 # Compute the non-linear power spectrum with CAMB (use Mead halofit)
 k_camb, pk_camb = C.camb_Pk(k = kk, z = zz, nonlinear = True, halofit = 'mead2020')

And then with COLIBRI via the :func:`colibri.nonlinear.HMcode2016` class:

.. code-block:: python

 # Use the `HMcode2020' class, which takes as arguments
 # Compute at first the linear power spectrum (for 'cb' and 'tot')
  k_l, pk_l = C.camb_XPk(z = zz, k = kk, var_1 =  ['cb','tot'], var_2 = ['cb','tot'])
  do_nonlinear = NL.HMcode2020(z            = zz,
                               k            = k_l,
                               pk_cc        = pk_cc,
                               pk_mm        = pk_mm,
                               cosmology    = C)
 pk_hf        = do_nonlinear.pk_nl

The comparison should look like the figure below, where the maximum deviation is below 1%.

.. image:: ../_static/nonlinear_power_spectrum.png
   :width: 700



