.. _nonlinear_test:

The Halofit operator
======================================

There are several ways to compute the non-linear power spectrum.
On one hand there is non-linear perturbation theory with its well-known limits; on the other hand the application of an operator which transform a linear spectrum into its non-linear counterpart.
This operator is commonly known as Halofit, which consists of functions built to mimick N-body simulations, and it is typically included in Boltzmann solvers.
In time, there have been different versions, from the first one of `arXiv:0207664 <https://arxiv.org/abs/astro-ph/0207664>`_ to the more recent `1208.2701 <https://arxiv.org/abs/1208.2701>`_ .

These operators are typically applied to the total matter power spectrum: however it has been shown (see e.g. `arXiv:1311.0866 <https://arxiv.org/abs/1311.0866>`_, `arXiv:1311.1212 <https://arxiv.org/abs/1311.1212>`_, `arXiv:1311.1514 <https://arxiv.org/abs/1311.1514>`_ ) that in massive neutrino cosmologies the fundamental field relevant for clustering is the cold dark matter plus baryons (CDM+b, exluding neutrinos).
Therefore, the Halofit operator should be applied to the CDM+b part only.

This is why the :func:`nonlinear.halofit_operator` class reproduces the Halofit version by Mead (`arXiv:1505.07833 <https://arxiv.org/abs/1505.07833>`_ with the :math:`\Lambda\mathrm{CDM}` corrections applied in `arXiv:1602.02154 <https://arxiv.org/abs/1602.02154>`_ ).

We provide a file in ``tests`` named ``test_nonlinear.py`` that compares the Halofit provided by CAMB with the one implemented in the classes :func:`nonlinear.halofit_operator` and :func:`nonlinear.nonlinear_pk`.

We first define a ``cosmo`` instance, without massive neutrinos to have a direct comparison with the Halofit implemented in CAMB:

.. code-block:: python

 import colibri.cosmology as cc
 import colibri.nonlinear as NL


 C = cc.cosmo(Omega_m = 0.3089,
              M_nu = 0.0,
              Omega_b = 0.0486,
              As = 2.14e-9,
              ns = 0.9667,
              h = 0.6774)

We compute at this point the power spectrum with three different methods

.. code-block:: python

 # 1) Compute the non-linear power spectrum with CAMB (use Mead halofit)
 k_camb, pk_camb = C.camb_Pk(k = kk, z = zz, nonlinear = True, halofit = 'mead')

 # 2) Use the `halofit_operator' class, which takes as arguments
 # Compute at first the linear power spectrum (in LCDM 'cb' and 'tot' is the same)
 k_l, pk_l    = C.camb_Pk(z = zz, k = kk, var_1 = 'cb', var_2 = 'cb')
 # Compute non-linear power spectrum
 do_nonlinear = NL.halofit_operator(z = zz, k = k_l, pk = pk_l,
                                    field = 'cb', BAO_smearing = False, cosmology = C)
 # Take spectra
 pk_hf        = do_nonlinear.pk_nl

 # 3) Use the `nonlinear_pk' module, which inherits the class `cosmo'
 # `nonlinear_pk' instance
 HF = NL.nonlinear_pk(k = kk, z = zz, code = 'camb', BAO_smearing = False,
                      kwargs_code = code_arguments, cosmology = C)
 # Get all results
 k          = HF.k          # Scales
 pk_nl      = HF.pk_nl      # Non-linear power spectrum
 pk_l       = HF.pk_l       # Linear power spectrum
 pk_nl_cbcb = HF.pk_nl_cbcb # cdm+b non-linear power spectrum
 pk_cbcb    = HF.pk_cbcb    # cdm+b power spectrum
 pk_cbnu    = HF.pk_cbnu    # cross power spectrum
 pk_nunu    = HF.pk_nunu    # neutrino power spectrum
 pk_nw      = HF.pk_nw      # no wiggles power spectrum
 pk_dw      = HF.pk_dw      # de-wiggled power spectrum

The comparison should look like the figure below, where the maximum deviation is below 1%.

.. image:: ../_static/nonlinear_spectrum.png
   :width: 700



