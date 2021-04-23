.. _fourier_test:

Fourier and Hankel analysis
======================================

Two files in the folder ``tests`` (namely ``test_fft.py`` and ``test_fftlog.py``) show some brief and documented example of how to use the functions contained in :ref:`fourier`.
Alternatively, in the folder ``notebooks`` the equivalent Python notebooks can be found.
Here we just want to show how to use the `FFTlog`-based functions :func:`colibri.fourier.iFFT_iso_3D` function to switch from Fourier space and compute correlation functions in real space.

We know that the two-point correlation function is the 3D Fourier transform of the power spectrum, i.e.

.. math::

 \xi(r) = \int \frac{d^3\mathbf{k}}{(2\pi)^3} \ P(k) \ e^{i \mathbf{k\cdot r}} = \int_0^\infty \frac{dk \ k^2}{2\pi^2} \ P(k) \ \frac{\sin kr}{kr}

where the second equality follows from statistical isotropy of the Universe.
Mathematically, the equation above is similar to an inverse Hankel transform of order 1/2 (with some coefficients in front).

To compute that, we just need a few lines

.. code-block:: python

 import numpy as np
 import colibri.cosmology as cc
 import matplotlib.pyplot as plt
 import colibri.fourier as FF

 C = cc.cosmo()

 # Linear matter power spectrum
 k, pk = C.camb_Pk(z = [0., 1., 2])

 # Fourier transforms for each redshift
 xi = []
 for i in range(len(pk)):
     r_tmp, xi_tmp = FF.iFFT_iso_3D(k, pk[i])
     xi.append(xi_tmp)

 # Save array of correlation functions
 r  = r_tmp
 xi = np.array(xi)

Here is the outcome: the BAO feature is clearly visible.

.. image:: ../_static/correlation_function.png
   :width: 700

