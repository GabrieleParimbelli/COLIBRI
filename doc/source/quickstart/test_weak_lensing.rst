.. _weak_lensing_test:

Generate weak lensing spectra
===================================

Here is show an example of how to compute shear power spectra using the class :func:`~colibri.weak_lensing.weak_lensing`.
We want to compute the shear power spectrum, which in a flat Universe is given by:

.. math::

 C^{(ij)}(\ell) = \int_{z_{min}}^{z_{max}} dz \ \frac{c}{H(z)} \ \frac{W^{(i)}(z) W^{(j)}(z)}{\chi^2(z)} \ P\left(k = \frac{\ell}{\chi(z)}, z\right),

where in an ideal case :math:`z_{min}=0` and :math:`z_{max}=\infty`, :math:`W^{(i)}(z)` is the window function for a given redshift bin and :math:`P(k,z)` is the total matter non-linear power spectrum.

Initialization
---------------

To start, initialize a :func:`~colibri.cosmology.cosmo` instance and assign it to a :func:`~colibri.weak_lensing.weak_lensing` instance.

.. code-block:: python

 import cosmology as cc
 import weak_lensing as wlc
 C = cc.cosmo()
 S = wlc.weak_lensing(cosmology = C, z_limits = (0., 5.))

The quantity ``z_limits`` tells the code which are the limits of integration.

Power spectrum
---------------

As we saw, there are two main ingredients for the shear power spectrum, namely the matter power spectrum and the window functions.

The former is loaded through the line

.. code-block:: python

 S.load_power_spectra(z = np.linspace(0., 5., 51),
                      k = np.logspace(-4., 2., 101),
                      code = 'camb', nonlinear = True)

that generates a 2D interpolated object.

If one wants to use an already computed power spectrum (that comes e.g. from simulations), this function must be called with the additional argument ``power_spectra``, set to some 2D array of shape ``(len(z), len(k))`` and in this case ``k`` and ``z`` must be the scales and redshifts at which that array is computed:

.. code-block:: python

 S.load_power_spectra(z = np.linspace(0., 5., 51),
                      k = np.logspace(-4., 2., 101),
                      power_spectra = pkz)	# 'pkz' must have a shape (51,101)

.. note::

 The previous step can also be skipped. When the function :func:`~colibri.weak_lensing.weak_lensing.shear_power_spectrum` is called, the power spectra will be loaded automatically using the default values of the :func:`~colibri.weak_lensing.weak_lensing.load_power_spectrum` or with arguments given in a dictionary ``kwargs_power_spectra``.

Window functions
------------------

The routine :func:`~colibri.weak_lensing.weak_lensing.load_window_functions` takes as argument a list of galaxies distributions, i.e. callable functions whose first argument must be redshift, and transforms them into interpolated objects which will be used to compute the shear power spectra.

As an example, we assume a distribution given by :func:`~colibri.weak_lensing.weak_lensing.euclid_distribution`, with 3 redshift bins with edges [0, 0.72], [0.72, 1.11], [1.11, 5].
The window functions will be loaded with the line

.. code-block:: python

 S.load_window_functions(galaxy_distributions = [[S.euclid_distribution, {'zmin': 0.00, 'zmax': 0.72}],
                                                 [S.euclid_distribution, {'zmin': 0.72, 'zmax': 1.11}],
                                                 [S.euclid_distribution, {'zmin': 1.11, 'zmax': 5.00}]])

As can be seen, the ``galaxy_distribution`` argument is a nested list: each element is a in turn list whose first element is a callable function and the second its arguments, oraganized in a dictionary.

Of course, users can define their own distribution function (the code normalizes it automatically), provided that the first argument of the function is redshift.

The code above generates two lists of length ``len(galaxy_distribution)`` of interpolated objects, ``self.window_function`` and ``self.window_function_IA``.

Shear power spectrum
---------------------

Finally, the shear power spectrum is computed with

.. code-block:: python

 ll = np.geomspace(2., 4.e4, 51)
 Cl = S.shear_power_spectrum(l = ll, IA = 'LA', kwargs_IA = {'A_IA': -1.3})

The ``l`` argument sets the multipoles at which the spectrum must be computed; the ``IA`` argument sets the intrinsic alignment model used, implemented with the arguments contained in ``kwargs_IA`` (if ``IA = None`` all the terms relative to intrinsic alignment are set to zero).
See the function :func:`~colibri.weak_lensing.weak_lensing.intrinsic_alignment_kernel` for all the relevant info.

The returned object is a dictionary that contains 3 keys: ``GG``, ``GI``, ``II`` that represent the cosmological signal, the cross spectrum with intrinsic alignment effect and the pure intrinsic alignment signal, respectively.
Each of these keys is a 3D array, in this case of shape ``(3, 3, 51)``, containing the quantity :math:`C^{(ij)}(\ell)`.

.. image:: ../_static/shear_spectrum.png
   :width: 700

Shear correlation functions
----------------------------

Equivalently, the two shear correlation functions can be computed with

.. code-block:: python

 theta = np.geomspace(1., 100., 51)
 xi_plus, xi_minus = S.shear_correlation_functions(theta = theta, IA = 'LA', kwargs_IA = {'A_IA': -1.3})


