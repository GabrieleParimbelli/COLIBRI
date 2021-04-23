.. _limber_test:

Generate weak lensing spectra
===================================

Here is show an example of how to compute shear and galaxy power spectra using the class :func:`colibri.limber.limber_3x2pt` .
In a flat Universe such spectra are given by:

.. math::

 C_{AB}^{(ij)}(\ell) = \int_{z_{min}}^{z_{max}} dz \ \frac{c}{H(z)} \ \frac{W_A^{(i)}(z) W_B^{(j)}(z)}{\chi^2(z)} \ P\left(k = \frac{\ell}{\chi(z)}, z\right),

where :math:`A` and :math:`B` can either be weak lensing or galaxy clustering. In an ideal case :math:`z_{min}=0` and :math:`z_{max}=\infty`, :math:`W_A^{(i)}(z)` is the window function for a given redshift bin and :math:`P(k,z)` is the total matter non-linear power spectrum.

Initialization
---------------

To start, initialize a :func:`colibri.cosmology.cosmo` instance and assign it to a :func:`colibri.limber.limber_3x2pt` instance.

.. code-block:: python

 import colibri.cosmology as cc
 import colibri.limber as LL
 C = cc.cosmo()
 S = LL.limber_3x2pt(cosmology = C, z_limits = [0.01, 5.])

The quantity ``z_limits`` tells the code which are the limits of integration.

Power spectrum
---------------

As we saw, there are three main ingredients for the power spectrum, namely the matter power spectrum, galaxy bias and the window functions.

The former two are loaded together through the following lines

.. code-block:: python

 # Scales and redshifts
 kk = np.geomspace(1e-4, 1e2, 301)
 zz = np.linspace(0., 5., 51)
 # Compute power spectrum table
 _, pkz = C.camb_Pk(z = zz, k = kk, nonlinear = True, halofit = 'mead2020') # 'pkz' must have a shape (51,301)
 # Compute galaxy bias table at same scales and redshifts
 ZZ,KK = np.meshgrid(zz,kk,indexing='ij')
 bkz = (1.+ZZ)**0.5
 # Load power spectrum and bias
 S.load_power_spectra(z = zz, k = kk, power_spectra = pkz, galaxy_bias = bkz)

where first we generate a table of power spectra and a galaxy bias. These must have the dimensions ``(len(z), len(k))`` and in this case ``k`` and ``z`` must be the scales and redshifts at which that array is computed:


Window functions
------------------

The window functions for galaxy clustering and weak lensing depend on the source galaxy distribution.
The routine :func:`colibri.limber.limber_3x2pt.load_window_functions` takes two arguments: a list or array of redshifts and the value of the (unnormalized) galaxy distributions evaluated at those redshifts.
The number of bins is computed from the shape of the given array.

As an example, we assume a distribution given by :func:`colibri.limber.limber_3x2pt.euclid_distribution`, with 3 redshift bins with edges [0, 0.72], [0.72, 1.11], [1.11, 5].
The window functions will be loaded with the line

.. code-block:: python

 # Load galaxy distributions
 bin_edges = [0.01, 0.72, 1.11, 5.00]
 nbins     = len(bin_edges)-1
 z_tmp     = np.linspace(S.z_min, S.z_max, 201) # Sample at least every 0.025
 nz_tmp    = [S.euclid_distribution(z=z_tmp,zmin = bin_edges[i], zmax = bin_edges[i+1]) for i in range(nbins)]
 # Window functions
 S.load_window_functions(z = z_tmp, nz = nz_tmp)


The code above generates two dictionaries of length ``nbins`` of interpolated objects ``self.window_function``.

Angular power spectra
-------------------------------

Finally, the shear power spectrum is computed with

.. code-block:: python

 ll     = np.geomspace(2., 1e4, 51)
 Cl     = S.limber_angular_power_spectra(l            = ll,
                                         do_WL        = True,
                                         do_IA        = True,
                                         do_GC        = True,
                                         A_IA = -1.3, beta_IA = 0., eta_IA = 0., lum_IA = 1.)

The ``l`` argument sets the multipoles at which the spectrum must be computed; ``do_WL``, ``do_IA``, ``do_GC`` are the three flags switching on/off weak lensing, intrinsic alignment and galaxy clustering.
The parameters ``A_IA``, ``beta_IA``, ``eta_IA``, ``lum_IA`` are keyword arguments for the intrinsic alignment term.
The full extended-non-linear alignment model is implemented, for all the relevant parameter and info, see :func:`colibri.limber.limber_3x2pt.intrinsic_alignment_kernel` .

The returned object is a dictionary that contains the following keys: ``gg``, ``gI``, ``II``, ``LL``, ``GL``, ``GG``.
The first three represent the cosmological signal of cosmic shear, the cross spectrum with intrinsic alignment effect, the pure intrinsic alignment signal, respectively.
The ``LL`` key is the sum of the previous three, ``GL`` is the galaxy-galaxy lensing signal and the ``GG`` is the galaxy clustering angular power spectrum.
Each of these keys is a 3D array, in this case of shape ``(3, 3, 51)``, containing the quantity :math:`C^{(ij)}(\ell)`.
With the settings above, ``GL`` and ``GG`` will be zero, since ``do_GC`` is set to ``False``

.. image:: ../_static/limber_spectrum.png
   :width: 700

Shear correlation functions
----------------------------

Equivalently, the two shear correlation functions can be computed with

.. code-block:: python

 theta = np.geomspace(1., 100., 51)  # in arcmin
 xi    = S.angular_correlation_functions(theta = theta, do_WL = True, do_IA = True, do_GC = True, A_IA = -1.3)



