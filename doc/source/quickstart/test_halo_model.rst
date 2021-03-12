.. _halo_model_test:

Testing the halo model
======================

A different approach to compute the non-linear matter power spectrum is provided by the halo model, which became popular at the beginning of the 2000.
This code implements all the useful functions for this purpose, using the Sheth-Tormen mass function.
Despite it returns a poor description of matter clustering, it can be easily expanded to account for galaxies and/or qualitatively predicting covariance matrices for cosmological observables (for a review see `arXiv:0206508 <https://arxiv.org/abs/astro-ph/0206508>`_ ).


Matter power spectrum
---------------------

Let us start defining a :func:`colibri.halo.halo` instance:


.. code-block:: python

 import colibri.cosmology as cc
 import colibri.halo as hc

 # Define a halo instance
 H = hc.halo(z = [0., 1., 2.],              # 3 redshifts
             k = np.logspace(-4., 1., 201), # Scales in h/Mpc
             code = 'camb',                 # Boltzmann code with which to compute linear P(k)
             BAO_smearing = False,          # Smooth BAO due to non-linearities
             cosmology = cc.cosmo())        # ``cosmo`` instance


The halo power spectrum is then computed with the following lines

.. code-block:: python

 # Load power spectrum according to the halo model
 H.halo_Pk(kwargs_mass_function = {'a': 0.707, 'p': 0.3},   # arguments to pass to the Sheth-Tormen mass function
           kwargs_concentration = {'c0': 9., 'b': 0.13})    # arguments to pass to the concentration parameter c0*(M/Mstar)**(-b)

 # Store 1-halo, 2-halo and total terms
 oneh = H.Pk['matter']['1-halo']       # array of shape (len(z),len(k))
 twoh = H.Pk['matter']['2-halo']       # array of shape (len(z),len(k))
 tot  = H.Pk['matter']['total halo']   # array of shape (len(z),len(k))

The comparison with CAMB Halofit (for our 3 different redshifts) is the following:

.. image:: ../_static/halo_spectrum.png
   :width: 700



Galaxy power spectrum: real space
---------------------------------

The galaxy power spectrum in real space is computed using the Halo Occupation Distribution (HOD) method.
In this pictures, halos are populated of galaxies depending exclusively on their mass.
Galaxies are divided into central (max. 1 per halo) and satellites (from 0 to infinity): the 1-halo and 2-halo terms are computed by counting pairs of galaxies in the same and different halos, respectively.

To compute the power spectrum, one must provide the average amount of galaxies per halo.
This is done in a few steps here.
First, we define a :func:`colibri.galaxy.galaxy` instance:

.. code-block:: python

 import colibri.cosmology as cc
 import colibri.galaxy as gc

 # Define a halo instance
 G = gc.galaxy(z = [0., 1., 2.],              # 3 redshifts
               k = np.logspace(-4., 2., 201), # Scales in h/Mpc
               code = 'camb',                 # Boltzmann code with which to compute linear P(k)
               BAO_smearing = False,          # Smooth BAO due to non-linearities
               cosmology = cc.cosmo())        # ``cosmo`` instance


Second, we give to the code the 'ingredients' for the HOD.
This is done through the function :func:`colibri.galaxy.galaxy.load_HOD`

.. code-block:: python

 G.load_HOD(kind_satellite   = G.power_law,
            kwargs_satellite = {'log_Mcut':[13., 12.5, 12.3], 'log_M1': [13., 13.4, 13.], 'alpha': [1., 1.5, 1.]},
            kind_central     = G.logistic_function,
            kwargs_central   = {'log_Mmin': [13., 12.4, 11.4], 'sigma_logM': [0.8, 0.5, 0.6]})

The arguments ``kind_satellite`` and ``kind_central`` are callable functions whose first argument `must be the halo mass` in units of :math:`M_\odot/h`.
In this case we have used the functions :func:`colibri.galaxy.galaxy.power_law` and :func:`colibri.galaxy.galaxy.logistic_function`.
The arguments ``kwargs_satellite`` and ``kwargs_central`` are dictionaries that contain all the remaining arguments to pass to the above functions.
Each value of each key must be a list of same size of the redshifts required, otherwise the code will return an ``AssertionError``.

The last step consists of the actual computation of the galaxy power spectrum, which is done like in the ``halo`` case:

.. code-block:: python

 G.galaxy_Pk(kwargs_mass_function = {'a': 0.707, 'p': 0.3},
             kwargs_concentration = {'c0': 9., 'b': 0.13})

 pg1 = G.Pk['galaxies']['real space']['1-halo']
 pg2 = G.Pk['galaxies']['real space']['2-halo']
 pgt = G.Pk['galaxies']['real space']['total halo']


.. note::

 The last two steps can be combined together.
 In fact, when the function :func:`colibri.galaxy.galaxy.galaxy_Pk` is called, it will search the quantities ``self.Ncen`` and ``self.Nsat`` which are generated only after the call of :func:`colibri.galaxy.galaxy.load_HOD`.
 If these are not found, the code will try to compute them on the fly, provided that the arguments of :func:`colibri.galaxy.galaxy.load_HOD` are given.
 In other words, the last two steps can be gathered in:

 .. code-block:: python

  G.galaxy_Pk(kind_satellite       = G.power_law,
              kwargs_satellite     = {'log_Mcut':[13., 12.5, 12.3], 'log_M1': [13., 13.4, 13.], 'alpha': [1., 1.5, 1.]},
              kind_central         = G.logistic_function,
              kwargs_central       = {'log_Mmin': [13., 12.4, 11.4], 'sigma_logM': [0.8, 0.5, 0.6]},
              kwargs_mass_function = {'a': 0.707, 'p': 0.3},
              kwargs_concentration = {'c0': 9., 'b': 0.13})

.. image:: ../_static/galaxy_spectrum.png
   :width: 700


Galaxy power spectrum: redshift space
-------------------------------------

Galaxies in surveys are observed in redshift-space due to the fact that we only measure the recession velocity along the line-of-sight.
The class :func:`colibri.RSD.RSD` provides routines able to compute the power spectrum in redshift-space using different configurations of independent variables:

 - in :math:`(k,\mu)` space
 - in multipole space, using the Legendre expansion
 - in :math:`(k_\parallel,k_\perp)` space


The ``RSD`` instance is called as follows:

.. code-block:: python

 import colibri.cosmology as cc
 import colibri.RSD as rsd

 Z = rsd.RSD(z            = [0., 1., 2.],                    # Redshifts
             k            = np.geomspace(0.0005, 10., 101),  # Scales in h/Mpc
             mu           = np.linspace(0., 1., 31),         # Cosine of angles with LOS
             k_par        = np.linspace(0.01, 1., 51),       # Scales parallel in h/Mpc
             k_perp       = np.linspace(0.01, 1., 31),       # Scales perpendicular in h/Mpc
             BAO_smearing = True,                            # Smooth BAO feature in non-linearities
             cosmology    = cc.cosmo())                      # Cosmology


With our 3 redshifts, we define the following useful quantities

.. code-block:: python

 # Galaxy biases
 bb = [1.30, 2.60, 3.34]    # Galaxy biases
 # Growth rates
 ff = C.Omega_m_z(zz)**0.55
 # Velocity dispersions (in km/s)
 sv = [200., 200., 200.]
 # HOD functions (like in ``galaxy``)
 HOD_central_kind, HOD_satellite_kind = Z.logistic_function, Z.power_law
 HOD_central_parameters   = {'log_Mmin': [12., 12.5, 12.], 'sigma_logM': [0.8, 0.5, 0.2]}
 HOD_satellite_parameters = {'log_Mcut': [13., 12.5, 12.], 'log_M1': [14., 13.2, 13.4], 'alpha': [1., 1.25, 1.5]}
 # Kind of damping functions for fingers-of-god effect
 FoG_damping = 'Lorentzian' # Kind of damping due to fingers-of-God (choose between 'Lorentzian' and 'Gaussian')
 # Base power spectrum is the non-linear matter one (computed with Halofit) (choose among 'linear', 'nonlinear', 'HOD', 'halo model')
 RSD_model = 'nonlinear'


At this point the galaxy redshift-space power spectrum can be computed in 3 ways:


 * in :math:`(k,\mu)` space

 .. code-block:: python

  Z.galaxy_RSD_Pk(bias                 = bb,                        # Galaxy bias (used only if model = 'HOD' or 'halo model')
                  growth_rate          = ff,                        # Growth rate f = dln(D)/dln(a)
                  velocity_dispersion  = sv,                        # Average velocity dispersion of galaxies in halos
                  model                = RSD_model,                 # Model to compute RSD
                  kwargs_mass_function = {'a': 0.707, 'p': 0.3},    # Parameters to compute halo mass function (used only if model = 'HOD' or 'halo model')
                  kwargs_concentration = {'c0': 9., 'b': 0.13},     # Parameters to compute concentration parameter (used only if model = 'HOD' or 'halo model')
                  fingers_of_god       = FoG_damping,               # Kind of damping ('Lorentzian' or 'Gaussian', used only if model != 'halo model')
                  kind_central         = HOD_central_kind,          # Function to compute central galaxies (1st arguments must be mass in Msun/h)
                  kwargs_central       = HOD_central_parameters,    # Remaining arguments to pass to kind_central
                  kind_satellite       = HOD_satellite_kind,        # Function to compute satellite galaxies (1st arguments must be mass in Msun/h)
                  kwargs_satellite     = HOD_satellite_parameters)  # Remaining arguments to pass to kind_satellite


 - in multipole space, using the Legendre expansion

 .. code-block:: python

  Z.galaxy_RSD_Pk_multipoles(l                    = [0,2,4]                     # Multipoles to compute (monopole, quadrupole, hexadecapole)
                             bias                 = bb,                         # Galaxy bias (used only if model = 'HOD' or 'halo model')
                             growth_rate          = ff,                         # Growth rate f = dln(D)/dln(a)
                             velocity_dispersion  = sv,                         # Average velocity dispersion of galaxies in halos
                             model                = RSD_model,                  # Model to compute RSD
                             kwargs_mass_function = {'a': 0.707, 'p': 0.3},     # Parameters to compute halo mass function (used only if model = 'HOD' or 'halo model')
                             kwargs_concentration = {'c0': 9., 'b': 0.13},      # Parameters to compute concentration parameter (used only if model = 'HOD' or 'halo model')
                             fingers_of_god       = FoG_damping,                # Kind of damping ('Lorentzian' or 'Gaussian', used only if model != 'halo model')
                             kind_central         = HOD_central_kind,           # Function to compute central galaxies (1st arguments must be mass in Msun/h)
                             kwargs_central       = HOD_central_parameters,     # Remaining arguments to pass to kind_central
                             kind_satellite       = HOD_satellite_kind,         # Function to compute satellite galaxies (1st arguments must be mass in Msun/h)
                             kwargs_satellite     = HOD_satellite_parameters)   # Remaining arguments to pass to kind_satellite



 - in :math:`(k_\parallel,k_\perp)` space

 .. code-block:: python

  Z.galaxy_RSD_Pk_2D(bias                 = bb,                        # Galaxy bias (used only if model = 'HOD' or 'halo model')
                     growth_rate          = ff,                        # Growth rate f = dln(D)/dln(a)
                     velocity_dispersion  = sv,                        # Average velocity dispersion of galaxies in halos
                     model                = RSD_model,                 # Model to compute RSD
                     kwargs_mass_function = {'a': 0.707, 'p': 0.3},    # Parameters to compute halo mass function (used only if model = 'HOD' or 'halo model')
                     kwargs_concentration = {'c0': 9., 'b': 0.13},     # Parameters to compute concentration parameter (used only if model = 'HOD' or 'halo model')
                     fingers_of_god       = FoG_damping,               # Kind of damping ('Lorentzian' or 'Gaussian', used only if model != 'halo model')
                     kind_central         = HOD_central_kind,          # Function to compute central galaxies (1st arguments must be mass in Msun/h)
                     kwargs_central       = HOD_central_parameters,    # Remaining arguments to pass to kind_central
                     kind_satellite       = HOD_satellite_kind,        # Function to compute satellite galaxies (1st arguments must be mass in Msun/h)
                     kwargs_satellite     = HOD_satellite_parameters)  # Remaining arguments to pass to kind_satellite


The result should be the following:

.. image:: ../_static/rsd_spectrum.png
   :width: 700









