import colibri.constants as const
import colibri.cosmology as cc
import colibri.halo as hc
import numpy as np
import scipy


# class galaxy: is a 'halo' subclass

class galaxy (hc.halo):
	"""
	The ``galaxy`` class inherits all the functions from :func:`colibri.halo.halo` and has the final
	goal to compute the galaxy power spectrum in the Halo Occupation Distribution (HOD)
	prescription.
	As such, the class contains some routies to compute the average number of central/
	satellite galaxies per halo.
	The call of a ``galaxy`` instance takes the following arguments:

	:type z: array
	:param z: Redshifts.

	:param k: Array of scales in :math:`h/\mathrm{Mpc}`.
	:type k: array

	:param code: Boltzmann solver to compute the linear power spectrum. Choose among `'camb'`, `'class'`, `'eh'` (for Eisenstein-Hu). N.B. If Eisenstein-Hu is selected, effects of massive neutrinos and evolving dark energy cannot be accounted for, as such spectrum is a good approximation for LCDM cosmologies only.
	:type code: string, default = `'camb'`

	:param BAO_smearing: Whether to damp the BAO feature due to non-linearities.
	:type BAO_smearing: boolean, default = True

	:param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
	:type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``


	:return: The initialization automatically calls a ``halo`` instance (see :func:`colibri.halo.halo`) and therefore all the quantities described there are also available here. Also, the key ``['galaxies']['real space']`` is added to the ``self.Pk`` dictionary.

	"""

	def __init__(self,
			z,
			k,
			code = 'camb',
			BAO_smearing = False,
			cosmology = cc.cosmo()):

		# Initialize halo class (with linear matter power spectrum)
		hc.halo.__init__(self, z = z, k = k, code = code, BAO_smearing = BAO_smearing, cosmology = cosmology)

		# Initialize galaxy power spectrum
		self.Pk['galaxies']               = {}		
		self.Pk['galaxies']['real space'] = {}

	#-----------------------------------------------------------------------------------------
	# EXAMPLES OF HALO OCCUPATION DISTRIBUTION
	#-----------------------------------------------------------------------------------------

	def logistic_function(self, M, log_Mmin, sigma_logM):
		"""
		Example function of HOD for central galaxies in a halo of mass `M`. A step-like function with a transition region of a certain width.

		.. math::

			N_{cen}(M) = \\frac{1}{2} \left[1+\mathrm{erf}\left(\\frac{\log M - \log M_{min}}{\sigma_{\log M}}\\right)\\right]


		:type M: array
		:param M: Masses in :math:`M_\odot/h`

		:type log_Mmin: float
		:param log_Mmin: Mass for which the probability of having a central galaxy in a halo of a certain mass is 1/2.

		:type sigma_logM: float
		:param sigma_logM: Width of the transition from 0 to 1 central galaxy in a halo.

		:return: array   
		"""
		arg = np.log10(M/10.**log_Mmin)/sigma_logM
		galaxies = 0.5*(1+scipy.special.erf(arg))
		return galaxies

	def step_function(self, M, logMmin):
		"""
		Example function of HOD for central galaxies in a halo of mass `M`: a step function.

		:type M: array
		:param M: Masses in :math:`M_\odot/h`

		:type log_Mmin: float
		:param log_Mmin: Mass for which the probability of having a central galaxy in a halo of a certain mass is 1/2.

		:return: array   
		"""
		return self.logistic_function(M, log_Mmin = log_Mmin, sigma_logM = 1e-5)

	def power_law(self, M, log_Mcut, log_M1, alpha):
		"""
		Typical HOD for satellite galaxies. A power-law with a cutoff here is used.

		.. math::

			N_{sat}(M) = \left(\\frac{M-M_{cut}}{M_1}\\right)^\\alpha

		:type M: array
		:param M: Masses in :math:`M_\odot/h`

		:type log_Mcut: float
		:param log_Mcut: Cutoff mass below which there are no satellite galaxies.

		:type log_M1: float
		:param log_M1: Scale mass for satellite galaxies in a halo.

		:type alpha: float
		:param alpha: Exponent of the power law.

		:return: array 
		"""
		M1   = 10.**log_M1
		Mcut = 10.**log_Mcut
		M    = np.array([M])
		ii   = M > Mcut
		sat  = M*0
		sat[ii] = ((M[ii]-Mcut)/M1)**alpha
		return sat

	def power_law_step(self, M, log_M1, alpha):
		"""
		Typical HOD for satellite galaxies. No low-mass cutoff is used.

		.. math::

			N_{sat}(M) = \left(\\frac{M}{M_1}\\right)^\\alpha

		:type M: array
		:param M: Masses in :math:`M_\odot/h`

		:type log_M1: float
		:param log_M1: Scale mass for satellite galaxies in a halo.

		:type alpha: float
		:param alpha: Exponent of the power law.

		:return: array 
		"""
		M1   = 10.**log_M1
		return (M/M1)**alpha

	#-----------------------------------------------------------------------------------------
	# NUMBER OF GALAXIES: HALO OCCUPATION DISTRIBUTION
	#-----------------------------------------------------------------------------------------
	def load_HOD(self, kind_central = None, kind_satellite = None, kwargs_central = {}, kwargs_satellite = {}):
		"""
		This method loads the number of central and satellite galaxies for every redshift. It is an essential ingredient of the halo model for galaxies. One can load it here or while calling  the :func:`colibri.galaxy.galaxy.galaxy_Pk` function.

		:type kind_central: callable, default = None
		:param kind_central: First argument must be mass (in :math:`M_\odot/h`), other specified by kwargs_central.

		:type kind_satellite: callable, default = None
		:param kind_satellite: First argument must be mass (in :math:`M_\odot/h`), other specified by kwargs_satellite.

		:type kwargs_central: dictionary
		:param  kwargs_central: Every key is a keyword parameter for the galaxy distribution and each value is a list of length equal to the size of the redshift required.

		:type kwargs_satellite: dictionary
		:param kwargs_satellite: Every key is a keyword parameter for the galaxy distribution and each value is a list of length equal to the size of the redshift required.

		:return: Nothing, but it loads the quantities:

		 - ``self.Ncen`` (`array`) - Number of central galaxies evaluated at ``self.mass``
		 - ``self.Nsat`` (`array`) - Number of satellite galaxies evaluated at ``self.mass``
		 - ``self.Ntot`` (`array`) - Number of total galaxies evaluated at ``self.mass``.

		N.B. To avoid having satellite galaxies in halos which do not have a central, the number of total galaxies is computed in the following way:

		.. math::

		 N_{tot}(M) = N_{cen}(M) \left(1+N_{sat}(M)\\right)

		"""
		# Initialize N_central and N_satellite
		self.Ncen = np.zeros((self.nz, self.nm))
		self.Nsat = np.zeros((self.nz, self.nm))

		keys_cen, values_cen = kwargs_central.keys(),   np.atleast_1d(list(kwargs_central.values())).T
		keys_sat, values_sat = kwargs_satellite.keys(), np.atleast_1d(list(kwargs_satellite.values())).T

		# Check parameters are of right shape
		# If more than one redshift
		if self.nz != 1:
			assert values_cen.shape == ((self.nz, len(keys_cen))), "Parameters for central are of wrong shape."
			assert values_sat.shape == ((self.nz, len(keys_sat))), "Parameters for central are of wrong shape."
		# Otherwise just "promote" one more dimension to avoid troubles with indices
		else:
			values_cen = [values_cen]
			values_sat = [values_sat]

		# Fill the number of galaxies for every redshift
		for iz in range(self.nz):

			# Create a new dictionary for every redshift
			new_dict_cen = dict(zip(keys_cen, list(values_cen[iz])))
			new_dict_sat = dict(zip(keys_sat, list(values_sat[iz])))

			# Fill the arrays
			self.Ncen[iz] = kind_central  (self.mass, **new_dict_cen)
			self.Nsat[iz] = kind_satellite(self.mass, **new_dict_sat)*self.Ncen[iz]


		self.Ntot = self.Ncen + self.Nsat

	#-----------------------------------------------------------------------------------------
	# AVERAGE NUMBER OF GALAXIES
	#-----------------------------------------------------------------------------------------
	def average_galaxy_density(self, **kwargs):
		"""
		Average number of galaxies given a HOD model at all the redshifts required. This function can be called only after having called :func:`colibri.galaxy.galaxy.load_HOD`.

		:param kwargs: Keyword arguments to pass to :func:`colibri.halo.halo.mass_fun_ST`

		:return: array of shape ``len(z)`` in units of :math:`h^3 \mathrm{Mpc}^{-3}`
		"""
		M       = self.mass
		dlnM    = np.log(M[1]/M[0])
		density = self.load_halo_mass_function(**kwargs)
		
		integrand = density*self.Ntot*M
		n_bar = np.trapz(integrand, x = np.log(M))
		return n_bar

	#-----------------------------------------------------------------------------------------
	# GALAXY POWER SPECTRUM
	#-----------------------------------------------------------------------------------------
	def galaxy_Pk(self,
				  kind_central = None,
				  kwargs_central = {},
				  kind_satellite = None,
				  kwargs_satellite = {},
				  kwargs_mass_function = {},
				  kwargs_concentration = {}):
		"""
		It computes the galaxy power spectrum in the HOD prescription along with the 1-halo and 2-halo terms.

		.. math::

		  P_g^{(1h)}(k) = \\frac{1}{\\bar{n_g}^2} \int_0^\infty dM \ \\frac{dn}{dM} \ \left[2 \ N_{cen}(M) \ N_{sat}(M) \ u(k|M) + N_{sat}^2(M) \ u^2(k,M)\\right]

		.. math::

		  P_g^{(2h)}(k) = \left[\\frac{1}{\\bar{n_g}} \int_0^\infty dM \ \\frac{dn}{dM} \ b(M) \ N_{tot}(M) \ u(k,M) \\right]^2 P_{lin}(k)

		:type kind_central: callable, default = None
		:param kind_central: First argument must be mass (in :math:`M_\odot/h`), other specified by kwargs_central.

		:type kind_satellite: callable, default = None
		:param kind_satellite: First argument must be mass (in :math:`M_\odot/h`), other specified by kwargs_satellite.

		:type kwargs_central: dictionary
		:param  kwargs_central: Every key is a keyword parameter for the galaxy distribution and each value is a list of length equal to the size of the redshift required.

		:type kwargs_satellite: dictionary
		:param kwargs_satellite: Every key is a keyword parameter for the galaxy distribution and each value is a list of length equal to the size of the redshift required.

		:param kwargs_mass_function: Keyword arguments to pass to :func:`colibri.halo.halo.mass_fun_ST`.
		:type kwargs_mass_function: dictionary, default = {}

		:param kwargs_concentration: Keyword arguments to pass to :func:`colibri.halo.halo.conc`.
		:type kwargs_concentration: dictionary, default = {}

		:return: Nothing, but the following keys are added to the ``self.Pk`` dictionary

		 - ``['galaxies']['real space']['1-halo']`` (`2D array of shape` ``(len(z), len(k))`` ) - 1-halo term of the matter power spectrum
		 - ``['galaxies']['real space']['2-halo']`` (`2D array of shape` ``(len(z), len(k))`` ) - 2-halo term of the matter power spectrum
		 - ``['galaxies']['real space']['total halo']`` (`2D array of shape` ``(len(z), len(k))`` ) - Sum of 1-halo and 2-halo terms.

		"""

		# Defining all the quantities needed
		nu      = self.nu()
		bias    = self.halo_bias_ST(nu, **kwargs_mass_function)
		M       = self.mass
		dlnM    = np.log(M[1]/M[0])
		dndM    = self.load_halo_mass_function(**kwargs_mass_function)
		k       = self.k		
		try:
			Nc      = self.Ncen
			Ns      = self.Nsat
		except AttributeError:
			self.load_HOD(kind_central = kind_central, kind_satellite = kind_satellite, kwargs_central = kwargs_central, kwargs_satellite = kwargs_satellite)
			Nc      = self.Ncen
			Ns      = self.Nsat
		n_avg   = self.average_galaxy_density(**kwargs_mass_function)
		r_s     = self.R_s(M)
		c       = self.conc(M, **kwargs_concentration)
		nfw     = np.zeros((self.nz, self.nk, self.nm))
		for iz in range(self.nz):
			for i in range(self.nk):
				nfw[iz, i] = self.u_NFW(c[iz], k[i]*r_s[iz])
		power   = self.Pk['matter']['linear']

		# normalization and scale of cutoff
		normalization = self.norm_2h(bias, **kwargs_mass_function)
		k_star        = 0.01*(1.+self.z)

		# Filling power spectrum array
		P_g_1h = np.zeros_like(power)
		P_g_2h = np.zeros_like(power)

		for iz in range(self.nz):
			for ik in range(self.nk):
				integrand_1h  = (1./n_avg[iz]**2.*dndM[iz]*(2.*Nc[iz]*Ns[iz]*nfw[iz,ik]+Ns[iz]**2.*nfw[iz,ik]**2))*M
				integrand_2h  = (1./n_avg[iz]*dndM[iz]*bias[iz]*(Nc[iz]+Ns[iz]*nfw[iz,ik]))*M
				P_g_1h[iz,ik] = np.trapz(integrand_1h, dx = dlnM)*(1-np.exp(-(self.k[ik]/k_star[iz])**2.))
				P_g_2h[iz,ik] = np.trapz(integrand_2h, dx = dlnM)**2.*normalization[iz]**2*power[iz,ik]

		# Galaxy power spectrum dictionary
		self.Pk['galaxies']['real space']['1-halo']     = P_g_1h
		self.Pk['galaxies']['real space']['2-halo']     = P_g_2h
		self.Pk['galaxies']['real space']['total halo'] = P_g_1h + P_g_2h

		del P_g_1h, P_g_2h




