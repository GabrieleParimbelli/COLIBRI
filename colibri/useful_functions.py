import numpy as np
import scipy.interpolate as si
import scipy.fftpack as sfft
import scipy.optimize
import sys
from six.moves import xrange
try:   import fftlog
except ImportError: pass

#-----------------------------------------------------------------------------------------
# EXTRAPOLATE
#-----------------------------------------------------------------------------------------
def extrapolate(x, y, xmin, xmax, order):
	"""
	This function extrapolates a given function (`y`) defined in some points (`x`) to some other
	points external to the extension of `x` itself. The extrapolation is a power-law of an
	order which must be specified in input
	The points `x` must be linearly spaced, `xmin` and `xmax` must be smaller and greater than
	`x.min()` and `x.max()` respectively.

	:type x: array/list
	:param x: Abscissa of the function. Must be linearly spaced.

	:type y: array/list
	:param y: Ordinates (evaluated at `x`) of the function.

	:type xmin: float
	:param xmin: Minimum abscissa where to extend the array.

	:type xmax: float
	:param xmax: Maximum abscissa where to extend the array.

	:type order: float
	:param order: Order of the power-law to use.

	Returns
	-------

	x_ext: array
	    Extended/extrapolated abscissa.

	y_ext: array
	    Extended/extrapolated ordinate.

	"""
	# Step in linear-space
	assert np.allclose(np.diff(x), np.diff(x)[0], rtol = 1e-3), "'x' array not linearly spaced"
	dx   = np.diff(x)[0]

	# Linear interpolation in log-space (i.e. power law)		
	low_fit  = np.polyfit(x[:4] , y[:4] , order)
	high_fit = np.polyfit(x[-4:], y[-4:], order)

	# New arrays to which extrapolate
	x_low  = np.arange(xmin, x[1], dx)
	y_low  = np.polyval(low_fit, x_low)
	x_high = np.arange(x[-1] + dx, xmax, dx) 
	y_high = np.polyval(high_fit, x_high)

	# Concatenating the arrays. These are the 'k' and the 'P(k)' arrays I will use to compute sigma^2
	x_ext = np.concatenate([x_low, x, x_high])
	y_ext = np.concatenate([y_low, y, y_high])

	return x_ext, y_ext


#-----------------------------------------------------------------------------------------
# EXTRAPOLATE LOG
#-----------------------------------------------------------------------------------------
def extrapolate_log(x, y, xmin, xmax):
	"""
	This function extrapolates a given function (`y`) defined in some points (`x`) to some other
	points external to the extension of `x` itself. The extrapolation is a power-law of an
	order which must be specified in input
	The points `x` must be log-spaced, `xmin` and `xmax` must be smaller and greater than
	`x.min()` and `x.max()` respectively.

	:type x: array/list
	:param x: Abscissa of the function. Must be linearly spaced.

	:type y: array/list
	:param y: Ordinates (evaluated at `x`) of the function.

	:type xmin: float
	:param xmin: Minimum abscissa where to extend the array.

	:type xmax: float
	:param xmax: Maximum abscissa where to extend the array.

	Returns
	-------

	x_ext: array
	    Extended/extrapolated abscissa.

	y_ext: array
	    Extended/extrapolated ordinate.
	"""
	# Step in log-space
	assert np.allclose(np.diff(np.log(x)), np.diff(np.log(x))[0], rtol = 1e-3), "'x' array not log-spaced"
	dx   = x[1]/x[0]
	dlnx = np.log(dx)

	# Linear interpolation in log-space (i.e. power law)		
	low_fit  = np.polyfit(np.log(x[:2]) , np.log(y[:2]) , 1)
	high_fit = np.polyfit(np.log(x[-2:]), np.log(y[-2:]), 1)

	# New arrays to which extrapolate
	lnx_low  = np.arange(np.log(xmin), np.log(x[0]), dlnx)
	lny_low  = np.polyval(low_fit, lnx_low)
	lnx_high = np.arange(np.log(x[-1]*dx), np.log(xmax), dlnx) 
	lny_high = np.polyval(high_fit, lnx_high)

	# Switching to lin-space instead of log-space
	x_low  = np.exp(lnx_low)
	y_low  = np.exp(lny_low)
	x_high = np.exp(lnx_high)
	y_high = np.exp(lny_high)

	# Concatenating the arrays. These are the 'k' and the 'P(k)' arrays I will use to compute sigma^2
	x_ext = np.concatenate([x_low, x, x_high])
	y_ext = np.concatenate([y_low, y, y_high])

	return x_ext, y_ext


#-------------------------------------------------------------------------------
# NEUTRINO MASSES
#-------------------------------------------------------------------------------
def neutrino_masses(M_nu, hierarchy = 'normal'):
	"""
	Value of neutrino masses according to particle physics and the Solar Neutrino Experiment.
	Taken from `Pylians <https://github.com/franciscovillaescusa/Pylians>`_ codes by Francisco Villaescusa-Navarro.

	:type M_nu: float
	:param M_nu: Value of the sum of neutrino masses (in :math:`eV`).

	:type hierarchy: string, default = `'normal'`
	:param hierarchy: Set the neutrino hierarchy.

	 - `'normal'`, `'Normal'`, `'NH'`, `'N'`, `'n'` for normal hierarchy.
	 - `'inverted'`, `'Inverted'`, `'IH'`, `'I'`, `'i'` for inverted hierarchy.
	 - `'degenerate'`, `'Degenerate'`, `'DH'`, `'deg'`, `'D'`, `'d'` for degenerate hierarchy.

	Returns
	----------

	m1, m2, m3: values of the three neutrino masses (in :math:`eV`).
	"""
	# Difference of square masses
	delta21 = 7.5e-5
	delta31 = 2.45e-3

	# Minimum masses for NH, IH
	M_NH_min = np.sqrt(delta21)+np.sqrt(delta31)
	M_IH_min = np.sqrt(delta31)+np.sqrt(delta21+delta31)

	# Proceed depending on hierarchy
	if   hierarchy in ['normal', 'Normal', 'NH', 'N', 'n']:
		if M_nu<M_NH_min:
			raise ValueError('Normal hierarchy non allowed for M_nu = %.4f eV' %M_nu)
		else:
			m1_fun = lambda x: M_nu - x - np.sqrt(delta21+x**2) - np.sqrt(delta31+x**2)
			m1 = scipy.optimize.brentq(m1_fun, 0.0, M_nu)
			m2 = np.sqrt(delta21+m1**2)
			m3 = np.sqrt(delta31+m1**2)

	elif hierarchy in ['inverted', 'Inverted', 'IH', 'I', 'i']:
		if M_nu<M_IH_min:
			raise ValueError('Inverted hierarchy non allowed for M_nu = %.4f eV' %M_nu)
		else:
			m3_fun = lambda x: M_nu - x - np.sqrt(delta31+x**2) - np.sqrt(delta21+np.sqrt(delta31+x**2)**2)
			m3 = scipy.optimize.brentq(m3_fun, 0.0, M_nu)
			m1 = np.sqrt(delta31+m3**2)
			m2 = np.sqrt(delta21+m1**2)

	elif hierarchy in ['degenerate', 'Degenerate', 'DH', 'deg', 'D', 'd']:
		m1, m2, m3 = M_nu/3., M_nu/3., M_nu/3.

	else:
		raise NameError("Hierarchy not recognized")

	return m1, m2, m3

#-------------------------------------------------------------------------------
# TOP-HAT WINDOW FUNCTION IN FOURIER SPACE
#-------------------------------------------------------------------------------
def TopHat_window(x):
	"""
	Top-hat window function in Fourier space.

	:param x: Abscissa
	:type x: array

	:return: array
	"""
	return 3./(x)**3*(np.sin(x)-x*np.cos(x))


#-------------------------------------------------------------------------------
# SMOOTH
#-------------------------------------------------------------------------------
def smooth(y, box_pts):
	"""
	This routine smooths an array of a certain range of points.

	:type y: array
	:param y: Array to smooth.

	:type box_pts: int
	:param box_pts: Number of points over which to smooth.

	:return: array
	"""
	box = np.ones(box_pts)/box_pts
	y_smooth = np.convolve(y, box, mode='same')
	return y_smooth

#-------------------------------------------------------------------------------
# NYQUIST FREQUENCY
#-------------------------------------------------------------------------------
def Nyquist_frequency(boxsize, grid):
	"""
	This routine returns the Nyquist frequency of a cosmological box where the density field is computed with a grid of a certain size.

	:type boxsize: float
	:param boxsize: Size of the cubic box in :math:`\mathrm{Mpc}/h`.

	:type grid: int
	:param grid: Thickness of grid.

	:return: float
	"""
	return np.pi/(boxsize/grid)


#-------------------------------------------------------------------------------
# FUNDAMENTAL FREQUENCY
#-------------------------------------------------------------------------------
def fundamental_frequency(boxsize):
	"""
	This routine returns the fundamental frequency of a cosmological box where the density field is computed with a grid of a certain size.

	:type boxsize: float
	:param boxsize: Size of the cubic box in :math:`\mathrm{Mpc}/h`.

	:return: float
	"""
	return 2.*np.pi/boxsize


#-------------------------------------------------------------------------------
# FULL SKY
#-------------------------------------------------------------------------------
def full_sky():
	"""
	Total square degrees in the full sky.

	:return: float
	"""
	return 4.*np.pi*(180./np.pi)**2.

#-------------------------------------------------------------------------------
# SKY FRACTION
#-------------------------------------------------------------------------------
def sky_fraction(area):
	"""
	Returns the sky fraction given the survey size in square degrees.

	:type area: float
	:param area: Survey area in square degrees.

	:return: float
	"""
	return area/full_sky()


#-----------------------------------------------------------------------------------------
# BARYON FEEDBACK
#-----------------------------------------------------------------------------------------
def feedback_suppression(k, z, log_Mc, eta_b, z_c):
	"""
	Suppression of the matter power spectrum according to the Baryon Correction Model
	(Schneider et al., 2015).

	.. warning::

	 This function also exists in the class :func:`colibri.cosmology.cosmo()`.

	:param k: Scales in units of :math:`h/\mathrm{Mpc}`.
	:type k: array

	:param z: Redshifts.
	:type z: array

	:param log_Mc: Feedback mass: all halos below the mass of 10.**log_Mc are stripped of their gas.
	:type log_Mc: float>12.1

	:param eta_b: Ratio between the thermal velocity of the gas and the escape velocity from the halo.
	:type eta_b: float

	:param z_c: Scale redshift of feedback.
	:type z_c: float


	:return: 2D array of shape ``(len(z), len(k))``
	"""
	K,Z = np.meshgrid(k,z)

	# Model is valid only for eta_b > 0
	if eta_b <= 0.: raise ValueError("eta_b must be grater than 0.")

	# Stellar component
	ks = 55.
	stellar = 1. + (K/ks)**2.
	
	# Baryon suppression
	B0 = 0.105*log_Mc - 1.27
	assert B0>0., "log_Mc must be grater than 12.096"
	B = B0*1./(1.+(Z/z_c)**2.5)

	k_g = 0.7*((1.-B)**4.)*eta_b**(-1.6)
	scale_ratio = K/k_g

	suppression = B/(1.+scale_ratio**3.)+(1.-B)

	return suppression*stellar

#-------------------------------------------------------------------------------
# WDM suppression
#-------------------------------------------------------------------------------
def WDM_suppression(k, z, M_wdm, Omega_cdm, h, nonlinear = False):
	"""
	Suppression of the matter power spectrum due to (thermal) warm dark matter. In the linear
	case, the formula by https://arxiv.org/pdf/astro-ph/0501562.pdf is followed;
	otherwise the formula by https://arxiv.org/pdf/1107.4094.pdf is used.
	The linear formula is an approximation strictly valid only at :math:`k < 5-10 \ h/\mathrm{Mpc}`.
	The nonlinear formula has an accuracy of 2% level at :math:`z < 3` and for masses larger than 0.5 keV.

	.. warning::

	 This function also exists in the class :func:`colibri.cosmology.cosmo()`, where ``Omega_cdm`` and ``h`` are set to the values fixed at initialization of the class.

	.. warning::

	 This function returns the total suppression in power. To obtain the suppression in the transfer function, take the square root of the output.

	:param k: Scales in units of :math:`h/\mathrm{Mpc}`.
	:type k: array

	:param z: Redshifts.
	:type z: array

	:param M_wdm: Mass of the warm dark matter particle in keV.
	:type M_wdm: float

	:param Omega_cdm: Total matter density parameter today
	:type Omega_cdm: float

	:param h: Hubble constant in units of 100 km/s/Mpc
	:type h: float

	:param nonlinear: Whether to return non-linear transfer function.
	:type nonlinear: boolean, default = False

	:return: 2D array of shape ``(len(z), len(k))``
	"""
	K,Z = np.meshgrid(k,z)
	if not nonlinear:
		alpha_linear = 0.049*M_wdm**(-1.11)*(Omega_cdm/0.25)**0.11*(h/0.7)**1.22 # Mpc/h
		nu           = 1.12
		return (1.+(alpha_linear*K)**(2.*nu))**(-10./nu)

	else:
		nu, l, s = 3., 0.6, 0.4
		alpha    = 0.0476*(1./M_wdm)**1.85*((1.+Z)/2.)**1.3 # Mpc/h
		return (1.+(alpha*K)**(nu*l))**(-s/nu)

