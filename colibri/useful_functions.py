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
	lnx_low  = np.arange(np.log(xmin), np.log(x[1]), dlnx)
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

