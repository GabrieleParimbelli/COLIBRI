import colibri.constants as const
import colibri.cosmology as cc
import numpy as np
import scipy.special as ss
import scipy.interpolate as si
import scipy.integrate as sint
import scipy.optimize as so
import colibri.fourier as FF
from six.moves import xrange
import colibri.nonlinear as NL
from math import sqrt

#==================
# CLASS LIMBER
#==================
class limber():
    """
    The class :func:`colibri.limber.limber` contains all the functions useful to compute
    the angular power spectra and correlation functions in the flat sky and Limber's approximation.
    It also contains routines to compute widely-used window functions as well as a routine to add custom ones.
    At initialization it takes as inputs a redshift range for integration and a
    :func:`colibri.cosmology.cosmo` instance.


    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``

    :param z_limits: Lower and upper limit of integration along the line of sight. Both numbers must be non-negative and the first number must be smaller than the second. If the lower limit is set to 0, it will be enhanced by 1e-10 to avoid divergences at the origin of the lightcone.
    :type z_limits: 2-uple or list/array of length 2, default = (0., 5.)


    .. warning::

     All the power spectra are computed in the Limber approximation and the window function are assumed to be dependent only on redshift and not on scales (see e.g. :func:`colibri.limber.limber.load_lensing_window_functions`). Typically the scale dependence of the window functions can be factorized out (e.g. ISW effect, different orders of cosmological perturbation theory...) and in this code it can be added to the power spectrum (see :func:`colibri.limber.limber.load_power_spectra`).

    """

    #-----------------------------------------------------------------------------------------
    # INITIALIZATION FUNCTION
    #-----------------------------------------------------------------------------------------
    def __init__(self,
                 z_limits = (0., 5.),
                 cosmology = cc.cosmo()):

        # Cosmology
        self.cosmology = cosmology

        # Redshifts
        assert len(z_limits) == 2, "Limits of integration must be a 2-uple or a list of length 2, with z_min at 1st place and z_max at 2nd place"
        assert z_limits[0] < z_limits[1], "z_min (lower limit of integration) must be smaller than z_max (upper limit)"

        # Minimum and maximum redshifts
        self.z_min = z_limits[0]
        self.z_max = z_limits[1]

        # Remove possible infinity at z = 0.
        if self.z_min == 0.: self.z_min += 1e-10    # Remove singularity of 1/chi(z) at z = 0

        # Set the array of integration (set number of redshift so that there is dz = 0.125)
        self.dz_min = 0.0625
        self.nz_min = int((self.z_max-self.z_min)/self.dz_min+1)
        self.nz_integration = int((self.z_max - self.z_min)*self.nz_min + 2)
        self.z_integration  = np.linspace(self.z_min, self.z_max, self.nz_integration)

        # Array of redshifts for computing window integrals (set number of redshift so that there is an integral at least each dz = 0.025)
        self.dz_windows = 0.025
        self.z_windows  = np.arange(self.z_min, self.z_max+self.dz_windows, self.dz_windows)
        self.nz_windows = len(np.atleast_1d(self.z_windows))

        # Distances (in Mpc/h)
        self.geometric_factor         = self.geometric_factor_f_K(self.z_integration)
        self.geometric_factor_windows = self.geometric_factor_f_K(self.z_windows)

        # Hubble parameters (in km/s/(Mpc/h))
        self.Hubble         = self.cosmology.H_massive(self.z_integration)/self.cosmology.h
        self.Hubble_windows = self.cosmology.H_massive(self.z_windows)    /self.cosmology.h

        # Factor c/H(z)/f_K(z)^2
        self.c_over_H_over_chi_squared = const.c/self.Hubble/self.geometric_factor**2.

        # Initialize window functions
        self.window_function = {}


    #-----------------------------------------------------------------------------------------
    # LOAD_PK
    #-----------------------------------------------------------------------------------------
    def load_power_spectra(self, k, z, power_spectra):
        """
        This routine interpolates the total matter power spectrum (using the CDM prescription) in scales (units of :math:`h/\mathrm{Mpc}`) and redshifts. `power_spectra` and `galaxy_bias` must be a 2D array of shape ``(len(z), len(k))`` which contains the power spectrum (in units of :math:`(\mathrm{Mpc}/h)^3`) and galaxy bias evaluated at the scales and redshifts specified above.

        :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
        :type k: array

        :param z: Redshifts at which power spectrum must be/is computed.
        :type z: array

        :param power_spectra: It must be a 2D array of shape ``(len(z), len(k))`` which contains the power spectrum (in units of :math:`(\mathrm{Mpc}/h)^3`) evaluated at the scales and redshifts specified above.
        :type power_spectra: 2D NumPy array

        :return: Nothing, but two 2D-interpolated object ``self.power_spectra_interpolator`` and ``self.galaxy_bias_interpolator`` containing :math:`P(k,z)` in units of :math:`(\mathrm{Mpc}/h)^3` and :math:`b(k,z)` are created
        """

        # Select scales and redshifts
        self.k     = np.atleast_1d(k)
        self.z     = np.atleast_1d(z)
        self.k_min = k.min()
        self.k_max = k.max()

        # Interpolate
        kind_of_interpolation = 'cubic' if (len(self.z)>3) and (len(self.k)>3) else 'linear'
        self.power_spectra_interpolator = si.interp2d(self.k, self.z,
                                                      power_spectra,
                                                      kind_of_interpolation,
                                                      bounds_error = False, fill_value = 0.)

    #-----------------------------------------------------------------------------------------
    # EXAMPLES OF GALAXY PDF'S
    #-----------------------------------------------------------------------------------------
    def euclid_distribution(self, z, zmin, zmax, a = 2.0, b = 1.5, z_med = 0.9, step = 5e-3):
        """
        Example function for the distribution of source galaxy. This distribution in particular is expected to be used in the Euclid mission:

        .. math::

            n(z) \propto z^a \ \exp{\left[-\left(\\frac{z}{z_{med}/\sqrt 2}\\right)^b\\right]}

        This distribution will eventually be normalized such that its integral on all redshifts is 1.

        :param z: Redshifts.
        :type z: array

        :param zmin: Lower edge of the bin (a small width of 0.005 will be applied for convergence reasons).
        :type zmin: float

        :param zmax: Upper edge of the bin (a small width of 0.005 will be applied for convergence reasons).
        :type zmax: float

        :param a: Parameter of the distribution.
        :type a: float, default = 2.0

        :param b: Parameter of the distribution.
        :type b: float, default = 1.5

        :param z_med: Median redshift of the distribution.
        :type z_med: float, default = 0.9

        :param step: width of the cutoff (better to avoid a sharp one for numerical reasons, better set it to be at least 0.001)
        :type step: float, default = 0.005

        :return: array
        """
        # from median redshift to scale-redshift
        z_0 = z_med/sqrt(2.)
        # Heaviside-like function
        lower = 0.5*(1.+np.tanh((z-zmin)/step))
        upper = 0.5*(1.+np.tanh((zmax-z)/step))
        # Galaxy distribution
        n = (z/z_0)**a*np.exp(-(z/z_0)**b)*lower*upper
        return n

    def euclid_distribution_with_photo_error(self, z, zmin, zmax, a = 2.0, b = 1.5, z_med = 0.9, f_out = 0.1, c_b = 1.0, z_b = 0.0, sigma_b = 0.05, c_o = 1.0, z_o = 0.1, sigma_o = 0.05):
        """
        Example function for the distribution of source galaxy. This distribution in particular is expected to be used in the Euclid mission. Here also the effect of photometric errors is included.

        .. math::

         n^{(i)}(z) \propto \int_{z_i^-}^{z_i^+} dy \ z^a \ \exp{\left[-\left(\\frac{z}{z_{med}/\sqrt 2}\\right)^b\\right]} \ p_\mathrm{ph}(y|z)

        where

        .. math::


         p_\mathrm{ph}(y|z) = \\frac{1-f_\mathrm{out}}{\sqrt{2\pi}\sigma_b(1+z)} \ \exp\left[-\\frac{1}{2} \left(\\frac{z-c_b y -z_b}{\sigma_b(1+z)}\\right)^2\\right] +

         + \\frac{f_\mathrm{out}}{\sqrt{2\pi}\sigma_o(1+z)} \ \exp\left[-\\frac{1}{2} \left(\\frac{z-c_o y -z_o}{\sigma_o(1+z)}\\right)^2\\right]

        :param z: Redshifts.
        :type z: array

        :param zmin: Lower edge of the bin.
        :type zmin: float

        :param zmax: Upper edge of the bin.
        :type zmax: float

        :param a: Parameter of the distribution.
        :type a: float, default = 1.5

        :param b: Parameter of the distribution.
        :type b: float, default = 1.5

        :param z_med: Median redshift of the distribution.
        :type z_med: float, default = 0.9

        :param f_out: Fraction of outliers
        :type f_out: float, default = 0.1

        :param c_b: Parameter of the Gaussian (normalization) representing the uncertainty on the photometric error for in-liers.
        :type c_b: float, default = 1.0

        :param z_b: Parameter of the Gaussian (scale-redshift) representing the uncertainty on the photometric error for in-liers.
        :type z_b: float, default = 0.0

        :param sigma_b: Parameter of the Gaussian (width) representing the uncertainty on the photometric error for in-liers.
        :type sigma_b: float, default = 0.05

        :param c_o: Parameter of the Gaussian (normalization) representing the uncertainty on the photometric error for out-liers.
        :type c_o: float, default = 1.0

        :param z_o: Parameter of the Gaussian (scale-redshift) representing the uncertainty on the photometric error for out-liers.
        :type z_o: float, default = 0.1

        :param sigma_o: Parameter of the Gaussian (width) representing the uncertainty on the photometric error for out-liers.
        :type sigma_o: float, default = 0.05

        :return: array
        """
        # from median redshift to scale-redshift
        z_0       = z_med/sqrt(2.)
        gal_distr = (z/z_0)**a*np.exp(-(z/z_0)**b)
        # Photometric error function
        distr_in  = (1.-f_out)/(2.*c_b)*(ss.erf((z - c_b*zmin - z_b)/(sqrt(2.)*sigma_b*(1.+z)))-ss.erf((z - c_b*zmax - z_b)/(sqrt(2.)*sigma_b*(1.+z))))
        distr_out =    (f_out)/(2.*c_o)*(ss.erf((z - c_o*zmin - z_o)/(sqrt(2.)*sigma_o*(1.+z)))-ss.erf((z - c_o*zmax - z_o)/(sqrt(2.)*sigma_o*(1.+z))))
        photo_err_func = distr_in + distr_out
        return photo_err_func*gal_distr

    def gaussian_distribution(self, z, mean, sigma):
        """
        Example function for the distribution of source galaxy. Here we use a Gaussian galaxy distribution

        :param z: Redshifts.
        :type z: array

        :param mean: Mean redshift of the distribution.
        :type mean: float

        :param sigma: Width of the Gaussian
        :type sigma: float

        :return: array
        """
        exponent = -0.5*((z-mean)/sigma)**2.
        return np.exp(exponent)


    def constant_distribution(z, zmin, zmax, step = 5e-3):
        """
        Example function for the distribution of source galaxy. Here we use a constant distribution of sources.

        :param z: Redshifts.
        :type z: array

        :param zmin: Lower edge of the bin.
        :type zmin: float

        :param zmax: Upper edge of the bin.
        :type zmax: float

        :param step: width of the cutoff (better to avoid a sharp one for numerical reasons, better set it to be at least 0.001)
        :type step: float, default = 0.005

        :return: array
        """
        # Heaviside-like function
        lower = 0.5*(1.+np.tanh((z-zmin)/step))
        upper = 0.5*(1.+np.tanh((zmax-z)/step))
        # Galaxy distribution
        n = z**0.*lower*upper
        return n


    #-------------------------------------------------------------------------------
    # COMOVING DISTANCE
    #-------------------------------------------------------------------------------
    def comoving_distance(self, z, z0 = 0.):
        """
        Comoving distance between two redshifts. It assumes neutrinos as matter, which is a good
        approximation at low redshifts. In fact, this latter assumption introduces a bias of less
        than 0.02% at :math:`z<10` for even the lowest neutrino masses allowed by particle physics.

        :param z: Redshifts.
        :type z: array

        :param z0: Pivot redshift.
        :type z0: float, default = 0

        :return: array
        """
        c = const.c
        z = np.atleast_1d(z)
        length = len(z)
        result = np.zeros(length)
        for i in xrange(length):
            result[i], _ = sint.quad(lambda x: c*1./(self.cosmology.H_massive(x)/self.cosmology.h), z0, z[i], epsabs = 1e-8)

        return result

    #-------------------------------------------------------------------------------
    # GEOMETRIC FACTOR
    #-------------------------------------------------------------------------------
    def geometric_factor_f_K(self, z, z0 = 0.):
        """
        Geometric factor (distance) between two given redshifts ``z`` and ``z0``.
        It assumes neutrinos as matter, which is a good approximation at low redshifts.
        In fact, this latter assumption introduces a bias of less than 0.02% at :math:`z<10`
        for even the lowest neutrino masses allowed by particle physics.

        :param z: Redshifts.
        :type z: array

        :param z0: Pivot redshift.
        :type z0: float, default = 0

        :return: array
        """
        # Curvature in (h/Mpc)^2 units. Then I will take the sqrt and it will
        # go away with comoving_distance(z), giving a final result in units of Mpc/h
        K = self.cosmology.K
        chi_z = self.comoving_distance(z, z0)
        # Change function according to sign of K
        if K == 0.:
            return chi_z #Mpc/h
        elif K > 0.:
            return 1./K**0.5*np.sin(K**0.5*chi_z) #Mpc/h
        else:
            return 1./np.abs(K)**0.5*np.sinh(np.abs(K)**0.5*chi_z) #Mpc/h

    #-----------------------------------------------------------------------------------------
    # SHEAR WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_shear_window_functions(self, z, nz, name = 'shear'):
        """
        This function computes the window function for cosmic shear given the galaxy distribution in input.
        The function automatically normalizes the galaxy distribution such that the integral over
        redshifts is 1.
        The routine adds a key (specified in the ``name`` argument) to the ``self.window_function`` dictionary.
        Given a galxy distruibution in a redshift bin :math:`n^{(i)}(z)`, the equation is:

        .. math::

           W^{(i)}_\gamma(z) = \\frac{3}{2}\Omega_m \ \\frac{H_0^2}{c^2} \ f_K[\chi(z)] (1+z) \int_z^\infty dx \ n^{(i)}(x) \ \\frac{f_K[\chi(z-x)]}{f_K[\chi(z)]}


        :param z: array or list of redshift at which the galaxy distribution ``nz`` is evaluated
        :type z: 1-D array, default = None

        :param nz: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
        :type nz: 2-D array with shape ``(n_bins, len(z))``, default = None

        :param name: name of the key to add to the dictionary
        :type name: string, default = 'shear'

        An example call can be, for 3 bins all with a :func:`colibri.limber.limber.euclid_distribution` with default arguments for ``a`` and ``b`` but different bin edges ``zmin``, ``zmax``:

        .. code-block:: python

           bin_edges = [0.00, 0.72, 1.11, 5.00]
           nbins     = len(bin_edges)-1
           z_w       = np.linspace(0., 6., 1001)
           nz_w      = [S.euclid_distribution(z = z_w, a = 2.0, b = 1.5, zmin = bin_edges[i], zmax = bin_edges[i+1]) for i in range(nbins)]
           S.load_shear_window_functions(z = z_w, nz = nz_w)

        :return: A key of a given name is added to the ``self.window_function`` dictionary

        """
        nz = np.array(nz)
        z  = np.array(z)
        assert np.all(np.diff(z)<self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" %(self.dz_windows)
        assert nz.ndim == 2, "'nz' must be 2-dimensional" 
        assert (nz.shape)[1] == z.shape[0], "Length of each 'nz[i]' must be the same of 'z'"
        assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" %(self.z_min)
        assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" %(self.z_max)

        # Call a simpler function if Omega_K == 0.
        if self.cosmology.Omega_K == 0.:
            self.load_shear_window_functions_flat(z,nz,name)
        # Otherwise compute window function in curved geometry
        else:
            # Set number of bins, normalize them, find constant in front
            n_bins = len(nz)
            norm_const = sint.simps(nz, x = z, axis = 1)
            constant = 3./2.*self.cosmology.Omega_m*(self.cosmology.H0/self.cosmology.h/const.c)**2.*(1.+self.z_windows)*self.geometric_factor_windows

            # Initialize windows
            self.window_function[name]  = []

            # Set windows
            for galaxy_bin in xrange(n_bins):
                # Select which is the function and which are the arguments and do the integral for window function
                n_z = si.interp1d(z,nz[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
                integral = list(map(lambda z_i: sint.quad(lambda x: n_z(x)*self.geometric_factor_f_K(x,z_i)/self.geometric_factor_f_K(x), z_i, self.z_max, epsrel = 1.e-3)[0], self.z_windows))
                # Fill temporary window functions with real values
                window_function_tmp    = constant*integral/norm_const[galaxy_bin]
                # Interpolate (the Akima interpolator avoids oscillations around the zero due to the cubic spline)
                try:
                    self.window_function[name].append(si.interp1d(self.z_windows, window_function_tmp, 'cubic', bounds_error = False, fill_value = 0.))
                except ValueError:
                    self.window_function[name].append(si.Akima1DInterpolator(self.z_windows, window_function_tmp))

    def load_shear_window_functions_flat(self, z, nz, name = 'shear'):
        # Set number of bins, normalize them, find constant in front
        n_bins = len(nz)
        norm_const = sint.simps(nz, x = z, axis = 1)
        constant = 3./2.*self.cosmology.Omega_m*(self.cosmology.H0/self.cosmology.h/const.c)**2.*(1.+self.z_windows)*self.geometric_factor_windows

        # Initialize window
        self.window_function[name]  = []
        # Set windows
        chi_max = self.cosmology.comoving_distance(self.z_max)
        for galaxy_bin in xrange(n_bins):
            # Select which is the function and which are the arguments
            tmp_interp = si.interp1d(z,nz[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
            n_z_array  = tmp_interp(self.z_windows)
            n_z_interp = si.interp1d(self.geometric_factor_windows, n_z_array*(self.Hubble_windows/const.c), 'cubic', bounds_error = False, fill_value = 0.)
            # Do the integral for window function
            integral = list(map(lambda chi_i: sint.quad(lambda chi: n_z_interp(chi)*(1.-chi_i/chi), chi_i, chi_max, epsrel = 1.e-3)[0], self.geometric_factor_windows))
            # Fill temporary window functions with real values
            window_function_tmp    = constant*integral/norm_const[galaxy_bin]
            # Interpolate (the Akima interpolator avoids oscillations around the zero due to the cubic spline)
            try:
                self.window_function[name].append(si.interp1d(self.z_windows, window_function_tmp, 'cubic', bounds_error = False, fill_value = 0.))
            except ValueError:
                self.window_function[name].append(si.Akima1DInterpolator(self.z_windows, window_function_tmp))


    #-----------------------------------------------------------------------------------------
    # INTRINSIC ALIGNMENT WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_IA_window_functions(self, z, nz, A_IA = 1.0, eta_IA = 0.0, beta_IA = 0.0, lum_IA=1.0, name = 'IA'):
        """
        This function computes the window function for intrinsic alignment given a galaxy distribution.
        The function automatically normalizes the galaxy distribution such that the integral over
        redshifts is 1.
        The routine adds a key (specified in the ``name`` argument) to the ``self.window_function`` dictionary.
        Given a galxy distruibution in a redshift bin :math:`n^{(i)}(z)`, the equation is:

        .. math::

           W^{(i)}_\mathrm{IA}(z) = -\\frac{A_\mathrm{IA} \mathcal C_1 \Omega_\mathrm m}{D_1(k,z)}(1+z)^{\eta_\mathrm{IA}} \left[\\frac{L(z)}{L_*(z)}\\right]^{\\beta_\mathrm{IA}} \ n^{(i)}(z) \\frac{H(z)}{c}


        :param z: array or list of redshift at which the galaxy distribution ``nz`` is evaluated
        :type z: 1-D array, default = None

        :param nz: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
        :type nz: 2-D array with shape ``(n_bins, len(z))``, default = None

        :param A_IA: Intrinsic alignment amplitude.
        :type A_IA: float, default = 1

        :param eta_IA: Exponent for redshift dependence of intrinsic alignment.
        :type eta_IA: float, default = 0

        :param beta_IA: Exponent for luminosity dependence of intrinsic alignment.
        :type beta_IA: float, default = 0

        :param lum_IA: Relative luminosity of galaxies :math:`L(z)/L_*(z)`.
        :type lum_IA: float or callable whose **only** argument is :math:`z`, default = 1

        :param name: name of the key to add to the dictionary
        :type name: string, default = 'IA'

        An example call can be, for 3 bins all with a :func:`colibri.limber.limber.euclid_distribution` with default arguments for ``a`` and ``b`` but different bin edges ``zmin``, ``zmax``:

        .. code-block:: python

           bin_edges = [0.00, 0.72, 1.11, 5.00]
           nbins     = len(bin_edges)-1
           z_w       = np.linspace(0., 6., 1001)
           nz_w      = [S.euclid_distribution(z = z_w, a = 2.0, b = 1.5, zmin = bin_edges[i], zmax = bin_edges[i+1]) for i in range(nbins)]
           S.load_IA_window_functions(z = z_w, nz = nz_w, A_IA = 1, eta_IA = 0, beta_IA = 0, lum_IA = 1)

        :return: A key of a given name is added to the ``self.window_function`` dictionary

        """
        nz = np.array(nz)
        z  = np.array(z)
        n_bins = len(nz)
        norm_const = sint.simps(nz, x = z, axis = 1)
        assert np.all(np.diff(z)<self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" %(self.dz_windows)
        assert nz.ndim == 2, "'nz' must be 2-dimensional" 
        assert (nz.shape)[1] == z.shape[0], "Length of each 'nz[i]' must be the same of 'z'"
        assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" %(self.z_min)
        assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" %(self.z_max)
        # Initialize window
        self.window_function[name] = []
        # IA kernel
        F_IA = self.intrinsic_alignment_kernel(self.z_integration,A_IA,eta_IA,beta_IA,lum_IA)
        # Compute window
        for galaxy_bin in xrange(n_bins):
            tmp_interp = si.interp1d(z,nz[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
            self.window_function[name].append(si.interp1d(self.z_integration, tmp_interp(self.z_integration)*F_IA*self.Hubble/const.c/norm_const[galaxy_bin], 'cubic', bounds_error = False, fill_value = 0.))

    #-----------------------------------------------------------------------------------------
    # LENSING WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_lensing_window_functions(self, z, nz, A_IA = 0.0, eta_IA = 0.0, beta_IA = 0.0, lum_IA=1.0, name = 'lensing'):
        """
        This function computes the window function for lensing (comprehensive of shear and intrinsic
        alignment) given a galaxy distribution.
        The function automatically normalizes the galaxy distribution such that the integral over
        redshifts is 1.
        The routine adds a key (specified in the ``name`` argument) to the ``self.window_function`` dictionary.
        See :func:`colibri.limber.limber.load_shear_window_functions` 
        and :func:`colibri.limber.limber.load_IA_window_functions` for the equations.


        :param z: array or list of redshift at which the galaxy distribution ``nz`` is evaluated
        :type z: 1-D array, default = None

        :param nz: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
        :type nz: 2-D array with shape ``(n_bins, len(z))``, default = None

        :param A_IA: Intrinsic alignment amplitude.
        :type A_IA: float, default = 1

        :param eta_IA: Exponent for redshift dependence of intrinsic alignment.
        :type eta_IA: float, default = 0

        :param beta_IA: Exponent for luminosity dependence of intrinsic alignment.
        :type beta_IA: float, default = 0

        :param lum_IA: Relative luminosity of galaxies :math:`L(z)/L_*(z)`.
        :type lum_IA: float or callable whose **only** argument is :math:`z`, default = 1

        :param name: name of the key to add to the dictionary
        :type name: string, default = 'lensing'

        An example call can be, for 3 bins all with a :func:`colibri.limber.limber.euclid_distribution` with default arguments for ``a`` and ``b`` but different bin edges ``zmin``, ``zmax``:

        .. code-block:: python

           bin_edges = [0.00, 0.72, 1.11, 5.00]
           nbins     = len(bin_edges)-1
           z_w       = np.linspace(0., 6., 1001)
           nz_w      = [S.euclid_distribution(z = z_w, a = 2.0, b = 1.5, zmin = bin_edges[i], zmax = bin_edges[i+1]) for i in range(nbins)]
           S.load_lensing_window_functions(z = z_w, nz = nz_w, A_IA = 1, eta_IA = 0, beta_IA = 0, lum_IA = 1)

        :return: A key of a given name is added to the ``self.window_function`` dictionary

        """
        # Compute shear and IA
        self.load_shear_window_functions(z,nz,name='shear_temporary_for_lensing')
        self.load_IA_window_functions(z,nz,A_IA,eta_IA,beta_IA,lum_IA,name='IA_temporary_for_lensing')
        n_bins = len(self.window_function['shear_temporary_for_lensing'])

        # Initialize window
        self.window_function[name] = []
        for galaxy_bin in xrange(n_bins):
            WL = self.window_function['shear_temporary_for_lensing'][galaxy_bin](self.z_windows)+self.window_function['IA_temporary_for_lensing'][galaxy_bin](self.z_windows)
            try:
                self.window_function[name].append(si.interp1d(self.z_windows, WL, 'cubic', bounds_error = False, fill_value = 0.))
            except ValueError:
                self.window_function[name].append(si.Akima1DInterpolator(self.z_windows, WL))   
        del self.window_function['shear_temporary_for_lensing']
        del self.window_function['IA_temporary_for_lensing']

    #-----------------------------------------------------------------------------------------
    # GALAXY CLUSTERING WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_galaxy_clustering_window_functions(self, z, nz, bias = 1.0, name = 'galaxy'):
        """
        This function computes the window function for galaxy clustering given a galaxy distribution.
        The function automatically normalizes the galaxy distribution such that the integral over
        redshifts is 1.
        The routine adds a key (specified in the ``name`` argument) to the ``self.window_function`` dictionary.
        Given a galxy distruibution in a redshift bin :math:`n^{(i)}(z)`, the equation is:

        .. math::

           W^{(i)}_\mathrm{G}(z) = b(z) \ n^{(i)}(z) \\frac{H(z)}{c}


        :param z: array or list of redshift at which the galaxy distribution ``nz`` is evaluated
        :type z: 1-D array, default = None

        :param nz: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
        :type nz: 2-D array with shape ``(n_bins, len(z))``, default = None

        :param bias: Galaxy bias.
        :type bias: float or array, same length of ``nz``, default = 1

        :param name: name of the key to add to the dictionary
        :type name: string, default = 'galaxy'


        An example call can be, for 3 bins all with a :func:`colibri.limber.limber.euclid_distribution` with default arguments for ``a`` and ``b`` but different bin edges ``zmin``, ``zmax``:

        .. code-block:: python

           bin_edges = [0.00, 0.72, 1.11, 5.00]
           nbins     = len(bin_edges)-1
           z_w       = np.linspace(0., 6., 1001)
           nz_w      = [S.euclid_distribution(z = z_w, a = 2.0, b = 1.5, zmin = bin_edges[i], zmax = bin_edges[i+1]) for i in range(nbins)]
           S.load_galaxy_clustering_window_functions(z = z_w, nz = nz_w, bias = 1)

        :return: A key of a given name is added to the ``self.window_function`` dictionary

        """
        nz = np.array(nz)
        z  = np.array(z)
        n_bins = len(nz)
        norm_const = sint.simps(nz, x = z, axis = 1)
        assert np.all(np.diff(z)<self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" %(self.dz_windows)
        assert nz.ndim == 2, "'nz' must be 2-dimensional" 
        assert (nz.shape)[1] == z.shape[0], "Length of each 'nz[i]' must be the same of 'z'"
        assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" %(self.z_min)
        assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" %(self.z_max)

        if   isinstance(bias, float): bias = bias*np.ones(n_bins)
        elif isinstance(bias, int)  : bias = np.float(bias)*np.ones(n_bins)
        else:                         assert len(bias)==n_bins, "Number of bias factors different from number of bins"
        # Initialize window
        self.window_function[name] = []
        # Compute window
        for galaxy_bin in xrange(n_bins):
            tmp_interp = si.interp1d(z,nz[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
            self.window_function[name].append(si.interp1d(self.z_integration, tmp_interp(self.z_integration)*self.Hubble/const.c/norm_const[galaxy_bin]*bias[galaxy_bin], 'cubic', bounds_error = False, fill_value = 0.))

    #-----------------------------------------------------------------------------------------
    # HI WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_HI_window_functions(self, z, nz, Omega_HI = 6.25e-4, bias = 1., name = 'HI'):
        """
        This function computes the window function for HI brightness temperature given a galaxy distribution.
        The function automatically normalizes the galaxy distribution such that the integral over
        redshifts is 1.
        The routine adds a key (specified in the ``name`` argument) to the ``self.window_function`` dictionary.
        Given a galxy distruibution in a redshift bin :math:`n^{(i)}(z)`, the equation is:

        .. math::

           W^{(i)}_\mathrm{HI}(z) = b(z) \ T_b(z) \ D(z) \ \ n^{(i)}(z) \\frac{H(z)}{c}


        :param z: array or list of redshift at which the galaxy distribution ``nz`` is evaluated
        :type z: 1-D array, default = None

        :param nz: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
        :type nz: 2-D array with shape ``(n_bins, len(z))``, default = None

        :param Omega_HI: HI density parameter.
        :type Omega_HI: float, default = 6.25e-4

        :param bias: Galaxy bias.
        :type bias: float or array, same length of ``nz``, default = 1

        :param name: name of the key to add to the dictionary
        :type name: string, default = 'HI'


        An example call can be, for 3 bins all with a :func:`colibri.limber.limber.euclid_distribution` with default arguments for ``a`` and ``b`` but different bin edges ``zmin``, ``zmax``:

        .. code-block:: python

           bin_edges = [0.00, 0.72, 1.11, 5.00]
           nbins     = len(bin_edges)-1
           z_w       = np.linspace(0., 6., 1001)
           nz_w      = [S.euclid_distribution(z = z_w, a = 2.0, b = 1.5, zmin = bin_edges[i], zmax = bin_edges[i+1]) for i in range(nbins)]
           S.load_HI_window_functions(z = z_w, nz = nz_w, Omega_HI = 0.00063, bias = 1)

        :return: A key of a given name is added to the ``self.window_function`` dictionary

        """

        nz = np.array(nz)
        z  = np.array(z)
        n_bins = len(nz)
        norm_const = sint.simps(nz, x = z, axis = 1)
        assert np.all(np.diff(z)<self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" %(self.dz_windows)
        assert nz.ndim == 2, "'nz' must be 2-dimensional" 
        assert (nz.shape)[1] == z.shape[0], "Length of each 'nz[i]' must be the same of 'z'"
        assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" %(self.z_min)
        assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" %(self.z_max)

        if   isinstance(bias, float): bias = bias*np.ones(n_bins)
        elif isinstance(bias, int)  : bias = np.float(bias)*np.ones(n_bins)
        else:                         assert len(bias)==n_bins, "Number of bias factors different from number of bins"
        # Initialize window
        self.window_function[name] = []
        # Compute window
        Dz = self.cosmology.growth_factor_scale_independent(self.z_integration)
        Tz = self.brightness_temperature_HI(self.z_integration,Omega_HI)
        for galaxy_bin in xrange(n_bins):
            tmp_interp = si.interp1d(z,nz[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
            self.window_function[name].append(si.interp1d(self.z_integration, tmp_interp(self.z_integration)*self.Hubble/const.c/norm_const[galaxy_bin]*bias[galaxy_bin]*Tz*Dz, 'cubic', bounds_error = False, fill_value = 0.))

    #-----------------------------------------------------------------------------------------
    # CMB LENSING WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_CMB_lensing_window_functions(self, z, nz, z_LSS = 1089., name = 'CMB lensing'):
        """
        This function computes the window function for CMB lensing given a galaxy distribution.
        The function automatically normalizes the galaxy distribution such that the integral over
        redshifts is 1.
        The routine adds a key (specified in the ``name`` argument) to the ``self.window_function`` dictionary.
        Given a galxy distruibution in a redshift bin :math:`n^{(i)}(z)`, the equation is:

        .. math::

           W^{(i)}_\mathrm{CMB}(z) = \\frac{3}{2}\Omega_m \ \\frac{H_0^2}{c^2} \ f_K[\chi(z)] (1+z) \ n^{(i)}(z) \\frac{H(z)}{c} \ \\frac{f_K[\chi(z_{LSS})]-f_K[\chi(z)]}{f_K[\chi(z_{LSS})]}


        :param z: array or list of redshift at which the galaxy distribution ``nz`` is evaluated
        :type z: 1-D array, default = None

        :param nz: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
        :type nz: 2-D array with shape ``(n_bins, len(z))``, default = None

        :param z_LSS: last-scattering surface redshift.
        :type z_LSS: float, default = 1089

        :param name: name of the key to add to the dictionary
        :type name: string, default = 'CMB lensing'


        An example call can be, for 3 bins all with a :func:`colibri.limber.limber.euclid_distribution` with default arguments for ``a`` and ``b`` but different bin edges ``zmin``, ``zmax``:

        .. code-block:: python

           bin_edges = [0.00, 0.72, 1.11, 5.00]
           nbins     = len(bin_edges)-1
           z_w       = np.linspace(0., 6., 1001)
           nz_w      = [S.euclid_distribution(z = z_w, a = 2.0, b = 1.5, zmin = bin_edges[i], zmax = bin_edges[i+1]) for i in range(nbins)]
           S.load_CMB_window_functions(z = z_w, nz = nz_w, z_LSS = 1089.)

        :return: A key of a given name is added to the ``self.window_function`` dictionary

        """
        nz = np.array(nz)
        z  = np.array(z)
        n_bins = len(nz)
        norm_const = sint.simps(nz, x = z, axis = 1)
        constant = 3./2.*self.cosmology.Omega_m*(self.cosmology.H0/self.cosmology.h/const.c)**2.*(1.+self.z_windows)*self.geometric_factor_windows
        assert np.all(np.diff(z)<self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" %(self.dz_windows)
        assert nz.ndim == 2, "'nz' must be 2-dimensional" 
        assert (nz.shape)[1] == z.shape[0], "Length of each 'nz[i]' must be the same of 'z'"
        assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" %(self.z_min)
        assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" %(self.z_max)

        # Initialize window
        self.window_function[name] = []
        
        # Comoving distance to last scattering surface
        com_dist_LSS = self.geometric_factor_f_K(z_LSS)
        # Comoving distances to redshifts
        for galaxy_bin in xrange(n_bins):
            tmp_interp = si.interp1d(z,nz[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
            self.window_function[name].append(si.interp1d(self.z_windows, constant*tmp_interp(self.z_windows)/norm_const[galaxy_bin]*self.Hubble_windows/const.c*(com_dist_LSS-self.geometric_factor_windows)/com_dist_LSS, 'cubic', bounds_error = False, fill_value = 0.))


    #-----------------------------------------------------------------------------------------
    # ADD WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_custom_window_functions(self, z, window, name):
        """
        This function loads a custom window function and adds the key to the dictionary
        The window function in input must already be normalized.

        :param z: array or list of redshift at which the galaxy distribution ``nz`` is evaluated
        :type z: 1-D array, default = None

        :param window: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
        :type window: 2-D array with shape ``(n_bins, len(z))``, default = None

        :param name: name of the key to add to the dictionary
        :type name: string

        :return: A key of a given name is added to the ``self.window_function`` dictionary

        """
        nz = np.array(window)
        z  = np.array(z)
        n_bins = len(nz)  
        assert np.all(np.diff(z)<self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" %(self.dz_windows)
        assert nz.ndim == 2, "'nz' must be 2-dimensional" 
        assert (nz.shape)[1] == z.shape[0], "Length of each 'nz[i]' must be the same of 'z'"
        assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" %(self.z_min)
        assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" %(self.z_max)   
         # Initialize window
        self.window_function[name] = []
        # Compute window
        for galaxy_bin in xrange(n_bins):
            self.window_function[name].append(si.interp1d(z,window[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.))

    #-----------------------------------------------------------------------------------------
    # CORRECTION FUNCTION FOR INTRINSIC ALIGNMENT
    #-----------------------------------------------------------------------------------------
    def intrinsic_alignment_kernel(self, z, A_IA = 0., eta_IA = 0., beta_IA = 0., lum_IA = 1.):
        # Constants in front
        C1     = 0.0134
        front  = -A_IA*C1*self.cosmology.Omega_m
        # Growth factors
        growth = self.cosmology.growth_factor_scale_independent(z = z)
        # Relative luminosity is either a function or a float
        if   callable(lum_IA):          rel_lum = lum_IA(z)
        elif isinstance(lum_IA, float): rel_lum = lum_IA
        else:                           raise TypeError("'lum_IA' must be either a float or a function with redshift as the only argument.")
        return front/growth*(1.+z)**eta_IA*rel_lum**beta_IA

    #-----------------------------------------------------------------------------------------
    # BRIGHTNESS TEMPERATURE OF HI
    #-----------------------------------------------------------------------------------------
    def brightness_temperature_HI(self,z,Omega_HI):
        TB0        = 44. # micro-K
        Hz         = self.cosmology.H_massive(z)
        Omega_HI_z = Omega_HI*(1.+z)**3.*self.cosmology.H0**2./Hz**2.
        return TB0*Omega_HI_z*self.cosmology.h/2.45e-4*(1.+z)**2.*self.cosmology.H0/Hz


    #-----------------------------------------------------------------------------------------
    # ANGULAR SPECTRA
    #-----------------------------------------------------------------------------------------
    def limber_angular_power_spectra(self, l, windows = None):
        """
        This function computes the angular power spectra (using the Limber's and the flat-sky approximations) for the window function specified.
        Given two redshift bins `i` and `j` the equation is

        .. math::

          C^{(ij)}(\ell) = \int_0^\infty dz \ \\frac{c}{H(z)} \ \\frac{W^{(i)}(z) W^{(j)}(z)}{f_K^2[\chi(z)]} \ P\left(\\frac{\ell}{f_K[\chi(z)]}, z\\right),

        where :math:`P(k,z)` is the matter power spectrum and :math:`W^{(i)}(z)` are the window functions.

        :param l: Multipoles at which to compute the shear power spectra.
        :type l: array

        :param windows: which spectra (auto and cross) must be computed. If set to ``None`` all the spectra will be computed.
        :type windows: list of strings, default = ``None``

        :return: dictionary whose keys are combinations of window functions specified in ``windows``. Each key is a 3-D array whose entries are ``Cl[bin i, bin j, multipole l]``
        """

        # Check existence of power spectrum
        try: self.power_spectra_interpolator
        except AttributeError: raise AttributeError("Power spectra have not been loaded yet")

        # Check convergence with (l, k, z):
        assert np.atleast_1d(l).min() > self.k_min*self.geometric_factor_f_K(self.z_min), "Minimum 'l' is too low. Extend power spectra to lower k_min? Use lower z_min for power spectrum?"
        assert np.atleast_1d(l).max() < self.k_max*self.geometric_factor_f_K(self.z_max), "Maximum 'l' is too high. Extend power spectra to higher k_max? Use higher z_max for power spectrum?"

        # Check window functions to use
        if windows is None:
            windows_to_use = self.window_function
        else:
            windows_to_use = {}
            for key in windows:
                try:             windows_to_use[key] = self.window_function[key]
                except KeyError: raise KeyError("Requested window function '%s' not known" %key)

        # Check window functions 
        keys   = windows_to_use.keys()
        if len(keys) == 0: raise AttributeError("No window function has been computed!")
        nkeys  = len(keys)
        n_bins = [len(windows_to_use[key]) for key in keys]

        # 1) Define lengths and quantities
        zz       = self.z_integration
        n_l      = len(np.atleast_1d(l))
        n_z      = self.nz_integration
        HH       = self.Hubble
        cH_chi2  = self.c_over_H_over_chi_squared
        Cl       = {}
        for i,ki in enumerate(keys):
            for j,kj in enumerate(keys):
                Cl['%s-%s' %(ki,kj)] = np.zeros((n_bins[i],n_bins[j], n_l))

        # 2) Load power spectra
        power_spectra = self.power_spectra_interpolator
        PS_lz = np.zeros((n_l, n_z))
        for il in xrange(n_l):
            for iz in range(n_z):
                PS_lz[il,iz] = power_spectra(l[il]/self.geometric_factor[iz], zz[iz])

        # 3) load Cls given the source functions
        # 1st key (from 1 to N_keys)
        for index_X in xrange(nkeys):
            key_X = list(keys)[index_X]
            W_X = np.array([windows_to_use[key_X][i](zz) for i in range(n_bins[index_X])])
            # 2nd key (from 1st key to N_keys)
            for index_Y in xrange(index_X,nkeys):
                key_Y = list(keys)[index_Y]
                W_Y = np.array([windows_to_use[key_Y][i](zz) for i in range(n_bins[index_Y])])
                # Symmetry C_{AA}^{ij} == C_{AA}^{ji}
                if key_X == key_Y:
                    for bin_i in xrange(n_bins[index_X]):
                        for bin_j in xrange(bin_i, n_bins[index_Y]):
                            Cl['%s-%s' %(key_X,key_Y)][bin_i,bin_j] = [sint.simps(cH_chi2*W_X[bin_i]*W_Y[bin_j]*PS_lz[xx], x = zz) for xx in range(n_l)]
                            Cl['%s-%s' %(key_X,key_Y)][bin_j,bin_i] = Cl['%s-%s' %(key_X,key_Y)][bin_i,bin_j]
                # Symmetry C_{AB}^{ij} == C_{BA}^{ji}
                else:
                    for bin_i in xrange(n_bins[index_X]):
                        for bin_j in xrange(n_bins[index_Y]):
                            Cl['%s-%s' %(key_X,key_Y)][bin_i,bin_j] = [sint.simps(cH_chi2*W_X[bin_i]*W_Y[bin_j]*PS_lz[xx], x = zz) for xx in range(n_l)]
                            Cl['%s-%s' %(key_Y,key_X)][bin_j,bin_i] = Cl['%s-%s' %(key_X,key_Y)][bin_i,bin_j]
        return Cl


    #-----------------------------------------------------------------------------------------
    # CORRELATION FUNCTIONS
    #-----------------------------------------------------------------------------------------
    def limber_angular_correlation_functions(self, theta, l, Cl, order):
        """
        This function computes the angular correlation function from an angular power spectrum. The equation is as follows

        .. math::

            \\xi^{(ij)}_{XY}(\\theta) = \int_0^\infty \\frac{d\ell}{2\pi} \ \ell \ C_{XY}^{(ij)}(\ell) \ J_{\\nu} (\ell\\theta),

        where :math:`\\nu` is the order of the transform and it changes from observable to observable.

        .. warning::

         For example, for shear :math:`\\nu=0` or :math:`\\nu=4`, for galaxy clustering :math:`\\nu=0` and for galaxy-galaxy lensing :math:`\\nu=2`.

        :param theta: Angles (in :math:`\mathrm{arcmin}`) where to compute the shear correlation functions
        :type theta: array

        :param l: Multipoles at which the spectrum is computed
        :type l: array

        :param Cl: 3D array, where first and second dimensions are the bins and the third is the multipoles, i.e. ``Cl[bin i, bin j, multipole l]``. The last dimension has to have the same length as ``l``.
        :type Cl: 3D array

        :param order: Order of Hankel transform.
        :type order: float

        :return: 3D array containing ``xi[bin i, bin j, angle theta]``

        """
        # 1) Define and check lengths and quantities
        l,theta,Cl = np.atleast_1d(l), np.atleast_1d(theta), np.atleast_3d(Cl)
        n_theta,n_l,nbins_i,nbins_j = len(theta),len(l),len(Cl),len(Cl[0])
        assert len(Cl[0,0]) == n_l, "Shapes of multipoles and spectra do not match"

        # 2) Check consistency of angles and multipoles
        ratio            = 30.  #(1/30 of minimum and maximum to avoid oscillations)
        theta_min_from_l = 60*180/np.pi/np.max(l)  # arcmin
        theta_max_from_l = 60*180/np.pi/np.min(l)  # arcmin
        assert theta.min() > theta_min_from_l, "Minimum theta is too small to obtain convergent results for the correlation function"
        assert theta.max() < theta_max_from_l, "Maximum theta is too large to obtain convergent results for the correlation function"


        # 3) Initialize arrays
        NN = 8192
        xi = np.zeros((nbins_i,nbins_j,n_theta))

        # 4) Hankel transform (ORDER!!!!)
        for bin_i in xrange(nbins_i):
            for bin_j in xrange(nbins_j):
                theta_tmp, xi_tmp = FF.Hankel(l, Cl[bin_i,bin_j]/(2.*np.pi), order = order, N = NN)
                xi_interp = si.interp1d(theta_tmp*180./np.pi*60., xi_tmp, 'cubic', bounds_error = False, fill_value = 0.)
                xi[bin_i,bin_j] = xi_interp(theta)
        del xi_tmp

        return xi

