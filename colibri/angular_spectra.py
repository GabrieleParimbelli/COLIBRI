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
# CLASS ANGULAR SPECTRA
#==================
class angular_spectra():
    """
    The class :func:`~colibri.angular_spectra.angular_spectra` contains all the functions useful to compute
    the angular power spectra and correlation functions in the flat sky and Limber's approximation.
    It also computes window functions and galaxy PDF. The initialization requires the redshifts and
    scales to integrate (this choice is dictated by the fact that in this way one can use power spectra
    from simulations). Also routines to compute the intrinsic alignment terms are present.


    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`~colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``

    :param z_limits: Lower and upper limit of integration along the line of sight. This arguments avoids useless integration at high redshift, where there are essentially no galaxies. Both numbers must be non-negative and the first number must be smaller than the second. If the lower limit is set to 0, it will be enhanced by 1e-10 to avoid divergences at the origin of the lightcone.
    :type z_limits: 2-uple or list/array of length 2, default = (0., 5.)

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

        # Set the array of integration (set number of redshift so that there is dz = 0.0125)
        nz_min = 80
        self.nz_integration = int((self.z_max - self.z_min)*nz_min + 2)
        self.z_integration  = np.linspace(self.z_min, self.z_max, self.nz_integration)

        # Array of redshifts for computing window integrals (set number of redshift so that there is an integral at least each dz = 0.025)
        dz_windows = 0.025
        self.z_windows  = np.arange(self.z_min, self.z_max+dz_windows, dz_windows)
        self.nz_windows = len(np.atleast_1d(self.z_windows))


        # Distances (in Mpc/h)
        self.geometric_factor         = self.geometric_factor_f_K(self.z_integration)
        self.geometric_factor_windows = self.geometric_factor_f_K(self.z_windows)

        # Hubble parameters (in km/s/(Mpc/h))
        self.Hubble         = self.cosmology.H_massive(self.z_integration)/self.cosmology.h
        self.Hubble_windows = self.cosmology.H_massive(self.z_windows)    /self.cosmology.h


    #-----------------------------------------------------------------------------------------
    # LOAD_PK
    #-----------------------------------------------------------------------------------------
    def load_power_spectra(self, k, z, power_spectra = None, nonlinear = True, code = 'Class'):
        """
        This routine interpolates the total matter power spectrum (using the CDM prescription) in scales (units of :math:`h/\mathrm{Mpc}`) and redshifts. If `power_spectra` is set to ``None``, the power spectra are first computed at the scales and redshifts required and the interpolated; if `power_spectra` is not ``None``, it must be a 2D array of shape ``(len(z), len(k))`` which contains the power spectrum (in units of :math:`(\mathrm{Mpc}/h)^3`) evaluated at the scales and redshifts specified above.

        :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
        :type k: array

        :param z: Redshifts at which power spectrum must be/is computed.
        :type z: array

        :param power_spectra: If given, it must be a 2D array of shape ``(len(z), len(k))`` which contains the power spectrum (in units of :math:`(\mathrm{Mpc}/h)^3`) evaluated at the scales and redshifts specified above.
        :type power_spectra: 2D NumPy array, default = ``None``

        :param nonlinear: Whether to return nonlinear power spectra. Used only if ``power_spectra`` is not ``None``.
        :type nonlinear: boolean, default = ``True``

        :param code: Which code use to compute the linear power spectrum. Used only if ``power_spectra`` is not ``None``.

         - [`'Class'`, `'class'`, `'Xclass'`, `'XClass'`] for `CLASS <http://class-code.net/>`_
         - [`'Camb'`, `'CAMB'`, `'camb'`, `'Xcamb'`, `'XCamb'`, `'XCAMB'`] for `CAMB <https://camb.info/>`_
         - [`'EH'`, `'eh'`, `'Eisenstein-Hu'`] for Eisenstein-Hu (then turned non-linear with :func:`~colibri.nonlinear.HMcode2016`)

        :type code: string, default = `'Class'`.


        :return: Nothing, but a 2D-interpolated object ``self.power_spectra_interpolator`` containing :math:`P(k,z)` in units of :math:`(\mathrm{Mpc}/h)^3` is created
        """

        # Select scales and redshifts
        self.k     = np.atleast_1d(k)
        self.z     = np.atleast_1d(z)
        self.nk    = len(self.k)
        self.nz    = len(self.z)
        self.k_min = k.min()
        self.k_max = k.max()
        self.z_int_min = z.min()
        self.z_int_max = z.max()

        # Assertion on redshifts: they must be at least as extended as the integration limits
        #assert self.z_int_max >= self.z_max, "Maximum redshift for power spectrum must be at least as large as the maximum redshift for integration (at least %.2f)" %(self.z_max)
        #assert self.z_int_min <= self.z_min, "Minimum redshift for power spectrum must be at least as small as the minimum redshift for integration (at most %.2f)"  %(self.z_min)

        # Select kind of interpolation
        if self.nz > 3: kind_of_interpolation = 'cubic'
        else:           kind_of_interpolation = 'linear'

        # If a power_spectra 2D array is not provide, compute power spectra using CAMB/Class
        if power_spectra is None:

            # Fractions
            f_nu  = np.sum(self.cosmology.f_nu)
            f_cb  = self.cosmology.f_cb

            # Linear power spectrum
            if code in ['Class', 'class', 'Xclass', 'XClass']:
                k_lin, pk_lin = self.cosmology.class_XPk(z = self.z, k = self.k, var_1 = ['cb', 'nu'], var_2 = ['cb', 'nu'])
            elif code in ['Camb', 'CAMB', 'camb', 'Xcamb', 'XCamb', 'XCAMB']:
                k_lin, pk_lin = self.cosmology.camb_XPk(z = self.z, k = self.k, var_1 = ['cb', 'nu'], var_2 = ['cb', 'nu'])
            elif code in ['EH', 'eh', 'Eisenstein-Hu']:
                try:
                    k_lin, pk_eh = self.cosmology.EisensteinHu_Pk(z = self.z, k = self.k, sigma_8 = self.cosmology.sigma_8)
                except TypeError:
                    raise TypeError("sigma_8 value is currently set to None in self.cosmology. Set it to a float.")
                pk_lin = {}
                pk_lin['cb-cb'] = pk_eh
                pk_lin['cb-nu'] = np.zeros_like(pk_eh)
                pk_lin['nu-nu'] = np.zeros_like(pk_eh)
            else:
                raise NameError("Boltzmann solver not recognized")

            # Do linear or non-linear as required
            if nonlinear == False:
                pk_m          = f_cb**2.*pk_lin['cb-cb'] + 2.*f_nu*f_cb*pk_lin['cb-nu'] + f_nu**2.*pk_lin['nu-nu']
                self.power_spectra_interpolator = si.interp2d(self.k, self.z, pk_m, kind_of_interpolation, bounds_error = False, fill_value = 0.)
            else:
                # Non-linear 'cdm+b' power spectrum
                pk_lin_cb     = pk_lin['cb-cb']
                do_nonlinear  = NL.HMcode2016(z = self.z, k = self.k, pk = pk_lin_cb, cosmology = self.cosmology)
                k_hf, pk_hf   = do_nonlinear.k, do_nonlinear.pk_nl
                # Total matter non-linear power spectrum
                pk_m          = f_cb**2.*pk_hf + 2.*f_nu*f_cb*pk_lin['cb-nu'] + f_nu**2.*pk_lin['nu-nu']
                # Interpolate
                self.power_spectra_interpolator = si.interp2d(self.k, self.z, pk_m, kind_of_interpolation, bounds_error = False, fill_value = 0.)

        else:
                # Interpolate
                self.power_spectra_interpolator = si.interp2d(self.k, self.z, power_spectra, kind_of_interpolation, bounds_error = False, fill_value = 0.)



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
        :type a: float, default = 1.5

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
        Comoving distance between two redshifts. It assumes neutrinos as matter.
        This latter assumption introduces a bias of less than 0.02% at :math:`z<10` for even the lowest
        neutrino masses allowed by particle physics.

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
        Geometric factor (distance) between two given redshifts ``z`` and ``z0``. It assumes neutrinos as matter. This latter assumption introduces a bias of less than 0.02 % at :math:`z<10` for even the lowest neutrino masses allowed by particle physics.

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
    # WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_window_functions(self, galaxy_distributions = None, z = None, nz = None):
        """
        This function computes the window function to use in the shear power spectrum, evaluated at
        ``self.z_integration``. It automatically normalizes the galaxy distribution function such that the
        integral over redshifts is 1. It does not return anything but it creates the quantities
        ``self.window_function`` and ``self.window_function_IA`` (for shear and intrinsic aligment signals).
        Given a galaxy distribution :math:`n^{(i)}(z)` in the `i`-th bin:

        .. math::

           W^{(i)}(z) = \\frac{3}{2}\Omega_m \ \\frac{H_0^2}{c^2} \ f_K[\chi(z)] (1+z) \int_z^\infty dx \ n^{(i)}(x) \ \\frac{f_K[\chi(z-x)]}{f_K[\chi(z)]}


        .. math::

           W^{(i)}_{IA}(z) = n^{(i)}(z) \\frac{H(z)}{c}

        :param galaxy_distributions: The distribution of galaxies in each bin for which the shear spectra need to be computed. A list of functions containing the distribution of galaxies in each bin considered. Every element of this list is in turn a list. The first element is the name of a defined function describing a galaxy distribution in redshift. The first argument of said function must be redshift. All the other arguments must be specified by the second element of the list, which is a dictionary.
        :type galaxy_distributions: nested list, default = None

        :param z: array or list of redshift at which the galaxy distribution ``nz`` is evaluated
        :type z: 1-D array, default = None

        :param nz: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
        :type nz: 2-D array with shape ``(n_bins, len(z))``, default = None

        An example call can be, for 3 bins all with a :func:`~colibri.angular_spectra.euclid_distribution` with default arguments for ``a`` and ``b`` but different bin edges ``zmin``, ``zmax``:

        .. code-block:: python

           self.load_window_functions(galaxy_distributions = [[self.euclid_distribution, {'zmin': 0.00, 'zmax': 0.72}],
                                                              [self.euclid_distribution, {'zmin': 0.72, 'zmax': 1.11}],
                                                              [self.euclid_distribution, {'zmin': 1.11, 'zmax': 5.00}]])

        Alternatively, one can pre-compute a 2-D array containing the galaxy distributions and feed it into the function.

        .. code-block:: python

           bin_edges = [0.00, 0.72, 1.11, 5.00]
           nbins     = len(bin_edges)-1
           z_w       = np.linspace(0., 6., 1001)
           nz_w      = [S.euclid_distribution(z = z_w, a = 2.0, b = 1.5, zmin = bin_edges[i], zmax = bin_edges[i+1], step = 1e-4) for i in range(nbins)]
           S.load_window_functions(z = z_w, nz = nz_w)

        :return: Nothing, but the quantities ``self.window_function`` and ``self.window_function_IA`` will be created: these are lists of length ``len(galaxy_distributions)`` of interpolated functions from ``z_limits.min()`` to ``z_limits.max()``.

        """

        if (galaxy_distributions is None and z is None and nz is None):
            raise ValueError("Either 'galaxy distribution' or 'z' AND 'nz' must be different from None")
        if nz is not None and z is not None:
            nz = np.array(nz)
            z  = np.array(z)
            assert nz.ndim == 2, "'nz' must be 2-dimensional" 
            assert (nz.shape)[1] == z.shape[0], "Length of each 'nz[i]' must be the same of 'z'"

        # Call a simpler function if Omega_K == 0.
        if self.cosmology.Omega_K == 0.:
            self.load_window_functions_flat(galaxy_distributions, z, nz)
        # Otherwise compute window function in curved geometry
        else:
            # Set number of bins and normalize them
            if galaxy_distributions is not None:
                n_bins = len(galaxy_distributions)
                norm_const = [sint.quad(lambda x: galaxy_distributions[i][0](x, **galaxy_distributions[i][1]), 0., np.inf)[0] for i in range(n_bins)]
            else:
                n_bins = len(nz)
                norm_const = sint.simps(nz, x = z, axis = 1)

            # Initialize arrays of windows
            self.window_function    = []
            self.window_function_IA = []
            constant = 3./2.*self.cosmology.Omega_m*(self.cosmology.H0/self.cosmology.h/const.c)**2.*(1.+self.z_windows)*self.geometric_factor_windows

            # Set windows
            for galaxy_bin in xrange(n_bins):
                # Select which is the function and which are the arguments and do the integral for window function
                if galaxy_distributions is not None:
                    n_z  = galaxy_distributions[galaxy_bin][0]
                    args = galaxy_distributions[galaxy_bin][1]
                    integral = list(map(lambda z_i: sint.quad(lambda x: n_z(x, **args)*self.geometric_factor_f_K(x,z_i)/self.geometric_factor_f_K(x), z_i, self.z_max, epsrel = 1.e-3)[0], self.z_windows))
                else:
                    n_z = si.interp1d(z,nz[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
                    args = {}
                    integral = list(map(lambda z_i: sint.quad(lambda x: n_z(x)*self.geometric_factor_f_K(x,z_i)/self.geometric_factor_f_K(x), z_i, self.z_max, epsrel = 1.e-3)[0], self.z_windows))
                # Fill temporary window functions with real values
                window_function_tmp    = constant*integral/norm_const[galaxy_bin]
                # Interpolate (the Akima interpolator avoids oscillations around the zero due to the cubic spline)
                try:
                    self.window_function.append(si.interp1d(self.z_windows, window_function_tmp, 'cubic', bounds_error = False, fill_value = 0.))
                except ValueError:
                    self.window_function.append(si.Akima1DInterpolator(self.z_windows, window_function_tmp))
                self.window_function_IA.append(si.interp1d(self.z_integration, n_z(self.z_integration, **args)*self.Hubble/const.c/norm_const[galaxy_bin], 'cubic', bounds_error = False, fill_value = 0.))

    def load_window_functions_flat(self, galaxy_distributions, z = None, nz = None):
        """
        This function does the same as :func:`~colibri.angular_spectra.load_window_functions` but for a flat Universe, with a speed-up in the calculation of a factor ~4.
        When the function :func:`~colibri.angular_spectra.load_window_functions` is called and :math:`\Omega_\mathrm{K}` is set to 0, this function is what is actually being run.

        :param galaxy_distributions: The distribution of galaxies in each bin for which the shear spectra need to be computed. A list of functions containing the distribution of galaxies in each bin considered. Every element of this list is in turn a list. The first element is the name of a defined function describing a galaxy distribution in redshift. The first argument of said function must be redshift. All the other arguments must be specified by the second element of the list, which is a dictionary.
        :type galaxy_distributions: nested list 

        :param z: array or list of redshift at which the galaxy distribution ``nz`` is evaluated
        :type z: 1-D array, default = None

        :param nz: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
        :type nz: 2-D array with shape ``(n_bins, len(z))``, default = None

        :return: Nothing, but the quantities ``self.window_function`` and ``self.window_function_IA`` will be created: these are lists of length ``len(galaxy_distributions)`` of interpolated functions from ``z_limits.min()`` to ``z_limits.max()``.


        """

        if (galaxy_distributions is None and z is None and nz is None):
            raise ValueError("Either 'galaxy distribution' or 'z' AND 'nz' must be different from None")
        if nz is not None and z is not None:
            nz = np.array(nz)
            z  = np.array(z)
            assert nz.ndim == 2, "'nz' must be 2-dimensional" 
            assert (nz.shape)[1] == z.shape[0], "Length of each 'nz[i]' must be the same of 'z'"
        # Set number of bins and normalize them
        if galaxy_distributions is not None:
            n_bins = len(galaxy_distributions)
            norm_const = [sint.quad(lambda x: galaxy_distributions[i][0](x, **galaxy_distributions[i][1]), 0., np.inf)[0] for i in range(n_bins)]
        else:
            n_bins = len(nz)
            norm_const = sint.simps(nz, x = z, axis = 1)

        # Initialize arrays of windows
        self.window_function    = []
        self.window_function_IA = []
        constant = 3./2.*self.cosmology.Omega_m*(self.cosmology.H0/self.cosmology.h/const.c)**2.*(1.+self.z_windows)*self.geometric_factor_windows

        # Set windows
        chi_max = self.cosmology.comoving_distance(self.z_max)
        for galaxy_bin in xrange(n_bins):
            # Select which is the function and which are the arguments
            if galaxy_distributions is not None:
                n_z_array  = galaxy_distributions[galaxy_bin][0](self.z_windows, **galaxy_distributions[galaxy_bin][1])
            else:
                tmp_interp = si.interp1d(z,nz[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
                n_z_array  = tmp_interp(self.z_windows)
            n_z_interp = si.interp1d(self.geometric_factor_windows, n_z_array*(self.Hubble_windows/const.c), 'cubic', bounds_error = False, fill_value = 0.)
            # Do the integral for window function
            integral = list(map(lambda chi_i: sint.quad(lambda chi: n_z_interp(chi)*(1.-chi_i/chi), chi_i, chi_max, epsrel = 1.e-3)[0], self.geometric_factor_windows))
            # Fill temporary window functions with real values
            window_function_tmp    = constant*integral/norm_const[galaxy_bin]
            # Interpolate (the Akima interpolator avoids oscillations around the zero due to the cubic spline)
            try:
                self.window_function.append(si.interp1d(self.z_windows, window_function_tmp, 'cubic', bounds_error = False, fill_value = 0.))
            except ValueError:
                self.window_function.append(si.Akima1DInterpolator(self.z_windows, window_function_tmp))
            if galaxy_distributions is not None:
                self.window_function_IA.append(si.interp1d(self.z_integration, galaxy_distributions[galaxy_bin][0](self.z_integration, **galaxy_distributions[galaxy_bin][1])*self.Hubble/const.c/norm_const[galaxy_bin], 'cubic', bounds_error = False, fill_value = 0.))
            else:
                self.window_function_IA.append(si.interp1d(self.z_integration, tmp_interp(self.z_integration)*self.Hubble/const.c/norm_const[galaxy_bin], 'cubic', bounds_error = False, fill_value = 0.))

    #-----------------------------------------------------------------------------------------
    # GALAXY BIAS
    #-----------------------------------------------------------------------------------------
    def load_galaxy_bias(self, bias_function, **kwargs):
        """
        It loads an interpolator for galaxy bias. Use it only if you are computing angular galaxy clustering. For how this code is built, this function must be called after:func:`~colibri.angular_spectra.load_power_spectra`

        :param bias_function: bias function for galaxies :math:`b(k,z)`.
        :type bias_function: a function whose two first arguments are scale k (in :math:`h/\mathrm{Mpc}`) and redshift :math:`z`.

        :return: an interpolated object in 2D

        """
        # Compute galaxy bias from function
        K,Z = np.meshgrid(self.k, self.z)
        bias_array = bias_function(K, Z, **kwargs)
        # Select kind of interpolation
        if self.nz > 3: kind_of_interpolation = 'cubic'
        else:           kind_of_interpolation = 'linear'
        # Create interpolator
        self.galaxy_bias_interpolator = si.interp2d(self.k, self.z, bias_array, kind_of_interpolation, bounds_error = False, fill_value = 0.)


    #-----------------------------------------------------------------------------------------
    # CORRECTION FUNCTION FOR INTRINSIC ALIGNMENT
    #-----------------------------------------------------------------------------------------
    def intrinsic_alignment_kernel(self, k, z, A_IA, eta_IA = 0., beta_IA = 0., lum_IA = 1.):
        """
        Intrinsic alignment correction function.

        :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
        :type k: array

        :param z: Redshifts at which the function must be computed.
        :type z: array

        :param A_IA: Intrinsic alignment amplitude.
        :type A_IA: float

        :param eta_IA: Exponent for redshift dependence of intrinsic alignment.
        :type eta_IA: float, default = 0

        :param beta_IA: Exponent for luminosity dependence of intrinsic alignment.
        :type beta_IA: float, default = 0

        :param rlum_IA: Relative luminosity of galaxies w.r.t. :math:`L_*`.
        :type lum_IA: float or array of same length as `z`, default = 1

        :param kwargs: Keyword arguments for the IA function.

        :return: 2D array containing the correction function :math:`F(k,z)` for intrinsic alignment
        """
        # Constants in front
        C1     = 0.0134
        front  = -A_IA*C1*self.cosmology.Omega_m
        # Growth factors
        Z,K    = np.meshgrid(z,k,indexing='ij')
        growth = self.cosmology.D_cbnu(k = np.array(k), z = z)
        # Relative luminosity is either a function or a float
        if   callable(lum_IA):          rel_lum = lum_IA(Z)
        elif isinstance(lum_IA, float): rel_lum = lum_IA
        else:                           raise TypeError("'lum_IA' must be either a float or a function with redshift as the only argument.")
        return front/growth*(1.+Z)**eta_IA*rel_lum**beta_IA

    #-----------------------------------------------------------------------------------------
    # SPECTRA SHEAR CLUSTERING
    #-----------------------------------------------------------------------------------------
    def compute_angular_power_spectra(self,
                                      l,
                                      do_WL                = True,
                                      do_IA                = False,
                                      do_GC                = False,
                                      kwargs_power_spectra = {},
                                      kwargs_IA            = {}):
        """
        This function computes the shear/clustering angular power spectra using the Limber's and the flat-sky approximations.
        Given two redshift bins `i` and `j` the equation is

        .. math::

          C^{(ij)}(\ell) = \int_0^\infty dz \ \\frac{c}{H(z)} \ \\frac{W^{(i)}(z) W^{(j)}(z)}{f_K^2[\chi(z)]} \ P\left(\\frac{\ell}{f_K[\chi(z)]}, z\\right),

        where :math:`P(k,z)` is the matter power spectrum and :math:`W^{(i)}(z)` are the window functions for either shear, galaxy clustering or intrinsic alignment.

        :param l: Multipoles at which to compute the shear power spectra.
        :type l: array

        :param do_WL: Whether to compute the shear power spectrum.
        :type do_WL: boolean, default = ``True``

        :param do_IA: Whether to compute the intrinsic alignment terms.
        :type do_IA: boolean, default = ``False``

        :param do_GC: Whether to compute the galaxy clustering power spectra.
        :type do_GC: boolean, default = ``False``

        :param kwargs_power_spectra: Keyword arguments to pass to ``self.load_power_spectra`` (used only if ``power_spectra == None``).
        :type kwargs_power_spectra: dictionary, default = {}
            
        :param kwargs_IA: Keyword arguments to pass to intrinsic alignment model (used only if ``do_IA`` is ``True``)
        :type kwargs_IA: dictionary, default = {}

        :return: power spectrum :math:`C^{(ij)}(\ell)` for all bin pairs whose window function was generated by :func:`~colibri.angular_spectra.load_window_functions`.
        :rtype:  dictionary containing the following keys:

            - ``'gg'``: shear power spectrum;
            - ``'gI'``: galaxy-IA term;
            - ``'II'``: IA-IA term;
            - ``'LL'``: the sum of the three weak lensing terms;
            - ``'GL'``: galaxy-galaxy lensing term;
            - ``'GG'``: galaxy clustering term.

            Each key is a 3-D array whose entries are ``Cl[bin i, bin j, multipole l]``
        """

        # 0a) Check window functions have been loaded
        try:
            n_bin    = len(self.window_function)
        except AttributeError:
            raise AttributeError("Load window functions using the self.load_window_functions method")

        # 1) Define lengths and quantities
        zz       = self.z_integration
        n_l      = len(np.atleast_1d(l))
        n_z      = self.nz_integration
        HH       = self.Hubble
        Cl       = {'gg': np.zeros((n_bin, n_bin, n_l)),
                    'gI': np.zeros((n_bin, n_bin, n_l)),
                    'II': np.zeros((n_bin, n_bin, n_l)),
                    'LL': np.zeros((n_bin, n_bin, n_l)),
                    'GL': np.zeros((n_bin, n_bin, n_l)),
                    'GG': np.zeros((n_bin, n_bin, n_l))}

        # 2) Load power spectra, F_IA, bias
        # (one can use simulations in this way)
        try:
            power_spectra = self.power_spectra_interpolator
        except AttributeError:
            print("Power spectra not found, loading them using 'load_power_spectra' method with arguments from 'kwargs_power_spectra'")
            self.load_power_spectra(**kwargs_power_spectra)
            power_spectra = self.power_spectra_interpolator

        # Check convergence with (l, k, z):
        assert np.atleast_1d(l).min() > self.k_min*self.geometric_factor_f_K(self.z_min), "Minimum 'l' is too low. Extend to lower k_min? Use lower z_min?"
        assert np.atleast_1d(l).max() < self.k_max*self.geometric_factor_f_K(self.z_max), "Maximum 'l' is too high. Extend to higher k_max? Use higher z_min?"

        # Fill arrays of interpolated power spectra, IA and bias (\ell, z)
        PS_lz = np.zeros((n_l, n_z))
        F_lz  = np.zeros((n_l, n_z))
        b_lz  = np.zeros((n_l, n_z))
        if do_WL and do_IA:
            F_IA = self.intrinsic_alignment_kernel(k = self.k, z = zz, **kwargs_IA)
        else:
            F_IA = np.zeros((n_z,self.nk))
        F_IA_interp = si.interp2d(self.k, zz, F_IA, kind = 'cubic', bounds_error = False, fill_value = 0.)
        for il in xrange(n_l):
            for iz in range(n_z):
                PS_lz[il,iz] = power_spectra(l[il]/self.geometric_factor[iz], zz[iz])
                F_lz[il,iz]  = F_IA_interp  (l[il]/self.geometric_factor[iz], zz[iz])
        if do_GC:
            assert hasattr(self, "galaxy_bias_interpolator"), "Load bias function with 'load_galaxy_bias' before compute galaxy clustering"
            for il in xrange(n_l):
                for iz in range(n_z):
                    b_lz[il,iz] = self.galaxy_bias_interpolator(l[il]*1./self.geometric_factor[iz], zz[iz])

        # 3) load W(z) given the source functions
        windows    = np.array([self.window_function   [i](zz) for i in xrange(n_bin)])
        IA_windows = np.array([self.window_function_IA[i](zz) for i in xrange(n_bin)])
        g_windows  = np.array([self.window_function_IA[i](zz) for i in xrange(n_bin)])

        # 4) Compute shear spectra
        if do_WL:
            for i in xrange(n_bin):
                for j in xrange(n_bin):
                    Cl['gg'][i,j] = [sint.simps(const.c/HH*windows[i]*windows[j]*PS_lz[xx]/self.geometric_factor**2., x = zz) for xx in range(len(np.atleast_1d(l)))]

        # 5) Compute intrinsic alignment spectra
        if do_WL and do_IA:
            for i in xrange(n_bin):
                for j in xrange(n_bin):
                    for il in xrange(n_l):
                        Cl['gI'][i,j,il] = sint.simps(const.c/HH*(IA_windows[i]*windows[j] + IA_windows[j]*windows[i])*F_lz[il]*PS_lz[il]/self.geometric_factor**2., x = zz)
                        Cl['II'][i,j,il] = sint.simps(const.c/HH*IA_windows[i]*IA_windows[j]*F_lz[il]**2.*PS_lz[il]/self.geometric_factor**2., x = zz)

        # 6) Compute lensing spectra (GG+GI+II)
        Cl['LL'] = Cl['gg'] + Cl['gI'] + Cl['II']

        # 7) Compute galaxy clustering and galaxy-galaxy lensing
        if do_GC:
            for i in xrange(n_bin):
                for j in xrange(n_bin):
                    for il in xrange(n_l):
                        Cl['GG'][i,j,il] = sint.simps(const.c/HH*g_windows[i]*g_windows[j]*b_lz[il]**2.*PS_lz[il]/self.geometric_factor**2., x = zz)
                        if do_WL:
                            Cl['GL'][i,j,il] = sint.simps(const.c/HH*b_lz[il]*g_windows[i]*(windows[j]+F_lz[il]*IA_windows[j])*PS_lz[il]/self.geometric_factor**2., x = zz)

        return Cl


    #-----------------------------------------------------------------------------------------
    # CORRELATION FUNCTIONS SHEAR CLUSTERING
    #-----------------------------------------------------------------------------------------
    def compute_angular_correlation_functions(self,
                                              theta,
                                              do_WL = True,
                                              do_IA = False,
                                              do_GC = False,
                                              kwargs_power_spectra = {},
                                              kwargs_IA = {}):
        """
        This function computes the angular correlation functions for shear, intrinisic alignment and galaxy clustering using the Limber's and the flat-sky approximations. It first computes :func:`~colibri.angular_spectra.compute_angular_power_spectra` and then computes its Hankel transform with :func:`~colibri.fourier.Hankel` (therefore this function requires the ``FFTlog`` package). The shear correlations function, the galaxy-galaxy lensing correlation function and the galaxy correlation function read, respectively

        .. math::

            \\xi_{+/-}^{(ij)}(\\theta) = \int_0^\infty \\frac{d\ell}{2\pi} \ \ell \ C_{GG}^{(ij)}(\ell) \ J_{0/4} (\ell\\theta)

        .. math::

            \gamma_t^{(ij)}(\\theta) = \int_0^\infty \\frac{d\ell}{2\pi} \ \ell \ C_{gG}^{(ij)}(\ell) \ J_{2} (\ell\\theta)

        .. math::

            w^{(ij)}(\\theta) = \int_0^\infty \\frac{d\ell}{2\pi} \ \ell \ C^{(ij)}_{gg}(\ell) \ J_{0} (\ell\\theta)

        :param theta: Angles (in :math:`\mathrm{arcmin}` units) where to compute the shear correlation functions
        :type theta: array

        :param do_WL: Whether to compute the shear power spectrum.
        :type do_WL: boolean, default = ``True``

        :param do_IA: Whether to compute the intrinsic alignment terms.
        :type do_IA: boolean, default = ``False``

        :param do_GC: Whether to compute the galaxy clustering power spectra.
        :type do_GC: boolean, default = ``False``

        :param kwargs_power_spectra: Keyword arguments to pass to ``self.load_power_spectra`` (used only if ``power_spectra == None``).
        :type kwargs_power_spectra: dictionary, default = {}
            
        :param kwargs_IA: Keyword arguments to pass to intrinsic alignment model (used only if ``do_IA`` is ``True``)
        :type kwargs_IA: dictionary, default = {}


        :return: power spectrum :math:`C^{(ij)}(\ell)` for all bin pairs whose window function was generated by :func:`~colibri.angular_spectra.load_window_functions`.
        :rtype:  dictionary containing the following keys:

            - ``'gg+'``: shear correlation function (transformed with :math:`J_0`);
            - ``'gI+'``: galaxy-IA term (transformed with :math:`J_0`);
            - ``'II+'``: IA-IA term (transformed with :math:`J_0`);
            - ``'LL+'``: the sum of the three weak lensing terms (transformed with :math:`J_0`);
            - ``'gg-'``: shear correlation function (transformed with :math:`J_4`);
            - ``'gI-'``: galaxy-IA term (transformed with :math:`J_4`);
            - ``'II-'``: IA-IA term (transformed with :math:`J_4`);
            - ``'LL-'``: the sum of the three weak lensing terms (transformed with :math:`J_4`);
            - ``'GL'`` : galaxy-galaxy lensing term (transformed with :math:`J_2`);
            - ``'GG'`` : galaxy clustering term (transformed with :math:`J_2`).

            Each key is a 3-D array whose entries are ``Cl[bin i, bin j, multipole l]``
        """

        # 1) Define lengths and quantities
        l_min, l_max = 180./np.pi*60./np.max(theta)*0.95, 180./np.pi*60./np.min(theta)*1.05
        l            = np.geomspace(l_min, l_max, 128)
        zz           = self.z_integration
        n_theta      = len(np.atleast_1d(theta))
        n_z          = self.nz_integration
        HH           = self.Hubble

        # 2) Compute shear spectra
        Cl = self.compute_angular_power_spectra(l                    = l,
                                                do_WL                = do_WL,
                                                do_IA                = do_IA,
                                                do_GC                = do_GC,
                                                kwargs_power_spectra = kwargs_power_spectra,
                                                kwargs_IA            = kwargs_IA)

        # 3) Initialize arrays
        NN = 8192
        n_bin   = len(self.window_function)
        xi_tmp  = {'gg+': np.zeros((n_bin, n_bin, NN)),  # J_0
                   'gI+': np.zeros((n_bin, n_bin, NN)),  # J_0
                   'II+': np.zeros((n_bin, n_bin, NN)),  # J_0
                   'LL+': np.zeros((n_bin, n_bin, NN)),  # J_0
                   'gg-': np.zeros((n_bin, n_bin, NN)),  # J_4
                   'gI-': np.zeros((n_bin, n_bin, NN)),  # J_4
                   'II-': np.zeros((n_bin, n_bin, NN)),  # J_4
                   'LL-': np.zeros((n_bin, n_bin, NN)),  # J_4
                   'GL' : np.zeros((n_bin, n_bin, NN)),  # J_2
                   'GG' : np.zeros((n_bin, n_bin, NN))}  # J_0



        xi      = {'gg+': np.zeros((n_bin, n_bin, n_theta)),  # J_0
                   'gI+': np.zeros((n_bin, n_bin, n_theta)),  # J_0
                   'II+': np.zeros((n_bin, n_bin, n_theta)),  # J_0
                   'LL+': np.zeros((n_bin, n_bin, n_theta)),  # J_0
                   'gg-': np.zeros((n_bin, n_bin, n_theta)),  # J_4
                   'gI-': np.zeros((n_bin, n_bin, n_theta)),  # J_4
                   'II-': np.zeros((n_bin, n_bin, n_theta)),  # J_4
                   'LL-': np.zeros((n_bin, n_bin, n_theta)),  # J_4
                   'GL' : np.zeros((n_bin, n_bin, n_theta)),  # J_2
                   'GG' : np.zeros((n_bin, n_bin, n_theta))}  # J_0

        # 4) Hankel transform
        for i in range(n_bin):
            for j in range(n_bin):
                if do_WL:
                    theta_tmp, xi_tmp['gg+'][i,j] = FF.Hankel(l, Cl['gg'][i,j]/(2.*np.pi), order = 0, N = NN)
                    theta_tmp, xi_tmp['gg-'][i,j] = FF.Hankel(l, Cl['gg'][i,j]/(2.*np.pi), order = 4, N = NN)
                if do_WL and do_IA:
                    for component in ['gI', 'II']:
                        theta_tmp, xi_tmp[component+'+'][i,j] = FF.Hankel(l, Cl[component][i,j]/(2.*np.pi), order = 0, N = NN)
                        theta_tmp, xi_tmp[component+'-'][i,j] = FF.Hankel(l, Cl[component][i,j]/(2.*np.pi), order = 4, N = NN)
                xi_tmp['LL+'] = xi_tmp['gg+'] + xi_tmp['gI+'] + xi_tmp['II+']
                xi_tmp['LL-'] = xi_tmp['gg-'] + xi_tmp['gI-'] + xi_tmp['II-']
                if do_GC:
                    theta_tmp, xi_tmp['GG'][i,j] = FF.Hankel(l, Cl[component][i,j]/(2.*np.pi), order = 0, N = NN)
                    if do_WL:
                        theta_tmp, xi_tmp['GL'][i,j] = FF.Hankel(l, Cl[component][i,j]/(2.*np.pi), order = 2, N = NN)


        # 5) Transform temporary angles in arcmin
        theta_tmp *= 180./np.pi*60.

        # 6) Interpolate
        for i in range(n_bin):
            for j in range(n_bin):
                for comp in xi_tmp.keys():
                    xi_interp  = si.interp1d(theta_tmp, xi_tmp [comp][i,j], 'cubic', bounds_error = False, fill_value = 0.)
                    xi[comp][i,j] = xi_interp(theta)
        del xi_tmp

        return xi


