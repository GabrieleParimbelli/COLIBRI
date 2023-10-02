import numpy as np
from scipy.special import *
import scipy.interpolate as si
import colibri.constants as const
import colibri.cosmology as cc
import colibri.useful_functions as UF

class halo:
    """
    The class ``halo`` computes the non-linear power spectrum in the halo model and useful
    quantities related to it, such as mass functions, densities of virialized objects and
    concentration parameters. For a review, see `arXiv:0206508 <https://arxiv.org/abs/astro-ph/0206508>`_ .

    :type z: array
    :param z: Redshifts.

    :param k: Array of scales in :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param code: Boltzmann solver to compute the linear power spectrum. Choose among `'camb'`, `'class'`, `'eh'` (for Eisenstein-Hu). N.B. If Eisenstein-Hu is selected, effects of massive neutrinos and evolving dark energy cannot be accounted for, as such spectrum is a good approximation for LCDM cosmologies only.
    :type code: string, default = `'camb'`

    :param BAO_smearing: Whether to damp the BAO feature due to non-linearities.
    :type BAO_smearing: boolean, default = False

    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``


    :return: The initialization automatically computes the linear matter power spectrum. After initialization, the following quantities will be available:

     - ``self.k_ext`` (`array`) - Extended array of scales spanning :math:`10^{-6} - 10^8 \ h/\mathrm{Mpc}`.

     - ``self.pk_ext`` (`2D array`) - Extended power spectra of shape ``(len(z), len(k_ext))`` in units of :math:`(\mathrm{Mpc}/h)^3`.

     - ``self.mass`` (`array` ``np.logspace(2., 18., 512)``) - Array of masses in :math:`M_\odot/h` at which some quantities will be evaluated.

     - ``self.Pk`` (`dictionary`) - The only keys available after initialization are

         - ``self.Pk['matter']['linear']``: 2D array of shape ``(len(z), len(k))`` containing the linear matter power spectra in :math:`(\mathrm{Mpc}/h)^3`.
         - ``self.Pk['matter']['no-wiggle']``: 2D array of shape ``(len(z), len(k))`` containing the no-wiggle linear matter power spectra in :math:`(\mathrm{Mpc}/h)^3`. It is an array of zeros if ``BAO_smearing`` is ``False``.
         - ``self.Pk['matter']['de-wiggled']``: 2D array of shape ``(len(z), len(k))`` containing the de-wiggled linear matter power spectra in :math:`(\mathrm{Mpc}/h)^3`. It is an array of zeros if ``BAO_smearing`` is ``False``.


    """

    def __init__(self,
            z,
            k,
            code = 'camb',
            BAO_smearing = False,
            cosmology = cc.cosmo()):

        # Reading all cosmological parameters
        self.cosmology = cosmology

        # Code
        self.code = code

        # Smear BAO in non-linearities
        self.BAO_smearing = BAO_smearing

        # Overdensity spherical collapse
        self.delta_sc = 3./20.*(12.*np.pi)**(2./3.)
        
        # Redshift and scales at which all must be computed
        self.z  = np.atleast_1d(z)
        self.k  = np.atleast_1d(k)
        self.nz = len(self.z)
        self.nk = len(self.k)

        # Growth factor
        self.growth_factor = self.cosmology.growth_factor_scale_independent(self.z)

        # Power spectrum dictionary initialization
        self.Pk = {}
        self.Pk['matter'] = {}

        # Load linear power spectrum
        self.load_Pk()

        # Initialize mass
        self.nm   = 512
        self.mass = np.logspace(2., 18., self.nm)

        # Peak height
        self.peak_height = self.nu()

        # virial_radii
        self.rv = self.R_v(self.mass)


    #-----------------------------------------------------------------------------------------
    # OVERDENSITY AT COLLAPSE
    #-----------------------------------------------------------------------------------------
    def Delta_v(self, z):
        """
        Overdensity of a virialized halo

        :type z: array, default = 0
        :param z: Redshifts.

        :return: array
        """
        omz = self.cosmology.Omega_m_z(self.z)
        return 18.*np.pi**2.*(1.+0.399*(1./omz-1.))

    #-----------------------------------------------------------------------------------------
    # SMOOTHING RADIUS
    #-----------------------------------------------------------------------------------------
    def smoothing_radius(self, M):
        """
        Radius containing a mass :math:`M`

        :type M: array
        :param M: Masses in :math:`M_\odot/h`

        :return: array   
        """
        rho = self.cosmology.rho(0.)
        return (3*M/(4*np.pi*rho))**(1./3.)


    #-----------------------------------------------------------------------------------------
    # LOAD_PK
    #-----------------------------------------------------------------------------------------
    def load_Pk(self):
        """
        Loads the linear power spectrum with the code specified in initialization. The result is extrapolated to ``k_ext``.


        :return: Nothing, but it creates (or overwrites) the keys ``['matter']['linear']``, ``['matter']['no-wiggle']``, ``['matter']['no-wiggle']`` in the ``self.Pk`` dictionary

        """
        if self.code == 'camb':
            k, Pk_L = self.cosmology.camb_Pk(z = self.z, k = self.k)
        elif self.code == 'eh':
            k, Pk_L = self.cosmology.EisensteinHu_Pk(z = self.z, k = self.k)
        elif self.code == 'class':
            k, Pk_L = self.cosmology.class_Pk(z = self.z, k = self.k)
        else:
            raise NameError("unknown Boltzmann solver")
        self.Pk['matter']['linear'] = Pk_L

        # Compute BAO damping
        self.sv2 = [1./(6.*np.pi**2.)*np.trapz(k*Pk_L[iz], x = np.log(k)) for iz in range(self.nz)]

        # No-wiggle and de-wiggled power spectra
        if self.BAO_smearing:
            self.Pk['matter']['no-wiggle']  = np.zeros((self.nz, self.nk))
            self.Pk['matter']['de-wiggled'] = np.zeros((self.nz, self.nk))
            for iz in range(self.nz):
                self.Pk['matter']['no-wiggle'][iz]  = self.cosmology.remove_bao(k, Pk_L[iz], k_low = 0.01, k_high = 0.45)
                self.Pk['matter']['de-wiggled'][iz] = (self.Pk['matter']['linear'][iz]-self.Pk['matter']['no-wiggle'][iz])*np.exp(-k**2.*self.sv2[iz]) + self.Pk['matter']['no-wiggle'][iz]

        # Expand every redshift
        pk_ext = []
        for i in range(self.nz):
            k_ext, pk_tmp = UF.extrapolate_log(k, self.Pk['matter']['linear'][i], 1e-6, 1e8)
            pk_ext.append(pk_tmp)
        self.k_ext  = k_ext
        self.pk_ext = np.array(pk_ext)    


    #-----------------------------------------------------------------------------------------
    # SIGMA^2
    #-----------------------------------------------------------------------------------------
    def sigma2(self):
        """
        Mass variance in spheres, evaluated at the masses of the initialization which are transforms into radii.

        :return: 2D array containing :math:`\sigma^2(z,M)`, where `M` = ``self.mass`` and redshifts are given in the initialization.
        """        
        # Scales and power
        kappa   = self.k_ext
        P_kappa = self.pk_ext*(self.cosmology.growth_factor_CDM_baryons_neutrinos(z=0., k=kappa)[0]/self.cosmology.growth_factor_CDM_baryons_neutrinos(z=self.z, k=kappa)[0])**2.
        dlnk    = np.log(kappa[1]/kappa[0])        

        # Smoothing radii
        R   = self.smoothing_radius(self.mass)
        
        # Integration in log-bins (with numpy)
        integrand = np.zeros((self.nz, len(R),len(kappa)))
        integral  = np.zeros((self.nz, len(R)))
        for iz in range(self.nz):
            for ir in range(len(R)):
                integrand[iz, ir] = kappa**3.*P_kappa[iz]/(2.*np.pi**2.)*UF.TopHat_window(kappa*R[ir])**2.
                integral[iz, ir]  = np.trapz(integrand[iz,ir], dx = dlnk)

        return integral


    #-----------------------------------------------------------------------------------------
    # NU-MASS RELATION
    #-----------------------------------------------------------------------------------------
    def nu(self):
        """
        Peak height, i.e. :math:`1.686/\sigma(z,M)`.

        :return: D array containing :math:`\\nu(z,M)`, where `M` = ``self.mass`` and redshifts are given in the initialization.
        """
        return np.transpose(self.delta_sc/np.transpose((self.sigma2())**.5))

    #-----------------------------------------------------------------------------------------
    # SHETH-TORMEN MASS FUNCTION
    #-----------------------------------------------------------------------------------------
    def mass_fun_ST(self, nu, a = 0.707, p = 0.3):
        """
        This routine returns the universal mass function by `Sheth-Tormen <https://arxiv.org/pdf/astro-ph/9901122.pdf>`_ as function of the peak height.

        :type nu: array
        :param nu: Peak height.

        :type a: float, default = 0.707
        :param a: Sheth-Tormen parameter.

        :type p: array, default = 0.3
        :param p: Sheth-Tormen parameter.

        :return: 2D array containing :math:`f_\\nu^{ST}(z,\\nu)`, where redshifts are given in the initialization.
        """
        n = nu**2.
        A = 1./(1. + 2.**(-p)*gamma(0.5-p)/np.sqrt(np.pi))
        ST = A * np.sqrt(2.*a*n/np.pi) * (1.+1./(a*n)**p) * np.exp(-a*n/2.)
        return ST

    #-----------------------------------------------------------------------------------------
    # SHETH-TORMEN BIAS
    #-----------------------------------------------------------------------------------------
    def halo_bias_ST(self, nu, a = 0.707, p = 0.3):
        """
        This routine returns the linear `Sheth-Tormen <https://arxiv.org/pdf/astro-ph/9901122.pdf>`_ halo bias.

        :type nu: array
        :param nu: Peak height.

        :type a: float, default = 0.707
        :param a: Sheth-Tormen parameter.

        :type p: array, default = 0.3
        :param p: Sheth-Tormen parameter.

        :return: 2D array containing :math:`b_\\nu^{ST}(z,\\nu)`, where redshifts are given in the initialization.
        """
        d_sc  = self.delta_sc
        return 1. + (a*nu**2.-1.)/d_sc + 2.*p/d_sc/(1.+(a*nu**2.)**p)

    #-----------------------------------------------------------------------------------------
    # HALO MASS FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_halo_mass_function(self, **kwargs):
        """
        This routine returns the halo mass function at the points specified by the array ``self.mass`` in the initialization.

        :param kwargs: Keyword arguments to pass to :func:`colibri.halo.halo.mass_fun_ST`.

        :return: 2D array containing  the halo mass function in units of :math:`h^4 \ \mathrm{Mpc}^{-3}  \ M_\odot^{-1}.` Its shape is ``(len(z), len(M)``, where `M` = ``self.mass`` and redshifts are given in the initialization.

        """    
        # Masses and nu's
        m  = self.mass
        nu = self.peak_height

        # dln(nu)/dln(m) (adding last component as equal to second-to-last)
        dln_nu = np.log(nu[:,1:]/nu[:,:-1])
        dln_m  = np.log(m[1]/m[0])
        ln_der = dln_nu/dln_m
        ln_der = np.append(ln_der[:,:], np.transpose([ln_der[:,-1]]), axis = 1)

        # ST mass function
        mass_fun = self.mass_fun_ST(nu, **kwargs)

        # Matter density today
        rho = self.cosmology.rho(0.)
        
        # Halo mass function
        hmf = rho/m**2.*ln_der*mass_fun

        return hmf


    #-----------------------------------------------------------------------------------------
    # M STAR
    #-----------------------------------------------------------------------------------------
    def M_star(self):
        """
        This routine computes the typical halo mass at redshift `z = 0`. This is defined as the
        mass for which :math:`\\nu = \\frac{1.686}{\sigma(z=0,M)} = 1`. Result in :math:`M_\odot/h`.
        """
        # Compute it at any redshift
        nu = self.peak_height[0]*self.growth_factor[0]
        func = si.interp1d(nu, self.mass)
        value = func(1.)
        return value


    #-----------------------------------------------------------------------------------------
    # FOURIER TRANSFORM OF NFW PROFILE
    #-----------------------------------------------------------------------------------------
    def u_NFW(self, c, x):
        """
        It returns the Navarro-Frenk-White (NFW) profile in Fourier space, normalized such that its integral is equal to unity.

        :param x: Abscissa.
        :type x: array

        :type c: float
        :param c: Concentration parameter.

        :return: array of size ``len(x)``
        """
        (Si_1,Ci_1) = sici(x)
        (Si_2,Ci_2) = sici((1.+c)*x)
        den         = np.log(1.+c)-c*1./(1.+c)
        num1        = np.sin(x)*(Si_2-Si_1)
        num2        = np.sin(c*x)
        num3        = np.cos(x)*(Ci_2-Ci_1)
        return 1./den*(num1+num3-num2*1./((1.+c)*x))


    #-----------------------------------------------------------------------------------------
    # CONCENTRATION PARAMETER
    #-----------------------------------------------------------------------------------------
    def conc(self, M, c0 = 9., b = 0.13):
        """
        This parameter enters in the Fourier transform of the NFW profile. It is defined as
        the ratio between the virial radius of the halo and the scale radius that appears in 
        the definition of the NFW density profile in configuration space.
        The concentration parameter has been shown to follow a log-normal distribution with mean

        .. math:

         c(M,z) = \\frac{c0}{1+z} \ \left(\\frac{M}{M_*}\\right)^{-b)}

        :type M: array
        :param M: Masses in :math:`M_\odot/h`

        :type c0: float
        :param c0: Normalization of concentration parameter formula.

        :type b: float
        :param b: Slope of concentration parameter formula. N.B. Notice that the actual exponent in the formula is `-b`.

        :return: array of size ``len(M)``
        """
        scale_mass = self.M_star()
        conc = np.zeros((self.nz, len(np.atleast_1d(M))))
        for i in range(self.nz):
            conc[i] = c0/(1.+self.z[i])*(M/scale_mass)**(-b)
        return conc



    #-----------------------------------------------------------------------------------------
    # VIRIAL RADIUS
    #-----------------------------------------------------------------------------------------
    def R_v(self, M):
        """
        Virial radius of a halo, in units of :math:`\mathrm{Mpc}/h`.

        :type M: array
        :param M: Masses in :math:`M_\odot/h`

        :return: array of size ``len(M)``
        """
        rho = self.cosmology.rho(0.)
        dv  = self.Delta_v(self.z)
        rv  = np.zeros((self.nz, len(np.atleast_1d(M))))
        for i in range(self.nz):
            rv[i] = ((3.*M)/(4.*np.pi*rho*dv[i]))**(1./3.)
        return rv


    #-----------------------------------------------------------------------------------------
    # SCALE RADIUS
    #-----------------------------------------------------------------------------------------
    def R_s(self, M, kwargs_concentration = {}):
        """
        Scale radius of a halo given the mass in Msun/h.


        :type M: array
        :param M: Masses in :math:`M_\odot/h`

        :param kwargs_concentration: Keyword arguments to pass to :func:`colibri.halo.halo.conc`.
        :type kwargs_concentration: dictionary, default = {}


        :return: array of size len(M) in Mpc/h
        """
        cc = self.conc(M, **kwargs_concentration)
        rs = np.zeros_like(cc)
        rv = self.R_v(M)
        for i in range(self.nz):
            rs[i] = rv[i]/cc[i]
        return rs

    #-----------------------------------------------------------------------------------------
    # NORMALIZATION 2 HALO
    #-----------------------------------------------------------------------------------------
    def norm_2h(self, bias, **kwargs):
        """
        Normalization to the 2-halo term (due to integration effects).

        :param kwargs: Keyword arguments to pass to :func:`colibri.halo.halo.mass_fun_ST`.

        :return: array of size ``len(z)``
        """    
        M    = self.mass
        dlnM = np.log(M[1]/M[0])
        nu   = self.peak_height
        dndM = self.load_halo_mass_function(**kwargs)
        rho  = self.cosmology.rho(0.)
        
        value = np.trapz(M**2./rho*dndM*bias, dx = dlnM)
        return 1./value


    #-----------------------------------------------------------------------------------------
    # HALO POWER SPECTRUM
    #-----------------------------------------------------------------------------------------
    def halo_Pk(self, kwargs_mass_function = {}, kwargs_concentration = {}):
        """
        It returns the halo power spectrum split in the 1-halo and 2-halo terms.

        .. math::

          P^{(1h)}(k) =  \int_0^\infty dM \ \\frac{M^2}{\\bar{\\rho}^2} \ \\frac{dn}{dM} \ u^2(k,M)

        .. math::

          P^{(2h)}(k) = \left[ \int_0^\infty dM \ \\frac{M}{\\bar{\\rho}} \\frac{dn}{dM} \ b(M) \ u(k,M) \\right]^2 P_{lin}(k)

        :param kwargs_mass_function: Keyword arguments to pass to :func:`colibri.halo.halo.mass_fun_ST`.
        :type kwargs_mass_function: dictionary, default = {}

        :param kwargs_concentration: Keyword arguments to pass to :func:`colibri.halo.halo.conc`.
        :type kwargs_concentration: dictionary, default = {}

        :return: Nothing, but the following keys are added to the ``self.Pk`` dictionary

         - ``['matter']['1-halo']`` (`2D array of shape` ``(len(z), len(k))`` ) - 1-halo term of the matter power spectrum
         - ``['matter']['2-halo']`` (`2D array of shape` ``(len(z), len(k))`` ) - 2-halo term of the matter power spectrum
         - ``['matter']['total halo']`` (`2D array of shape` ``(len(z), len(k))`` ) - Sum of 1-halo and 2-halo terms.
        """    
        # I define all the quantities I need to compute the power spectrum, so I compute
        # all of them once for all
        nu      = self.peak_height
        bias    = self.halo_bias_ST(nu, **kwargs_mass_function)
        M       = self.mass
        dlnM    = np.log(M[1]/M[0])
        dndM    = self.load_halo_mass_function(**kwargs_mass_function)
        k       = self.k
        r_s     = self.R_s(M)
        rho     = self.cosmology.rho(0.)
        c       = self.conc(M, **kwargs_concentration)
        nfw     = np.zeros((self.nz, self.nk, self.nm))
        for iz in range(self.nz):
            for i in range(self.nk):
                nfw[iz, i] = self.u_NFW(c[iz], k[i]*r_s[iz])

        # Normalization for 2-halo term
        normalization = self.norm_2h(bias, **kwargs_mass_function)

        # One-halo term damping at very large scales
        k_damp = 0.01*(1.+self.z)

        # Linear power spectrum
        if self.BAO_smearing:
            pk_linear = self.Pk['matter']['de-wiggled']
        else:
            pk_linear = self.Pk['matter']['linear']
        
        P_1h = np.zeros_like(pk_linear)
        P_2h = np.zeros_like(pk_linear)

        for iz in range(self.nz):
            for ik in range(len(self.k)):
                integrand_1h = ((M/rho)**2*dndM[iz]*nfw[iz,ik]**2)*M
                integrand_2h = (M/rho*dndM[iz]*nfw[iz,ik]*bias[iz])*M
                P_1h[iz,ik]  = np.trapz(integrand_1h, dx = dlnM)*(1.-np.exp(-(self.k[ik]/k_damp[iz])**2.))
                P_2h[iz,ik]  = np.trapz(integrand_2h, dx = dlnM)**2.*normalization[iz]**2.*pk_linear[iz,ik]

        self.Pk['matter']['1-halo']     = P_1h
        self.Pk['matter']['2-halo']     = P_2h
        self.Pk['matter']['total halo'] = P_1h + P_2h





