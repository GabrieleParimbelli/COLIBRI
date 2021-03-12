import colibri.constants as const
import colibri.cosmology as cc
import numpy as np
import colibri.useful_functions as UF
import scipy
import scipy.special
import scipy.interpolate as si
import scipy.integrate as sint
from six.moves import xrange

########################################################################################################################
# HMcode2016: applies Halofit to a given power spectrum
########################################################################################################################


class HMcode2016():
    """
    The class ``HMcode2016`` transforms a linear input power spectrum to its non-linear counterpart using
    the Halofit model by Mead et al. (see `arXiv:1505.07833 <https://arxiv.org/abs/1505.07833>`_ and
    `arXiv:1602.02154 <https://arxiv.org/abs/1602.02154>`_).
    By calling this class, a noninear power spectrum is returned. It accepts the following arguments,
    with the default values specified:


    :param z: Redshift.
    :type z: float, default = 0.0

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param pk: Linear power spectra evaluated in ``z`` and ``k`` in units of :math:`(\mathrm{Mpc}/h)^3`.
    :type pk: 2D array of shape ``(len(z), len(k))``

    :param field: Which field to use as fundamental density, i.e. what power spectrum ``pk`` is :it should be chosen between `'cb'` for cold dark matter plus baryons and `'tot'` for total matter.
    :type field: string, default = `'cb'`

    :param BAO_smearing: Whether to damp the BAO feature due to non-linearities.
    :type BAO_smearing: boolean, default = True

    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``


    When the instance is called, the array ``self.mass = np.logspace(0., 18., 512)``, i.e. an array of masses spanning from :math:`1 M_\odot/h` to :math:`10^{18} M_\odot/h` is created, where all the mass functions are computed.

        
    :return: Nothing, but the quantity ``self.pk_nl`` is generated, a 2D array of shape ``(len(z), len(k))`` containing the non-linear matter power spectra in units of :math:`(\mathrm{Mpc}/h)^3`.

    """

    def __init__(self,
                 z,
                 k,
                 pk,
                 field = 'cb',
                 BAO_smearing = True,
                 cosmology = cc.cosmo()):

        # Assertion on k
        assert len(k)>100,     "k must have a length greater than 100 points"
        assert k.max()>=10.,   "Maximum wavenumber must be greater than 10 Mpc/h in order to achieve convergence"
        assert k.min()<=0.001, "Minimum wavenumber must be lowerer than 0.001 h/Mpc in order to achieve convergence"

        # Reading all cosmological parameters
        self.Omega_m      = cosmology.Omega_m
        self.Omega_cdm    = cosmology.Omega_cdm
        self.Omega_b      = cosmology.Omega_b
        self.Omega_lambda = cosmology.Omega_lambda
        self.Omega_K      = cosmology.Omega_K
        self.Omega_nu     = np.sum(cosmology.Omega_nu)
        self.Omega_cb     = cosmology.Omega_cb
        self.M_nu         = cosmology.M_nu
        self.N_nu         = cosmology.N_nu
        self.N_eff        = cosmology.N_eff
        self.Gamma_nu     = cosmology.Gamma_nu
        self.massive_nu   = cosmology.massive_nu
        self.massless_nu  = cosmology.N_eff - cosmology.massive_nu
        self.log10_As     = cosmology.log10_As
        self.As           = cosmology.As
        self.ns           = cosmology.ns
        self.w0           = cosmology.w0
        self.wa           = cosmology.wa
        self.T_cmb        = cosmology.T_cmb
        self.h            = cosmology.h
        self.H0           = 100.*self.h
        self.f_nu         = np.sum(cosmology.f_nu)
        self.f_cb         = cosmology.f_cb
        self.cosmology    = cosmology

        # Minimum halo concentration by Mead et al.
        self.A_bar  = 3.13

        # Redshift and scales at which all must be computed
        self.nz   = len(np.atleast_1d(z))
        self.nk   = len(np.atleast_1d(k))
        self.z    = np.atleast_1d(z)
        self.k    = np.atleast_1d(k)
        self.pk   = pk

        # Introduce smearing if required
        if BAO_smearing:
            self.pk_nw = [self.cosmology.remove_bao(self.k, self.pk[i]) for i in range(self.nz)]
            sv2        = [1./(6.*np.pi**2.)*np.trapz(self.k*self.pk[i], x = np.log(self.k)) for i in range(self.nz)]
            self.pk_dw = [(self.pk[i]-self.pk_nw[i])*np.exp(-self.k**2.*sv2[i]) + self.pk_nw[i] for i in range(self.nz)]
        else:
            self.pk_nw = self.pk
            self.pk_dw = self.pk
    

        if np.shape(pk) != (self.nz,self.nk):
            raise IndexError("pk must be of shape (len(z), len(k))")

        if self.nz == 1:
            self.z = np.asarray([z])
        else:
            self.z = np.asarray(z)
        self.k = np.asarray(k)

        # cdm+b density
        if field == 'cb':
            self.rho_field  = self.cosmology.rho_crit(0.)*self.cosmology.Omega_cb
        elif field == 'tot':
            self.rho_field  = self.cosmology.rho_crit(0.)*self.cosmology.Omega_m
        else:
            raise ValueError("Field not recognized: choose between 'cb' and 'tot.")

        # Initialize mass
        self.mass   = np.logspace(0., 18., 512)
        self.lnmass = np.log(self.mass)
        self.dlnm   = np.log(self.mass[1]/self.mass[0])
        self.nm     = np.size(self.mass)
        self.rr     = self.radius_of_mass(self.mass)

        self.compute_nonlinear_pk()


    #-----------------------------------------------------------------------------------------
    # nonlinear_pk
    #-----------------------------------------------------------------------------------------
    def compute_nonlinear_pk(self):
        """
        It returns the non-linear power spectra at ``self.k`` and ``self.z``.

        Returns
        -------

        self.k: array
            Scales (in :math:`h/\mathrm{Mpc}`).

        self.pk_nl: 2D array
            Non-linear power spectra (in :math:`(\mathrm{Mpc}/h)^3`).
        """

        # Compute sigma8 and sigma^2
        self.sig8 = np.zeros(self.nz)
        self.sig2 = np.zeros((self.nz, self.nm))
        for i in xrange(self.nz):
            self.sig8[i] = self.cosmology.compute_sigma_8(k = self.k, pk = self.pk[i])
            self.sig2[i] = self.sigma2(k = self.k, pk = self.pk[i])
        
        # Compute sigma_d at R = 100 and R = 0  (only for cb)
        self.sigd100 = self.sigma_d(R = 100.)
        self.sigd    = self.sigma_d(R = 1e-3)
        
        # Omega_m(z)
        self.omz    = self.cosmology.Omega_m_z(self.z)
        self.ocz    = self.cosmology.Omega_m_z(self.z)*self.Omega_cb/self.Omega_m

        # Parameters fitted by Mead et al.
        self.Deltav = self.Delta_v(self.ocz)
        self.deltac = self.delta_c(self.sig8, self.ocz)
        self.fdamp  = self.fd(self.sigd100)
        self.eta    = self.eta_bloat(self.sig8)
        self.k_star = self.k_s(self.sigd)

        # nu(z, M)
        self.nu = (self.deltac/(self.sig2.T)**0.5).T

        # Redshift of formation
        self.zf = self.z_form()

        # n_eff(z) and quasi-linear softening
        # concentration parameter
        # virial radius
        self.n_eff = np.zeros(self.nz)
        self.conc  = np.zeros((self.nz, self.nm))
        self.rv    = np.zeros((self.nz, self.nm))
        for i in xrange(self.nz):
            # Find the mass at which sigma(M) = delta_c
            sig_int_2  = si.interp1d(np.log10(self.mass), self.sig2[i]-self.deltac[i]**2., 'cubic')
            try:               M_1 = 10.**(scipy.optimize.root(sig_int_2, 13.-1.75*(1+self.z[i]))['x'][0])
            except ValueError: M_1 = 10.**(13.-1.75*(1+self.z[i])) # "interpolated value"
            # Spline the sigma^2(M) function and take derivative at M_1
            s2_spl      = si.InterpolatedUnivariateSpline(self.lnmass, np.log(self.sig2[i]), k = 4)
            spl_logder  = s2_spl.derivative()
            logder      = spl_logder(np.log(M_1))
            # effective spectral index
            self.n_eff[i] = - 3. - 3.*logder
            # concentration
            self.conc[i] = self.c_bull(self.zf[i],self.z[i])
            # virial radius
            self.rv[i]   = ((3*self.mass)/(4*np.pi*self.rho_field*self.Deltav[i]))**(1./3.)

        # quasi-linear softening
        self.alpha = self.alp(self.n_eff)

        # scale radius
        self.rs = self.rv/self.conc
        
        # nfw profile, already normalized for bloating
        u = np.zeros((self.nz, self.nm, self.nk))
        eta_tmp = np.array([self.eta for x in xrange(self.nm)]).T
        R_bloat = self.nu**eta_tmp*self.rs
        for i in xrange(self.nz):
            for j in xrange(self.nm):
                u[i,j] = self.u_NFW(self.conc[i,j], self.k*R_bloat[i,j])

        # halo mass function
        hmf = self.dndM()

        # power spectrum
        arg_tanh = np.outer(self.k, self.sigd)/np.sqrt(self.fdamp)
        tanh2 = np.tanh(arg_tanh.T)**2.
        self.pk_1h   = np.zeros((self.nz, self.nk))
        self.pk_2h   = np.zeros((self.nz, self.nk))
        self.pk_nl   = np.zeros((self.nz, self.nk))
        for iz in xrange(self.nz):
            for ik in xrange(self.nk):
                integrand            = ((self.mass/self.rho_field)**2.*hmf[iz]*u[iz,:,ik]**2.)*self.mass
                self.pk_1h[iz,ik] = np.trapz(integrand, x = np.log(self.mass))
            self.pk_1h[iz]  *= (1. - np.exp(-self.k/self.k_star[iz])**2.)**3. # BETTER WITH THIS EXPONENT, THAT IN MEAD ET AL. IS NOT PRESENT!!
            self.pk_2h[iz]   = self.pk_dw[iz]*(1.-self.fdamp[iz]*tanh2[iz])
            self.pk_nl[iz]   = (self.pk_1h[iz]**self.alpha[iz] + self.pk_2h[iz]**self.alpha[iz])**(1./self.alpha[iz])

        return self.k, self.pk_nl


    #-----------------------------------------------------------------------------------------
    # SMOOTHING RADIUS
    #-----------------------------------------------------------------------------------------
    def radius_of_mass(self, M):
        """
        This function computes the radius :math:`\mathrm{Mpc}/h` which contains a certain amount of mass.

        :param M: Masses in :math:`M_\odot/h`
        :type M: array

        :return: array
        """
        return (3.*M/(4.*np.pi*self.rho_field))**(1./3.)


    #-----------------------------------------------------------------------------------------
    # SIGMA^2
    #-----------------------------------------------------------------------------------------
    def sigma2(self, k, pk):
        """
        Mass variance in spheres as a function of mass itself

        :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
        :type k: array

        :param pk: Power spectrum in units of :math:`(\mathrm{Mpc}/h)^3`.  If not given, it is computed at the given redshift with CAMB.
        :type pk: array

        :return: array of shape ``len(self.mass)``, where ``self.mass`` is given in :func:`colibri.nonlinear.HMcode2016`.
        """

        # Power spectrum
        k_ext, pk_ext = UF.extrapolate_log(k, pk, 1e-6, 1e8)

        # Scales and power
        kappa   = k_ext
        P_kappa = pk_ext
        dlnk    = np.log(kappa[1]/kappa[0])        
        
        # Integration in log-bins (with numpy)
        integral  = np.zeros(len(self.rr))
        for i in xrange(len(self.rr)):
            integrand = kappa**3.*P_kappa/(2.*np.pi**2.)*UF.TopHat_window(kappa*self.rr[i])**2.
            integral[i]  = np.trapz(integrand, dx = dlnk)

        return integral

    #-----------------------------------------------------------------------------------------
    # SIGMA_d(R)
    #-----------------------------------------------------------------------------------------
    def sigma_d(self, R):
        """
        Computes the displacement with a window function of size R

        :param R: Radii in :math:`\mathrm{Mpc}/h`.
        :type R: array

        :return: array of shape ``(len(self.z), len(R))``
        """    
    
        integral = np.zeros(self.nz)
        for i in xrange(self.nz):
            # Scales and power
            k_ext, pk_ext = UF.extrapolate_log(self.k, self.pk[i], 1e-6, 1e8)

            dlnk = np.log(k_ext[1]/k_ext[0])

            # Integration in log-bins (with numpy)
            integrand = 1./3.*k_ext**3.*pk_ext/(2.*np.pi**2.)/k_ext**2.*UF.TopHat_window(k_ext*R)**2.
            integral[i]  = np.trapz(integrand, dx = dlnk)

        return integral**.5

    #-----------------------------------------------------------------------------------------
    # SIGMA_d0(R)
    #-----------------------------------------------------------------------------------------
    def sigma_d0(self):
        """
        Computes the displacement with a constant window function.

        :return: array of length ``len(self.z)``
        """        
        integral = np.zeros(self.nz)
        for i in xrange(self.nz):
            # Scales and power
            k_ext, pk_ext = UF.extrapolate_log(self.k, self.pk[i], 1e-6, 1e8)

            dlnk = np.log(k_ext[1]/k_ext[0])

            # Integration in log-bins (with numpy)
            integrand = 1./3.*k_ext**3.*pk_ext/(2.*np.pi**2.)/k_ext**2.
            integral[i]  = np.trapz(integrand, dx = dlnk)

        return integral**.5


    #-----------------------------------------------------------------------------------------
    # DELTA_c
    #-----------------------------------------------------------------------------------------
    def delta_c(self, sig8, omm):
        """
        Computes the critical density at collapse as function of :math:`\Omega_m(z)` and :math:`\sigma_8(z)`

        Parameters
        ----------

        :param sig8: Normalization of power spectrum
        :type sig8: array

        :param omm: Matter density parameters
        :type omm: array, same size of ``sig8``

        :return: array
        """
        return (1.59 + 0.0314*np.log(sig8))*(1.+0.0123*np.log10(omm))*(1.+0.262*self.cosmology.f_nu_tot)

    #-----------------------------------------------------------------------------------------
    # DELTA_v
    #-----------------------------------------------------------------------------------------
    def Delta_v(self, omm):
        """
        Computes the overdensity of a collapsed object as function of :math:`\Omega_m(z)`.

        :param omm: Matter density parameters
        :type omm: array

        :return: array
        """
        return 418.*omm**(-0.352)*(1.+0.916*self.cosmology.f_nu_tot)

    #-----------------------------------------------------------------------------------------
    # ALPHA
    #-----------------------------------------------------------------------------------------
    def alp(self, neff):
        """
        Computes the quasi-linear softening as function of effective spectral index.

        Parameters
        ----------

        :type neff: array
        :param neff: Effective spectral index

        :return: array
        """
        return 3.24*1.85**neff

    #-----------------------------------------------------------------------------------------
    # FD
    #-----------------------------------------------------------------------------------------
    def fd(self, sigd100):
        """
        Computes the 2-halo damping parameter

        :param sigd100: Displacement with a window function of size 100 :math:`\mathrm{Mpc}/h`.
        :type sigd100: array

        :return: array
        """
        return 0.0095*sigd100**1.37

    #-----------------------------------------------------------------------------------------
    # ETA_BLOAT
    #-----------------------------------------------------------------------------------------
    def eta_bloat(self, sig8):
        """
        Computes the halo bloating parameter

        :type sig8: array
        :param sig8: Normalization of power spectrum, must be same size of 'self.z'.

        :return: array
        """
        return 0.98-0.12*self.A_bar-0.3*sig8

    #-----------------------------------------------------------------------------------------
    # K_S
    #-----------------------------------------------------------------------------------------
    def k_s(self, sigd):
        """
        Computes the 1-halo damping parameter

        :param sigd: Displacement with no window function.
        :type sigd: array

        :return: array
        """
        return 0.584*sigd**(-1)

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

        (Si_1,Ci_1) = scipy.special.sici(x)
        (Si_2,Ci_2) = scipy.special.sici((1.+c)*x)
        den  = np.log(1.+c)-c*1./(1.+c)
        num1 = np.sin(x)*(Si_2-Si_1)
        num2 = np.sin(c*x)
        num3 = np.cos(x)*(Ci_2-Ci_1)
        return 1./den*(num1+num3-num2*1./((1.+c)*x))

    #-----------------------------------------------------------------------------------------
    # REDSHIFT OF FORMATION OF HALOS
    #-----------------------------------------------------------------------------------------
    def z_form(self):
        """
        Redshift of formation of a halo of its mass.

        :return: 2D array of shape ``(len(self.z), len(self.mass))``, where ``self.mass`` is given in :func:`colibri.nonlinear.HMcode2016`.
        """
        frac  = 0.01
        fm    = frac*self.mass
        z_tmp = np.linspace(0., 30., 1001)
        res   = np.zeros((self.nz, self.nm))
        rhs   = np.zeros((self.nz, self.nm))
        
        Dzf = self.cosmology.D_1(z_tmp)
        zf_D = si.interp1d(Dzf, z_tmp, 'cubic')

        for iz in xrange(self.nz):
            m_ext, sig_ext = UF.extrapolate_log(self.mass, self.sig2[iz]**0.5, 1.e-1*frac*self.mass[0], 1.e1*self.mass[-1])
            sig_int        = si.interp1d(m_ext, sig_ext, 'cubic')
            s_fmz          = sig_int(fm)
            rhs[iz]        = self.cosmology.D_1(self.z[iz])*self.deltac[iz]/s_fmz
            for im in xrange(self.nm):
                try:
                    res[iz, im] = zf_D(rhs[iz,im])
                    if zf_D(rhs[iz,im]) < self.z[iz]:
                        res[iz, im] = self.z[iz]
                except ValueError:    res[iz, im] = self.z[iz]

        return res
        
    #-----------------------------------------------------------------------------------------
    # CONCENTRATION PARAMETER
    #-----------------------------------------------------------------------------------------
    def c_bull(self, zf, z):
        """
        This parameter enters in the Fourier transform of the NFW profile. It is defined as the ratio between the virial radius of the halo and the scale radius that appears in the definition of the NFW density profile in configuration space. The concentration parameter has been shown to follow a log-normal distribution with mean given by `arXiv:9908159 <https://arxiv.org/abs/astro-ph/9908159>`_.

        :param zf: Redshift of formation.
        :type zf: float

        :param z: Redshifts.
        :type z: array

        :return: array of size ``len(z)``
        """
        return self.A_bar*(1.+zf)/(1.+z)    

    #-----------------------------------------------------------------------------------------
    # M STAR
    #-----------------------------------------------------------------------------------------
    def M_star(self):
        """
        This routine computes the typical halo mass at redshift :math:`z=0`. This is defined as the
        mass for which the peak height :math:`\\frac{\delta_c}{\sigma(M)}` is unity. Result in :math:`M_\odot/h`.
        """
        nu = self.nu[0]
        func = si.interp1d(nu, self.mass, 'cubic')
        value = func(1.)
        return value


    #-----------------------------------------------------------------------------------------
    # SHETH-TORMEN MASS FUNCTION
    #-----------------------------------------------------------------------------------------
    def ST_mass_fun(self, nu):
        """
        This routine returns the universal mass function by Sheth-Tormen as function of the peak height :math:`\\frac{\delta_c}{\sigma(M)}`.

        :type nu: array
        :param nu: Peak height

        :return: array of same shape of ``nu``.
        """
        a = 0.707
        p = 0.3
        n = nu**2.
        A = 1./(1. + 2.**(-p)*scipy.special.gamma(0.5-p)/np.sqrt(np.pi))
        ST = A * np.sqrt(2.*a*n/np.pi) * (1.+1./(a*nu**2.)**p) * np.exp(-a*nu**2./2.)
        return ST


    #-----------------------------------------------------------------------------------------
    # HALO MASS FUNCTION
    #-----------------------------------------------------------------------------------------
    def dndM(self):
        """
        This routine returns the Sheth-Tormen halo mass function at the points specified by the array ``self.mass`` in the initialization.

        :return: array in :math:`h^4 \ \mathrm{Mpc}^{-3} \ M_\odot^{-1}` of size ``(len(self.z), len(self.mass))``, where ``self.mass`` is given in :func:`colibri.nonlinear.HMcode2016`.
        """    
        m    = self.mass
        hmf  = np.zeros((self.nz, self.nm))
        for i in xrange(self.nz):    
            nu = self.nu[i]

            # derivative
            log_der = np.gradient(nu, self.dlnm, edge_order = 2)/nu
            
            # ST mass function
            mass_fun = self.ST_mass_fun(nu)

            # Halo mass function
            hmf[i] = self.rho_field/m**2.*log_der*mass_fun
        
        return hmf

########################################################################################################################
# HMcode2020: applies Halofit to a given power spectrum
########################################################################################################################

class HMcode2020():
    """
    The class ``HMcode2020`` transforms a linear input power spectrum to its non-linear counterpart using
    the Halofit model by Mead et al. 2020 (see `arXiv:2009.01858 <https://arxiv.org/abs/2009.01858>`_ .
    By calling this class, a noninear power spectrum is returned. It accepts the following arguments,
    with the default values specified:


    :param z: Redshift.
    :type z: float, default = 0.0

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param pk: Linear power spectra evaluated in ``z`` and ``k`` in units of :math:`(\mathrm{Mpc}/h)^3`.
    :type pk: 2D array of shape ``(len(z), len(k))``

    :param field: Which field to use as fundamental density, i.e. what power spectrum ``pk`` is :it should be chosen between `'cb'` for cold dark matter plus baryons and `'tot'` for total matter.
    :type field: string, default = `'cb'`

    :param BAO_smearing: Whether to damp the BAO feature due to non-linearities.
    :type BAO_smearing: boolean, default = True

    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``


    When the instance is called, the array ``self.mass = np.logspace(0., 18., 512)``, i.e. an array of masses spanning from :math:`1 M_\odot/h` to :math:`10^{18} M_\odot/h` is created, where all the mass functions are computed.

        
    :return: Nothing, but the quantity ``self.pk_nl`` is generated, a 2D array of shape ``(len(z), len(k))`` containing the non-linear matter power spectra in units of :math:`(\mathrm{Mpc}/h)^3`.

    """

    def __init__(self,
                 z,
                 k,
                 pk,
                 field = 'cb',
                 BAO_smearing = True,
                 cosmology = cc.cosmo()):

        # Assertion on k
        assert len(k)>100,     "k must have a length greater than 100 points"
        assert k.max()>=10.,   "Maximum wavenumber must be greater than 10 Mpc/h in order to achieve convergence"
        assert k.min()<=0.001, "Minimum wavenumber must be lowerer than 0.001 h/Mpc in order to achieve convergence"

        # Reading all cosmological parameters
        self.Omega_m      = cosmology.Omega_m
        self.Omega_cdm    = cosmology.Omega_cdm
        self.Omega_b      = cosmology.Omega_b
        self.Omega_lambda = cosmology.Omega_lambda
        self.Omega_K      = cosmology.Omega_K
        self.Omega_nu     = np.sum(cosmology.Omega_nu)
        self.Omega_cb     = cosmology.Omega_cb
        self.M_nu         = cosmology.M_nu
        self.N_nu         = cosmology.N_nu
        self.N_eff        = cosmology.N_eff
        self.Gamma_nu     = cosmology.Gamma_nu
        self.massive_nu   = cosmology.massive_nu
        self.massless_nu  = cosmology.N_eff - cosmology.massive_nu
        self.log10_As     = cosmology.log10_As
        self.As           = cosmology.As
        self.ns           = cosmology.ns
        self.w0           = cosmology.w0
        self.wa           = cosmology.wa
        self.T_cmb        = cosmology.T_cmb
        self.h            = cosmology.h
        self.H0           = 100.*self.h
        self.f_nu         = np.sum(cosmology.f_nu)
        self.f_cb         = cosmology.f_cb
        self.cosmology    = cosmology

        # Minimum halo concentration by Mead et al.
        self.A_bar  = 5.196

        # Redshift and scales at which all must be computed
        self.nz   = len(np.atleast_1d(z))
        self.nk   = len(np.atleast_1d(k))
        self.z    = np.atleast_1d(z)
        self.k    = np.atleast_1d(k)
        self.pk   = pk

        # Introduce smearing if required
        if BAO_smearing:
            self.pk_nw = [self.cosmology.remove_bao(self.k, self.pk[i]) for i in range(self.nz)]
            sv2        = [1./(6.*np.pi**2.)*np.trapz(self.k*self.pk[i], x = np.log(self.k)) for i in range(self.nz)]
            self.pk_dw = [(self.pk[i]-self.pk_nw[i])*np.exp(-self.k**2.*sv2[i]) + self.pk_nw[i] for i in range(self.nz)]
        else:
            self.pk_nw = self.pk
            self.pk_dw = self.pk
    

        if np.shape(pk) != (self.nz,self.nk):
            raise IndexError("pk must be of shape (len(z), len(k))")

        if self.nz == 1:
            self.z = np.asarray([z])
        else:
            self.z = np.asarray(z)
        self.k = np.asarray(k)

        # cdm+b density
        if field == 'cb':
            self.rho_field  = self.cosmology.rho_crit(0.)*self.cosmology.Omega_cb
        elif field == 'tot':
            self.rho_field  = self.cosmology.rho_crit(0.)*self.cosmology.Omega_m
        else:
            raise ValueError("Field not recognized: choose between 'cb' and 'tot.")

        # Initialize mass
        self.mass   = np.logspace(0., 18., 512)
        self.lnmass = np.log(self.mass)
        self.dlnm   = np.log(self.mass[1]/self.mass[0])
        self.nm     = np.size(self.mass)
        self.rr     = self.radius_of_mass(self.mass)

        # Growth factor and integrated growth factor
        self.g, self.G = self.growth_factors(self.z)

        self.compute_nonlinear_pk()


    #-----------------------------------------------------------------------------------------
    # nonlinear_pk
    #-----------------------------------------------------------------------------------------
    def compute_nonlinear_pk(self):
        """
        It returns the non-linear power spectra at ``self.k`` and ``self.z``.

        Returns
        -------

        self.k: array
            Scales (in :math:`h/\mathrm{Mpc}`).

        self.pk_nl: 2D array
            Non-linear power spectra (in :math:`(\mathrm{Mpc}/h)^3`).
        """

        # Compute sigma8 and sigma^2
        self.sig8 = np.zeros(self.nz)
        self.sig2 = np.zeros((self.nz, self.nm))
        for i in xrange(self.nz):
            self.sig8[i] = self.cosmology.compute_sigma_8(k = self.k, pk = self.pk[i])
            self.sig2[i] = self.sigma2(k = self.k, pk = self.pk[i])
        
        # Omega_m(z)
        self.omz    = self.cosmology.Omega_m_z(self.z)
        self.ocz    = self.cosmology.Omega_m_z(self.z)*self.Omega_cb/self.Omega_m

        # Parameters fitted by Mead et al.
        self.Deltav = self.Delta_v(self.ocz)
        self.deltac = self.delta_c(self.sig8, self.ocz)
        self.fdamp  = self.fd(self.sig8)
        self.eta    = self.eta_bloat(self.sig8)
        self.k_star = self.k_s(self.sig8)
        self.k_damp = self.k_d(self.sig8)
        self.nd     = 2.85

        # nu(z, M)
        self.nu = (self.deltac/(self.sig2.T)**0.5).T

        # Redshift of formation
        self.zf = self.z_form()

        # n_eff(z) and quasi-linear softening
        # concentration parameter
        # virial radius
        self.n_eff = np.zeros(self.nz)
        self.conc  = np.zeros((self.nz, self.nm))
        self.rv    = np.zeros((self.nz, self.nm))
        for i in xrange(self.nz):
            # Find the mass at which sigma(M) = delta_c
            sig_int_2  = si.interp1d(np.log10(self.mass), self.sig2[i]-self.deltac[i]**2., 'cubic')
            try:               M_1 = 10.**(scipy.optimize.root(sig_int_2, 13.-1.75*(1+self.z[i]))['x'][0])
            except ValueError: M_1 = 10.**(13.-1.75*(1+self.z[i])) # "interpolated value"
            # Spline the sigma^2(M) function and take derivative at M_1
            s2_spl      = si.InterpolatedUnivariateSpline(self.lnmass, np.log(self.sig2[i]), k = 4)
            spl_logder  = s2_spl.derivative()
            logder      = spl_logder(np.log(M_1))
            # effective spectral index
            self.n_eff[i] = - 3. - 3.*logder
            # concentration
            self.conc[i] = self.c_bull(self.zf[i],self.z[i])
            # virial radius
            self.rv[i]   = ((3*self.mass)/(4*np.pi*self.rho_field*self.Deltav[i]))**(1./3.)

        # quasi-linear softening
        self.alpha = self.alp(self.n_eff)

        # scale radius
        self.rs = self.rv/self.conc
        
        # nfw profile, already normalized for bloating
        u = np.zeros((self.nz, self.nm, self.nk))
        eta_tmp = np.array([self.eta for x in xrange(self.nm)]).T
        R_bloat = self.nu**eta_tmp*self.rs
        for i in xrange(self.nz):
            for j in xrange(self.nm):
                u[i,j] = self.u_NFW(self.conc[i,j], self.k*R_bloat[i,j])

        # halo mass function
        hmf = self.dndM()

        # power spectrum
        self.pk_1h   = np.zeros((self.nz, self.nk))
        self.pk_2h   = np.zeros((self.nz, self.nk))
        self.pk_nl   = np.zeros((self.nz, self.nk))
        for iz in xrange(self.nz):
            for ik in xrange(self.nk):
                integrand            = ((self.mass/self.rho_field)**2.*hmf[iz]*u[iz,:,ik]**2.)*self.mass
                self.pk_1h[iz,ik] = np.trapz(integrand, x = np.log(self.mass))*(1.-self.cosmology.f_nu_tot)**2.
            self.pk_1h[iz]  *= (self.k/self.k_star[iz])**4/(1.+(self.k/self.k_star[iz])**4)
            self.pk_2h[iz]   = self.pk_dw[iz]*(1.-self.fdamp[iz]*(self.k/self.k_damp[iz])**self.nd/(1.+(self.k/self.k_damp[iz])**self.nd))
            self.pk_nl[iz]   = (self.pk_1h[iz]**self.alpha[iz] + self.pk_2h[iz]**self.alpha[iz])**(1./self.alpha[iz])

        return self.k, self.pk_nl

    #-----------------------------------------------------------------------------------------
    # GROWTH FACTORS
    #-----------------------------------------------------------------------------------------
    def growth_factors(self, z):
        """
        Computes the growth factors as defined in the paper by Mead et al.

        :param z: Redshifts.
        :type z: array

        :return: two arrays of shape ``len(self.z)``
        """    
        G = np.zeros_like(z)
        
        def dU_dx(U,a):
            g, y = U[0], U[1]
            om  = self.cosmology.Omega_m
            fac_omz = 1.5*om*a**(-3.)/(om*a**(-3.) + (1.-om))
            return y, -(3.-fac_omz)*y/a + fac_omz*g/a**2.
        U0    = [0.001, 1.]
        a_tmp = np.linspace(0.001, 1., 1001)
        g_tmp = sint.odeint(dU_dx, U0, a_tmp)[:,0]
        g_interp = si.interp1d(a_tmp, g_tmp)
        g = g_interp(1./(1.+z))
        for iz in range(len(np.atleast_1d(z))):
            G[iz], _ = sint.quad(lambda a: g_interp(a)/a, 0.001, 1./(1.+z[iz]))

        return g, G


    #-----------------------------------------------------------------------------------------
    # SMOOTHING RADIUS
    #-----------------------------------------------------------------------------------------
    def radius_of_mass(self, M):
        """
        This function computes the radius :math:`\mathrm{Mpc}/h` which contains a certain amount of mass.

        :param M: Masses in :math:`M_\odot/h`
        :type M: array

        :return: array
        """
        return (3.*M/(4.*np.pi*self.rho_field))**(1./3.)


    #-----------------------------------------------------------------------------------------
    # SIGMA^2
    #-----------------------------------------------------------------------------------------
    def sigma2(self, k, pk):
        """
        Mass variance in spheres as a function of mass itself

        :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
        :type k: array

        :param pk: Power spectrum in units of :math:`(\mathrm{Mpc}/h)^3`.  If not given, it is computed at the given redshift with CAMB.
        :type pk: array

        :return: array of shape ``len(self.mass)``, where ``self.mass`` is given in :func:`colibri.nonlinear.HMcode2016`.
        """

        # Power spectrum
        k_ext, pk_ext = UF.extrapolate_log(k, pk, 1e-6, 1e8)

        # Scales and power
        kappa   = k_ext
        P_kappa = pk_ext
        dlnk    = np.log(kappa[1]/kappa[0])        
        
        # Integration in log-bins (with numpy)
        integral  = np.zeros(len(self.rr))
        for i in xrange(len(self.rr)):
            integrand = kappa**3.*P_kappa/(2.*np.pi**2.)*UF.TopHat_window(kappa*self.rr[i])**2.
            integral[i]  = np.trapz(integrand, dx = dlnk)

        return integral

    #-----------------------------------------------------------------------------------------
    # SIGMA_d(R)
    #-----------------------------------------------------------------------------------------
    def sigma_d(self, R):
        """
        Computes the displacement with a window function of size R

        :param R: Radii in :math:`\mathrm{Mpc}/h`.
        :type R: array

        :return: array of shape ``(len(self.z), len(R))``
        """    
    
        integral = np.zeros(self.nz)
        for i in xrange(self.nz):
            # Scales and power
            k_ext, pk_ext = UF.extrapolate_log(self.k, self.pk[i], 1e-6, 1e8)

            dlnk = np.log(k_ext[1]/k_ext[0])

            # Integration in log-bins (with numpy)
            integrand = 1./3.*k_ext**3.*pk_ext/(2.*np.pi**2.)/k_ext**2.*UF.TopHat_window(k_ext*R)**2.
            integral[i]  = np.trapz(integrand, dx = dlnk)

        return integral**.5

    #-----------------------------------------------------------------------------------------
    # SIGMA_d0(R)
    #-----------------------------------------------------------------------------------------
    def sigma_d0(self):
        """
        Computes the displacement with a constant window function.

        :return: array of length ``len(self.z)``
        """        
        integral = np.zeros(self.nz)
        for i in xrange(self.nz):
            # Scales and power
            k_ext, pk_ext = UF.extrapolate_log(self.k, self.pk[i], 1e-6, 1e8)

            dlnk = np.log(k_ext[1]/k_ext[0])

            # Integration in log-bins (with numpy)
            integrand = 1./3.*k_ext**3.*pk_ext/(2.*np.pi**2.)/k_ext**2.
            integral[i]  = np.trapz(integrand, dx = dlnk)

        return integral**.5


    #-----------------------------------------------------------------------------------------
    # DELTA_c
    #-----------------------------------------------------------------------------------------
    def delta_c(self, sig8, omm):
        """
        Computes the critical density at collapse as function of :math:`\Omega_m(z)` and :math:`\sigma_8(z)`

        Parameters
        ----------

        :param sig8: Normalization of power spectrum
        :type sig8: array

        :param omm: Matter density parameters
        :type omm: array, same size of ``sig8``

        :return: array
        """
        #return (1.59 + 0.0314*np.log(sig8))*(1.+0.0123*np.log10(omm))*(1.+0.262*self.cosmology.f_nu_tot)
        p10,p11,p12,p13 = -0.0069, -0.0208, 0.0312,0.0021
        p20,p21,p22,p23 = 0.0001, -0.0647, 0.0417, 0.0646
        alpha1,alpha2 = 1., 0.
        factor = 1.686*(1.-0.041*self.cosmology.f_nu_tot)
        f_1 = lambda x,y: p10 + p11*(1.-x) + p12*(1.-x)**2. + p13*(1.-y)
        f_2 = lambda x,y: p20 + p21*(1.-x) + p22*(1.-x)**2. + p23*(1.-y)
        summation = 1.+f_1((1.+self.z)*self.g,(1.+self.z)*self.G)*np.log10(omm)**alpha1+f_2((1.+self.z)*self.g,(1.+self.z)*self.G)*np.log10(omm)**alpha2
        return factor*summation
        
    #-----------------------------------------------------------------------------------------
    # DELTA_v
    #-----------------------------------------------------------------------------------------
    def Delta_v(self, omm):
        """
        Computes the overdensity of a collapsed object as function of :math:`\Omega_m(z)`.

        :param omm: Matter density parameters
        :type omm: array

        :return: array
        """
        p30,p31,p32,p33 = -0.79, -10.17, 2.51, 6.51
        p40,p41,p42,p43 = -1.89, 0.38, 18.8, -15.87
        alpha3,alpha4 = 1., 2.
        factor = 177.7*(1.+0.763*self.cosmology.f_nu_tot)
        f_3 = lambda x,y: p30 + p31*(1.-x) + p32*(1.-x)**2. + p33*(1.-y)
        f_4 = lambda x,y: p40 + p41*(1.-x) + p42*(1.-x)**2. + p43*(1.-y)
        summation = 1.+f_3((1.+self.z)*self.g,(1.+self.z)*self.G)*np.log10(omm)**alpha3+f_4((1.+self.z)*self.g,(1.+self.z)*self.G)*np.log10(omm)**alpha4
        return factor*summation

    #-----------------------------------------------------------------------------------------
    # ALPHA
    #-----------------------------------------------------------------------------------------
    def alp(self, neff):
        """
        Computes the quasi-linear softening as function of effective spectral index.

        Parameters
        ----------

        :type neff: array
        :param neff: Effective spectral index

        :return: array
        """
        return 1.875*(1.603)**neff

    #-----------------------------------------------------------------------------------------
    # FD
    #-----------------------------------------------------------------------------------------
    def fd(self, sig8):
        """
        Computes the 2-halo damping parameter

        :param sig8: sigma8 parameter.
        :type sig8: array

        :return: array
        """
        return 0.2696*sig8**0.9403

    #-----------------------------------------------------------------------------------------
    # ETA_BLOAT
    #-----------------------------------------------------------------------------------------
    def eta_bloat(self, sig8):
        """
        Computes the halo bloating parameter

        :type sig8: array
        :param sig8: Normalization of power spectrum, must be same size of 'self.z'.

        :return: array
        """
        return 0.1281*sig8**(-0.3644)

    #-----------------------------------------------------------------------------------------
    # K_S
    #-----------------------------------------------------------------------------------------
    def k_s(self, sig8):
        """
        Computes the 1-halo damping parameter

        :param sig8: sigma8 parameter.
        :type sig8: array

        :return: array
        """
        return 0.05618*sig8**(-1.013)

    #-----------------------------------------------------------------------------------------
    # K_D
    #-----------------------------------------------------------------------------------------
    def k_d(self, sig8):
        """
        Computes the 2-halo damping parameter

        :param sig8: sigma8 parameter.
        :type sig8: array

        :return: array
        """
        return 0.05699*sig8**(-1.089)

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

        (Si_1,Ci_1) = scipy.special.sici(x)
        (Si_2,Ci_2) = scipy.special.sici((1.+c)*x)
        den  = np.log(1.+c)-c*1./(1.+c)
        num1 = np.sin(x)*(Si_2-Si_1)
        num2 = np.sin(c*x)
        num3 = np.cos(x)*(Ci_2-Ci_1)
        return 1./den*(num1+num3-num2*1./((1.+c)*x))

    #-----------------------------------------------------------------------------------------
    # REDSHIFT OF FORMATION OF HALOS
    #-----------------------------------------------------------------------------------------
    def z_form(self):
        """
        Redshift of formation of a halo of its mass.

        :return: 2D array of shape ``(len(self.z), len(self.mass))``, where ``self.mass`` is given in :func:`colibri.nonlinear.HMcode2016`.
        """
        frac  = 0.01
        fm    = frac*self.mass
        z_tmp = np.linspace(0., 30., 1001)
        res   = np.zeros((self.nz, self.nm))
        rhs   = np.zeros((self.nz, self.nm))
        
        Dzf = self.cosmology.D_1(z_tmp)
        zf_D = si.interp1d(Dzf, z_tmp, 'cubic')

        for iz in xrange(self.nz):
            m_ext, sig_ext = UF.extrapolate_log(self.mass, self.sig2[iz]**0.5, 1.e-1*frac*self.mass[0], 1.e1*self.mass[-1])
            sig_int        = si.interp1d(m_ext, sig_ext, 'cubic')
            s_fmz          = sig_int(fm)
            rhs[iz]        = self.cosmology.D_1(self.z[iz])*self.deltac[iz]/s_fmz
            for im in xrange(self.nm):
                try:
                    res[iz, im] = zf_D(rhs[iz,im])
                    if zf_D(rhs[iz,im]) < self.z[iz]:
                        res[iz, im] = self.z[iz]
                except ValueError:    res[iz, im] = self.z[iz]

        return res
        
    #-----------------------------------------------------------------------------------------
    # CONCENTRATION PARAMETER
    #-----------------------------------------------------------------------------------------
    def c_bull(self, zf, z):
        """
        This parameter enters in the Fourier transform of the NFW profile. It is defined as the ratio between the virial radius of the halo and the scale radius that appears in the definition of the NFW density profile in configuration space. The concentration parameter has been shown to follow a log-normal distribution with mean given by `arXiv:9908159 <https://arxiv.org/abs/astro-ph/9908159>`_.

        :param zf: Redshift of formation.
        :type zf: float

        :param z: Redshifts.
        :type z: array

        :return: array of size ``len(z)``
        """
        return self.A_bar*(1.+zf)/(1.+z)    

    #-----------------------------------------------------------------------------------------
    # M STAR
    #-----------------------------------------------------------------------------------------
    def M_star(self):
        """
        This routine computes the typical halo mass at redshift :math:`z=0`. This is defined as the
        mass for which the peak height :math:`\\frac{\delta_c}{\sigma(M)}` is unity. Result in :math:`M_\odot/h`.
        """
        nu = self.nu[0]
        func = si.interp1d(nu, self.mass, 'cubic')
        value = func(1.)
        return value


    #-----------------------------------------------------------------------------------------
    # SHETH-TORMEN MASS FUNCTION
    #-----------------------------------------------------------------------------------------
    def ST_mass_fun(self, nu):
        """
        This routine returns the universal mass function by Sheth-Tormen as function of the peak height :math:`\\frac{\delta_c}{\sigma(M)}`.

        :type nu: array
        :param nu: Peak height

        :return: array of same shape of ``nu``.
        """
        a = 0.707
        p = 0.3
        n = nu**2.
        A = 1./(1. + 2.**(-p)*scipy.special.gamma(0.5-p)/np.sqrt(np.pi))
        ST = A * np.sqrt(2.*a*n/np.pi) * (1.+1./(a*nu**2.)**p) * np.exp(-a*nu**2./2.)
        return ST


    #-----------------------------------------------------------------------------------------
    # HALO MASS FUNCTION
    #-----------------------------------------------------------------------------------------
    def dndM(self):
        """
        This routine returns the Sheth-Tormen halo mass function at the points specified by the array ``self.mass`` in the initialization.

        :return: array in :math:`h^4 \ \mathrm{Mpc}^{-3} \ M_\odot^{-1}` of size ``(len(self.z), len(self.mass))``, where ``self.mass`` is given in :func:`colibri.nonlinear.HMcode2016`.
        """    
        m    = self.mass
        hmf  = np.zeros((self.nz, self.nm))
        for i in xrange(self.nz):    
            nu = self.nu[i]

            # derivative
            log_der = np.gradient(nu, self.dlnm, edge_order = 2)/nu
            
            # ST mass function
            mass_fun = self.ST_mass_fun(nu)

            # Halo mass function
            hmf[i] = self.rho_field/m**2.*log_der*mass_fun
        
        return hmf



########################################################################################################################
# HMcode2020_cdm: applies Halofit to a given power spectrum
########################################################################################################################

class HMcode2020_cdm():
    """
    The class ``HMcode2020_cdm`` transforms a linear input power spectrum to its non-linear counterpart using
    the Halofit model by Mead et al. 2020 (see `arXiv:2009.01858 <https://arxiv.org/abs/2009.01858>`_ .
    By calling this class, a noninear power spectrum is returned. It accepts the following arguments,
    with the default values specified:


    :param z: Redshift.
    :type z: float, default = 0.0

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param pk_cc: Linear cold dark matter power spectra evaluated at ``z`` and ``k`` in units of :math:`(\mathrm{Mpc}/h)^3`.
    :type pk_cc: 2D array of shape ``(len(z), len(k))``

    :param pk_mm: Linear total matter power spectra evaluated at ``z`` and ``k`` in units of :math:`(\mathrm{Mpc}/h)^3`.
    :type pk_mm: 2D array of shape ``(len(z), len(k))``

    :param BAO_smearing: Whether to damp the BAO feature due to non-linearities.
    :type BAO_smearing: boolean, default = True

    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``


    When the instance is called, the array ``self.mass = np.logspace(0., 18., 512)``, i.e. an array of masses spanning from :math:`1 M_\odot/h` to :math:`10^{18} M_\odot/h` is created, where all the mass functions are computed.

        
    :return: Nothing, but the quantity ``self.pk_nl`` is generated, a 2D array of shape ``(len(z), len(k))`` containing the non-linear matter power spectra in units of :math:`(\mathrm{Mpc}/h)^3`.

    """
    def __init__(self,
                 z,
                 k,
                 pk_cc,
                 pk_mm,
                 BAO_smearing = True,
                 cosmology = cc.cosmo()):

        # Assertion on k
        assert len(k)>100,     "k must have a length greater than 100 points"
        assert k.max()>=10.,   "Maximum wavenumber must be greater than 10 Mpc/h in order to achieve convergence"
        assert k.min()<=0.001, "Minimum wavenumber must be lowerer than 0.001 h/Mpc in order to achieve convergence"

        # Reading all cosmological parameters
        self.Omega_m      = cosmology.Omega_m
        self.Omega_cdm    = cosmology.Omega_cdm
        self.Omega_b      = cosmology.Omega_b
        self.Omega_lambda = cosmology.Omega_lambda
        self.Omega_K      = cosmology.Omega_K
        self.Omega_nu     = np.sum(cosmology.Omega_nu)
        self.Omega_cb     = cosmology.Omega_cb
        self.M_nu         = cosmology.M_nu
        self.N_nu         = cosmology.N_nu
        self.N_eff        = cosmology.N_eff
        self.Gamma_nu     = cosmology.Gamma_nu
        self.massive_nu   = cosmology.massive_nu
        self.massless_nu  = cosmology.N_eff - cosmology.massive_nu
        self.log10_As     = cosmology.log10_As
        self.As           = cosmology.As
        self.ns           = cosmology.ns
        self.w0           = cosmology.w0
        self.wa           = cosmology.wa
        self.T_cmb        = cosmology.T_cmb
        self.h            = cosmology.h
        self.H0           = 100.*self.h
        self.f_nu         = np.sum(cosmology.f_nu)
        self.f_cb         = cosmology.f_cb
        self.cosmology    = cosmology

        # Minimum halo concentration by Mead et al.
        self.A_bar  = 5.20

        # Redshift and scales at which all must be computed
        self.nz    = len(np.atleast_1d(z))
        self.nk    = len(np.atleast_1d(k))
        self.z     = np.atleast_1d(z)
        self.k     = np.atleast_1d(k)
        self.pk_cc = pk_cc
        self.pk_mm = pk_mm

        # Introduce smearing if required
        if BAO_smearing:
            self.pk_nw = [self.cosmology.remove_bao(self.k, self.pk_mm[i]) for i in range(self.nz)]
            sv2        = [1./(6.*np.pi**2.)*np.trapz(self.k*self.pk_mm[i], x = np.log(self.k)) for i in range(self.nz)]
            self.pk_dw = [(self.pk_mm[i]-self.pk_nw[i])*np.exp(-self.k**2.*sv2[i]) + self.pk_nw[i] for i in range(self.nz)]
        else:
            self.pk_nw = self.pk_mm
            self.pk_dw = self.pk_mm
    

        if np.shape(pk_mm) != (self.nz,self.nk):
            raise IndexError("pk must be of shape (len(z), len(k))")

        if self.nz == 1:
            self.z = np.asarray([z])
        else:
            self.z = np.asarray(z)
        self.k = np.asarray(k)

        # cdm+b density
        self.rho_field  = self.cosmology.rho_crit(0.)*self.cosmology.Omega_cb

        # Initialize mass
        self.mass   = np.logspace(0., 18., 512)
        self.lnmass = np.log(self.mass)
        self.dlnm   = np.log(self.mass[1]/self.mass[0])
        self.nm     = np.size(self.mass)
        self.rr     = self.radius_of_mass(self.mass)

        # Growth factor and integrated growth factor
        self.g, self.G = self.growth_factors(self.z)

        self.compute_nonlinear_pk()


    #-----------------------------------------------------------------------------------------
    # nonlinear_pk
    #-----------------------------------------------------------------------------------------
    def compute_nonlinear_pk(self):
        """
        It returns the non-linear power spectra at ``self.k`` and ``self.z``.

        Returns
        -------

        self.k: array
            Scales (in :math:`h/\mathrm{Mpc}`).

        self.pk_nl: 2D array
            Non-linear power spectra (in :math:`(\mathrm{Mpc}/h)^3`).
        """
        # Compute sigma8 and sigma^2
        self.sig8 = np.zeros(self.nz)
        self.sig2 = np.zeros((self.nz, self.nm))
        for i in xrange(self.nz):
            self.sig8[i] = self.cosmology.compute_sigma_8(k = self.k, pk = self.pk_cc[i])
            self.sig2[i] = self.sigma2(k = self.k, pk = self.pk_cc[i])
        
        # Omega_m(z)
        self.omz    = self.cosmology.Omega_m_z(self.z)
        self.ocz    = self.cosmology.Omega_m_z(self.z)*self.Omega_cb/self.Omega_m

        # Parameters fitted by Mead et al.
        self.Deltav = self.Delta_v(self.ocz)
        self.deltac = self.delta_c(self.sig8, self.ocz)
        self.fdamp  = self.fd(self.sig8)
        self.eta    = self.eta_bloat(self.sig8)
        self.k_star = self.k_s(self.sig8)
        self.k_damp = self.k_d(self.sig8)
        self.nd     = 2.85

        # nu(z, M)
        self.nu = (self.deltac/(self.sig2.T)**0.5).T

        # Redshift of formation
        self.zf = self.z_form()

        # n_eff(z) and quasi-linear softening
        # concentration parameter
        # virial radius
        self.n_eff = np.zeros(self.nz)
        self.conc  = np.zeros((self.nz, self.nm))
        self.rv    = np.zeros((self.nz, self.nm))
        for i in xrange(self.nz):
            # Find the mass at which sigma(M) = delta_c
            sig_int_2  = si.interp1d(np.log10(self.mass), self.sig2[i]-self.deltac[i]**2., 'cubic')
            try:               M_1 = 10.**(scipy.optimize.root(sig_int_2, 13.-1.75*(1+self.z[i]))['x'][0])
            except ValueError: M_1 = 10.**(13.-1.75*(1+self.z[i])) # "interpolated value"
            # Spline the sigma^2(M) function and take derivative at M_1
            s2_spl      = si.InterpolatedUnivariateSpline(self.lnmass, np.log(self.sig2[i]), k = 4)
            spl_logder  = s2_spl.derivative()
            logder      = spl_logder(np.log(M_1))
            # effective spectral index
            self.n_eff[i] = - 3. - 3.*logder
            # concentration
            self.conc[i] = self.c_bull(self.zf[i],self.z[i])
            # virial radius
            self.rv[i]   = ((3*self.mass)/(4*np.pi*self.rho_field*self.Deltav[i]))**(1./3.)

        # quasi-linear softening
        self.alpha = self.alp(self.n_eff)

        # scale radius
        self.rs = self.rv/self.conc
        
        # nfw profile, already normalized for bloating
        u = np.zeros((self.nz, self.nm, self.nk))
        eta_tmp = np.array([self.eta for x in xrange(self.nm)]).T
        R_bloat = self.nu**eta_tmp*self.rs
        for i in xrange(self.nz):
            for j in xrange(self.nm):
                u[i,j] = self.u_NFW(self.conc[i,j], self.k*R_bloat[i,j])

        # halo mass function
        hmf = self.dndM()

        # power spectrum
        self.pk_1h   = np.zeros((self.nz, self.nk))
        self.pk_2h   = np.zeros((self.nz, self.nk))
        self.pk_nl   = np.zeros((self.nz, self.nk))
        for iz in xrange(self.nz):
            for ik in xrange(self.nk):
                integrand            = ((self.mass/self.rho_field)**2.*hmf[iz]*u[iz,:,ik]**2.)*self.mass
                self.pk_1h[iz,ik] = np.trapz(integrand, x = np.log(self.mass))*(1.-self.cosmology.f_nu_tot)**2.
            self.pk_1h[iz]  *= (self.k/self.k_star[iz])**4/(1.+(self.k/self.k_star[iz])**4)
            self.pk_2h[iz]   = self.pk_dw[iz]*(1.-self.fdamp[iz]*(self.k/self.k_damp[iz])**self.nd/(1.+(self.k/self.k_damp[iz])**self.nd))
            self.pk_nl[iz]   = (self.pk_1h[iz]**self.alpha[iz] + self.pk_2h[iz]**self.alpha[iz])**(1./self.alpha[iz])

        return self.k, self.pk_nl

    #-----------------------------------------------------------------------------------------
    # GROWTH FACTORS
    #-----------------------------------------------------------------------------------------
    def growth_factors(self, z):
        """
        Computes the growth factors as defined in the paper by Mead et al.

        :param z: Redshifts.
        :type z: array

        :return: two arrays of shape ``len(self.z)``
        """       
        G = np.zeros_like(z)
        
        def dU_dx(U,a):
            g, y = U[0], U[1]
            om  = self.cosmology.Omega_m
            fac_omz = 1.5*om*a**(-3.)/(om*a**(-3.) + (1.-om))
            return y, -(3.-fac_omz)*y/a + fac_omz*g/a**2.
        U0    = [0.001, 1.]
        a_tmp = np.linspace(0.001, 1., 1001)
        g_tmp = sint.odeint(dU_dx, U0, a_tmp)[:,0]
        g_interp = si.interp1d(a_tmp, g_tmp)
        g = g_interp(1./(1.+z))
        for iz in range(len(np.atleast_1d(z))):
            G[iz], _ = sint.quad(lambda a: g_interp(a)/a, 0.001, 1./(1.+z[iz]))

        return g, G


    #-----------------------------------------------------------------------------------------
    # SMOOTHING RADIUS
    #-----------------------------------------------------------------------------------------
    def radius_of_mass(self, M):
        """
        This function computes the radius :math:`\mathrm{Mpc}/h` which contains a certain amount of mass.

        :param M: Masses in :math:`M_\odot/h`
        :type M: array

        :return: array
        """
        return (3.*M/(4.*np.pi*self.rho_field))**(1./3.)


    #-----------------------------------------------------------------------------------------
    # SIGMA^2
    #-----------------------------------------------------------------------------------------
    def sigma2(self, k, pk):
        """
        Mass variance in spheres as a function of mass itself

        :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
        :type k: array

        :param pk: Power spectrum in units of :math:`(\mathrm{Mpc}/h)^3`.  If not given, it is computed at the given redshift with CAMB.
        :type pk: array

        :return: array of shape ``len(self.mass)``, where ``self.mass`` is given in :func:`colibri.nonlinear.HMcode2016`.
        """

        # Power spectrum
        k_ext, pk_ext = UF.extrapolate_log(k, pk, 1e-6, 1e8)

        # Scales and power
        kappa   = k_ext
        P_kappa = pk_ext
        dlnk    = np.log(kappa[1]/kappa[0])        
        
        # Integration in log-bins (with numpy)
        integral  = np.zeros(len(self.rr))
        for i in xrange(len(self.rr)):
            integrand = kappa**3.*P_kappa/(2.*np.pi**2.)*UF.TopHat_window(kappa*self.rr[i])**2.
            integral[i]  = np.trapz(integrand, dx = dlnk)

        return integral

    #-----------------------------------------------------------------------------------------
    # DELTA_c
    #-----------------------------------------------------------------------------------------
    def delta_c(self, sig8, omm):
        """
        Computes the critical density at collapse as function of :math:`\Omega_m(z)` and :math:`\sigma_8(z)`

        Parameters
        ----------

        :param sig8: Normalization of power spectrum
        :type sig8: array

        :param omm: Matter density parameters
        :type omm: array, same size of ``sig8``

        :return: array
        """

        p10,p11,p12,p13 = -0.0069, -0.0208, 0.0312,0.0021
        p20,p21,p22,p23 = 0.0001, -0.0647, 0.0417, 0.0646
        alpha1,alpha2 = 1., 0.
        factor = (3./20.)*(12.*np.pi)**(2./3.)*(1.-0.041*self.cosmology.f_nu_tot)
        f_1 = lambda x,y: p10 + p11*(1.-x) + p12*(1.-x)**2. + p13*(1.-y)
        f_2 = lambda x,y: p20 + p21*(1.-x) + p22*(1.-x)**2. + p23*(1.-y)
        summation = 1.+f_1((1.+self.z)*self.g,(1.+self.z)*self.G)*np.log10(omm)**alpha1+f_2((1.+self.z)*self.g,(1.+self.z)*self.G)*np.log10(omm)**alpha2
        return factor*summation
        
    #-----------------------------------------------------------------------------------------
    # DELTA_v
    #-----------------------------------------------------------------------------------------
    def Delta_v(self, omm):
        """
        Computes the overdensity of a collapsed object as function of :math:`\Omega_m(z)`.

        :param omm: Matter density parameters
        :type omm: array

        :return: array
        """
        p30,p31,p32,p33 = -0.79, -10.17, 2.51, 6.51
        p40,p41,p42,p43 = -1.89, 0.38, 18.8, -15.87
        alpha3,alpha4 = 1., 2.
        factor = 18.*np.pi**2.*(1.+0.763*self.cosmology.f_nu_tot)
        f_3 = lambda x,y: p30 + p31*(1.-x) + p32*(1.-x)**2. + p33*(1.-y)
        f_4 = lambda x,y: p40 + p41*(1.-x) + p42*(1.-x)**2. + p43*(1.-y)
        summation = 1.+f_3((1.+self.z)*self.g,(1.+self.z)*self.G)*np.log10(omm)**alpha3+f_4((1.+self.z)*self.g,(1.+self.z)*self.G)*np.log10(omm)**alpha4
        return factor*summation

    #-----------------------------------------------------------------------------------------
    # ALPHA
    #-----------------------------------------------------------------------------------------
    def alp(self, neff):
        """
        Computes the quasi-linear softening as function of effective spectral index.

        Parameters
        ----------

        :type neff: array
        :param neff: Effective spectral index

        :return: array
        """
        return 1.875*(1.603)**neff

    #-----------------------------------------------------------------------------------------
    # FD
    #-----------------------------------------------------------------------------------------
    def fd(self, sig8):
        """
        Computes the 2-halo damping parameter

        :param sigd100: Displacement with a window function of size 100 :math:`\mathrm{Mpc}/h`.
        :type sigd100: array

        :return: array
        """
        return 0.2696*sig8**0.9403

    #-----------------------------------------------------------------------------------------
    # ETA_BLOAT
    #-----------------------------------------------------------------------------------------
    def eta_bloat(self, sig8):
        """
        Computes the halo bloating parameter

        :type sig8: array
        :param sig8: Normalization of power spectrum, must be same size of 'self.z'.

        :return: array
        """
        return 0.1281*sig8**(-0.3644)

    #-----------------------------------------------------------------------------------------
    # K_S
    #-----------------------------------------------------------------------------------------
    def k_s(self, sig8):
        """
        Computes the 1-halo damping parameter

        :param sigd: Displacement with no window function.
        :type sigd: array

        :return: array
        """
        return 0.05618*sig8**(-1.013)

    #-----------------------------------------------------------------------------------------
    # K_D
    #-----------------------------------------------------------------------------------------
    def k_d(self, sig8):
        """
        Computes the 2-halo damping parameter

        :param sig8: sigma8 parameter.
        :type sig8: array

        :return: array
        """
        return 0.05699*sig8**(-1.089)

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
        (Si_1,Ci_1) = scipy.special.sici(x)
        (Si_2,Ci_2) = scipy.special.sici((1.+c)*x)
        den  = np.log(1.+c)-c*1./(1.+c)
        num1 = np.sin(x)*(Si_2-Si_1)
        num2 = np.sin(c*x)
        num3 = np.cos(x)*(Ci_2-Ci_1)
        return 1./den*(num1+num3-num2*1./((1.+c)*x))

    #-----------------------------------------------------------------------------------------
    # REDSHIFT OF FORMATION OF HALOS
    #-----------------------------------------------------------------------------------------
    def z_form(self):
        """
        Redshift of formation of a halo of its mass.

        :return: 2D array of shape ``(len(self.z), len(self.mass))``, where ``self.mass`` is given in :func:`colibri.nonlinear.HMcode2016`.
        """
        frac  = 0.01
        fm    = frac*self.mass
        z_tmp = np.linspace(0., 30., 1001)
        res   = np.zeros((self.nz, self.nm))
        rhs   = np.zeros((self.nz, self.nm))
        
        Dzf = self.cosmology.D_1(z_tmp)
        zf_D = si.interp1d(Dzf, z_tmp, 'cubic')

        for iz in xrange(self.nz):
            m_ext, sig_ext = UF.extrapolate_log(self.mass, self.sig2[iz]**0.5, 1.e-1*frac*self.mass[0], 1.e1*self.mass[-1])
            sig_int        = si.interp1d(m_ext, sig_ext, 'cubic')
            s_fmz          = sig_int(fm)
            rhs[iz]        = self.cosmology.D_1(self.z[iz])*self.deltac[iz]/s_fmz
            for im in xrange(self.nm):
                try:
                    res[iz, im] = zf_D(rhs[iz,im])
                    if zf_D(rhs[iz,im]) < self.z[iz]:
                        res[iz, im] = self.z[iz]
                except ValueError:    res[iz, im] = self.z[iz]

        return res
        
    #-----------------------------------------------------------------------------------------
    # CONCENTRATION PARAMETER
    #-----------------------------------------------------------------------------------------
    def c_bull(self, zf, z):
        """
        This parameter enters in the Fourier transform of the NFW profile. It is defined as the ratio between the virial radius of the halo and the scale radius that appears in the definition of the NFW density profile in configuration space. The concentration parameter has been shown to follow a log-normal distribution with mean given by `arXiv:9908159 <https://arxiv.org/abs/astro-ph/9908159>`_.

        :param zf: Redshift of formation.
        :type zf: float

        :param z: Redshifts.
        :type z: array

        :return: array of size ``len(z)``
        """
        return self.A_bar*(1.+zf)/(1.+z)    

    #-----------------------------------------------------------------------------------------
    # M STAR
    #-----------------------------------------------------------------------------------------
    def M_star(self):
        """
        This routine computes the typical halo mass at redshift :math:`z=0`. This is defined as the
        mass for which the peak height :math:`\\frac{\delta_c}{\sigma(M)}` is unity. Result in :math:`M_\odot/h`.
        """
        nu = self.nu[0]
        func = si.interp1d(nu, self.mass, 'cubic')
        value = func(1.)
        return value


    #-----------------------------------------------------------------------------------------
    # SHETH-TORMEN MASS FUNCTION
    #-----------------------------------------------------------------------------------------
    def ST_mass_fun(self, nu):
        """
        This routine returns the universal mass function by Sheth-Tormen as function of the peak height :math:`\\frac{\delta_c}{\sigma(M)}`.

        :type nu: array
        :param nu: Peak height

        :return: array of same shape of ``nu``.
        """
        a = 0.707
        p = 0.3
        n = nu**2.
        A = 1./(1. + 2.**(-p)*scipy.special.gamma(0.5-p)/np.sqrt(np.pi))
        ST = A * np.sqrt(2.*a*n/np.pi) * (1.+1./(a*nu**2.)**p) * np.exp(-a*nu**2./2.)
        return ST


    #-----------------------------------------------------------------------------------------
    # HALO MASS FUNCTION
    #-----------------------------------------------------------------------------------------
    def dndM(self):
        """
        This routine returns the Sheth-Tormen halo mass function at the points specified by the array ``self.mass`` in the initialization.

        :return: array in :math:`h^4 \ \mathrm{Mpc}^{-3} \ M_\odot^{-1}` of size ``(len(self.z), len(self.mass))``, where ``self.mass`` is given in :func:`colibri.nonlinear.HMcode2016`.
        """    
        m    = self.mass
        hmf  = np.zeros((self.nz, self.nm))
        for i in xrange(self.nz):    
            nu = self.nu[i]

            # derivative
            log_der = np.gradient(nu, self.dlnm, edge_order = 2)/nu
            
            # ST mass function
            mass_fun = self.ST_mass_fun(nu)

            # Halo mass function
            hmf[i] = self.rho_field/m**2.*log_der*mass_fun
        
        return hmf







###########################################################################################
# nonlinear: computing the non-linear matter power spectrum in massive neutrino cosmologies
###########################################################################################

class nonlinear_pk(cc.cosmo):
    """
    The class ``nonlinear_pk`` computes the total matter non-linear power spectrum given scales, redshifts
    and cosmological parameters. To do so, it uses the so-called `CDM prescription`, where the Halofit operator
    is applied to the linear CDM+baryons power spectrum only, while the parts including neutrinos are added
    (linearly) afterwards.
    It inherits all the functions from the ``cosmo`` class.
    It accepts the following arguments, with the default values specified:

    :param z: Redshift.
    :type z: float, default = 0.0

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param code: Boltzmann solver to compute the linear power spectrum, to be chosen between `'camb'` and `'class'`. Also Eisenstein-Hu can be used (by setting ``code = 'eh'``) but this method cannot deal with massive neutrinos and/or evolving dark energy.
    :type code: string, default = `'camb'`

    :param kwargs_code: Keyword arguments to pass to :func:`colibri.cosmology.cosmo.camb_XPk`, :func:`colibri.cosmology.cosmo.class_XPk` or :func:`colibri.cosmology.cosmo.EisensteinHu_Pk`.
    :type kwargs_code: dictionary, default = {}

    :param halofit: Version of halofit to use: currently available 'HMcode2016', 'HMcode2020'
    :type halofit: string, default = 'HMcode2016'

    :param BAO_smearing: Whether to damp the BAO feature due to non-linearities.
    :type BAO_smearing: boolean, default = True

    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``

    :return: Nothing is explicitly returned, but the following quantities are created:

     - ``self.k`` (`array`) - Array of scales (same as input) in :math:`h/\mathrm{Mpc}`).

     - ``self.z`` (`array`) - Array of redshifts (same as input).

     - ``self.pk_nl_cbcb`` (`2D array`) - Non-linear CDM+baryons power spectrum (in :math:`(\mathrm{Mpc}/h)^3`).

     - ``self.pk_cbcb`` (`2D array`) - Linear CDM+baryons power spectrum (in :math:`(\mathrm{Mpc}/h)^3`).

     - ``self.pk_cbnu`` (`2D array`) - Linear CDM+baryons/neutrino cross power spectrum (in :math:`(\mathrm{Mpc}/h)^3`).

     - ``self.pk_nunu`` (`2D array`) - Linear neutrino power spectrum (in :math:`(\mathrm{Mpc}/h)^3`).

     - ``self.pk_nw`` (`2D array`) - No-wiggle CDM+baryons linear power spectrum (in :math:`(\mathrm{Mpc}/h)^3`). If ``BAO_smearing`` is ``False``, it is equal to ``self.pk_cbcb``.

     - ``self.pk_dw`` (`2D array`) - De-wiggled CDM+baryons linear power spectrum (in :math:`(\mathrm{Mpc}/h)^3`). If ``BAO_smearing`` is ``False``, it is equal to ``self.pk_cbcb``.

     - ``self.pk_l`` (`2D array`) - Total matter linear power spectrum (in :math:`(\mathrm{Mpc}/h)^3`).

     - ``self.pk_nl`` (`2D array`) - Total matter non-linear power spectrum (in :math:`(\mathrm{Mpc}/h)^3`).
    """
    def __init__(self,
                 z = 0.,                            # Redshift
                 k = np.logspace(-4., 2., 1001),    # Scales for P(k)
                 code = 'camb',                     # Choose among camb or class
                 kwargs_code = {},                  # keyword arguments for camb/class/eh
                 halofit = 'HMcode2016',            # Halofit model to choose
                 BAO_smearing = True,               # Introduce BAO damping
                 cosmology = cc.cosmo()):           # Cosmo instance
        # Assertion on k
        assert len(k)>100,     "k must have a length greater than 100 points"
        assert k.max()>=10.,   "Maximum wavenumber must be greater than 10 Mpc/h in order to achieve convergence"
        assert k.min()<=0.001, "Minimum wavenumber must be lowerer than 0.001 h/Mpc in order to achieve convergence"

        # Initialize cosmology
        self.cosmology = cosmology

        # Initialize model
        self.halofit = halofit

        # Fractions
        fcb = self.cosmology.f_cb
        fnu = self.cosmology.f_nu_tot

        # Redshift and scales at which all must be computed
        self.z    = np.atleast_1d(z)
        self.k    = np.atleast_1d(k)
        self.nz   = len(self.z)
        self.nk   = len(self.k)

        # Temporary scales used to compute a convergent power spectrum (they will be interpolated later)
        self.k_tmp = np.logspace(-4., 2., 501)

        # Kind of interpolation
        if self.nz>3: kind_of_interpolation = 'cubic'
        else:         kind_of_interpolation = 'linear'

        # Compute linear auto- and cross-spectra
        if code in ['camb', 'Camb', 'CAMB']:
            pk_l    = self.cosmology.camb_XPk(k = self.k_tmp, z = self.z, nonlinear = False, var_1 = ['cb','nu'], var_2 = ['cb','nu'], **kwargs_code)[1]
            pk_cbcb = pk_l['cb-cb']
            pk_cbnu = pk_l['cb-nu']
            pk_nunu = pk_l['nu-nu']
        elif code in ['class', 'Class', 'CLASS']:
            pk_l    = self.cosmology.class_XPk(k = self.k_tmp, z = self.z, nonlinear = False, var_1 = ['cb','nu'], var_2 = ['cb','nu'], **kwargs_code)[1]
            pk_cbcb = pk_l['cb-cb']
            pk_cbnu = pk_l['cb-nu']
            pk_nunu = pk_l['nu-nu']
        elif code in ['EH', 'eh', 'EisensteinHu', 'Eisenstein-Hu']:
            pk_l    = self.cosmology.EisensteinHu_Pk(k = self.k_tmp, z = self.z, **kwargs_code)[1]
            pk_cbcb = pk_l
            pk_cbnu = np.zeros_like(pk_l)
            pk_nunu = np.zeros_like(pk_l)
        else:
            raise NameError("Code not recognized. Choose among 'CAMB', 'CLASS' and 'EH'.")

        # Smooth BAO if required
        if BAO_smearing:
            pk_nw = [self.cosmology.remove_bao(self.k_tmp, pk_cbcb[i]) for i in range(self.nz)]
            sv2   = [1./(6.*np.pi**2.)*np.trapz(self.k_tmp*pk_cbcb[i], x = np.log(self.k_tmp)) for i in range(self.nz)]
            pk_dw = [(pk_cbcb[i]-pk_nw[i])*np.exp(-self.k_tmp**2.*sv2[i]) + pk_nw[i] for i in range(self.nz)]
        else:
            pk_nw = pk_cbcb
            pk_dw = pk_cbcb

        # Use halofit operator on P_{cb-cb}(k)
        if self.halofit == 'HMcode2016':
            HO = HMcode2016(z = self.z, k = self.k_tmp, pk = pk_cbcb, field = 'cb', BAO_smearing = BAO_smearing, cosmology = self.cosmology)
        elif self.halofit == 'HMcode2020':
            HO = HMcode2020(z = self.z, k = self.k_tmp, pk = pk_cbcb, field = 'cb', BAO_smearing = BAO_smearing, cosmology = self.cosmology)
        else:
            raise ValueError('model not recognized')

        # Set nonlinear quantities, de-wiggled and no-wiggle power spectra
        pk_nl_cbcb   = HO.pk_nl     # Non-linear cb-cb power spectrum
        pk_nw_cbcb   = HO.pk_nw     # No wiggles linear cb-cb power spectrum
        pk_dw_cbcb   = HO.pk_dw     # De-wiggled linear cb-cb power spectrum
        pk_nl        = fcb**2.*pk_nl_cbcb + 2.*fnu*fcb*pk_cbnu + fnu**2.*pk_nunu    # Non-linear total matter power spectrum

        # Interpolate everything
        pk_cbcb_int    = si.interp2d(self.k_tmp, self.z, pk_cbcb,    kind_of_interpolation)
        pk_cbnu_int    = si.interp2d(self.k_tmp, self.z, pk_cbnu,    kind_of_interpolation)
        pk_nunu_int    = si.interp2d(self.k_tmp, self.z, pk_nunu,    kind_of_interpolation)
        pk_nw_int      = si.interp2d(self.k_tmp, self.z, pk_nw_cbcb, kind_of_interpolation)
        pk_dw_int      = si.interp2d(self.k_tmp, self.z, pk_dw_cbcb, kind_of_interpolation)
        pk_nl_cbcb_int = si.interp2d(self.k_tmp, self.z, pk_nl_cbcb, kind_of_interpolation)
        pk_nl_int      = si.interp2d(self.k_tmp, self.z, pk_nl,      kind_of_interpolation)

        # Evaluate interpolations at required values of scales
        self.pk_cbcb    = pk_cbcb_int   (self.k, self.z)
        self.pk_cbnu    = pk_cbnu_int   (self.k, self.z)
        self.pk_nunu    = pk_nunu_int   (self.k, self.z)
        self.pk_nw      = pk_nw_int     (self.k, self.z)
        self.pk_dw      = pk_dw_int     (self.k, self.z)
        self.pk_nl_cbcb = pk_nl_cbcb_int(self.k, self.z)
        self.pk_nl      = pk_nl_int     (self.k, self.z)

        # Interpolate linear power spectrum (as dictionary)
        self.pk_l = {}
        self.pk_l['cb-cb'] = self.pk_cbcb
        self.pk_l['cb-nu'] = self.pk_cbnu
        self.pk_l['nu-nu'] = self.pk_nunu




