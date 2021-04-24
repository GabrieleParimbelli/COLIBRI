import colibri.constants as const
import colibri.cosmology as cc
import numpy as np
import colibri.useful_functions as UF
import scipy.special as ss
import scipy.interpolate as si
import scipy.integrate as sint
import scipy.misc as sm
import scipy.optimize as so
import warnings
from six.moves import xrange

########################################################################################################################
# HMcode2016: applies Halofit to a given power spectrum
########################################################################################################################
class HMcode2016():
    """
    The class ``HMcode2016`` transforms a linear input power spectrum to its non-linear counterpart using
    the Halofit model by Mead et al. (see `arXiv:1505.07833 <https://arxiv.org/abs/1505.07833>`_ .
    For massive neutrino cosmologies, one should apply this operator to the cold dark matter (plus baryons) only and then add the weighted contributions of the neutrino part.
    By calling this class, a non-linear power spectrum is returned. It accepts the following arguments,
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
        assert len(k)>200,     "k must have a length greater than 200 points"
        assert k.max()>=10.,   "Maximum wavenumber must be greater than 10 Mpc/h in order to achieve convergence"
        assert k.min()<=0.001, "Minimum wavenumber must be lower than 0.001 h/Mpc in order to achieve convergence"

        # Reading all cosmological parameters
        self.f_nu         = np.sum(cosmology.f_nu)
        self.cosmology    = cosmology

        # Raise warning if neutrino mass is not zero:
        if np.any(self.cosmology.M_nu!=0.):
            warnings.warn("Neutrino mass is different from zero. This version of HMcode2016 works with zero neutrino mass, maybe better to use TakaBird or HMcode2020?")

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
            self.pk_nw = [self.cosmology.remove_bao(self.k, self.pk[i],self.cosmology.k_eq()) for i in range(self.nz)]
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
        self.ocz    = self.cosmology.Omega_m_z(self.z)*self.cosmology.Omega_cb/self.cosmology.Omega_m

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
            try:               M_1 = 10.**(so.root(sig_int_2, 13.-1.75*(1+self.z[i]))['x'][0])
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
        return (3.*M/(4.*np.pi*self.rho_field))**(1./3.)


    #-----------------------------------------------------------------------------------------
    # SIGMA^2
    #-----------------------------------------------------------------------------------------
    def sigma2(self, k, pk):

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
        return (1.59 + 0.0314*np.log(sig8))*(1.+0.0123*np.log10(omm))*(1.+0.262*self.cosmology.f_nu_tot)

    #-----------------------------------------------------------------------------------------
    # DELTA_v
    #-----------------------------------------------------------------------------------------
    def Delta_v(self, omm):
        return 418.*omm**(-0.352)*(1.+0.916*self.cosmology.f_nu_tot)

    #-----------------------------------------------------------------------------------------
    # ALPHA
    #-----------------------------------------------------------------------------------------
    def alp(self, neff):
        return 3.24*1.85**neff

    #-----------------------------------------------------------------------------------------
    # FD
    #-----------------------------------------------------------------------------------------
    def fd(self, sigd100):
        return 0.0095*sigd100**1.37

    #-----------------------------------------------------------------------------------------
    # ETA_BLOAT
    #-----------------------------------------------------------------------------------------
    def eta_bloat(self, sig8):
        return 0.98-0.12*self.A_bar-0.3*sig8

    #-----------------------------------------------------------------------------------------
    # K_S
    #-----------------------------------------------------------------------------------------
    def k_s(self, sigd):
        return 0.584*sigd**(-1)

    #-----------------------------------------------------------------------------------------
    # FOURIER TRANSFORM OF NFW PROFILE
    #-----------------------------------------------------------------------------------------
    def u_NFW(self, c, x):
        (Si_1,Ci_1) = ss.sici(x)
        (Si_2,Ci_2) = ss.sici((1.+c)*x)
        den  = np.log(1.+c)-c*1./(1.+c)
        num1 = np.sin(x)*(Si_2-Si_1)
        num2 = np.sin(c*x)
        num3 = np.cos(x)*(Ci_2-Ci_1)
        return 1./den*(num1+num3-num2*1./((1.+c)*x))

    #-----------------------------------------------------------------------------------------
    # REDSHIFT OF FORMATION OF HALOS
    #-----------------------------------------------------------------------------------------
    def z_form(self):
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
        return self.A_bar*(1.+zf)/(1.+z)    

    #-----------------------------------------------------------------------------------------
    # M STAR
    #-----------------------------------------------------------------------------------------
    def M_star(self):
        nu = self.nu[0]
        func = si.interp1d(nu, self.mass, 'cubic')
        value = func(1.)
        return value


    #-----------------------------------------------------------------------------------------
    # SHETH-TORMEN MASS FUNCTION
    #-----------------------------------------------------------------------------------------
    def ST_mass_fun(self, nu):
        a = 0.707
        p = 0.3
        n = nu**2.
        A = 1./(1. + 2.**(-p)*ss.gamma(0.5-p)/np.sqrt(np.pi))
        ST = A * np.sqrt(2.*a*n/np.pi) * (1.+1./(a*nu**2.)**p) * np.exp(-a*nu**2./2.)
        return ST


    #-----------------------------------------------------------------------------------------
    # HALO MASS FUNCTION
    #-----------------------------------------------------------------------------------------
    def dndM(self):
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
    the Halofit model by Mead et al. (see `arXiv:2009.01858 <https://arxiv.org/pdf/2009.01858.pdf>`_ ).

    .. warning::

     It is quite slower than the one installed in CAMB or Class.


    By calling this class, the total matter non-linear power spectrum is returned. It accepts the following arguments,
    with the default values specified:

    :param z: Redshift.
    :type z: float, default = 0.0

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param pk_cc: Linear cold dark matter (plus baryons) power spectra evaluated in ``z`` and ``k`` in units of :math:`(\mathrm{Mpc}/h)^3`.
    :type pk_cc: 2D array of shape ``(len(z), len(k))``

    :param pk_mm: Linear total matter power spectra evaluated in ``z`` and ``k`` in units of :math:`(\mathrm{Mpc}/h)^3`.
    :type pk_mm: 2D array of shape ``(len(z), len(k))``

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
                 cosmology = cc.cosmo()):

        # Assertion on k
        assert len(k)>200,     "k must have a length greater than 200 points"
        assert k.max()>=10.,   "Maximum wavenumber must be greater than 10 Mpc/h in order to achieve convergence"
        assert k.min()<=0.001, "Minimum wavenumber must be lower than 0.001 h/Mpc in order to achieve convergence"

        # Reading all cosmological parameters
        self.f_nu         = np.sum(cosmology.f_nu[np.where(cosmology.M_nu!=0.)])
        self.cosmology    = cosmology

        # Redshift and scales at which all must be computed
        self.nz    = len(np.atleast_1d(z))
        self.nk    = len(np.atleast_1d(k))
        self.z     = np.atleast_1d(z)
        self.k     = np.atleast_1d(k)
        self.pk_cc = pk_cc
        self.pk_mm = pk_mm

        # Introduce smearing if required
        self.pk_nw = np.array([self.cosmology.remove_bao(self.k,self.pk_mm[i],self.cosmology.k_eq()) for i in range(self.nz)])
        sv2        = np.expand_dims([1./(6.*np.pi**2.)*np.trapz(self.k*self.pk_mm[i],x=np.log(self.k)) for i in range(self.nz)],1)
        self.pk_dw = self.pk_mm-(1.-np.exp(-self.k**2.*sv2))*(self.pk_mm-self.pk_nw)

        if np.shape(pk_cc) != (self.nz,self.nk):
            raise IndexError("pk_cc must be of shape (len(z), len(k))")
        if np.shape(pk_mm) != (self.nz,self.nk):
            raise IndexError("pk_mm must be of shape (len(z), len(k))")

        # cdm+b density
        self.rho_field  = self.cosmology.rho_crit(0.)*self.cosmology.Omega_cb*(1.+self.f_nu)

        # Initialize mass
        self.mass    = np.logspace(0., 18., 512)
        self.lnmass  = np.log(self.mass)
        self.logmass = np.log10(self.mass)
        self.dlnm    = np.log(self.mass[1]/self.mass[0])
        self.nm      = len(self.mass)
        self.rr      = self.cosmology.radius_of_mass(self.mass,var='cb',window='th')

        self.compute_nonlinear_pk()


    #-----------------------------------------------------------------------------------------
    # nonlinear_pk
    #-----------------------------------------------------------------------------------------
    def compute_nonlinear_pk(self):
        # Compute sigma8 and sigma^2
        sig8_cc = np.array([self.cosmology.compute_sigma_8(k=self.k,pk=self.pk_cc[iz]) for iz in range(self.nz)])
        sig2_cc = self.cosmology.mass_variance(self.logmass,k=self.k,pk=self.pk_cc)

        # Compute growth factors
        g_growth, G_growth = self.growth_factors(self.z)
        
        # Omega_m(z)
        omz = self.cosmology.Omega_m_z(self.z)

        # Overdensities
        Deltav = self.Delta_v(self.z, self.f_nu, omz, g_growth, G_growth)
        deltac = self.delta_c(self.z, self.f_nu, omz, g_growth, G_growth)

        # Parameters fitted by Mead et al.
        kdamp  = 0.05699*sig8_cc**(-1.0890)
        f_2h   = 0.26960*sig8_cc**( 0.9403)
        kstar  = 0.05617*sig8_cc**(-1.0130)
        eta    = 0.12810*sig8_cc**(-0.3644)
        nd_2h  = 2.85300
        B_halo = 5.19600
        
        # nu(z, M)
        peak_height = (deltac/sig2_cc.T**0.5).T

        # Redshift of formation
        zf = self.z_form(self.z, self.mass, deltac, sig2_cc)

        # Concentration
        conc = B_halo*(1+zf)/np.expand_dims(1.+self.z, 1)

        # Virial radius
        rv = ((3*np.expand_dims(self.mass,0))/(4*np.pi*self.rho_field*np.expand_dims(Deltav,1)))**(1./3.)

        # Scale radius
        rs = rv/conc

        # n_eff(z) and quasi-linear softening
        n_eff_cc = np.zeros(self.nz)
        for i in xrange(self.nz):
            # Find the mass at which sigma(M) = delta_c
            sig_int_2  = si.interp1d(self.logmass, sig2_cc[i]-deltac[i]**2.,'cubic', fill_value=0.,bounds_error=False)
            try:               M_1 = 10.**(so.root(sig_int_2, 13.-1.75*(1+self.z[i]))['x'][0])
            except ValueError: M_1 = 10.**(13.-1.75*(1+self.z[i])) # "interpolated value"
            # Spline the sigma^2(M) function and take derivative at M_1
            s2_spl      = si.InterpolatedUnivariateSpline(self.lnmass, np.log(sig2_cc[i]), k = 3)
            spl_logder  = s2_spl.derivative()
            logder      = spl_logder(np.log(M_1))
            # effective spectral index
            n_eff_cc[i] = - 3. - 3.*logder

        # Quasi-linear softening
        alpha  = np.expand_dims(1.875*1.603**n_eff_cc,1)
        
        # NFW profile, already normalized for bloating and corrected for neutrino fraction
        u_NFW = np.zeros((self.nz, self.nm, self.nk))
        eta_tmp = np.array([eta for x in xrange(self.nm)]).T
        R_bloat = peak_height**eta_tmp*rs
        for ik in range(self.nk):
            u_NFW[:,:,ik] = self.FFT_NFW_profile(conc, self.k[ik]*R_bloat)*(1-self.f_nu)

        # Halo mass function
        hmf = self.dndM(self.z, self.mass, peak_height)

        # power spectrum
        k_over_kdamp = np.outer(1./kdamp,self.k)
        k_over_kstar = np.outer(1./kstar,self.k)
        self.pk_1h   = np.zeros((self.nz, self.nk))
        self.pk_2h   = np.zeros((self.nz, self.nk))
        for iz in xrange(self.nz):
            for ik in xrange(self.nk):
                integrand            = ((self.mass/self.rho_field)**2.*hmf[iz]*u_NFW[iz,:,ik]**2.)*self.mass
                self.pk_1h[iz,ik] = np.trapz(integrand,x=self.lnmass)
        self.pk_2h  = self.pk_dw*(1.-f_2h*k_over_kdamp**nd_2h/(1.+k_over_kdamp**nd_2h))
        self.pk_1h *= k_over_kstar**4./(1.+k_over_kstar**4.)
        self.pk_nl  = (self.pk_1h**alpha + self.pk_2h**alpha)**(1./alpha)

    #-----------------------------------------------------------------------------------------
    # SIMPLIFIED HUBBLE PARAMETER AND Omega_m(a) (needed to speed up growth factor calculations)
    #-----------------------------------------------------------------------------------------
    def Hubble(self, a):
        return self.cosmology.H0*(self.cosmology.Omega_m*a**(-3) + 1.-self.cosmology.Omega_m)**0.5

    def Omega_m_a(self, a):
        return self.cosmology.Omega_m*a**(-3.)*(self.cosmology.H0/self.Hubble(a))**2.


    #-----------------------------------------------------------------------------------------
    # NON-NORMALIZED GROWTH FACTORS
    #-----------------------------------------------------------------------------------------
    def growth_factors(self, z):

        # Functions to integrate
        def derivatives(y, a):
            # Function 
            g,omega,G = y
            # Derivatives
            Oma  = self.Omega_m_a(a)
            dydt = [omega,-(3-1.5*Oma)*omega/a+1.5*Oma*g/a**2.,g/a]
            return dydt
        # Initial conditions
        epsilon = 0.01
        y0      = [epsilon, 1., epsilon]
        # Steps of integral
        a = np.sort(np.append([epsilon], 1/(1.+np.array(z))))
        # Solution
        g,_,G = sint.odeint(derivatives, y0, a).T
        # Remove first (z=99)
        g,G=np.flip(g[1:]),np.flip(G[1:])
        return g,G

    #-----------------------------------------------------------------------------------------
    # FUNCTIONS FOR delta_c AND Delta_V
    #-----------------------------------------------------------------------------------------
    def f1(self,x,y):
        p10,p11,p12,p13=-0.0069,-0.0208,0.0312,0.0021
        return p10+p11*(1-x)+p12*(1-x)**2.+p13*(1-y)
    def f2(self,x,y):
        p20,p21,p22,p23=0.0001,-0.0647,-0.0417,0.0646
        return p20+p21*(1-x)+p22*(1-x)**2.+p23*(1-y)
    def f3(self,x,y):
        p30,p31,p32,p33=-0.79,-10.17,2.51,6.51
        return p30+p31*(1-x)+p32*(1-x)**2.+p33*(1-y)
    def f4(self,x,y):
        p40,p41,p42,p43=-1.89,0.38,18.8,-15.87
        return p40+p41*(1-x)+p42*(1-x)**2.+p43*(1-y)


    #-----------------------------------------------------------------------------------------
    # CRITICAL DENSITY FOR COLLAPSE - LINEAR
    #-----------------------------------------------------------------------------------------
    def delta_c(self, z, f_nu, Omz, g, G):
        a = 1/(1+z)
        alpha_1, alpha_2 = 1,0
        delta_c0 = 1.686*(1-0.041*f_nu)
        factor = 1.+(self.f1(g/a,G/a)*np.log10(Omz)**alpha_1+self.f2(g/a,G/a)*np.log10(Omz)**alpha_2)
        return delta_c0*factor

    #-----------------------------------------------------------------------------------------
    # CRITICAL DENSITY FOR COLLAPSE - NON-LINEAR
    #-----------------------------------------------------------------------------------------
    def Delta_v(self, z, f_nu, Omz, g, G):
        a = 1/(1+z)
        alpha_3, alpha_4 = 1,2
        Delta_v0 = 177.7*(1+0.763*f_nu)
        factor = 1.+(self.f3(g/a,G/a)*np.log10(Omz)**alpha_3+self.f4(g/a,G/a)*np.log10(Omz)**alpha_4)
        return Delta_v0*factor


    #-----------------------------------------------------------------------------------------
    # REDSHIFT OF FORMATION OF HALOS
    #-----------------------------------------------------------------------------------------
    def z_form(self, z, mass, deltac, sig2):
        frac  = 0.01
        fm    = frac*mass
        nm    = len(np.atleast_1d(mass))
        nz    = len(z)
        z_tmp = np.linspace(0., 30., 1001)
        res   = np.zeros((nz, nm))
        rhs   = np.zeros((nz, nm))
        
        Dzf  = self.cosmology.D_1(z_tmp)
        zf_D = si.interp1d(Dzf, z_tmp, 'cubic')

        for iz in xrange(nz):
            m_ext, sig_ext = UF.extrapolate_log(mass, sig2[iz]**0.5, 1.e-1*frac*mass[0], 1.e1*mass[-1])
            sig_int        = si.interp1d(m_ext, sig_ext, 'cubic')
            s_fmz          = sig_int(fm)
            rhs[iz]        = self.cosmology.D_1(z[iz])*deltac[iz]/s_fmz
            for im in xrange(nm):
                try:
                    res[iz, im] = zf_D(rhs[iz,im])
                    if zf_D(rhs[iz,im]) < z[iz]:
                        res[iz, im] = z[iz]
                except ValueError: res[iz, im] = z[iz]

        return res
        
    #-----------------------------------------------------------------------------------------
    # FOURIER TRANSFORM OF NFW PROFILE
    #-----------------------------------------------------------------------------------------
    def FFT_NFW_profile(self, c, x):
        (Si_1,Ci_1) = ss.sici(x)
        (Si_2,Ci_2) = ss.sici((1.+c)*x)
        den  = np.log(1.+c)-c*1./(1.+c)
        num1 = np.sin(x)*(Si_2-Si_1)
        num2 = np.sin(c*x)
        num3 = np.cos(x)*(Ci_2-Ci_1)
        return 1./den*(num1+num3-num2*1./((1.+c)*x))

    #-----------------------------------------------------------------------------------------
    # SHETH-TORMEN MASS FUNCTION
    #-----------------------------------------------------------------------------------------
    def ST_mass_fun(self, nu):
        a = 0.707
        p = 0.3
        n = nu**2.
        A = 1./(1. + 2.**(-p)*ss.gamma(0.5-p)/np.sqrt(np.pi))
        ST = A * np.sqrt(2.*a*n/np.pi) * (1.+1./(a*n)**p)*np.exp(-a*n/2.)
        return ST


    #-----------------------------------------------------------------------------------------
    # HALO MASS FUNCTION
    #-----------------------------------------------------------------------------------------
    def dndM(self, z, M, peak_height):
        nz, nm = len(np.atleast_1d(z)), len(np.atleast_1d(M))
        dlnm   = np.diff(np.log(M))[0]
        hmf    = np.zeros((nz, nm))
        mass_fun = self.ST_mass_fun(peak_height)
        for iz in xrange(nz):    
            # derivative
            log_der = np.gradient(peak_height[iz], dlnm, edge_order = 2)/peak_height[iz]
            # Halo mass function
            hmf[iz] = self.rho_field/M**2.*log_der*mass_fun[iz]
        return hmf




########################################################################################################################
# Takahashi
########################################################################################################################
class Takahashi():
    """
    The class ``Takahashi`` transforms a linear input power spectrum to its non-linear counterpart using
    the Halofit model by Takahashi et al. (see `arXiv:1208.2701 <https://arxiv.org/abs/1208.2701>`_ ).
    By calling this class, a non-linear power spectrum is returned. It accepts the following arguments,
    with the default values specified:


    :param z: Redshift.
    :type z: float, default = 0.0

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param pk: Linear power spectra evaluated in ``z`` and ``k`` in units of :math:`(\mathrm{Mpc}/h)^3`.
    :type pk: 2D array of shape ``(len(z), len(k))``

    :param BAO_smearing: Whether to damp the BAO feature due to non-linearities.
    :type BAO_smearing: boolean, default = True

    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``

        
    :return: Nothing, but the quantity ``self.pk_nl`` is generated, a 2D array of shape ``(len(z), len(k))`` containing the non-linear matter power spectra in units of :math:`(\mathrm{Mpc}/h)^3`.

    """

    def __init__(self,
                 z,
                 k,
                 pk,
                 BAO_smearing = True,
                 cosmology = cc.cosmo()):

        # Assertion on k
        assert len(k)>200,     "k must have a length greater than 200 points"
        assert k.max()>=10.,   "Maximum wavenumber must be greater than 10 Mpc/h in order to achieve convergence"
        assert k.min()<=0.001, "Minimum wavenumber must be lower than 0.001 h/Mpc in order to achieve convergence"

        # Reading all cosmological parameters
        self.w0           = cosmology.w0
        self.wa           = cosmology.wa
        self.cosmology    = cosmology

        # Raise warning if neutrino mass is not zero:
        if np.any(self.cosmology.M_nu!=0.):
            warnings.warn("Neutrino mass is different from zero. The Takahashi halofit works best with zero neutrino mass, maybe better to use TakaBird?")

        # Redshift and scales at which all must be computed
        self.nz   = len(np.atleast_1d(z))
        self.nk   = len(np.atleast_1d(k))
        self.z    = np.atleast_1d(z)
        self.k    = np.atleast_1d(k)
        self.pk   = pk

        # Introduce smearing if required
        if BAO_smearing:
            self.pk_nw = [self.cosmology.remove_bao(self.k, self.pk[i],self.cosmology.k_eq()) for i in range(self.nz)]
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
        self.rho_field  = self.cosmology.rho_crit(0.)*self.cosmology.Omega_m

        # Initialize mass
        self.mass    = np.logspace(0., 18., 512)
        self.lnmass  = np.log10(self.mass)
        self.logmass = np.log10(self.mass)
        self.dlnm    = np.log(self.mass[1]/self.mass[0])
        self.nm      = np.size(self.mass)
        self.rr      = self.cosmology.radius_of_mass(self.mass, window = 'g')
        self.lnr     = np.log(self.rr)

        # Mass variance, non-linear scales, effective indices
        self.sigma2  = self.mass_variance(k=self.k,pk=self.pk)
        self.k_nl    = self.nonlinear_scale()
        self.n_eff   = self.effective_index()
        self.C_eff   = self.effective_curvature()
        # Omega matter as function of redshifts
        self.Omz     = self.cosmology.Omega_m_z(self.z)
        self.Olz     = self.cosmology.Omega_lambda_z(self.z)
        # Fitting factors
        frac        = self.Olz/(1.-self.Omz)
        self.f1     = frac*self.Omz**(-0.0307) + (1.-frac)*self.Omz**(-0.0732)
        self.f2     = frac*self.Omz**(-0.0585) + (1.-frac)*self.Omz**(-0.1423)
        self.f3     = frac*self.Omz**( 0.0743) + (1.-frac)*self.Omz**( 0.0725)
        self.an     = 10**(1.5222+2.8553*self.n_eff+2.3706*self.n_eff**2+0.9903*self.n_eff**3+ \
                      0.2250*self.n_eff**4-0.6038*self.C_eff +0.1749*self.Olz*(1.+self.w0+self.wa*self.z/(1.+self.z)))
        self.bn     = 10**(-0.5642 + 0.5864*self.n_eff + 0.5716*self.n_eff**2 - 1.5474*self.C_eff +0.2279*self.Olz*(1.+self.w0+self.wa*self.z/(1.+self.z)))
        self.cn     = 10**(0.3698 + 2.0404*self.n_eff + 0.8161*self.n_eff**2 + 0.5869*self.C_eff)
        self.gamman = 0.1971-0.0843*self.n_eff+0.8460*self.C_eff
        self.alphan = abs(6.0835+1.3373*self.n_eff-0.1959*self.n_eff**2-5.5274*self.C_eff)
        self.betan  = 2.0379-0.7354*self.n_eff + 0.3157*self.n_eff**2+1.2490*self.n_eff**3+ \
            0.3980*self.n_eff**4-0.1682*self.C_eff
        self.mun    = 0.0
        self.nun    = 10**(5.2105+3.6902*self.n_eff)

        # Compute!
        self.compute_nonlinear_pk()

    #-----------------------------------------------------------------------------------------
    # MASS VARIANCE
    #-----------------------------------------------------------------------------------------
    def mass_variance(self, k, pk):
        return self.cosmology.mass_variance_multipoles(logM   = self.logmass,
                                                       k      = k,
                                                       pk     = pk,
                                                       var    = 'tot',
                                                       window = 'g')

    #-----------------------------------------------------------------------------------------
    # k_SIGMA
    #-----------------------------------------------------------------------------------------
    def nonlinear_scale(self):
        nz = self.nz
        # Mass variance
        sigma2_array = self.sigma2
        s2_interp    = si.interp1d(self.rr,sigma2_array,'cubic',bounds_error=False,fill_value='extapolate')

        # Find k_sigma
        k_sigma = np.zeros(nz)
        for iz in range(nz):
            # Set min and max values possible
            Rmin,Rmax = 0.01, 10.0  #Mpc/h
            found = False;  precision = 1e-5
            while not(found):
                R = 0.5*(Rmin + Rmax)
                sigma2 = s2_interp(R)[iz]
                if abs(sigma2-1.0)<precision:  found = True
                elif sigma2>1.0:  Rmin = R
                else:             Rmax = R
            k_sigma[iz] = 1.0/R  #h/Mpc
        return k_sigma

    #-----------------------------------------------------------------------------------------
    # EFFECTIVE INDEX
    #-----------------------------------------------------------------------------------------
    def effective_index(self):
        nz = self.nz
        # Mass variance
        sigma2_array  = self.sigma2
        # ln(sigma^2) as function of ln(R)
        ln_s2_interp = si.interp1d(self.lnr,np.log(sigma2_array),'cubic', bounds_error=False,fill_value='extrapolate')
        # Find 1st derivative
        dln_sigma2_dlnR = sm.derivative(ln_s2_interp, self.lnr, dx=1e-5, n=1, order=3)
        # Interpolate derivative and evaluate it at 1/k_sigma
        dln_sigma2_dlnR_interp = si.interp1d(self.rr,dln_sigma2_dlnR,'cubic', bounds_error=False,fill_value='extrapolate')
        n_eff = np.array([-3. - dln_sigma2_dlnR_interp(1./self.k_nl[iz])[iz] for iz in range(nz)])
        return n_eff

    #-----------------------------------------------------------------------------------------
    # EFFECTIVE CURVATURE
    #-----------------------------------------------------------------------------------------
    def effective_curvature(self):
        nz = self.nz
        # Mass variance
        sigma2_array  = self.sigma2
        # ln(sigma^2) as function of ln(R)
        ln_s2_interp = si.interp1d(self.lnr,np.log(sigma2_array),'cubic', bounds_error=False,fill_value='extrapolate')
        # Find 1st derivative
        dln2_sigma2_dlnR2 = sm.derivative(ln_s2_interp, self.lnr, dx=1e-5, n=2, order=7)
        # Interpolate derivative and evaluate it at 1/k_sigma
        dln2_sigma2_dlnR2_interp = si.interp1d(self.rr,dln2_sigma2_dlnR2,'cubic', bounds_error=False,fill_value='extrapolate')
        C_eff = np.array([- dln2_sigma2_dlnR2_interp(1./self.k_nl[iz])[iz] for iz in range(nz)])
        return C_eff

    #-----------------------------------------------------------------------------------------
    # COMPUTE NONLINEAR P(k)
    #-----------------------------------------------------------------------------------------
    def compute_nonlinear_pk(self):
        self.pk_nl  = np.zeros_like(self.pk)

        an     = np.expand_dims(self.an,    0).T
        bn     = np.expand_dims(self.bn,    0).T
        cn     = np.expand_dims(self.cn,    0).T
        alphan = np.expand_dims(self.alphan,0).T
        betan  = np.expand_dims(self.betan, 0).T
        gamman = np.expand_dims(self.gamman,0).T
        mun    = np.expand_dims(self.mun,   0).T
        nun    = np.expand_dims(self.nun,   0).T
        k      = np.expand_dims(self.k,     1).T
        f1     = np.expand_dims(self.f1,    0).T
        f2     = np.expand_dims(self.f2,    0).T
        f3     = np.expand_dims(self.f3,    0).T
        # Proxy for scales
        y = np.outer(1./self.k_nl,self.k)
        # Linear term
        Delta_lin_2 = self.pk*k**3./(2.*np.pi**2.)
        # Quasi-linear term
        Delta_Q_2 = Delta_lin_2*(1.+Delta_lin_2)**betan/(1.+alphan*Delta_lin_2)*np.exp(-(y/4.+y**2./8.))
        # Halo term
        Delta_H_2_prime = an*y**(3*f1)/(1.+bn*y**f2+(cn*f3*y)**(3-gamman))
        Delta_H_2 = Delta_H_2_prime/(1.+mun/y+nun/y**2.)
        # Return power spectra
        self.pk_nl = 2.*np.pi**2.*(Delta_Q_2+Delta_H_2)/k**3.



########################################################################################################################
# TakaBird
########################################################################################################################
class TakaBird():
    """
    The class ``TakaBird`` transforms a linear input power spectrum to its non-linear counterpart using
    the Halofit model by Takahashi et al. (see `arXiv:1208.2701 <https://arxiv.org/abs/1208.2701>`_ ) corrected for the presence of massive neutrinos by Bird et al. (see `arXiv:1109.4416 <https://arxiv.org/abs/1109.4416>`_ ).
    By calling this class, a non-linear power spectrum is returned. It accepts the following arguments,
    with the default values specified:


    :param z: Redshift.
    :type z: float, default = 0.0

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param pk: Linear power spectra evaluated in ``z`` and ``k`` in units of :math:`(\mathrm{Mpc}/h)^3`.
    :type pk: 2D array of shape ``(len(z), len(k))``

    :param BAO_smearing: Whether to damp the BAO feature due to non-linearities.
    :type BAO_smearing: boolean, default = True

    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``

        
    :return: Nothing, but the quantity ``self.pk_nl`` is generated, a 2D array of shape ``(len(z), len(k))`` containing the non-linear matter power spectra in units of :math:`(\mathrm{Mpc}/h)^3`.

    """

    def __init__(self,
                 z,
                 k,
                 pk,
                 BAO_smearing = True,
                 cosmology = cc.cosmo()):

        # Assertion on k
        assert len(k)>200,     "k must have a length greater than 200 points"
        assert k.max()>=10.,   "Maximum wavenumber must be greater than 10 Mpc/h in order to achieve convergence"
        assert k.min()<=0.001, "Minimum wavenumber must be lower than 0.001 h/Mpc in order to achieve convergence"

        # Reading all cosmological parameters
        self.f_nu         = np.sum(cosmology.f_nu[np.where(cosmology.M_nu!=0.)])
        self.cosmology    = cosmology

        # Redshift and scales at which all must be computed
        self.nz   = len(np.atleast_1d(z))
        self.nk   = len(np.atleast_1d(k))
        self.z    = np.atleast_1d(z)
        self.k    = np.atleast_1d(k)
        self.pk   = pk

        # Introduce smearing if required
        if BAO_smearing:
            self.pk_nw = [self.cosmology.remove_bao(self.k, self.pk[i],self.cosmology.k_eq()) for i in range(self.nz)]
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
        self.rho_field  = self.cosmology.rho_crit(0.)*self.cosmology.Omega_m

        # Initialize mass
        self.mass    = np.logspace(0., 18., 512)
        self.lnmass  = np.log10(self.mass)
        self.logmass = np.log10(self.mass)
        self.dlnm    = np.log(self.mass[1]/self.mass[0])
        self.nm      = np.size(self.mass)
        self.rr      = self.cosmology.radius_of_mass(self.mass, window = 'g')
        self.lnr     = np.log(self.rr)

        # Mass variance, non-linear scales, effective indices
        self.sigma2  = self.mass_variance(k=self.k,pk=self.pk)
        self.k_nl    = self.nonlinear_scale()
        self.n_eff   = self.effective_index()
        self.C_eff   = self.effective_curvature()
        # Omega matter as function of redshifts
        self.Omz     = self.cosmology.Omega_m_z(self.z)
        self.Olz     = self.cosmology.Omega_lambda_z(self.z)
        # Fitting factors
        self.an     = 10**(1.4861+1.83693*self.n_eff+1.67618*self.n_eff**2+0.7940*self.n_eff**3.+            0.1670756*self.n_eff**4-0.620695*self.C_eff)
        self.bn     = 10**(0.9463+0.9466*self.n_eff+0.3084*self.n_eff**2-0.940*self.C_eff)
        self.cn     = 10**(-0.2807+0.6669*self.n_eff+0.3214*self.n_eff**2-0.0793*self.C_eff)
        self.gamman = 0.3159-0.0765*self.n_eff -0.8350*self.C_eff+0.86485+0.2989*self.n_eff+0.1631*self.C_eff
        self.alphan = 1.38848+0.3701*self.n_eff-0.1452*self.n_eff**2
        self.betan  = 0.8291+0.9854*self.n_eff+0.3400*self.n_eff**2+self.f_nu*(-6.4868+1.4373*self.n_eff**2)
        self.mun    = 10**(-3.54419+0.19086*self.n_eff)
        self.nun    = 10**(0.95897+1.2857*self.n_eff)

        frac        = self.Olz/(1.-self.Omz)
        self.f1     = frac*self.Omz**(-0.0307) + (1.-frac)*self.Omz**(-0.0732)
        self.f2     = frac*self.Omz**(-0.0585) + (1.-frac)*self.Omz**(-0.1423)
        self.f3     = frac*self.Omz**( 0.0743) + (1.-frac)*self.Omz**( 0.0725)
        # Compute!
        self.compute_nonlinear_pk()

    #-----------------------------------------------------------------------------------------
    # MASS VARIANCE
    #-----------------------------------------------------------------------------------------
    def mass_variance(self, k, pk):
        return self.cosmology.mass_variance_multipoles(logM   = self.logmass,
                                                       k      = k,
                                                       pk     = pk,
                                                       var    = 'tot',
                                                       window = 'g',
                                                       j=0,smooth=False,R_sm=0.)

    #-----------------------------------------------------------------------------------------
    # k_SIGMA
    #-----------------------------------------------------------------------------------------
    def nonlinear_scale(self):
        nz = self.nz
        # Mass variance
        sigma2_array = self.sigma2
        s2_interp    = si.interp1d(self.rr,sigma2_array,'cubic',bounds_error=False,fill_value='extapolate')

        # Find k_sigma
        k_sigma = np.zeros(nz)
        for iz in range(nz):
            # Set min and max values possible
            Rmin,Rmax = 0.01, 10.0  #Mpc/h
            found = False;  precision = 1e-5
            while not(found):
                R = 0.5*(Rmin + Rmax)
                sigma2 = s2_interp(R)[iz]
                if abs(sigma2-1.0)<precision:  found = True
                elif sigma2>1.0:  Rmin = R
                else:             Rmax = R
            k_sigma[iz] = 1.0/R  #h/Mpc
        return k_sigma

    #-----------------------------------------------------------------------------------------
    # EFFECTIVE INDEX
    #-----------------------------------------------------------------------------------------
    def effective_index(self):
        nz = self.nz
        # Mass variance
        sigma2_array  = self.sigma2
        # ln(sigma^2) as function of ln(R)
        ln_s2_interp = si.interp1d(self.lnr,np.log(sigma2_array),'cubic', bounds_error=False,fill_value='extrapolate')
        # Find 1st derivative
        dln_sigma2_dlnR = sm.derivative(ln_s2_interp, self.lnr, dx=1e-5, n=1, order=3)
        # Interpolate derivative and evaluate it at 1/k_sigma
        dln_sigma2_dlnR_interp = si.interp1d(self.rr,dln_sigma2_dlnR,'cubic', bounds_error=False,fill_value='extrapolate')
        n_eff = np.array([-3. - dln_sigma2_dlnR_interp(1./self.k_nl[iz])[iz] for iz in range(nz)])
        return n_eff

    #-----------------------------------------------------------------------------------------
    # EFFECTIVE CURVATURE
    #-----------------------------------------------------------------------------------------
    def effective_curvature(self):
        nz = self.nz
        # Mass variance
        sigma2_array  = self.sigma2
        # ln(sigma^2) as function of ln(R)
        ln_s2_interp = si.interp1d(self.lnr,np.log(sigma2_array),'cubic', bounds_error=False,fill_value='extrapolate')
        # Find 1st derivative
        dln2_sigma2_dlnR2 = sm.derivative(ln_s2_interp, self.lnr, dx=1e-5, n=2, order=7)
        # Interpolate derivative and evaluate it at 1/k_sigma
        dln2_sigma2_dlnR2_interp = si.interp1d(self.rr,dln2_sigma2_dlnR2,'cubic', bounds_error=False,fill_value='extrapolate')
        C_eff = np.array([- dln2_sigma2_dlnR2_interp(1./self.k_nl[iz])[iz] for iz in range(nz)])
        return C_eff

    #-----------------------------------------------------------------------------------------
    # COMPUTE NONLINEAR P(k)
    #-----------------------------------------------------------------------------------------
    def compute_nonlinear_pk(self):
        self.pk_nl  = np.zeros_like(self.pk)

        an     = np.expand_dims(self.an,    0).T
        bn     = np.expand_dims(self.bn,    0).T
        cn     = np.expand_dims(self.cn,    0).T
        alphan = np.expand_dims(self.alphan,0).T
        betan  = np.expand_dims(self.betan, 0).T
        gamman = np.expand_dims(self.gamman,0).T
        mun    = np.expand_dims(self.mun,   0).T
        nun    = np.expand_dims(self.nun,   0).T
        k      = np.expand_dims(self.k,     1).T
        f1     = np.expand_dims(self.f1,    0).T
        f2     = np.expand_dims(self.f2,    0).T
        f3     = np.expand_dims(self.f3,    0).T
        # Proxy for scales
        y = np.outer(1./self.k_nl,self.k)
        # Linear term
        Delta_lin_2 = self.pk*k**3./(2.*np.pi**2.)
        # Quasi-linear term
        Delta_lin_2_tilde = Delta_lin_2*(1.+47.48*self.f_nu*k**2./(1+1.5*k**2.))
        Delta_Q_2 = Delta_lin_2*(1.+Delta_lin_2_tilde)**betan/(1.+alphan*Delta_lin_2_tilde)*np.exp(-(y/4.+y**2./8.))
        # Halo term
        Delta_H_2_prime = an*y**(3*f1)/(1.+bn*y**f2+(cn*f3*y)**(3-gamman))
        Delta_H_2 = Delta_H_2_prime/(1.+mun/y+nun/y**2.)*(1+self.f_nu*0.977)
        # Return power spectra
        self.pk_nl = 2.*np.pi**2.*(Delta_Q_2+Delta_H_2)/k**3.


###########################################################################################
# nonlinear: computing the non-linear matter power spectrum in massive neutrino cosmologies
###########################################################################################

class nonlinear_pk():
    """
    The class ``nonlinear_pk`` computes the total matter non-linear power spectrum given scales, redshifts
    and cosmological parameters. To do so, it uses the so-called `CDM prescription`, where the Halofit operator
    is applied to the linear CDM+baryons power spectrum only, while the parts including neutrinos are added
    (linearly) afterwards.
    It accepts the following arguments, with the default values specified:

    :param z: Redshift.
    :type z: float, default = 0.0

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param code: Boltzmann solver to compute the linear power spectrum, to be chosen between `'camb'` and `'class'`. Also Eisenstein-Hu can be used (by setting ``code = 'eh'``) but this method cannot deal with massive neutrinos and/or evolving dark energy.
    :type code: string, default = `'camb'`

    :param kwargs_code: Keyword arguments to pass to :func:`colibri.cosmology.cosmo.camb_XPk`, :func:`colibri.cosmology.cosmo.class_XPk` or :func:`colibri.cosmology.cosmo.EisensteinHu_Pk`.
    :type kwargs_code: dictionary, default = {}

    :param halofit: Version of halofit to use: currently available 'HMcode2016', 'Takahashi'
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
        assert len(k)>200,     "k must have a length greater than 200 points"
        assert k.max()>=10.,   "Maximum wavenumber must be greater than 10 Mpc/h in order to achieve convergence"
        assert k.min()<=0.001, "Minimum wavenumber must be lower than 0.001 h/Mpc in order to achieve convergence"

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
            pk_nw = [self.cosmology.remove_bao(self.k_tmp, pk_cbcb[i],self.cosmology.k_eq()) for i in range(self.nz)]
            sv2   = [1./(6.*np.pi**2.)*np.trapz(self.k_tmp*pk_cbcb[i], x = np.log(self.k_tmp)) for i in range(self.nz)]
            pk_dw = [(pk_cbcb[i]-pk_nw[i])*np.exp(-self.k_tmp**2.*sv2[i]) + pk_nw[i] for i in range(self.nz)]
        else:
            pk_nw = pk_cbcb
            pk_dw = pk_cbcb

        # Use halofit operator on P_{cb-cb}(k)
        if   self.halofit == 'HMcode2016':
            HO = HMcode2016(z = self.z, k = self.k_tmp, pk = pk_cbcb, field = 'cb', BAO_smearing = BAO_smearing, cosmology = self.cosmology)
        elif self.halofit == 'Takahashi':
            HO = Takahashi(z = self.z, k = self.k_tmp, pk = pk_cbcb, BAO_smearing = BAO_smearing, cosmology = self.cosmology)
        elif self.halofit == 'HMcode2020':
            raise ValueError("HMcode2020 model is built to include neutrinos already, so it is useless to use it in the cold dark matter prescription. Use Takahashi or HMcode2016 to do a similar operation.")
        elif self.halofit == 'TakaBird':
            raise ValueError("TakaBird model is built to include neutrinos already, so it is useless to use it in the cold dark matter prescription. Use Takahashi or HMcode2016 to do a similar operation.")
        else:
            raise ValueError('non-linear model not recognized')

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




