import colibri.constants as const
import colibri.cosmology as cc
import numpy as np
import colibri.useful_functions as UF
import scipy.special as ss
import scipy.interpolate as si
import scipy.integrate as sint
import scipy.misc as sm
import scipy.optimize as so
from scipy.ndimage import gaussian_filter1d
import warnings

########################################################################################################################
# HMcode2016: applies Halofit to a given power spectrum
########################################################################################################################
class HMcode2016():
    """
    The class ``HMcode2016`` transforms a linear input power spectrum to its non-linear counterpart using
    the Halofit model by Mead et al. (see `arXiv:1602.02154 <https://arxiv.org/abs/1602.02154>`_ .
    By calling this class, a non-linear power spectrum is returned. It accepts the following arguments,
    with the default values specified:


    :param z: Redshift.
    :type z: array

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param pk: Linear power spectra evaluated in ``z`` and ``k`` in units of :math:`(\mathrm{Mpc}/h)^3`.
    :type pk: 2D array of shape ``(len(z), len(k))``

    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``


    When the instance is called, the array ``self.mass = np.logspace(2., 18., 512)``, i.e. an array of masses spanning from :math:`100 M_\odot/h` to :math:`10^{18} M_\odot/h` is created, where all the mass functions are computed.

        
    :return: Nothing, but the quantity ``self.pk_nl`` is generated, a 2D array of shape ``(len(z), len(k))`` containing the non-linear matter power spectra in units of :math:`(\mathrm{Mpc}/h)^3`.

    """

    def __init__(self,
                 z,
                 k,
                 pk,
                 cosmology = cc.cosmo()):

        # Assertion on k
        assert len(k)>200,     "k must have a length greater than 200 points"
        assert k.max()>=10.,   "Maximum wavenumber must be greater than 10 h/Mpc in order to achieve convergence"
        assert k.min()<=0.001, "Minimum wavenumber must be lower than 0.001 h/Mpc in order to achieve convergence"
        # Reading all cosmological parameters
        self.f_nu         = cosmology.f_nu
        self.cosmology    = cosmology
        # Minimum halo concentration by Mead et al.
        self.A_bar  = 3.13
        # Redshift and scales at which all must be computed
        self.nz   = len(np.atleast_1d(z))
        self.nk   = len(np.atleast_1d(k))
        self.z    = np.atleast_1d(z)
        self.k    = np.atleast_1d(k)
        self.pk   = pk
        # Extended versions of P(k) for convergence of mass variance
        pk_ext = []
        for iz in range(self.nz):
            k_ext, pk_tmp = UF.extrapolate_log(self.k,self.pk[iz],self.k.min(),1e4)
            pk_ext.append(pk_tmp)
        self.k_ext  = k_ext
        self.pk_ext = np.array(pk_ext)
        del pk_ext, k_ext
        if np.shape(pk) != (self.nz,self.nk):
            raise IndexError("pk must be of shape (len(z), len(k))")
        # density
        self.rho_field  = self.cosmology.rho_crit(0.)*self.cosmology.Omega_m
        # Initialize mass
        self.mass   = np.logspace(2., 18., 512)
        self.lnmass = np.log(self.mass)
        self.dlnm   = np.log(self.mass[1]/self.mass[0])
        self.nm     = np.size(self.mass)
        self.rr     = self.radius_of_mass(self.mass)
        # GO!
        self.compute_nonlinear_pk()

    #-----------------------------------------------------------------------------------------
    # nonlinear_pk
    #-----------------------------------------------------------------------------------------
    def compute_nonlinear_pk(self):
        # CDM Pk
        pk_cc     = self.pk_ext*(self.cosmology.growth_cb_unnormalized(self.k_ext,self.z)/self.cosmology.growth_cbnu_unnormalized(self.k_ext,self.z))**2.
        # sigma^2
        self.sig2 = self.cosmology.mass_variance(logM = np.log10(self.mass), k = self.k_ext, pk = pk_cc, var = 'tot', window = 'th')
        self.sig8 = self.cosmology.compute_sigma_8(k = self.k_ext, pk = self.pk_ext)
        # Compute sigma_d at R = 100 and R = 0  (only for cb)
        self.sigd100 = self.sigma_d(R = 100.)
        self.sigd    = self.sigma_d(R = 1e-4)
        # Omega_m(z)
        self.omz    = self.cosmology.Omega_m_z(self.z)
        # Parameters fitted by Mead et al.
        self.Deltav = self.Delta_v(self.omz)
        self.deltac = self.delta_c(self.sig8, self.omz)
        self.fdamp  = self.fd(self.sigd100)
        self.eta    = self.eta_bloat(self.sig8)
        self.k_star = self.k_s(self.sigd)
        # Peak height(z, M)
        self.nu = (self.deltac/(self.sig2.T)**0.5).T
        # Redshift of formation
        self.zf = self.z_form()
        # n_eff(z) and quasi-linear softening
        # concentration parameter
        # virial radius
        self.n_eff = np.zeros(self.nz)
        self.conc  = np.zeros((self.nz, self.nm))
        self.rv    = np.zeros((self.nz, self.nm))
        for i in range(self.nz):
            # Find the mass at which sigma(M) = delta_c
            sig_int_2  = si.interp1d(np.log10(self.mass), self.sig2[i]-self.deltac[i]**2., 'cubic',fill_value='extrapolate',bounds_error=False)
            M_1 = 10.**(so.root(sig_int_2, 13.)['x'][0])
            # Spline the sigma^2(M) function and take derivative at M_1
            s2_spl      = si.InterpolatedUnivariateSpline(self.lnmass, np.log(self.sig2[i]), k = 4)
            spl_logder  = s2_spl.derivative()
            logder      = spl_logder(np.log(M_1))
            # Effective spectral index
            self.n_eff[i] = - 3. - 3.*logder
            # Concentration
            self.conc[i] = self.c_bull(self.zf[i],self.z[i])
            # Virial radius
            self.rv[i]   = ((3*self.mass)/(4*np.pi*self.rho_field*self.Deltav[i]))**(1./3.)
        # Concentration difference for w0-wa
        if (self.cosmology.Omega_K!=0.) or (self.cosmology.w0!=-1.) or (self.cosmology.wa!=0.):
            z_corr_high   = 10.
            g_lcdm_high   = self.growth_factors_lcdm(z_corr_high)
            g_high        = self.growth_factors(z_corr_high)
            self.conc    *= np.expand_dims((g_high/g_lcdm_high)**1.5,1)
        # Quasi-linear softening
        self.alpha = self.alp(self.n_eff)
        # Scale radius
        self.rs = self.rv/self.conc
        # NFW profile, already normalized for bloating
        u = np.zeros((self.nz, self.nm, self.nk))
        eta_tmp = np.array([self.eta for x in range(self.nm)]).T
        R_bloat = self.nu**eta_tmp*self.rs
        for i in range(self.nz):
            for j in range(self.nm):
                u[i,j] = self.u_NFW(self.conc[i,j], self.k*R_bloat[i,j])
        # Halo mass function
        hmf = self.dndM()
        # Power spectra
        arg_tanh     = np.outer(self.k, self.sigd)/np.sqrt(self.fdamp)
        tanh2        = np.tanh(arg_tanh.T)**2.
        self.pk_1h   = np.zeros((self.nz, self.nk))
        self.pk_2h   = np.zeros((self.nz, self.nk))
        self.pk_nl   = np.zeros((self.nz, self.nk))
        for iz in range(self.nz):
            for ik in range(self.nk):
                integrand         = ((self.mass/self.rho_field)**2.*hmf[iz]*u[iz,:,ik]**2.)*self.mass
                self.pk_1h[iz,ik] = sint.simps(integrand, x = np.log(self.mass))
            # Power spectra (BETTER WITH '3' EXPONENT, THAT IN MEAD ET AL. IS NOT PRESENT!!)
            self.pk_1h[iz]  *= (1.-np.exp(-self.k/self.k_star[iz])**2.)**3.
            self.pk_2h[iz]   = self.pk[iz]*(1.-self.fdamp[iz]*tanh2[iz])
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
        for i in range(len(self.rr)):
            integrand = kappa**3.*P_kappa/(2.*np.pi**2.)*UF.TopHat_window(kappa*self.rr[i])**2.
            integral[i]  = np.trapz(integrand, dx = dlnk)
        return integral

    #-----------------------------------------------------------------------------------------
    # SIGMA_d(R)
    #-----------------------------------------------------------------------------------------
    def sigma_d(self, R):
        integral = np.zeros(self.nz)
        for i in range(self.nz):
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
        for i in range(self.nz):
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
        return (1.59 + 0.0314*np.log(sig8))*(1.+0.0123*np.log10(omm))*(1.+0.262*self.f_nu)

    #-----------------------------------------------------------------------------------------
    # DELTA_v
    #-----------------------------------------------------------------------------------------
    def Delta_v(self, omm):
        return 418.*omm**(-0.352)*(1.+0.916*self.f_nu)

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
        return 0.603-0.3*sig8

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
        Dzf  = self.growth_LCDM_base(z_tmp)
        zf_D = si.interp1d(Dzf, z_tmp, 'cubic')
        for iz in range(self.nz):
            m_ext, sig_ext = UF.extrapolate_log(self.mass, self.sig2[iz]**0.5, 1.e-1*frac*self.mass[0], 1.e1*self.mass[-1])
            sig_int        = si.interp1d(m_ext, sig_ext, 'cubic')
            s_fmz          = sig_int(fm)
            rhs[iz]        = self.growth_LCDM_base(self.z[iz])*self.deltac[iz]/s_fmz
            for im in range(self.nm):
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
        ST = A * np.sqrt(2.*a*n/np.pi) * (1.+1./(a*n)**p) * np.exp(-a*n/2.)
        return ST

    #-----------------------------------------------------------------------------------------
    # HALO MASS FUNCTION
    #-----------------------------------------------------------------------------------------
    def dndM(self):
        m    = self.mass
        hmf  = np.zeros((self.nz, self.nm))
        for i in range(self.nz):    
            nu = self.nu[i]
            # derivative
            log_der = np.gradient(nu, self.dlnm, edge_order = 2)/nu
            # ST mass function
            mass_fun = self.ST_mass_fun(nu)
            # Halo mass function
            hmf[i] = self.rho_field/m**2.*log_der*mass_fun
        return hmf

    #-----------------------------------------------------------------------------------------
    # SIMPLIFIED HUBBLE PARAMETER AND Omega_m(a) (needed to speed up growth factor calculations)
    #-----------------------------------------------------------------------------------------
    def Hubble(self, a):
        return (self.cosmology.Omega_m*a**(-3.) +
               (1-self.cosmology.Omega_m-self.cosmology.Omega_lambda)*a**-2. +
               (self.cosmology.Omega_lambda)*self.X_de(a))**0.5

    def AH(self, a):
        wde = self.cosmology.w0+(1.-a)*self.cosmology.wa
        return -0.5*(self.cosmology.Omega_m*a**(-3)+self.cosmology.Omega_lambda*(1.+3.*wde)*self.X_de(a))

    def Omega_m_a(self, a):
        return self.cosmology.Omega_m*a**(-3.)/self.Hubble(a)**2.

    def X_de(self, a):
        return a**(-3.*(1.+self.cosmology.w0+self.cosmology.wa))*np.exp(-3.*self.cosmology.wa*(1.-a))

    #-----------------------------------------------------------------------------------------
    # NON-NORMALIZED GROWTH FACTORS
    #-----------------------------------------------------------------------------------------
    def growth_factors(self, z):
        z = np.atleast_1d(z)
        # Functions to integrate
        def derivatives(y, a):
            # Function 
            g,omega = y
            # Derivatives
            Oma   = self.Omega_m_a(a)
            acce  = self.AH(a)
            dydt  = [omega,-(2+acce/self.Hubble(a)**2.)*omega/a+1.5*Oma*g/a**2.]
            return dydt
        # Initial conditions (z=99)
        epsilon = 0.01
        y0      = [0.01, 1.]
        # Steps of integral
        a = np.sort(np.concatenate(([epsilon],  1/(1.+z), [1.])))
        # Solution
        g,_ = sint.odeint(derivatives, y0, a).T
        # Divide by g(z=0)
        g  /= g[-1]
        # Remove first (z=99) and last (z=0)
        g = np.flip(g[1:-1])
        return g

    # for LCDM
    def growth_factors_lcdm(self, z):
        gam = 0.55
        z   = np.atleast_1d(z)
        Om  = self.cosmology.Omega_m
        Ol  = self.cosmology.Omega_lambda
        Oma = lambda a: Om*a**(-3.)/(Om*a**(-3)+(1-Om))
        result = np.exp(np.array([sint.quad(lambda a: Oma(a)**gam/a, 1., 1./(1.+zz))[0] for zz in z]))
        return result

    def growth_LCDM_base(self, z):
        a  = 1/(1.+z)
        Om = self.cosmology.Omega_m
        return a*ss.hyp2f1(1/3., 1., 11/6., -a**3/Om*(1.-Om))/ss.hyp2f1(1/3., 1., 11/6., -(1.-Om)/Om)


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
    :type z: array

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param pk_cc: Linear cold dark matter (plus baryons) power spectra evaluated in ``z`` and ``k`` in units of :math:`(\mathrm{Mpc}/h)^3`.
    :type pk_cc: 2D array of shape ``(len(z), len(k))``

    :param pk_mm: Linear total matter power spectra evaluated in ``z`` and ``k`` in units of :math:`(\mathrm{Mpc}/h)^3`.
    :type pk_mm: 2D array of shape ``(len(z), len(k))``

    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``


    When the instance is called, the array ``self.mass = np.logspace(2., 18., 512)``, i.e. an array of masses spanning from :math:`100 M_\odot/h` to :math:`10^{18} M_\odot/h` is created, where all the mass functions are computed.

        
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
        self.f_nu         = cosmology.f_nu
        self.cosmology    = cosmology
        # Redshift and scales at which all must be computed
        self.nz    = len(np.atleast_1d(z))
        self.nk    = len(np.atleast_1d(k))
        self.z     = np.atleast_1d(z)
        self.k     = np.atleast_1d(k)
        self.pk_cc = pk_cc
        self.pk_mm = pk_mm
        # Extended versions of P(k) for convergence of mass variance
        pk_cc_ext = []
        for iz in range(self.nz):
            k_ext, pk_tmp = UF.extrapolate_log(self.k,self.pk_cc[iz],self.k.min(),1e4)
            pk_cc_ext.append(pk_tmp)
        self.k_ext     = k_ext
        self.pk_cc_ext = np.array(pk_cc_ext)
        del pk_cc_ext, k_ext
        # Introduce smearing if required
        self.pk_nw = np.array([self.cosmology.remove_bao(self.k,self.pk_mm[i],self.cosmology.k_eq()*0.6)
                                 for i in range(self.nz)])
        #self.pk_nw = np.array([self.remove_bao_gaussian_filtering(k=self.k, pk=self.pk_mm[iz])
        #                         for iz in range(self.nz)])
        sv2        = np.expand_dims([1./(6.*np.pi**2.)*np.trapz(self.k*self.pk_mm[i],x=np.log(self.k)) for i in range(self.nz)],1)
        self.pk_dw = self.pk_mm-(1.-np.exp(-self.k**2.*sv2))*(self.pk_mm-self.pk_nw)
        if np.shape(pk_cc) != (self.nz,self.nk): raise IndexError("pk_cc must be of shape (len(z), len(k))")
        if np.shape(pk_mm) != (self.nz,self.nk): raise IndexError("pk_mm must be of shape (len(z), len(k))")
        # Matter density
        self.rho_field  = self.cosmology.rho_crit(0.)*self.cosmology.Omega_m
        # Initialize mass
        self.mass    = np.logspace(2., 18., 512)
        self.lnmass  = np.log(self.mass)
        self.logmass = np.log10(self.mass)
        self.dlnm    = np.log(self.mass[1]/self.mass[0])
        self.nm      = len(self.mass)
        self.rr      = self.cosmology.radius_of_mass(self.mass,var='cb',window='th')
        # GO!
        self.compute_nonlinear_pk()

    #-----------------------------------------------------------------------------------------
    # Gaussian filtering
    #-----------------------------------------------------------------------------------------
    def remove_bao_gaussian_filtering(self, k, pk, Lambda = 0.25):
        # Extrapolate
        kLinear, pLinear = UF.extrapolate_log(k, pk, 1e-6, 1e6)
        dqlog = np.diff(np.log10(kLinear))[0]
        # EH spectrum with rescaling
        pEH = self.cosmology.EisensteinHu_nowiggle_Pk(z=0, k=kLinear)[1][0]
        pEH /= pEH[0]/pLinear[0]
        # Smooth, interpolate and evaluate
        smoothPowerSpectrum     = gaussian_filter1d(pLinear/pEH, Lambda/dqlog)*pEH
        smoothPowerSpectrum_int = si.interp1d(kLinear,smoothPowerSpectrum,'cubic')
        smoothPowerSpectrum     = smoothPowerSpectrum_int(k)
        return smoothPowerSpectrum


    #-----------------------------------------------------------------------------------------
    # nonlinear_pk
    #-----------------------------------------------------------------------------------------
    def compute_nonlinear_pk(self):
        # Compute sigma8 and sigma^2
        # N.B. Computing sigma^2_cc but smoothing with total matter field returns better agreement than smoothing for the cb field
        sig8_cc = np.array([self.cosmology.compute_sigma_8(k=self.k_ext,pk=self.pk_cc_ext[iz]) for iz in range(self.nz)])
        sig2_cc = self.cosmology.mass_variance(self.logmass,k=self.k_ext,pk=self.pk_cc_ext,var='tot')
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
        if (self.cosmology.Omega_K!=0.) or (self.cosmology.w0!=-1.) or (self.cosmology.wa!=0.):
            z_corr_10         = 10.
            g_lcdm_10         = self.growth_factors_lcdm(z_corr_10)
            g_lcdm_z          = self.growth_factors_lcdm(self.z)
            g_10,_            = self.growth_factors(z_corr_10)
            conc             *= np.expand_dims(g_10/g_growth*g_lcdm_z/g_lcdm_10,1)
        # Virial radius
        rv = ((3*np.expand_dims(self.mass,0))/(4*np.pi*self.rho_field*np.expand_dims(Deltav,1)))**(1./3.)
        # Scale radius
        rs = rv/conc
        # n_eff(z) and quasi-linear softening
        n_eff_cc = np.zeros(self.nz)
        for i in range(self.nz):
            # Find the mass at which sigma(M) = delta_c
            sig_int_2   = si.interp1d(self.logmass, sig2_cc[i]-deltac[i]**2.,'cubic',
                                      fill_value='extrapolate',bounds_error=False)
            M_1         = 10.**(so.root(sig_int_2, 13.-1.75*(1+self.z[i]))['x'][0])
            # Spline the sigma^2(M) function and take derivative at M_1
            s2_spl      = si.InterpolatedUnivariateSpline(self.lnmass,np.log(sig2_cc[i]),k=3)
            spl_logder  = s2_spl.derivative()
            logder      = spl_logder(np.log(M_1))
            # effective spectral index
            n_eff_cc[i] = -3.-3.*logder
        # Quasi-linear softening
        alpha  = np.expand_dims(1.875*1.603**n_eff_cc,1)
        # NFW profile, already normalized for bloating and corrected for neutrino fraction
        u_NFW = np.zeros((self.nz, self.nm, self.nk))
        eta_tmp = np.array([eta for x in range(self.nm)]).T
        R_bloat = peak_height**eta_tmp*rs
        for ik in range(self.nk): u_NFW[:,:,ik] = self.FFT_NFW_profile(conc,self.k[ik]*R_bloat)*(1-self.f_nu)
        # Halo mass function
        hmf = self.dndM(self.z,self.mass,peak_height)
        # power spectrum
        k_over_kdamp = np.outer(1./kdamp,self.k)
        k_over_kstar = np.outer(1./kstar,self.k)
        self.pk_1h   = np.zeros((self.nz, self.nk))
        self.pk_2h   = np.zeros((self.nz, self.nk))
        for iz in range(self.nz):
            for ik in range(self.nk):
                integrand            = ((self.mass/self.rho_field)**2.*hmf[iz]*u_NFW[iz,:,ik]**2.)*self.mass
                self.pk_1h[iz,ik] = np.trapz(integrand,x=self.lnmass)
        self.pk_2h  = self.pk_dw*(1.-f_2h*k_over_kdamp**nd_2h/(1.+k_over_kdamp**nd_2h))
        self.pk_1h *= k_over_kstar**4./(1.+k_over_kstar**4.)
        self.pk_nl  = (self.pk_1h**alpha + self.pk_2h**alpha)**(1./alpha)

    #-----------------------------------------------------------------------------------------
    # SIMPLIFIED HUBBLE PARAMETER AND Omega_m(a) (needed to speed up growth factor calculations)
    #-----------------------------------------------------------------------------------------
    def Hubble(self, a):
        return (self.cosmology.Omega_m*a**(-3.) +
               (1-self.cosmology.Omega_m-self.cosmology.Omega_lambda)*a**(-2.) +
               self.cosmology.Omega_lambda*self.X_de(a))**0.5
    def AH(self, a):
        wde = self.cosmology.w0+(1.-a)*self.cosmology.wa
        return -0.5*(self.cosmology.Omega_m*a**(-3)+self.cosmology.Omega_lambda*(1.+3.*wde)*self.X_de(a))
    def Omega_m_a(self, a):
        return self.cosmology.Omega_m*a**(-3.)/self.Hubble(a)**2.
    def X_de(self, a):
        w0,wa=self.cosmology.w0,self.cosmology.wa
        return a**(-3.*(1.+w0+wa))*np.exp(-3.*wa*(1.-a))

    #-----------------------------------------------------------------------------------------
    # NON-NORMALIZED GROWTH FACTORS
    #-----------------------------------------------------------------------------------------
    def growth_factors(self, z):
        z = np.atleast_1d(z)
        # Functions to integrate
        def derivatives(y, a):
            # Function 
            g,omega,G=y
            # Derivatives
            Oma   = self.Omega_m_a(a)
            acce  = self.AH(a)
            Ha    = self.Hubble(a)
            dydt  = [omega,-(2+acce/Ha**2.)*omega/a+1.5*Oma*g/a**2.,g/a]
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
    def growth_factors_lcdm(self, z):
        gam = 0.55
        z   = np.atleast_1d(z)
        Om  = self.cosmology.Omega_m
        Ol  = self.cosmology.Omega_lambda
        Oma = lambda a: Om*a**(-3.)/(Om*a**(-3)+(1-Om))
        result = np.exp(np.array([sint.quad(lambda a: (Oma(a)**gam)/a, 1., 1./(1.+zz))[0] for zz in z]))
        return result
    def growth_LCDM_base(self, z):
        a  = 1/(1.+z)
        Om = self.cosmology.Omega_m
        return a*ss.hyp2f1(1/3., 1., 11/6., -a**3/Om*(1.-Om))/ss.hyp2f1(1/3., 1., 11/6., -(1.-Om)/Om)

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
        delta_c0 = (3./20.)*(12.*np.pi)**(2/3.)*(1-0.041*f_nu)
        factor = 1.+(self.f1(g/a,G/a)*np.log10(Omz)**alpha_1+self.f2(g/a,G/a)*np.log10(Omz)**alpha_2)
        return delta_c0*factor

    #-----------------------------------------------------------------------------------------
    # CRITICAL DENSITY FOR COLLAPSE - NON-LINEAR
    #-----------------------------------------------------------------------------------------
    def Delta_v(self, z, f_nu, Omz, g, G):
        a = 1/(1+z)
        alpha_3, alpha_4 = 1,2
        Delta_v0 = 18*np.pi**2.*(1+0.763*f_nu)
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
        Dzf   = self.growth_LCDM_base(z_tmp)
        zf_D  = si.interp1d(Dzf, z_tmp, 'cubic')
        for iz in range(nz):
            m_ext, sig_ext = UF.extrapolate_log(mass, sig2[iz]**0.5, 1.e-1*frac*mass[0], 1.e1*mass[-1])
            sig_int        = si.interp1d(m_ext, sig_ext, 'cubic')
            s_fmz          = sig_int(fm)
            aa             = 1/(1.+z[iz])
            rhs[iz]        = self.growth_LCDM_base(z[iz])*deltac[iz]/s_fmz
            for im in range(nm):
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
        nz, nm   = len(np.atleast_1d(z)), len(np.atleast_1d(M))
        dlnm     = np.diff(np.log(M))[0]
        hmf      = np.zeros((nz, nm))
        mass_fun = self.ST_mass_fun(peak_height)
        for iz in range(nz):    
            # derivative
            log_der = np.gradient(peak_height[iz], dlnm, edge_order = 2)/peak_height[iz]
            # Halo mass function
            hmf[iz] = self.rho_field/M**2.*log_der*mass_fun[iz]
        return hmf


########################################################################################################################
# halomodel: applies halo model to a given power spectrum
########################################################################################################################
class halomodel():
    """
    The class ``halomodel`` transforms a linear input power spectrum to its non-linear counterpart using
    the halo model implemented in CAMB.
    For the 'classic' halo model see the class :func:`colibri.halo.halo` and e.g. `arXiv:0206508 <https://arxiv.org/pdf/astro-ph/0206508.pdf>`_ for a theoretical insight.

    By calling this class, the total matter non-linear power spectrum is returned. It accepts the following arguments,
    with the default values specified:

    :param z: Redshift.
    :type z: array

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param pk: Linear total matter power spectra evaluated in ``z`` and ``k`` in units of :math:`(\mathrm{Mpc}/h)^3`.
    :type pk: 2D array of shape ``(len(z), len(k))``

    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``


    When the instance is called, the array ``self.mass = np.logspace(2., 18., 512)``, i.e. an array of masses spanning from :math:`100 M_\odot/h` to :math:`10^{18} M_\odot/h` is created, where all the mass functions are computed.

    :param a_ShethTormen: Sheth-Tormen parameter.
    :type a_ShethTormen: float, default = 0.707

    :param p_ShethTormen: Sheth-Tormen parameter.
    :type p_ShethTormen: float, default = 0.3
        
    :return: Nothing, but the quantity ``self.pk_nl`` is generated, a 2D array of shape ``(len(z), len(k))`` containing the non-linear matter power spectra in units of :math:`(\mathrm{Mpc}/h)^3`.

    """

    def __init__(self,
                 z,
                 k,
                 pk,
                 cosmology     = cc.cosmo(),
                 a_ShethTormen = 0.707,
                 p_ShethTormen = 0.3):
        # Assertion on k
        assert len(k)>200,     "k must have a length greater than 200 points"
        assert k.max()>=10.,   "Maximum wavenumber must be greater than 10 h/Mpc in order to achieve convergence"
        assert k.min()<=0.001, "Minimum wavenumber must be lower than 0.001 h/Mpc in order to achieve convergence"
        # Reading all cosmological parameters
        self.f_nu         = cosmology.f_nu
        self.cosmology    = cosmology
        # Redshift and scales at which all must be computed
        self.nz    = len(np.atleast_1d(z))
        self.nk    = len(np.atleast_1d(k))
        self.z     = np.atleast_1d(z)
        self.k     = np.atleast_1d(k)
        self.pk    = pk
        # Extended versions of P(k) for convergence of mass variance
        pk_ext = []
        for iz in range(self.nz):
            k_ext, pk_tmp = UF.extrapolate_log(self.k,self.pk[iz],self.k.min(),1e4)
            pk_ext.append(pk_tmp)
        self.k_ext     = k_ext
        self.pk_ext = np.array(pk_ext)
        del pk_ext, k_ext
        if np.shape(pk) != (self.nz,self.nk): raise IndexError("pk must be of shape (len(z), len(k))")
        # Matter density
        self.rho_field  = self.cosmology.rho_crit(0.)*self.cosmology.Omega_m
        # Fixed parameters
        self.delta_sc  = 3./20.*(12.*np.pi)**(2./3.)
        self.a         = a_ShethTormen
        self.p         = p_ShethTormen
        self.A_bar     = 4.
        # Initialize mass
        self.mass    = np.logspace(2., 18., 512)
        self.lnmass  = np.log(self.mass)
        self.logmass = np.log10(self.mass)
        self.dlnm    = np.log(self.mass[1]/self.mass[0])
        self.nm      = len(self.mass)
        self.rr      = self.cosmology.radius_of_mass(self.mass,var='cb',window='th')
        # GO!
        self.compute_nonlinear_pk()

    #-----------------------------------------------------------------------------------------
    # nonlinear_pk
    #-----------------------------------------------------------------------------------------
    def compute_nonlinear_pk(self):

        if self.f_nu != 0.:
            pk_cc     = self.pk_ext*(self.cosmology.growth_cb_unnormalized(self.k_ext,self.z)/self.cosmology.growth_cbnu_unnormalized(self.k_ext,self.z))**2.
        else:
            pk_cc = self.pk_ext
        # Compute sigma8 and sigma^2
        self.sig2 = self.cosmology.mass_variance(self.logmass,k=self.k_ext,pk=pk_cc,var='tot')
        # Omega_m(z)
        self.omz = self.cosmology.Omega_m_z(self.z)
        # Compute growth factors
        g_growth, G_growth = self.growth_factors(self.z)
        # Linear overdensity at collapse
        dc = self.delta_c(self.z,self.f_nu, self.omz, g_growth, G_growth)
        # Virial density
        Dv = self.Delta_v(self.z, self.f_nu, self.omz, g_growth, G_growth)
        # peak height
        peak_height = np.expand_dims(dc,1)/self.sig2**0.5
        # Virial radii
        rv = ((3*np.expand_dims(self.mass,0))/(4*np.pi*self.rho_field*np.expand_dims(Dv,1)))**(1./3.)
        # Concentration
        self.zf  = self.z_form()
        conc     = np.array([self.c_bull(self.zf[i],self.z[i]) for i in range(self.nz)])
        # Scale radii
        R_s = rv/conc
        # NFW profile, already normalized for bloating and corrected for neutrino fraction
        u_NFW = np.zeros((self.nz, self.nm, self.nk))
        for ik in range(self.nk): u_NFW[:,:,ik] = self.FFT_NFW_profile(conc, self.k[ik]*R_s)
        # Halo mass function
        hmf         = self.dndM(self.z,self.mass,peak_height,self.a,self.p)
        # Power spectrum
        M_over_rho  = np.expand_dims(self.mass/self.rho_field,0)
        M           = np.expand_dims(self.mass,0)
        self.pk_1h  = np.transpose([sint.simps(M_over_rho**2.*hmf*u_NFW[:,:,ik]**2.*M,x=self.lnmass, axis = 1)
                                     for ik in range(self.nk)])
        self.pk_2h  = self.pk
        self.pk_nl  = self.pk_1h + self.pk_2h

    #-----------------------------------------------------------------------------------------
    # SIMPLIFIED HUBBLE PARAMETER AND Omega_m(a) (needed to speed up growth factor calculations)
    #-----------------------------------------------------------------------------------------
    def Hubble(self, a):
        return (self.cosmology.Omega_m*a**(-3.) +
               (1-self.cosmology.Omega_m-self.cosmology.Omega_lambda)*a**-2. +
               (self.cosmology.Omega_lambda)*self.X_de(a))**0.5
    def AH(self, a):
        wde = self.cosmology.w0+(1.-a)*self.cosmology.wa
        return -0.5*(self.cosmology.Omega_m*a**-3+self.cosmology.Omega_lambda*(1.+3.*wde)*self.X_de(a))
    def Omega_m_a(self, a):
        return self.cosmology.Omega_m*a**(-3.)/self.Hubble(a)**2.
    def X_de(self, a):
        w0,wa=self.cosmology.w0,self.cosmology.wa
        return a**(-3.*(1.+w0+wa))*np.exp(-3.*wa*(1.-a))

    #-----------------------------------------------------------------------------------------
    # NON-NORMALIZED GROWTH FACTORS
    #-----------------------------------------------------------------------------------------
    def growth_factors(self, z):
        z = np.atleast_1d(z)
        # Functions to integrate
        def derivatives(y, a):
            # Function 
            g,omega,G=y
            # Derivatives
            Oma   = self.Omega_m_a(a)
            Om    = self.cosmology.Omega_m
            Ol    = self.cosmology.Omega_lambda
            w0,wa = self.cosmology.w0, self.cosmology.wa
            w_de  = w0+wa*(1.-a)
            acce  = self.AH(a)
            dydt  = [omega,-(2+acce/self.Hubble(a)**2.)*omega/a+1.5*Oma*g/a**2.,g/a]
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

    # for LCDM
    def growth_factors_lcdm(self, z):
        gam = 0.55
        z = np.atleast_1d(z)
        Om = self.cosmology.Omega_m
        Oma = lambda a: Om*a**(-3.)/(Om*a**(-3)+(1-Om))
        result = np.exp(np.array([sint.quad(lambda a: Oma(a)**gam/a, 1., 1./(1.+zz))[0] for zz in z]))
        return result

    #-----------------------------------------------------------------------------------------
    # REDSHIFT OF FORMATION OF HALOS
    #-----------------------------------------------------------------------------------------
    def z_form(self):
        frac  = 0.01
        fm    = frac*self.mass
        z_tmp = np.linspace(0., 30., 1001)
        res   = np.zeros((self.nz, self.nm))
        rhs   = np.zeros((self.nz, self.nm))
        Dzf = self.cosmology.growth_factor_scale_independent(z_tmp)
        zf_D = si.interp1d(Dzf, z_tmp, 'cubic')
        for iz in range(self.nz):
            m_ext, sig_ext = UF.extrapolate_log(self.mass, self.sig2[iz]**0.5, 1.e-1*frac*self.mass[0], 1.e1*self.mass[-1])
            sig_int        = si.interp1d(m_ext, sig_ext, 'cubic')
            s_fmz          = sig_int(fm)
            rhs[iz]        = self.cosmology.growth_factor_scale_independent(self.z[iz])*self.delta_sc/s_fmz
            for im in range(self.nm):
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
        delta_c0 = (3./20.)*(12.*np.pi)**(2/3.)*(1-0.041*f_nu)
        factor = 1.+(self.f1(g/a,G/a)*np.log10(Omz)**alpha_1+self.f2(g/a,G/a)*np.log10(Omz)**alpha_2)
        return delta_c0*factor

    #-----------------------------------------------------------------------------------------
    # CRITICAL DENSITY FOR COLLAPSE - NON-LINEAR
    #-----------------------------------------------------------------------------------------
    def Delta_v(self, z, f_nu, Omz, g, G):
        a = 1/(1+z)
        alpha_3, alpha_4 = 1,2
        Delta_v0 = 18*np.pi**2.*(1+0.763*f_nu)
        factor = 1.+(self.f3(g/a,G/a)*np.log10(Omz)**alpha_3+self.f4(g/a,G/a)*np.log10(Omz)**alpha_4)
        return Delta_v0*factor

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
    def ST_mass_fun(self,nu,a,p):
        n = nu**2.
        A = 1./(1. + 2.**(-p)*ss.gamma(0.5-p)/np.sqrt(np.pi))
        ST = A * np.sqrt(2.*a*n/np.pi) * (1.+1./(a*n)**p)*np.exp(-a*n/2.)
        return ST

    #-----------------------------------------------------------------------------------------
    # HALO MASS FUNCTION
    #-----------------------------------------------------------------------------------------
    def dndM(self, z, M, peak_height,a,p):
        nz, nm = len(np.atleast_1d(z)), len(np.atleast_1d(M))
        dlnm   = np.diff(np.log(M))[0]
        hmf    = np.zeros((nz, nm))
        mass_fun = self.ST_mass_fun(peak_height,a,p)
        for iz in range(nz):    
            # derivative
            log_der = np.gradient(peak_height[iz], dlnm, edge_order = 2)/peak_height[iz]
            # Halo mass function
            hmf[iz] = self.rho_field/M**2.*log_der*mass_fun[iz]
        return hmf

########################################################################################################################
# classic_halomodel: applies halo model to a given power spectrum
########################################################################################################################

class classic_halomodel():
    """
    The class ``classic_halomodel`` transforms a linear input power spectrum to its non-linear counterpart using the halo model as described in e.g. `arXiv:0206508 <https://arxiv.org/pdf/astro-ph/0206508.pdf>`_  .

    By calling this class, the total matter non-linear power spectrum is returned. It accepts the following arguments,
    with the default values specified:

    :param z: Redshift.
    :type z: array

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param pk: Linear total matter power spectra evaluated in ``z`` and ``k`` in units of :math:`(\mathrm{Mpc}/h)^3`.
    :type pk: 2D array of shape ``(len(z), len(k))``

    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``

    :param pivot_concentration: concentration parameter at typical mass.
    :type pivot_concentration: float, default = 9.

    :param slope_concentration: slope of the mass-concentration relation (minus sign already included, set to positive here).
    :type slope_concentration: float, default = 0.13

    :param a_ShethTormen: Sheth-Tormen parameter.
    :type a_ShethTormen: float, default = 0.707

    :param p_ShethTormen: Sheth-Tormen parameter.
    :type p_ShethTormen: float, default = 0.3
        
    :return: Nothing, but the quantity ``self.pk_nl`` is generated, a 2D array of shape ``(len(z), len(k))`` containing the non-linear matter power spectra in units of :math:`(\mathrm{Mpc}/h)^3`.

    """

    def __init__(self,
            z,
            k,
            pk,
            cosmology = cc.cosmo(),
            pivot_concentration = 9.,
            slope_concentration = 0.13,
            a_ShethTormen = 0.707,
            p_ShethTormen = 0.3):

        # Assertion on k
        assert len(k)>200,     "k must have a length greater than 200 points"
        assert k.max()>=10.,   "Maximum wavenumber must be greater than 10 h/Mpc in order to achieve convergence"
        assert k.min()<=0.001, "Minimum wavenumber must be lower than 0.001 h/Mpc in order to achieve convergence"
        # Reading all cosmological parameters
        self.cosmology    = cosmology
        self.w0           = cosmology.w0
        self.wa           = cosmology.wa
        self.f_nu         = cosmology.f_nu
        # Raise warning if neutrino mass is not zero:
        if np.any(self.cosmology.M_nu!=0.):
            warnings.warn("Neutrino mass is different from zero. The Takahashi halofit works best with zero neutrino mass, maybe better to use Takahashi or Bird?")
        # Halo model parameters
        self.pivot_concentration = pivot_concentration
        self.slope_concentration = slope_concentration
        self.a_ShethTormen       = a_ShethTormen
        self.p_ShethTormen       = p_ShethTormen
        # Overdensity spherical collapse
        self.delta_sc = 3./20.*(12.*np.pi)**(2./3.)
        # Redshift and scales at which all must be computed
        self.z  = np.atleast_1d(z)
        self.k  = np.atleast_1d(k)
        self.nz = len(self.z)
        self.nk = len(self.k)
        self.pk   = pk
        # Growth factor
        self.growth_factor = self.cosmology.growth_factor_scale_independent(self.z)
        # Initialize mass
        self.nm   = 512
        self.mass = np.logspace(2., 18., self.nm)
        # Peak height
        self.peak_height = self.nu()
        # virial_radii
        self.rv = self.R_v(self.mass)
        # GO!
        self.compute_nonlinear_pk(kwargs_mass_function = {'a' : self.a_ShethTormen,
                                                          'p' : self.p_ShethTormen},
                                  kwargs_concentration = {'c0': self.pivot_concentration,
                                                          'b' : self.slope_concentration})

    #-----------------------------------------------------------------------------------------
    # OVERDENSITY AT COLLAPSE
    #-----------------------------------------------------------------------------------------
    def Delta_v(self, z):
        omz = self.cosmology.Omega_m_z(self.z)
        return 18.*np.pi**2.*(1.+0.399*(1./omz-1.))

    #-----------------------------------------------------------------------------------------
    # SMOOTHING RADIUS
    #-----------------------------------------------------------------------------------------
    def smoothing_radius(self, M):
        rho = self.cosmology.rho(0.)
        return (3*M/(4*np.pi*rho))**(1./3.)

    #-----------------------------------------------------------------------------------------
    # SIGMA^2
    #-----------------------------------------------------------------------------------------
    def sigma2(self):
        kappa   = self.k
        P_kappa = self.pk*(self.cosmology.growth_factor_CDM_baryons_neutrinos(z=0., k=kappa)[0]/self.cosmology.growth_factor_CDM_baryons_neutrinos(z=self.z, k=kappa)[0])**2.
        dlnk    = np.log(kappa[1]/kappa[0])        
        R       = self.smoothing_radius(self.mass)
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
        return np.transpose(self.delta_sc/np.transpose((self.sigma2())**.5))

    #-----------------------------------------------------------------------------------------
    # SHETH-TORMEN MASS FUNCTION
    #-----------------------------------------------------------------------------------------
    def mass_fun_ST(self, nu, a = 0.707, p = 0.3):
        n = nu**2.
        A = 1./(1. + 2.**(-p)*ss.gamma(0.5-p)/np.sqrt(np.pi))
        ST = A * np.sqrt(2.*a*n/np.pi) * (1.+1./(a*n)**p) * np.exp(-a*n/2.)
        return ST

    #-----------------------------------------------------------------------------------------
    # SHETH-TORMEN BIAS
    #-----------------------------------------------------------------------------------------
    def halo_bias_ST(self, nu, a = 0.707, p = 0.3):
        d_sc  = self.delta_sc
        return 1. + (a*nu**2.-1.)/d_sc + 2.*p/d_sc/(1.+(a*nu**2.)**p)

    #-----------------------------------------------------------------------------------------
    # HALO MASS FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_halo_mass_function(self, **kwargs):
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
        # Compute it at any redshift
        nu = self.peak_height[0]*self.growth_factor[0]
        func = si.interp1d(nu, self.mass)
        value = func(1.)
        return value

    #-----------------------------------------------------------------------------------------
    # FOURIER TRANSFORM OF NFW PROFILE
    #-----------------------------------------------------------------------------------------
    def u_NFW(self, c, x):
        (Si_1,Ci_1) = ss.sici(x)
        (Si_2,Ci_2) = ss.sici((1.+c)*x)
        den         = np.log(1.+c)-c*1./(1.+c)
        num1        = np.sin(x)*(Si_2-Si_1)
        num2        = np.sin(c*x)
        num3        = np.cos(x)*(Ci_2-Ci_1)
        return 1./den*(num1+num3-num2*1./((1.+c)*x))


    #-----------------------------------------------------------------------------------------
    # CONCENTRATION PARAMETER
    #-----------------------------------------------------------------------------------------
    def conc(self, M, c0 = 9., b = 0.13):
        scale_mass = self.M_star()
        conc = np.zeros((self.nz, len(np.atleast_1d(M))))
        for i in range(self.nz):
            conc[i] = c0/(1.+self.z[i])*(M/scale_mass)**(-b)
        return conc

    #-----------------------------------------------------------------------------------------
    # VIRIAL RADIUS
    #-----------------------------------------------------------------------------------------
    def R_v(self, M):
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
        cc = self.conc(M, **kwargs_concentration)
        rs = np.zeros_like(cc)
        rv = self.R_v(M)
        for i in range(self.nz): rs[i] = rv[i]/cc[i]
        return rs

    #-----------------------------------------------------------------------------------------
    # NORMALIZATION 2 HALO
    #-----------------------------------------------------------------------------------------
    def norm_2h(self, bias, **kwargs):
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
    def compute_nonlinear_pk(self, kwargs_mass_function = {}, kwargs_concentration = {}):
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
        # Power spectrum
        P_1h = np.zeros_like(self.pk)
        P_2h = np.zeros_like(self.pk)
        for iz in range(self.nz):
            for ik in range(len(self.k)):
                integrand_1h = ((M/rho)**2*dndM[iz]*nfw[iz,ik]**2)*M
                integrand_2h = (M/rho*dndM[iz]*nfw[iz,ik]*bias[iz])*M
                P_1h[iz,ik]  = np.trapz(integrand_1h, dx = dlnM)*(1.-np.exp(-(self.k[ik]/k_damp[iz])**2.))
                P_2h[iz,ik]  = np.trapz(integrand_2h, dx = dlnM)**2.*normalization[iz]**2.*self.pk[iz,ik]
        self.pk_nl = P_1h + P_2h

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
    :type z: array

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param pk: Linear power spectra evaluated in ``z`` and ``k`` in units of :math:`(\mathrm{Mpc}/h)^3`.
    :type pk: 2D array of shape ``(len(z), len(k))``

    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``

        
    :return: Nothing, but the quantity ``self.pk_nl`` is generated, a 2D array of shape ``(len(z), len(k))`` containing the non-linear matter power spectra in units of :math:`(\mathrm{Mpc}/h)^3`.

    """

    def __init__(self,
                 z,
                 k,
                 pk,
                 cosmology = cc.cosmo()):

        # Assertion on k
        assert len(k)>200,     "k must have a length greater than 200 points"
        assert k.max()>=10.,   "Maximum wavenumber must be greater than 10 h/Mpc in order to achieve convergence"
        assert k.min()<=0.001, "Minimum wavenumber must be lower than 0.001 h/Mpc in order to achieve convergence"
        # Reading all cosmological parameters
        self.w0           = cosmology.w0
        self.wa           = cosmology.wa
        self.cosmology    = cosmology
        self.f_nu         = cosmology.f_nu
        # Redshift and scales at which all must be computed
        self.nz   = len(np.atleast_1d(z))
        self.nk   = len(np.atleast_1d(k))
        self.z    = np.atleast_1d(z)
        self.k    = np.atleast_1d(k)
        self.pk   = pk
        # Extended versions of P(k) for convergence of mass variance
        pk_ext = []
        for iz in range(self.nz):
            k_ext, pk_tmp = UF.extrapolate_log(self.k,self.pk[iz],self.k.min(),1e4)
            pk_ext.append(pk_tmp)
        self.k_ext  = k_ext
        self.pk_ext = np.array(pk_ext)
        del pk_ext, k_ext
        if np.shape(pk) != (self.nz,self.nk): raise IndexError("pk must be of shape (len(z), len(k))")
        # Matter density
        self.rho_field  = self.cosmology.rho_crit(0.)*self.cosmology.Omega_m
        # Initialize mass
        self.mass    = np.logspace(2., 18., 512)
        self.lnmass  = np.log10(self.mass)
        self.logmass = np.log10(self.mass)
        self.dlnm    = np.log(self.mass[1]/self.mass[0])
        self.nm      = np.size(self.mass)
        self.rr      = self.cosmology.radius_of_mass(self.mass, window = 'g')
        self.lnr     = np.log(self.rr)
        # Mass variance, non-linear scales, effective indices
        self.sigma2  = self.mass_variance(k=self.k_ext,pk=self.pk_ext)
        self.k_nl    = self.nonlinear_scale(precision = 1e-5)
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
            0.3980*self.n_eff**4-0.1682*self.C_eff + self.f_nu*(1.081 + 0.395*self.n_eff**2.)
        self.mun    = 0.0
        self.nun    = 10**(5.2105+3.6902*self.n_eff)
        # GO!
        self.compute_nonlinear_pk()

    #-----------------------------------------------------------------------------------------
    # MASS VARIANCE
    #-----------------------------------------------------------------------------------------
    def mass_variance(self, k, pk):
        return self.cosmology.mass_variance_multipoles(logM   = self.logmass,
                                                       k      = k,
                                                       pk     = pk,
                                                       var    = 'cb',
                                                       window = 'g')

    #-----------------------------------------------------------------------------------------
    # k_SIGMA
    #-----------------------------------------------------------------------------------------
    def nonlinear_scale(self, precision = 1e-5):
        nz = self.nz
        # Mass variance
        sigma2_array = self.sigma2
        s2_interp    = si.interp1d(self.rr,sigma2_array,'cubic',bounds_error=False,fill_value='extrapolate')
        # Find k_sigma
        k_sigma = np.zeros(nz)
        for iz in range(nz):
            count = 0
            # Set min and max values possible
            Rmin,Rmax = 0.01, 100.0  #Mpc/h
            found = False
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

########################################################################################################################
# Bird
########################################################################################################################
class Bird():
    """
    The class ``Bird`` transforms a linear input power spectrum to its non-linear counterpart using
    the original Halofit model by Smith et al. corrected for the presence of massive neutrinos by Bird et al. (see `arXiv:1109.4416 <https://arxiv.org/abs/1109.4416>`_ ).
    By calling this class, a non-linear power spectrum is returned. It accepts the following arguments,
    with the default values specified:


    :param z: Redshift.
    :type z: array

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param pk: Linear power spectra evaluated in ``z`` and ``k`` in units of :math:`(\mathrm{Mpc}/h)^3`.
    :type pk: 2D array of shape ``(len(z), len(k))``

    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``

        
    :return: Nothing, but the quantity ``self.pk_nl`` is generated, a 2D array of shape ``(len(z), len(k))`` containing the non-linear matter power spectra in units of :math:`(\mathrm{Mpc}/h)^3`.

    """

    def __init__(self,
                 z,
                 k,
                 pk,
                 cosmology = cc.cosmo()):

        # Assertion on k
        assert len(k)>200,     "k must have a length greater than 200 points"
        assert k.max()>=10.,   "Maximum wavenumber must be greater than 10 h/Mpc in order to achieve convergence"
        assert k.min()<=0.001, "Minimum wavenumber must be lower than 0.001 h/Mpc in order to achieve convergence"
        # Reading all cosmological parameters
        self.f_nu         = cosmology.f_nu
        self.cosmology    = cosmology
        # Assertions
        if self.cosmology.w0 != -1. or self.cosmology.wa != 0.:
            warnings.warn("This model does not currently support dynamic dark energy")
        # Redshift and scales at which all must be computed
        self.nz   = len(np.atleast_1d(z))
        self.nk   = len(np.atleast_1d(k))
        self.z    = np.atleast_1d(z)
        self.k    = np.atleast_1d(k)
        self.pk   = pk
        # Extended versions of P(k) for convergence of mass variance
        pk_ext = []
        for iz in range(self.nz):
            k_ext, pk_tmp = UF.extrapolate_log(self.k,self.pk[iz],self.k.min(),1e4)
            pk_ext.append(pk_tmp)
        self.k_ext  = k_ext
        self.pk_ext = np.array(pk_ext)
        del pk_ext, k_ext
        if np.shape(pk) != (self.nz,self.nk): raise IndexError("pk must be of shape (len(z), len(k))")
        # cdm+b density
        self.rho_field  = self.cosmology.rho_crit(0.)*self.cosmology.Omega_cb
        # Initialize mass
        self.mass    = np.logspace(2., 18., 512)
        self.lnmass  = np.log10(self.mass)
        self.logmass = np.log10(self.mass)
        self.dlnm    = np.log(self.mass[1]/self.mass[0])
        self.nm      = np.size(self.mass)
        self.rr      = self.cosmology.radius_of_mass(self.mass, window = 'g')
        self.lnr     = np.log(self.rr)
        # Mass variance, non-linear scales, effective indices
        self.sigma2  = self.mass_variance(k=self.k_ext,pk=self.pk_ext,var='cb',window='g')  # Smoothing P_mm(k) on cdm+b field seems to work best, but why??
        self.k_nl    = self.nonlinear_scale(sigma2_array=self.sigma2)
        self.n_eff   = self.effective_index(sigma2_array=self.sigma2)
        self.C_eff   = self.effective_curvature(sigma2_array=self.sigma2)
        # Omega matter as function of redshifts
        self.Omz     = self.cosmology.Omega_m_z(self.z)
        self.Olz     = self.cosmology.Omega_lambda_z(self.z)
        # Fitting factors
        extragamma  = 0.3159-0.0765*self.n_eff -0.8350*self.C_eff
        self.an     = 10**(1.4861+1.83693*self.n_eff+1.67618*self.n_eff**2+
                           0.7940*self.n_eff**3.+0.1670756*self.n_eff**4-0.620695*self.C_eff)
        self.bn     = 10**(0.9463+0.9466*self.n_eff+0.3084*self.n_eff**2-0.940*self.C_eff)
        self.cn     = 10**(-0.2807+0.6669*self.n_eff+0.3214*self.n_eff**2-0.0793*self.C_eff)
        self.gamman = 0.86485+0.2989*self.n_eff+0.1631*self.C_eff+extragamma
        self.alphan = 1.38848+0.3701*self.n_eff-0.1452*self.n_eff**2
        self.betan  = 0.8291+0.9854*self.n_eff+0.3400*self.n_eff**2+self.f_nu*(-6.4868+1.4373*self.n_eff**2)
        self.mun    = 10**(-3.54419+0.19086*self.n_eff)
        self.nun    = 10**(0.95897+1.2857*self.n_eff)
        frac        = self.Olz/(1.-self.Omz)
        self.f1     = frac*self.Omz**(-0.0307) + (1.-frac)*self.Omz**(-0.0732)
        self.f2     = frac*self.Omz**(-0.0585) + (1.-frac)*self.Omz**(-0.1423)
        self.f3     = frac*self.Omz**( 0.0743) + (1.-frac)*self.Omz**( 0.0725)
        # GO!
        self.compute_nonlinear_pk()

    #-----------------------------------------------------------------------------------------
    # MASS VARIANCE
    #-----------------------------------------------------------------------------------------
    def mass_variance(self, k, pk, var, window):
        return self.cosmology.mass_variance_multipoles(logM   = self.logmass,
                                                       k      = k,
                                                       pk     = pk,
                                                       var    = var,
                                                       window = window,
                                                       j=0,smooth=False,R_sm=0.)

    #-----------------------------------------------------------------------------------------
    # k_SIGMA
    #-----------------------------------------------------------------------------------------
    def nonlinear_scale(self, sigma2_array, precision = 1e-7):
        nz = self.nz
        # Mass variance
        s2_interp    = si.interp1d(self.rr,sigma2_array,'cubic',bounds_error=False,fill_value='extrapolate')
        # Find k_sigma
        k_sigma = np.zeros(nz)
        for iz in range(nz):
            # Set min and max values possible
            Rmin,Rmax = 0.01, 100.0  #Mpc/h
            found = False
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
    def effective_index(self,sigma2_array):
        nz = self.nz
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
    def effective_curvature(self,sigma2_array):
        nz = self.nz
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

