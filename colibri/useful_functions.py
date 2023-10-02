import numpy as np
import scipy.interpolate as si
import scipy.integrate as sint
import scipy.fftpack as sfft
import scipy.optimize
import sys

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
    assert np.allclose(np.diff(np.log(x)), np.diff(np.log(x))[0], rtol = 1e-2), "'x' array not log-spaced"
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

    m1, m2, m3: values of the three neutrino masses :math:`\mathrm{eV}`.
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
# PHASE SPACE DISTRIBUTIONS
#-------------------------------------------------------------------------------
def phase_space_distribution(momentum,mass,temperature,multiplicity,chemical_potential,sign='+'):
    """
    Returns either the Fermi-Dirac or the Bose-Einstein distribution.

    :param momentum: particle momentum in :math:`\mathrm{eV}`
    :type momentum: array

    :param mass: particle mass in :math:`\mathrm{eV}`
    :type mass: float

    :param temperature: temperature in :math:`\mathrm{K}`
    :type temperature: float

    :param multiplicity: spin multiplicity factor
    :type multiplicity: float

    :param chemical_potential: chemical potential in :math:`\mathrm{eV}`
    :type chemical_potential: float    

    :return: array in  :math:`\mathrm{eV}^{-3} s^{-3}`
    """

    # Check sign of distribution
    if   sign in ['+','p','pl','plus' ]:
        eps = 1
    elif sign in ['-','m','mn','minus']:
        eps = -1
    else:
        raise ValueError("String not recognized")

    # Check all except momenta are float
    assert isinstance(mass, float), "Mass must be a float"
    assert isinstance(temperature, float), "Temperature must be a float"
    assert isinstance(multiplicity, float), "Multiplicity must be a float"
    assert isinstance(chemical_potential, float), "Chemical potential must be a float"

    # constants
    Planck_constant    = 4.13566766e-15 # eV*s
    Boltzmann_constant = 8.61733034e-5  # eV/K

    # energy
    energy = (momentum**2.+mass**2.)**0.5
    beta   = 1/(Boltzmann_constant*temperature)
    # exponent
    p_over_T = beta*(energy-chemical_potential)

    return multiplicity/Planck_constant**3./(np.exp(p_over_T)+eps)


#-------------------------------------------------------------------------------
# WINDOW FUNCTIONS
#-------------------------------------------------------------------------------
def TopHat_window(x):
    """
    Top-hat window function in Fourier space.

    :param x: Abscissa
    :type x: array

    :return: array
    """
    return 3./(x)**3*(np.sin(x)-x*np.cos(x))

def Gaussian_window(x):
    """
    Gaussian window function.

    :param x: Abscissa.
    :type x: array

    :return: array.
    """
    return np.exp(-x**2./2.)

def Sharp_k_window(x, step=1e-2):
    """
    Sharp window function in Fourier space.

    :param x: Abscissa.
    :type x: array

    :param step: Transition width from 0 to 1.
    :type step: float, default = 1e-2

    :return: array.
    """
    return 0.5*(1.+2./np.pi*np.arctan((1.-x)/1e-2))

def Smooth_k_window(x, beta):
    """
    Smooth window function in Fourier space.

    :param x: Abscissa.
    :type x: array

    :param beta: Transition width from 0 to 1.
    :type beta: float, default = 1e-2

    :return: array.
    """
    return 1./(1.+x**beta)


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

#-------------------------------------------------------------------------------
# DDM suppression
#-------------------------------------------------------------------------------
def decaying_dark_matter_suppression(k, z, tau, f_ddm, Omega_b, Omega_m, h):
    """
    Suppression of the matter power spectrum due to decaying dark matter (DDM), according to Hubert et al., 2021. It predicts the correct shape at up to :math:`k>10 h/\mathrm{Mpc}` and :math:`z<2.35`.

    .. warning::

     This function returns the total suppression in power. To obtain the suppression in the transfer function, take the square root of the output.

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param z: Redshifts.
    :type z: array

    :param tau: Half-life time of the particle in Gyr
    :type tau: float

    :param f_ddm: fraction of DDM.
    :type f_ddm: float between 0 and 1

    :return: 2D array of shape ``(len(z), len(k))``
    """
    omega_b, omega_m = Omega_b*h**2., Omega_m*h**2.
    K,Z = np.meshgrid(k*h,z)
    a = 0.7208+2.027/tau+3.0310/(1+1.1*Z)-0.180
    b = 0.0120+2.786/tau+0.6699/(1+1.1*Z)-0.090
    p = 1.0450+1.225/tau+0.2207/(1+1.1*Z)-0.099
    q = 0.9922+1.735/tau+0.2154/(1+1.1*Z)-0.056
    u,v,w = omega_b/0.0216,h/0.6776,omega_m/0.14116
    alpha = (5.323-1.4644*u-1.391*v)+(-2.055+1.329*u+0.8672*v)*w+(0.2682-0.3509*u)*w**2
    beta  = 0.9260 + (0.05735-0.02690*v)*w + (-0.01373+0.006713*v)*w**2
    gamma = (9.553-0.7860*v)+(0.4884+0.1754*v)*w+(-0.2512+0.07558*v)*w**2
    eps_lin    = alpha/tau**beta*(1+0.105*Z)**(-gamma)
    eps_nonlin = eps_lin*(1.+a*K**p)/(1.+b*K**q)*f_ddm
    return 1.-eps_nonlin



#-------------------------------------------------------------------------------
# f(R) enhancement
#-------------------------------------------------------------------------------
def fR_correction(k, z, f_R0, nonlinear = True):
    """
    Enhancement of the matter power spectrum due to f(R) gravity. The formula used is from Winther et al. (2019)

    .. warning::

     This function returns the total suppression in power. To obtain the suppression in the transfer function, take the square root of the output.

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param z: Redshifts.
    :type z: array

    :param f_R0: modified gravity amplitude
    :type f_R0: float

    :param nonlinear: whether to return the linear or nonlinear enhancement
    :type nonlinear: boolean, default = True

    :return: 2D array of shape ``(len(z), len(k))``
    """
    # Substitute k_max above 10.
    if nonlinear: k_new = np.array([i if i<10. else 10. for i in k])
    else:         k_new = k
    # k,z arrays
    Z,K      = np.meshgrid(z,k_new,indexing='ij')
    a        = 1./(1.+Z)

    # Change f(R) within bounds
    fR0min = 1e-7
    fR0max = 1e-4
    if(f_R0 < fR0min): f_R0 = fR0min
    if(f_R0 > fR0max): f_R0 = fR0max

    # Non-linear enhancement
    if nonlinear:
        # Low values of f(R) and relative table
        fR0_low    = 5e-6
        param_low  = [ 0.768779  , -0.405375   ,  0.0075176,  0.0288574 , -0.0638224 , -0.401206  , 
                       0.369507  ,  0.109392   , -0.342089 ,  0.226376  , -0.107105  ,  0.0484649 ,
                      -0.024377  , -0.051962   , -0.0351849,  0.147194  ,  0.061761  , -0.131382  ,
                       0.00759035, -0.00101884 ,  0.0118011, -0.0296267 ,  0.025968  ,  0.076885  ,
                       0.0312734 ,  0.0293253  , -0.0141899,  0.109011  ,  0.0818948 , -0.0568241 ,
                       0.120272  ,  0.0249235  , -0.0298492,  0.0354401 , -0.262769  ,  0.230278  ,
                      -0.139116  , -0.132313   ,  0.13132  , -0.0565551 , -0.0338864 ,  0.0712653 ,
                       0.20246   , -0.116113   ,  0.102453 ,  0.0632254 ,  0.0694305 ,  0.00296431,
                       0.0522931 ,  0.0780708  , -0.0977045]
        # Medium values of f(R) and relative table
        fR0_mid    = 1e-5
        param_mid  = [ 0.936496   , -0.545832  ,  0.634804 , -0.0290649 , -0.0954373 , -0.342491  ,
                       0.491066   ,  0.297816  , -0.287142 , -0.0399919 ,  0.3037    ,  0.360959  ,
                       0.000615209, -0.00941931, -0.0181341,  0.376297  ,  0.486358  ,  0.0349385 ,
                       0.240066   ,  0.188202  ,  0.665834 ,  0.0122249 , -0.0343399 , -0.0520361 ,
                       0.261006   ,  0.525633  ,  0.266255 ,  0.393546  ,  0.29088   , -0.411491  ,
                       0.776609   ,  0.470777  , -0.681923 , -0.079589  , -0.282388  ,  0.53954   ,
                      -0.0930797  ,  0.0783781 ,  0.194957 ,  0.270378  ,  0.370288  ,  0.194857  ,
                       0.318637   ,  0.0457011 ,  0.139237 ,  0.033403  ,  0.0762982 , -0.0001047 , 
                      -0.00275824 ,  0.0461644 ,  0.189897 ]

        # High values of f(R) and relative table
        fR0_high   = 5e-5
        param_high = [ 0.572477   ,  0.254686  ,  1.21637  ,  0.00046274, -0.0901242 , -0.355849  ,
                       2.31154    ,  2.29822   , -0.483186 ,  0.4988    ,  0.36089   ,  0.0703424 ,
                       0.0257389  ,  0.0168936 , -0.030697 , -0.206992  ,  0.266084  ,  0.603357  ,
                       0.574264   , -0.30799   ,  0.831644 , -0.0093644 ,  0.00221153,  0.0076829 ,
                      -0.650381   ,  0.0179215 ,  0.927038 ,  0.77903   ,  0.919643  , -0.936328  ,
                       1.26756    ,  1.44477   , -1.44129  ,  0.219594  ,  0.353883  ,  1.02533   ,
                      -0.251705   ,  0.124875  ,  0.345995 , -0.146438  ,  0.0200251 ,  0.0892343 ,
                       0.284755   , -0.158286  ,  0.541178 , -0.0471913 ,  0.139772  , -0.134888  ,
                       0.0959162  ,  0.368186  , -0.157828 ]

        # r \propto log(f_R0)
        r_low  = np.log(f_R0/fR0_low)
        r_mid  = np.log(f_R0/fR0_mid)
        r_high = np.log(f_R0/fR0_high)

        # Find ratios
        ratio_low  = ratio_by_param(r_low , a, K, param_low )  # 1e-7 < fR0 < 5e-6
        ratio_mid  = ratio_by_param(r_mid , a, K, param_mid )  # 5e-6 < fR0 < 5e-5
        ratio_high = ratio_by_param(r_high, a, K, param_high)  # 1e-5 < fR0 < 1e-4

        # Return
        if   f_R0>5e-5: enhancement = ratio_high
        elif f_R0<5e-6: enhancement = ratio_low
        elif f_R0>1e-5: enhancement = ratio_mid + (ratio_high - ratio_mid) * (f_R0 - 1e-5)/(5e-5 - 1e-5)
        else:           enhancement = ratio_low + (ratio_mid  - ratio_low) * (f_R0 - 5e-6)/(1e-5 - 5e-6)

        # Change due to Omega_m
        #dom_om       = (Omega_m-0.3)/0.3
        #aaa          = 0.015
        #bbb          = 1.4
        #kstar        = 0.16*(1e-5/f_R0)**0.5
        #enhancement *= 1-aaa*dom_om*np.tanh((K/kstar)**bbb)

        # Change due to sigma_8
        #ds8_s8       = (sigma_8-0.8)/0.8
        #kst          = 1.2
        #enhancement *= 1+ds8_s8*K/(1+(K/kstar)**2)

    # Linear enhancement
    else:
        r   = np.log(f_R0/1e-5)
        K  *= np.sqrt(f_R0/1e-5)
        b_Z =  3.10000+ 2.34466*(a-1.)- 1.86362*(a-1.)**2.
        c_Z = 34.49510+28.86370*(a-1.)-13.13020*(a-1.)**2.
        d_Z =  0.14654- 0.01000*(a-1.)- 0.14944*(a-1.)**2.
        e_Z =  1.62807+ 0.71291*(a-1.)- 1.41003*(a-1.)**2.
        enhancement = 1. + (b_Z*K)**2./(1.+c_Z*K**2.) + d_Z*np.abs(np.log(K)*K/(K-1.))*np.arctan(e_Z*K)

    # There cannot be suppression
    enhancement[np.where(enhancement<1.0)] = 1.0

    return enhancement

def ratio_by_param(r,a,k,param):

    aminusone  = a-1.
    aminusone2 = aminusone**2.
    r2         = r**2.

    b0 = (param[ 0]) + (param[ 9])*r + (param[18])*r2 
    b1 = (param[ 1]) + (param[10])*r + (param[19])*r2
    b2 = (param[ 2]) + (param[11])*r + (param[20])*r2
    c0 = (param[ 3]) + (param[12])*r + (param[21])*r2 
    c1 = (param[ 4]) + (param[13])*r + (param[22])*r2
    c2 = (param[ 5]) + (param[14])*r + (param[23])*r2
    d0 = (param[ 6]) + (param[15])*r + (param[24])*r2 
    d1 = (param[ 7]) + (param[16])*r + (param[25])*r2
    d2 = (param[ 8]) + (param[17])*r + (param[26])*r2
    e0 = (1.0      ) + (param[27])*r + (param[30])*r2
    e1 = (0.0      ) + (param[28])*r + (param[31])*r2
    e2 = (0.0      ) + (param[29])*r + (param[32])*r2
    f0 = (param[33]) + (param[36])*r + (param[39])*r2 
    f1 = (param[34]) + (param[37])*r + (param[40])*r2
    f2 = (param[35]) + (param[38])*r + (param[41])*r2
    g0 = (param[42]) + (param[45])*r + (param[48])*r2 
    g1 = (param[43]) + (param[46])*r + (param[49])*r2
    g2 = (param[44]) + (param[47])*r + (param[50])*r2
      
    b = b0 + b1*aminusone + b2*aminusone2
    c = c0 + c1*aminusone + c2*aminusone2
    d = d0 + d1*aminusone + d2*aminusone2
    e = e0 + e1*aminusone + e2*aminusone2
    f = f0 + f1*aminusone + f2*aminusone2
    g = g0 + g1*aminusone + g2*aminusone2

    return 1.+b*(1.+c*k)/(1.+e*k)*np.abs(np.arctan(d*k))**(1.+f+g*k)


#-----------------------------------------------------------------------------------------
# SIGMA 8
#-----------------------------------------------------------------------------------------
def compute_sigma_8(k, pk):
    """
    This routine computes :math:`sigma_8`.

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array, default = []

    :param pk: Power spectrum in units of :math:`(\mathrm{Mpc}/h)^3`.
    :type pk: array, default = []

    :return: float
    """

    # Smoothing radius of 8 Mpc/h
    R = 8.
    # Assertions on scales, lengths and intervals
    assert np.max(k)>=np.pi/R*2., "k_max too low to obtain a convergent result. Use k_max >= pi/4 h/Mpc to obtain covergent results."
    assert np.min(k)<=0.001,      "k_min too high to obtain a convergent result. Use k_min <= 0.001 h/Mpc to obtain covergent results."
    assert len(k)>=100,           "size of 'k' too low to obtain a convergent result. Use at least 100 points."
    # Make 'k' 2D
    k2d = np.atleast_2d(k)
    # Top-hat window function
    W_kR = TopHat_window(k2d*R)
    # Integration in log-bins
    integral = sint.simps(k2d**3.*pk/(2.*np.pi**2.)*W_kR**2.,x=np.log(k),axis=1)
    return integral**.5

#-----------------------------------------------------------------------------------------
# T_WDM
#-----------------------------------------------------------------------------------------
def WDM_temperature_vs_cmb(omega_wdm,m_wdm):
    """
    This routine computes the temperature of thermal WDM in units of photon temperature given its mass and its density parameter.

    :param omega_wdm: reduced density parameter of WDM, :math:`\Omega_\mathrm{wdm} h^2`.
    :type omega_wdm: float

    :param m_wdm: WDM particle mass in :math:`\mathrm{eV}`.
    :type m_wdm: float

    :return: float
    """
    T_nu_over_T_cmb_std = 0.71611
    return T_nu_over_T_cmb_std * np.power(omega_wdm*93.14/m_wdm,1./3.)

def WDM_temperature(Omega_wdm,m_wdm,h=0.67,T_cmb=2.7255):
    """
    This routine computes the temperature of thermal WDM given its mass and its density parameter.

    :param Omega_wdm: reduced density parameter of WDM
    :type Omega_wdm: float

    :param m_wdm: WDM particle mass in :math:`\mathrm{eV}`.
    :type m_wdm: float

    :param h: Hubble parameter today in units of :math:`100 \ \mathrm{km/s/Mpc}`.
    :type h: float, default = 0.67

    :param T_cmb: CMB temperature today, in kelvin
    :type T_cmb: float, default = 2.7255 K

    :return: float, in kelvin.
    """
    return WDM_temperature_vs_cmb(np.array(Omega_wdm)*h**2.,np.array(m_wdm))*T_cmb

#-----------------------------------------------------------------------------------------
# OMEGA_WDM
#-----------------------------------------------------------------------------------------
def omega_wdm_from_mass_and_temperature(m_wdm,T_wdm_wrt_cmb):
    """
    This routine computes the reduced density parameter of WDM given its mass and its temperature.

    :param m_wdm: WDM particle mass in :math:`\mathrm{eV}`.
    :type m_wdm: float

    :param T_wdm_wrt_cmb: WDM temperature in units of photon temperature
    :type T_wdm_wrt_cmb: float

    :return: float, in kelvin.
    """
    T_nu_over_T_cmb_std = 0.71611
    return m_wdm/93.14*(T_wdm_wrt_cmb/T_nu_over_T_cmb_std)**3.

def Omega_wdm_from_mass_and_temperature(m_wdm,T_wdm,h=0.67,T_cmb=2.7255):
    """
    This routine computes the reduced density parameter of WDM given its mass and its temperature.

    :param m_wdm: WDM particle mass in :math:`\mathrm{eV}`.
    :type m_wdm: float

    :param T_wdm: WDM temperature in kelvin
    :type T_wdm: float

    :param h: Hubble parameter today in units of :math:`100 \ \mathrm{km/s/Mpc}`.
    :type h: float, default = 0.67

    :param T_cmb: CMB temperature today, in kelvin
    :type T_cmb: float, default = 2.7255 K

    :return: float, in kelvin.
    """
    return omega_wdm_from_mass_and_temperature(m_wdm,T_wdm/T_cmb)/h**2.

#-----------------------------------------------------------------------------------------
# ALCOCK-PACZYNSKI EFFECT
#-----------------------------------------------------------------------------------------
def AP_factors(z,cosmo,cosmo_fid,massive_nu_approx=True):
    """
    Compute the Alcock-Paczynski parallel and perpendicular factors.

    :param z: redshift.
    :type z: float

    :param cosmo: current cosmology for which parameters will be computed
    :type cosmo: :func:`colibri.cosmology.cosmo()` object

    :param cosmo_fid: fiducial cosmology for which probes are measured
    :type cosmo_fid: :func:`colibri.cosmology.cosmo()` object

    :param massive_nu_approx: whether to assume neutrinos behave as pure matter at all redshifts
    :type massive_nu_approx: boolean, default = True

    :return: 2 floats, :math:`q_\parallel` and  :math:`q_\perp`

    """
    q_par  = cosmo_fid.H(z)/cosmo.H(z)
    q_perp = cosmo    .angular_diameter_distance(z,massive_nu_approx)/ \
             cosmo_fid.angular_diameter_distance(z,massive_nu_approx)
    return q_par, q_perp

def AP_polar_coordinates_fourier_space(z,k_fid,mu_fid,cosmo,cosmo_fid,massive_nu_approx=True):
    """
    Compute the Alcock-Paczynski corrected polar coordinates in Fourier space.

    :param z: redshift.
    :type z: float

    :param k_fid: scales in :math:`h/\mathrm{Mpc}`
    :type k_fid: array

    :param mu_fid: cosines of angle w.r.t. line of sight.
    :type mu_fid: array

    :param cosmo: current cosmology for which parameters will be computed
    :type cosmo: :func:`colibri.cosmology.cosmo()` object

    :param cosmo_fid: fiducial cosmology for which probes are measured
    :type cosmo_fid: :func:`colibri.cosmology.cosmo()` object

    :param massive_nu_approx: whether to assume neutrinos behave as pure matter at all redshifts
    :type massive_nu_approx: boolean, default = True

    :return: two 2D arrays of shape ``(len(s_fid), len(mu_fid))`` containing scales and angles in the new cosmology

    """
    KK,MMUU      = np.meshgrid(k_fid,mu_fid,indexing='ij')
    q_par,q_perp = AP_factors(z,cosmo,cosmo_fid,massive_nu_approx)
    F_AP         = q_par/q_perp
    denominator  = (1.+MMUU**2*(F_AP**(-2.)-1.))**0.5
    k_prime      = KK/q_perp*denominator
    mu_prime     = MMUU/F_AP/denominator
    return k_prime,mu_prime

def AP_polar_coordinates_configuration_space(z,s_fid,mu_fid,cosmo,cosmo_fid,massive_nu_approx=True):
    """
    Compute the Alcock-Paczynski corrected polar coordinates in configuration space.

    :param z: redshift.
    :type z: float

    :param s_fid: scales in :math:`\mathrm{Mpc}/h`
    :type s_fid: array

    :param mu_fid: cosines of angle w.r.t. line of sight.
    :type mu_fid: array

    :param cosmo: current cosmology for which parameters will be computed
    :type cosmo: :func:`colibri.cosmology.cosmo()` object

    :param cosmo_fid: fiducial cosmology for which probes are measured
    :type cosmo_fid: :func:`colibri.cosmology.cosmo()` object

    :param massive_nu_approx: whether to assume neutrinos behave as pure matter at all redshifts
    :type massive_nu_approx: boolean, default = True

    :return: two 2D arrays of shape ``(len(s_fid), len(mu_fid))`` containing scales and angles in the new cosmology

    """
    SS,MMUU      = np.meshgrid(s_fid,mu_fid,indexing='ij')
    q_par,q_perp = AP_factors(z,cosmo,cosmo_fid,massive_nu_approx)
    F_AP         = q_par/q_perp
    denominator  = (1.+MMUU**2*(F_AP**2.-1.))**0.5
    s_prime      = SS*q_perp*denominator
    mu_prime     = MMUU*F_AP/denominator
    return s_prime,mu_prime

def AP_cartesian_coordinates_fourier_space(z,k_par_fid,k_perp_fid,cosmo,cosmo_fid,massive_nu_approx=True):
    """
    Compute the Alcock-Paczynski corrected Cartesian coordinates in Fourier space.

    :param z: redshift.
    :type z: float

    :param k_par_fid: parallel component of the separation wavevector in :math:`h/\mathrm{Mpc}`
    :type k_par_fid: array

    :param k_perp_fid: perpendicular component of the separation wavevector in :math:`h/\mathrm{Mpc}`
    :type k_perp_fid: array

    :param cosmo: current cosmology for which parameters will be computed
    :type cosmo: :func:`colibri.cosmology.cosmo()` object

    :param cosmo_fid: fiducial cosmology for which probes are measured
    :type cosmo_fid: :func:`colibri.cosmology.cosmo()` object

    :param massive_nu_approx: whether to assume neutrinos behave as pure matter at all redshifts
    :type massive_nu_approx: boolean, default = True

    :return: two 1D arrays containing scales in the new cosmology

    """
    q_par,q_perp = AP_factors(z,cosmo,cosmo_fid,massive_nu_approx)
    k_par_prime,k_perp_prime = k_par_fid/q_par, k_perp_fid/q_perp
    return k_par_prime,k_perp_prime

def AP_cartesian_coordinates_configuration_space(z,s_par_fid,s_perp_fid,cosmo,cosmo_fid,massive_nu_approx=True):
    """
    Compute the Alcock-Paczynski corrected Cartesian coordinates in configuration space.

    :param z: redshift.
    :type z: float

    :param s_par_fid: parallel component of the separation vector in :math:`\mathrm{Mpc}/h`
    :type s_par_fid: array

    :param s_perp_fid: perpendicular component of the separation vector in :math:`\mathrm{Mpc}/h`
    :type s_perp_fid: array

    :param cosmo: current cosmology for which parameters will be computed
    :type cosmo: :func:`colibri.cosmology.cosmo()` object

    :param cosmo_fid: fiducial cosmology for which probes are measured
    :type cosmo_fid: :func:`colibri.cosmology.cosmo()` object

    :param massive_nu_approx: whether to assume neutrinos behave as pure matter at all redshifts
    :type massive_nu_approx: boolean, default = True

    :return: two 1D arrays of shape containing scales in the new cosmology

    """
    q_par,q_perp = AP_factors(z,cosmo,cosmo_fid,massive_nu_approx)
    s_par_prime,s_perp_prime = s_par_fid/q_par, s_perp_fid/q_perp
    return s_par_prime,s_perp_prime
