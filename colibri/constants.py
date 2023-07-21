import numpy as np
import math

def explanatory():
    """
    This file contains a series of physical constants and conversion factors which are used in the other codes.
    Also useful quantities not strictly related to cosmology are defined.

    Conversion factors, distance

     - ``km_to_cm``: 1 km in cm
     - ``km_to_m``: 1 km in m 
     - ``pc_to_cm``: 1 pc in cm 
     - ``pc_to_m``: 1 pc in m 
     - ``pc_to_km``: 1 pc in km 
     - ``kpc_to_cm``: 1 kpc in cm 
     - ``kpc_to_m``: 1 kpc in m 
     - ``kpc_to_km``: 1 kpc in km 
     - ``Mpc_to_cm``: 1 Mpc in cm 
     - ``Mpc_to_m``: 1 Mpc in m 
     - ``Mpc_to_km``: 1 Mpc in km 

    Conversion factors, time

     - ``yr_to_s``: 1 yr in s 
     - ``Myr_to_s``: 1 Myr in s 

    Conversion factors, energy

     - ``eV``: electron-Volt in :math:`\mathrm{J}`
     - ``keV``: kilo Electron-Volt in :math:`\mathrm{J}` 
     - ``MeV``: mega Electron-Volt in :math:`\mathrm{J}` 
     - ``GeV``: giga Electron-Volt in :math:`\mathrm{J}` 
     - ``TeV``: tera Electron-Volt in :math:`\mathrm{J}` 

    Proton & electron mass & charges

     - ``mp``: proton mass in :math:`\mathrm{eV}`
     - ``mp_g``: proton mass in :math:`\mathrm{g}` 
     - ``mp_J``: proton mass in :math:`\mathrm{J}`
     - ``me``: eletron mass in :math:`\mathrm{eV^2}` 
     - ``me_g``: eletron mass in :math:`\mathrm{g}` 
     - ``me_J``: eletron mass in :math:`\mathrm{J}` 
     - ``q``: electron/proton charge in :math:`\mathrm{C}` 

    Neutrino properties

     - ``Delta_m21_squared``: difference of squared masses in :math:`\mathrm{eV^2}`
     - ``Delta_m32_squared_IH``: difference of squared masses in :math:`\mathrm{eV^2}`
     - ``Delta_m32_squared_NH``: difference of squared masses in :math:`\mathrm{eV^2}`
     - ``sin_theta_21_squared``: sine squared of mixing angle 
     - ``sin_theta_23_squared_IH``: sine squared of mixing angle, inverted hierarchy
     - ``sin_theta_23_squared_NH``: sine squared of mixing angle, normal hierarchy
     - ``sin_theta_13_squared``: sine squared of mixing angle 

    Constants of physics

     - ``G``: Newton's gravitational constant in units of :math:`\mathrm{Mpc} \ M_\odot (\mathrm{km/s})^2`
     - ``eps_0``: vacuum permittivity in :math:`\mathrm{F/m = C/V \ m}`
     - ``mu_0``: magnetic permeability in :math:`\mathrm{H/m = T \ m^2/A = V \ s/A}`  
     - ``kB``: Boltzmann constant in :math:`\mathrm{eV/K}`  
     - ``c``: speed of light in :math:`\mathrm{km/s}`
     - ``hP``: Planck constant in units of :math:`\mathrm{eV \ s}`
     - ``sSB``: Stefan-Boltzmann constant in :math:`\mathrm{W \ m^{-2} \ K^{-4}}` 
     - ``N_A``: Avogadro constant in :math:`\mathrm{mol^{-1}}`  

    Derived constants

     - ``e2``: electron/proton charge (squared!) in CGS units 
     - ``hPb``: reduced Planck constant (:math:`\\bar{h}`) in :math:`\mathrm{eV \ s}`
     - ``hPJ``: Planck constant in :math:`\mathrm{J \ s}`
     - ``hPJb``: reduced Planck constant (:math:`\\bar{h}`) in :math:`\mathrm{J \ s}`
     - ``sSB_eV``: Stefan-Boltzmann constant in :math:`\mathrm{eV \ m^{-2} \ K^{-4}}`
     - ``alpha_BB``: constant for blackbody energy density in :math:`\mathrm{J \ m^{-3} \ K^{-4}}`
     - ``R``: perfect gas constant in :math:`\mathrm{J \ mol^{-1} \ K^{-1}}`  
     - ``alpha_EM``: fine structure constant 
     - ``lambda_e``: Compton wavelength for electron in :math:`\mathrm{m}` 
     - ``r_e``: electron classical radius in :math:`\mathrm{m}` 
     - ``sigma_T``: Thomson scattering cross section in :math:`\mathrm{m}^2`
     - ``rhoch2``: critical density of the Universe :math:`h^2 \ M_\odot \ \mathrm{Mpc}^{-3}`

    Planck units

     - ``l_Pl``: Planck length in :math:`\mathrm{m}` 
     - ``t_Pl``: Planck time in :math:`\mathrm{s}`  
     - ``m_Pl``: Planck mass in :math:`\mathrm{g}` 
     - ``T_Pl``: Planck temperatur in :math:`\mathrm{K}`  
     - ``q_Pl``: Planck charge in :math:`\mathrm{C}`  

    Solar units

     - ``Msun``: solar mass in :math:`\mathrm{g}` 
     - ``Rsun``: solar radius in :math:`\mathrm{cm}` 
     - ``Tsun``: solar surface temperature in :math:`\mathrm{K}`
     - ``Lsun``: solar luminosity in :math:`\mathrm{erg \ s^{-1}}`
    """
    return 0

#-------------------------------
# Conversion factors: distance
#-------------------------------
km_to_cm  = 1e5                  # 1 km in cm
km_to_m   = 1e3                  # 1 km in m
pc_to_cm  = 3.085677581282e18    # 1 pc in cm
pc_to_m   = 3.085677581282e16    # 1 pc in m
pc_to_km  = 3.085667581282e13    # 1 pc in km
kpc_to_cm = 3.085677581282e21    # 1 kpc in cm
kpc_to_m  = 3.085677581282e19    # 1 kpc in m
kpc_to_km = 3.085667581282e16    # 1 kpc in km
Mpc_to_cm = 3.085677581282e24    # 1 Mpc in cm
Mpc_to_m  = 3.085677581282e22    # 1 Mpc in m
Mpc_to_km = 3.085667581282e19    # 1 Mpc in km

#-------------------------------
# Conversion factors: time
#-------------------------------
yr_to_s   = 3.1536e7        # 1 yr in s
Myr_to_s  = 3.1536e13       # 1 Myr in s

#-------------------------------
# Conversion factors: energy
#-------------------------------
eV        = 1.60217663e-19    # Electron-Volt in J
keV       = 10.**3.*eV        # Kilo Electron-Volt in J
MeV       = 10.**6.*eV        # Mega Electron-Volt in J
GeV       = 10.**9.*eV        # Giga Electron-Volt in J
TeV       = 10.**12.*eV       # Tera Electron-Volt in J

#-------------------------------
# Proton & electron mass & charges
#-------------------------------
mp        = 9.38e8          # Proton mass in eV
mp_g      = 1.6726e-24      # Proton mass in g
mp_J      = 0.938*GeV       # Proton mass in J
me        = 5.11e5          # Eletron mass in eV
me_g      = 9.10938356e-28  # Eletron mass in g
me_J      = 511.*keV        # Eletron mass in J
q         = 1.602176487e-19 # Electron/proton charge in C

#-------------------------------
# Neutrino properties
#-------------------------------
Delta_m21_squared       = 7.53e-5    # Difference of squared masses [eV^2] 
Delta_m32_squared_IH    = -2.56e-3   # Difference of squared masses [eV^2 
Delta_m32_squared_NH    = 2.51e-3    # Difference of squared masses [eV^2] 
sin_theta_21_squared    = 0.307      # Sine squared of mixing angle
sin_theta_23_squared_IH = 0.592      # Sine squared of mixing angle, inverted hierarchy (S = 1.1)
sin_theta_23_squared_NH = 0.597      # Sine squared of mixing angle, normal hierarchy (S = 1.1)
sin_theta_13_squared    = 2.12e-2    # Sine squared of mixing angle

#-------------------------------
# Constants of physics
#-------------------------------
G         = 4.302180824e-9      # Newton's gravitational constant in units of [Mpc/M_sun (km/s)^2]
eps_0     = 8.85418781762e-12   # Vacuum permittivity in [F/m = C/V m] 
mu_0      = 1.25663706144e-6    # Magnetic permeability in [H/m = T m^2/A = V s/A] 
kB        = 8.617342791e-5      # Boltzmann constant in [eV/K] 
c         = 2.99792458e5        # Speed of light in [km/s] 
hP        = 4.135667334e-15     # Planck constant in units of [eV s] 
N_A       = 6.022140857e23      # Avogadro constant [mol^-1] 

#-------------------------------
# Derived constants
#-------------------------------
G_mks      = 6.67428e-11                # Newton's gravitational constant in units of [m^3/kg/s^2]
sSB        = 2*np.pi**5.*(kB*eV)**4/(15*(hP*eV)**3.*(c*km_to_m)**2.) # Stefan-Boltzmann constant [W/m^2 K^4]
e2         = q**2./(4.*np.pi*eps_0)        # Electron/proton charge (squared!) in CGS units
hPb        = hP/(2.*np.pi)                 # Reduced Planck constant in [eV s]  ('h bar')
hPJ        = hP*eV                      # Planck constant in [J s] 
hPJb       = hPb*eV                     # Reduced Planck constant in [J s]  ('h bar')
sSB_eV     = sSB/eV                     # Stefan-Boltzmann constant [eV/s m^2 K^4] 
alpha_BB   = 4.*sSB/(c*km_to_m)         # Constant for blackbody energy density [J/m^3 K^4] 
R          = kB*eV*N_A                  # Perfect gas constant [J/mol K] 
alpha_EM   = e2/(hPJb*c*km_to_m)        # Fine structure constant
lambda_e   = hP*c*km_to_m/me            # Compton wavelength for electron in [m] 
r_e        = alpha_EM*lambda_e/(2.*np.pi)  # Electron classical radius in [m] 
sigma_T    = 8.*np.pi/3.*r_e**2.           # Thomson scattering cross section in [m^2] 
rhoch2     = (3.*100.**2.)/(8*np.pi*G)     # Critical density of the Universe [h^2 Msun/Mpc^3] 
rhoch2_mks = 1.8783472458e-31           # Critical density of the Universe [h^2 kg/m^3] 

#-------------------------------
# Planck units
#-------------------------------
l_Pl      = 1.616252e-35   # Planck length [m] 
t_Pl      = 5.39124e-44    # Planck time [s] 
m_Pl      = 2.17644e-5     # Planck mass [g] 
T_Pl      = 1.416785e32    # Planck temperature [K] 
q_Pl      = 1.87554587e-18 # Planck charge [C] 

#-------------------------------
# Solar units
#-------------------------------
Msun      = 1.989e33       # Solar mass [g] 
Rsun      = 6.957e10       # Solar radius [cm] 
Tsun      = 5772.          # Solar surface temperature [K] 
Lsun      = 3.848e33       # Solar luminosity [erg/s] 

