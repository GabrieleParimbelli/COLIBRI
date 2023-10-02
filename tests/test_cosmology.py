import colibri.cosmology as cc
import colibri.constants as const
import numpy as np
import matplotlib.pyplot as plt

# Settings for plots
plt.rc('font',size=20,family='serif')
plt.rc('text',usetex=True)


# Set an array of redshifts
# here in log-space from 0 to ~1e7
zz = np.geomspace(1., 1e7, 101)-1

# Precompute neutrino sector
mnu        = [0.05,0.01]

# WDM sector
mw         = [10.]
Ow         = [0.01]

# Number of non-cold species
mnu        = np.atleast_1d(mnu)
mw         = np.atleast_1d(mw)
delta_neff = 0.044
nn         = len(mnu)
nnu        = max(nn,3)
nw         = len(mw)
neff       = nnu + nw + delta_neff
nmassive   = nn + nw

# Colibri "cosmo" instance
C  = cc.cosmo(Omega_m   = 0.32,      # Total matter density today
              Omega_b   = 0.05,      # Baryonic matter density today
              As        = 2.1265e-9, # Amplitude of primordial fluctuations
              ns        = 0.96,      # Index of primordial fluctuations
              h         = 0.67,      # Hubble parameter
              Omega_K   = 0.0,       # Curvature parameter
              w0        = -1.0,      # Dark energy parameter of state today (CPL parametrization)
              wa        = 0.0,       # Variation of dark energy parameter of state (CPL parametrization)
              T_cmb     = 2.7255,    # CMB temperature in K
              N_eff     = neff,      # Effective number of relativistic species in the early Universe
              N_nu      = nnu,       # Number of active neutrinos (integer)
              M_nu      = mnu,       # List of neutrino masses in eV
              M_wdm     = mw,        # List of WDM species masses in eV
              Omega_wdm = Ow)        # List of WDM density parameters

print("> colibri 'cosmo' class called with following parameters:")
print("    Matter")
print("        Omega_m           : %.6f" %C.Omega_m)
print("        Omega_cdm         : %.6f" %C.Omega_cdm)
print("        Omega_b           : %.6f" %C.Omega_b)
print("        Omega_nu          : %s" %(''.join(str(C.Omega_nu))))
print("        Omega_wdm         : %s" %(''.join(str(C.Omega_wdm))))
print("    Relativistic species")
print("        T_cmb             : %.6f K" %C.T_cmb)
print("        Omega_gamma       : %.6f" %C.Omega_gamma)
print("        Omega_ur          : %.6f" %C.Omega_ur)
print("    Dark energy")
print("        Omega_DE          : %.6f" %C.Omega_lambda)
print("        w0                : %.6f" %C.w0)
print("        wa                : %.6f" %C.wa)
print("    Curvature")
print("        Omega_K           : %.6f" %C.Omega_K)
print("        K                 : %.6f (h/Mpc)^2" %C.K)
print("    Hubble expansion")
print("        h                 : %.6f" %C.h)
print("        H0                : %.6f km/s/Mpc" %C.H0)
print("    Initial conditions")
print("        As                : %.6e" %C.As)
print("        ns                : %.6f" %C.ns)

# Compute evolution of density parameters
Omega_de    = C.Omega_lambda_z(zz)
Omega_cdm   = C.Omega_cdm_z(zz)
Omega_b     = C.Omega_b_z(zz)
Omega_gamma = C.Omega_gamma_z(zz)
Omega_K     = C.Omega_K_z(zz)
Omega_wdm   = C.Omega_wdm_z(zz)
Omega_nu    = C.Omega_nu_z(zz)
Omega_ur    = C.Omega_ur_z(zz)
print("> Redshift dependence of cosmological parameters computed")

# Compute Hubble parameter, distances
Hz          = C.H(zz)
DC          = C.comoving_distance(zz)
DL          = C.luminosity_distance(zz)
DA          = C.angular_diameter_distance(zz)
print("> Hubble function and distances computed")

# Plot
fig, ax = plt.subplots(2,1,figsize=(9,8),sharex=True)
L,B,R,T = 0.13,0.13, 0.95, 0.84
plt.subplots_adjust(L,B,R,T,0.35,0)

# Plot Omegas
ax[0].loglog(1+zz, Omega_cdm  ,c='blue'     ,lw=2,label='cdm')
ax[0].loglog(1+zz, Omega_b    ,c='red'      ,lw=2,label='b')
ax[0].loglog(1+zz, Omega_de   ,c='black'    ,lw=2,label='$\Lambda$')
ax[0].loglog(1+zz, Omega_gamma,c='green'    ,lw=2,label='$\\gamma$')
ax[0].loglog(1+zz, Omega_ur   ,c='violet'   ,lw=2,label='ur')
if C.Omega_K>=0: ax[0].loglog(1+zz, Omega_K    ,c='goldenrod',lw=2,ls='-' ,label='K')
else           : ax[0].loglog(1+zz,-Omega_K    ,c='goldenrod',lw=2,ls='--',label='-K')
for i in range(nn): ax[0].loglog(1+zz, Omega_nu[i] ,c='magenta',lw=2)
for i in range(nw): ax[0].loglog(1+zz, Omega_wdm[i],c='cyan'   ,lw=2)
ax[0].loglog(np.nan,np.nan,c='magenta',lw=2,label='$\\nu$')
ax[0].loglog(np.nan,np.nan,c='cyan'   ,lw=2,label='wdm')

# Non-relativistic transition of different free-streaming species
for i in range(len(C.M_nu)):  ax[0].axvline(1890*C.M_nu[i], c='magenta',ls=':')
for i in range(len(C.M_wdm)): ax[0].axvline(C.M_wdm[i]/(const.kB*C.T_wdm[i])/np.sqrt(8./np.pi), c='cyan',ls=':')

# Plot H(z), distances
ax[1].loglog(1+zz, Hz/C.H0,'k',lw=2,label='$H(z)/H_0$')
ax[1].loglog(1+zz, DC     ,'b',lw=2,label='$\chi(z)$')
ax[1].loglog(1+zz, DL     ,'r',lw=2,label='$D_L(z)$')
ax[1].loglog(1+zz, DA     ,'g',lw=2,label='$D_A(z)$')

ax[1].set_xlabel('$1+z$')
ax[0].set_ylabel('$\Omega_i(z)$')

ax[0].legend(loc='lower center',
               bbox_to_anchor=(0.5,1.01),
               bbox_transform=ax[0].transAxes,ncol=4,fontsize=20)
ax[1].legend(loc='upper left',ncol=2,fontsize=16)

ax[0].set_xlim((1+zz).min(), (1+zz).max())
ax[0].set_ylim(3e-6, 5.)

plt.show()

