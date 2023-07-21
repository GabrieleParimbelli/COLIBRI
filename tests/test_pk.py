import numpy as np
import colibri.cosmology as cc

# Cosmology instance (default constructor with default parameters)
C  = cc.cosmo()

# Main settings
multiple_redshifts = True  # Sets if one wants a single redshift or many (see below which ones)
nonlinear          = False # Boolean for non-linear power spectra

print("------------------------------")
print("COMPUTING MATTER POWER SPECTRA")
print("------------------------------")
print("mutliple redshifts : %s" %multiple_redshifts)
print("nonlinear          : %s" %nonlinear)
print("------------------------------")

# Redshifts
if multiple_redshifts: z = np.linspace(0., 5., 6)
else:                  z = 0.
z = np.atleast_1d(z)
string_redshifts = ', '.join(str(x) for x in z) 

# Scales in h/Mpc
k = np.logspace(-4., 2., 1001)

# Compute power spectrum with CAMB
k_camb, pk_camb = C.camb_Pk(z = z,                   # Redshifts
                            k = k,                   # Scales
                            nonlinear = nonlinear,   # do nonlinear (as set above)
                            halofit = 'mead',        # use HMcode2016 if asked for nonlinear
                            var_1 = 'tot',           # Use total matter field for 1st variable
                            var_2 = 'tot')           # Use total matter field for 2nd variable
print(">> Done CAMB")

# Compute power spectrum with CAMB using cross spectra as a proxy
k_xcamb, pk_xcamb = C.camb_XPk(z = z,
                               k = k,
                               nonlinear = nonlinear,
                               halofit = 'mead',     # use HMcode2016 if asked for nonlinear
                               var_1 = ['tot'],
                               var_2 = ['tot'])
print(">> Done XCAMB")

# Compute power spectrum with Class
k_class, pk_class = C.class_Pk(z = z,                    # Redshifts
                               k = k,                    # Scales
                               nonlinear = nonlinear)    # do nonlinear (as set above)
print(">> Done CLASS")

# Compute power spectrum with Eisenstein-Hu formula (only linear, only for LCDM)
k_eh, pk_eh = C.EisensteinHu_Pk(z = z,              # Redshifts
                                k = k,              # Scales
                                sigma_8 = 0.831)    # normalization (As cannot be used here)
print(">> Done Eisenstein-Hu")

# Compute power spectrum with Eisenstein-Hu formula (without wiggles)
k_eh_nw, pk_eh_nw = C.EisensteinHu_nowiggle_Pk(z = z,              # Redshifts
                                               k = k,              # Scales
                                               sigma_8 = 0.831)    # normalization (As cannot be used here)
print(">> Done Eisenstein-Hu no wiggles")

# Plot
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family = 'serif', size = 25)

fig, ax = plt.subplots(2,1,sharex = True, figsize=(12,9))
plt.subplots_adjust(hspace = 0., wspace = 0.)
for i in range(len(np.atleast_1d(z))):
    ax[0].loglog(k_camb,  pk_camb[i],             'b',   lw = 3.0)
    ax[0].loglog(k_xcamb, pk_xcamb['tot-tot'][i], 'r-.', lw = 3.0)
    ax[0].loglog(k_class, pk_class[i],            'g--', lw = 3.0)
    ax[0].loglog(k_eh,    pk_eh[i],               'k:',  lw = 3.0)
    ax[0].loglog(k_eh_nw, pk_eh_nw[i],            'm:',  lw = 3.0)
ax[0].plot(0., 0., 'b',   lw = 3.0, label = 'CAMB')
ax[0].plot(0., 0., 'r-.', lw = 3.0, label = 'XCAMB')
ax[0].plot(0., 0., 'g--', lw = 3.0, label = 'CLASS')
ax[0].plot(0., 0., 'k:',  lw = 3.0, label = 'EH')
ax[0].plot(0., 0., 'm:',  lw = 3.0, label = 'EH no wiggle')
ax[0].legend(loc = 'lower left', fontsize = 16, ncol = 5)
ax[0].set_xlim(k.min(), k.max())
ax[0].set_ylabel('$P(k)$ $[(\mathrm{Mpc}/h)^3]$')
ax[0].text(0.98, 0.8, '$z =$ %s' %string_redshifts, ha='right', transform = ax[0].transAxes, bbox = dict(boxstyle='round', facecolor='white', alpha = 1.0))
ax[0].grid(True)

# Plot ratios
for i in range(len(np.atleast_1d(z))):
    ax[1].semilogx(k, (pk_xcamb['tot-tot'][i]/pk_camb[i]-1.)*100., 'r-.', lw = 3.0)
    ax[1].semilogx(k, (pk_class[i]           /pk_camb[i]-1.)*100., 'g--', lw = 3.0)
    ax[1].semilogx(k, (pk_eh[i]              /pk_camb[i]-1.)*100., 'k:',  lw = 3.0)
    ax[1].semilogx(k, (pk_eh_nw[i]           /pk_camb[i]-1.)*100., 'm:',  lw = 3.0)
ax[1].set_xlim(k.min(), k.max())
ax[1].set_ylim(-5.5, 5.5)
ax[1].set_xlabel('$k$ $[h/\mathrm{Mpc}]$')
ax[1].set_ylabel('$\\frac{\Delta P(k)}{P_\mathrm{camb}(k)} \ (\%)$')
ax[1].grid(True)
plt.show()
