import numpy as np
import colibri.cosmology as cc
import matplotlib.pyplot as plt

plt.rc('text',usetex=True)
plt.rc('font',size=15,family='serif')

#=============
# Plot settings
#=============
colors = ['r','b','g','y','m','c','k']

#=============
# Cosmology instance with default parameters
#=============
C=cc.cosmo()

#=============
# Scales, redshift, masses
#=============
zz   = np.linspace(0., 5., 6)
kk   = np.logspace(-3.,2.,501)
logM = np.arange(7.1,16.,0.1)
nz, nk, nm = len(np.atleast_1d(zz)), len(np.atleast_1d(kk)), len(np.atleast_1d(logM))

#=============
# Load linear power spectra
#=============
k,pk=C.camb_Pk(k=kk,z=zz)

#=============
# Mass variance in spheres
#=============
# Routine to compute mass variance in sphere of different masses.
# If one wants to use radii instead of masses, it is sufficient to run
#  M = C.masses_in_radius(R); logM = np.log10(M)
sigma_squared = C.mass_variance(logM = logM, k = k, pk = pk)

#=============
# Peak height
#=============
# Peak-background split quantity, i.e. nu = delta_c/sigma(M), where delta_c=1.686
# is the linear overdensity for collapse
nu_peak = C.peak_height(logM = logM, k = k, pk = pk)

#=============
# Sheth-Tormen
#=============
# Sheth-Tormen mass function (takes sigma(M), previously computed, as argument).
# Even if it returns things as function of sigma, we plot it as function of mass.
ShethTormen = C.ShethTormen_mass_function(sigma = sigma_squared**0.5, a = 0.707, p = 0.3)

#=============
# Halo mass function
#=============
# We use the Sheth-Tormen mass function.
# If 'mass_fun' == 'Tinker' or 'MICE', you should add 'z=zz' to the arguments of the function.
HMF = C.halo_mass_function(logM = logM, k = k, pk = pk, window = 'th', mass_fun = 'ShethTormen')

#=============
# Plot
#=============
fig, ax = plt.subplots(2,2,figsize=(12,12),sharex=True)
L,R,T,B=0.13,0.96,0.96,0.13
plt.subplots_adjust(left=L,right=R,top=T,bottom=B,wspace=0.25,hspace=0.)


for iz in range(nz):
    # Plot mass variance
    ax[0,0].loglog(10.**logM,sigma_squared[iz],c=colors[iz],label='$z=%i$' %(zz[iz]))

    # Plot peak height
    ax[0,1].loglog(10.**logM,nu_peak[iz],c=colors[iz])

    # Plot Sheth-Tormen
    ax[1,0].loglog(10.**logM,ShethTormen[iz],c=colors[iz])
    ax[1,0].set_ylim(1e-5,1e0)

    # Plot halo mass function
    ax[1,1].loglog(10.**logM,10.**logM*HMF[iz],c=colors[iz])
    ax[1,1].set_ylim(1e-12,1e3)

    # x limits
    ax[0,0].set_xlim(10.**logM.min(),10.**logM.max())

# Labels
for a in [0,1]:
    ax[1,a].set_xlabel('$M \ [M_\odot/h]$')
ax[0,0].set_ylabel('$\sigma^2(M)$')
ax[0,1].set_ylabel('$\\nu(M)=\\frac{\delta_c}{\sigma(M)}$')
ax[1,0].set_ylabel('$\\nu \ f_\mathrm{ST}(\\nu)$')
ax[1,1].set_ylabel('$M \\frac{\mathrm{d}n}{\mathrm{d}M} \ [h^3 \mathrm{Mpc}^{-3}]$')

# Legend and plot
ax[0,0].legend(loc='upper right',fontsize=12, ncol=2)
plt.show()







