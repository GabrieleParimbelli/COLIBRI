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
kk   = np.logspace(-4.,2.,1001)
logM = np.linspace(5.,16.,111)
nz, nk, nm = len(np.atleast_1d(zz)), len(np.atleast_1d(kk)), len(np.atleast_1d(logM))
#=============
# Load linear power spectra
#=============
k,pk=C.camb_Pk(z=zz)


#=============
# Mass variance in spheres
#=============
# The function returns a list of interpolated objects containing sigma^2[log10(M)] for each redshift
sigma2_interp = C.sigma2(k=k,pk=pk)
# Fill arrays
sigma_squared = np.zeros((nz,nm))
for iz in range(nz):
    sigma_squared[iz] = sigma2_interp[iz](logM)

#=============
# Peak height
#=============
# The function returns a list of interpolated objects containing nu^2[log10(M)] for each redshift
peak_height_interp = C.peak_height(k=k,pk=pk)
# Fill arrays
nu = np.zeros((nz,nm))
for iz in range(nz):
    nu[iz] = peak_height_interp[iz](logM)

#=============
# Sheth-Tormen
#=============
# Sheth-Tormen mass function (takes sigma(M), previously computed, as argument).
# Even if it returns things as function of sigma, we plot it as function of mass.
ShethTormen = C.ShethTormen_mass_function(sigma = sigma_squared**5., a = 0.707, p = 0.3)

#=============
# Halo mass function
#=============
# We use the Sheth-Tormen mass function.
# If 'mass_fun' == 'Tinker' or 'MICE', you should add 'z=zz' to the arguments of the function.
# The function returns a list of interpolated objects containing dn/dM[log10(M)] for each redshift
mass_functions = C.halo_mass_function(k = k, pk = pk, mass_fun = 'ShethTormen')
# Fill arrays
HMF = np.zeros((nz,nm))
for iz in range(nz):
    HMF[iz] = mass_functions [iz](logM)

#=============
# Plot
#=============
fig, ax = plt.subplots(2,2,figsize=(14,14),sharex=True)
L,R,T,B=0.13,0.96,0.96,0.13
plt.subplots_adjust(left=L,right=R,top=T,bottom=B,wspace=0.2,hspace=0.)

# Plot mass variance
for iz in range(nz):
    ax[0,0].loglog(10.**logM,sigma_squared[iz],c=colors[iz],label='$z=%i$' %(zz[iz]))

# Plot peak height
for iz in range(nz):
    ax[0,1].loglog(10.**logM,nu[iz],c=colors[iz])

# Plot Sheth-Tormen
for iz in range(nz):
    ax[1,0].loglog(10.**logM,ShethTormen[iz],c=colors[iz])
ax[1,0].set_ylim(1e-5,1e0)

# Plot Sheth-Tormen
for iz in range(nz):
    ax[1,1].loglog(10.**logM,10.**logM*HMF[iz],c=colors[iz])
ax[1,1].set_ylim(1e-12,1e3)

# x limits
ax[0,0].set_xlim(10.**logM.min(),10.**logM.max())

# Labels
for a in [0,1]:
    ax[1,a].set_xlabel('$M \ [M_\odot]$')
ax[0,0].set_ylabel('$\sigma^2(M)$')
ax[0,1].set_ylabel('$\\nu(M)=\\frac{\delta_c}{\sigma(M)}$')
ax[1,0].set_ylabel('$\\nu \ f_\mathrm{ST}(\\nu)$')
ax[1,1].set_ylabel('$M \\frac{\mathrm{d}n}{\mathrm{d}M} \ [h^3 \mathrm{Mpc}^{-3}]$')

# Legend and plot
ax[0,0].legend(loc='upper right',fontsize=12, ncol=2)
plt.show()







