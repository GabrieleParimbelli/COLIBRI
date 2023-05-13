import colibri.cosmology as cc
import numpy as np
import colibri.fourier as FF
import colibri.useful_functions as UF
import matplotlib.pyplot as plt
import scipy.interpolate as si

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size = 25)

######################
# Test of FFTlog
######################
# This routine computes the linear power spectrum, inverse Fourier transforms it
# and then goes back to Fourier space to check both the correlation function and
# the goodness of the inversion

# Define cosmology 
C = cc.cosmo()

# Compute linear power spectrum with camb
k, pk = C.camb_Pk(z = 0, k = np.logspace(-4., 2.5, 1001))
pk = pk[0]

# Extrapolate a power law at low- and high-ends
k, pk = UF.extrapolate_log(k, pk, xmin = 1e-10, xmax = 1e6)

# Compute correlation function through inverse FFT in 3D
r,xi = FF.iFFT_iso_3D(k, pk)

# Go back to Fourier space through FFTlog
k_fft, pk_fft = FF.FFT_iso_3D(r, xi)

# Get interpolation
pk_fft_int = si.interp1d(k_fft,pk_fft,'cubic',fill_value=0,bounds_error=False)
pk_fft_reconstructed = pk_fft_int(k)


# Plot
fig,ax=plt.subplots(2,1,figsize = (12,9),sharex=True)
L,B,R,T=0.13,0.13,0.95,0.95
plt.subplots_adjust(L,B,R,T,0,0)
ax[0].loglog(k,     pk,     '--', label = 'camb')
ax[0].loglog(k_fft, pk_fft, '-',  label = 'double FFT')
ax[1].semilogx(k, pk_fft_reconstructed/pk, '-')
ax[1].set_xlabel('$k \ [h/\mathrm{Mpc}]$', fontsize = 30)
ax[0].set_ylabel('$P(k) \ [(\mathrm{Mpc}/h)^3]$', fontsize = 30)
ax[1].set_ylabel('output/input', fontsize = 30)
ax[1].set_ylim(0.9,1.1)
ax[0].legend()
plt.show()

