import colibri.cosmology as cc
import numpy as np
import colibri.fourier as FF
import colibri.useful_functions as UF
import matplotlib.pyplot as plt

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size = 25)

######################
# Test of FFTlog
######################
# This routine computes the linear power spectrum, inverse Fourier transforms it
# and then goes back to Fourier space to check both the correlation function and
# the goodness of the inversion

# Define cosmology 
C = cc.cosmo(M_nu = 0.0)

# Compute linear power spectrum with camb
k, pk = C.camb_Pk(z = 0, k = np.logspace(-4., 2.5, 1001))
# The power spectrum is a matrix of shape (1,len(k)). Take the first element, i.e. an array of len(k)
pk = pk[0]
# Extrapolate a power law at low- and high-ends
k, pk = UF.extrapolate_log(k, pk, xmin = 1e-10, xmax = 1e6)

# Compute correlation function through inverse FFT in 3D
r,xi = FF.iFFT_iso_3D(k, pk)

# Go back to Fourier space through FFTlog
k_fft, pk_fft = FF.FFT_iso_3D(r, xi)

# Plot
plt.figure(figsize = (12,9))
plt.loglog(k,     pk,     '--', label = 'camb')
plt.loglog(k_fft, pk_fft, '-',  label = 'double FFT')
plt.xlabel('$k \ [h/\mathrm{Mpc}]$', fontsize = 30)
plt.ylabel('$P(k) \ [(\mathrm{Mpc}/h)^3]$', fontsize = 30)
plt.legend()
plt.show()

