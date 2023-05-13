import colibri.halo as hc
import colibri.cosmology as cc
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family = 'serif', size = 25)

#=====================
# Test of 'halo'
#=====================
# This routine uses the 'halo' class to compute the non-linear matter
# power spectrum in the halo model

# Use more redshifts?
more_redshifts = True

# Define a cosmology instance
C = cc.cosmo()

# Set redshifts
if more_redshifts: zz = [0., 1., 2.]
else:              zz = np.atleast_1d(0.)

# Some colors for plotting
colors = ['magenta', 'darkgreen', 'darkorange']

# Define a halo instance
H = hc.halo(z = zz,                         # Redshift
			k = np.logspace(-4., 1., 201),  # Scales in h/Mpc
			code = 'camb',                  # Boltzmann code with which to compute linear P(k)
			BAO_smearing = False,           # Smooth BAO due to non-linearities
			cosmology = C)

# Load power spectrum according to the halo model
H.halo_Pk(kwargs_mass_function = {'a': 0.707, 'p': 0.3},   # arguments to pass to the Sheth-Tormen mass function
		  kwargs_concentration = {'c0': 9., 'b': 0.13})    # arguments to pass to the concentration parameter c0*(M/Mstar)**(-b)

# Store 1-halo, 2-halo and total terms
oneh = H.Pk['matter']['1-halo']
twoh = H.Pk['matter']['2-halo']
tot  = H.Pk['matter']['total halo']

# Compute non-linear power spectrum CAMB-halofit for comparison
kcamb, pkcamb = C.camb_Pk(k = H.k, z = H.z, nonlinear = True)

# Plot
plt.figure(figsize=(12,9))
ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
ax2 = plt.subplot2grid((4,4), (3,0), colspan=4)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.subplots_adjust(wspace=0, hspace=0)

for iz in range(len(np.atleast_1d(zz))):

	# Plot lines
	ax1.loglog(H.k, oneh[iz], 'r:', lw = 2.0)
	ax1.loglog(H.k, twoh[iz], 'b:', lw = 2.0)
	ax1.loglog(H.k, tot [iz],     c = colors[iz], lw = 2.0, label = '$z = %.1f$' %zz[iz])
	ax1.loglog(kcamb, pkcamb[iz], c = colors[iz], lw = 2.0, ls = '--')

	# Ratios
	ax2.semilogx(H.k, (tot[iz]/pkcamb[iz]-1.)*100., c = colors[iz], lw = 2.0)
	ax2.set_xlabel('$k$ $[h/\mathrm{Mpc}]$')
	ax2.set_ylabel(r'$\left(\frac{P_\mathrm{halo}(k)}{P(k)}-1\right) \ [\%]$', fontsize = 25)
	ax2.set_xlim(H.k.min(), H.k.max())
	ax2.set_ylim(-35., 35.)
	ax2.grid(True)

# Only for the legend
ax1.plot(0., 0., 'r:' , label = '1 halo term')
ax1.plot(0., 0., 'b:' , label = '2 halo term')
ax1.plot(0., 0., 'k',   label = 'halo model')
ax1.plot(0., 0., 'k--', label = 'HALOFIT')	

ax1.set_ylabel('$P_X(k) \ [(\mathrm{Mpc}/h)^3]$')
ax1.set_xlim(H.k.min(), H.k.max())
ax1.set_ylim(3e-2, 1e5)
ax1.grid(True)

ax1.legend(loc='lower left', ncol = 2, fontsize = 20, framealpha = 1)
plt.show()


