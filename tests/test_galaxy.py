import colibri.cosmology as cc
import colibri.galaxy as gc
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family = 'serif', size = 25)

#=====================
# Test of 'galaxy'
#=====================
# This routine uses the class `galaxy' to compute the galaxy power spectrum
# using the halo model and the Halo Occupation Distribution prescription

# Settings
multiple_redshifts = True
colors = ['magenta', 'darkgreen', 'darkorange']

# Cosmology instance
C      = cc.cosmo()

# Choose redshifts
if multiple_redshifts:
	zz = [0., 1., 2.]
else:
	zz = 0.
	zz = np.atleast_1d(zz)


# Define a galaxy instance
G = gc.galaxy(z = zz,							# Redshift
			  k = np.logspace(-4., 2., 201), 	# Scales in h/Mpc
			  BAO_smearing = True,				# Smooth BAO feature in non-linearities
			  cosmology = C)					# Cosmology

# Compute the galaxy power spectrum
# The method has 6 arguments:
#    - kind_central: name of a function (method) whose first argument is mass (in units of Msun/h)
#      and that describes the amount of central galaxies as function of halo mass
#    - kwargs_central: a dictionary containing the remaining arguments to pass to kind_central. Each
#      arguments must be of the same length of the number of redshifts of the initialization.
#    - kind_satellite: name of a function (method) whose first argument is mass (in units of Msun/h)
#      and that describes the amount of satellite galaxies as function of halo mass
#    - kwargs_satellite: a dictionary containing the remaining arguments to pass to kind_satellite. Each
#      arguments must be of the same length of the number of redshifts of the initialization.
#    - kwargs_mass_function: dictionary with arguments to pass to the halo mass function
#    - kwargs_concentration: dictionary with arguments to pass to concentration parameter
#
#
# As an alternative, one can previously load the HOD model with the function:
# G.load_HOD(kind_satellite, kwargs_satellite, kind_central, kwargs_central)
# and then compute the galaxy power spectrum without specifying again the HOD
# G.galaxy_Pk(kwargs_mass_function, kwargs_concentration)
#
# N.B. The load_HOD method returns self.Ncen, self.Nsat and self.Ntot = self.Ncen + self.Nsat,
#      three arrays that tell (for each redshift) the number of galaxies in 512 mass values log-spaced
#      between 10^2 and 10^18 Msun/h.
#
# N.B. The number of satellite galaxies is by default multiplied by the number of centrals in order to avoid having
#      a satellite galaxy with no centrals.
if multiple_redshifts:
	G.galaxy_Pk(kind_satellite   = G.power_law, # ((M-Mcut)/M1)**alpha (when M>Mcut), ranges from 0 to infinity
				kwargs_satellite = {'log_Mcut': [12., 12.5, 12.3],
                                    'log_M1'  : [13., 13.4, 13.],
                                    'alpha'   : [1., 1.5, 1.]},
				kind_central     = G.logistic_function, # 1/2*{1+erf[(logM-log M_min)/sigma_logM]}, ranges from 0 to 1
				kwargs_central   = {'log_Mmin'  : [13., 12.4, 11.4],
                                    'sigma_logM': [0.8, 0.5, 0.6]},
				kwargs_mass_function = {'a': 0.707, 'p': 0.3},
		  		kwargs_concentration = {'c0': 9., 'b': 0.13})
else:
	G.galaxy_Pk(kind_satellite   = G.power_law,
				kwargs_satellite = {'log_Mcut':14., 'log_M1': 13., 'alpha': 1.},
				kind_central     = G.logistic_function,
				kwargs_central   = {'log_Mmin': 12., 'sigma_logM': 0.5},
				kwargs_mass_function = {'a': 0.707, 'p': 0.3},
		  		kwargs_concentration = {'c0': 9., 'b': 0.13})
pg1, pg2, pgt = G.Pk['galaxies']['real space']['1-halo'], G.Pk['galaxies']['real space']['2-halo'], G.Pk['galaxies']['real space']['total halo']

# Compute non-linear matter power spectrum (1-halo, 2-halo, total)
G.halo_Pk()
ph1, ph2, pht = G.Pk['matter']['1-halo'], G.Pk['matter']['2-halo'], G.Pk['matter']['total halo']

# Bias function b(k) = [P_g(k)/P_m(k)]^0.5
galaxy_bias = (pgt/pht)**.5


# Plot
plt.figure(figsize=(12,8))
ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
ax2 = plt.subplot2grid((4,4), (3,0), colspan=4)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.subplots_adjust(wspace=0, hspace=0)
for iz in range(len(np.atleast_1d(zz))):

	# Plot galaxy power spectrum
	ax1.loglog(G.k, pg1[iz], 'r:', lw = 2.0)
	ax1.loglog(G.k, pg2[iz], 'b:', lw = 2.0)
	ax1.loglog(G.k, pgt[iz], c = colors[iz], lw = 3.0, label = '$P_\mathrm{galaxy}(k), z = %.1f$' %zz[iz])

	# Plot power spectrum according to halo model
	ax1.loglog(G.k, pht[iz], c = colors[iz], ls = '--' , lw = 3.0)

	# Bias
	ax2.semilogx(G.k, galaxy_bias[iz], c = colors[iz], lw = 2.0)


ax1.plot(0., 0., c = 'k', lw = 3., ls = '--', label = '$P_\mathrm{halo \ model}(k)$')
ax1.set_ylabel('$P_X(k) \ [(\mathrm{Mpc}/h)^3]$')
ax1.set_xlim(G.k.min(), G.k.max())
ax1.set_ylim(3e-2, 1e6)
ax1.grid(True)

ax2.set_xlabel('$k$ $[h/\mathrm{Mpc}]$')
ax2.set_ylabel('$b(k)$')
ax2.set_xlim(G.k.min(), G.k.max())
ax2.set_ylim(-0.5, 5.5)
ax2.grid(True)

# Only for the legend
ax1.plot(0., 0., 'r:' , label = '1 halo term')
ax1.plot(0., 0., 'b:' , label = '2 halo term')

ax1.legend(loc='lower left', ncol = 2, fontsize = 20)
plt.show()



