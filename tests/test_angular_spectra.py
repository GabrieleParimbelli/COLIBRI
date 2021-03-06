import colibri.cosmology as cc
import colibri.angular_spectra as ang
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size = 40)

#########################
# Test of angular spectra class
#########################
number_of_bins     = 4
photometric_errors = True

#-----------------
# 1) Define a cosmology instance (with default values)
#-----------------
C = cc.cosmo()
print("> Cosmology loaded")
#-----------------

#-----------------
# 2) Define an angular_spectra instance.
#-----------------
# This takes as arguments:
#   - a cosmology instance:
#   - a 2-uple or a list of length 2, whose values are the lower and upper limit of integration in redshift
S = ang.angular_spectra(cosmology = C, z_limits = [0., 5.])
print("> Shear instance loaded")
#-----------------

#-----------------
# 3) Load power spectra
#-----------------
# The routine 'load_power_spectra' interpolates the power spectra at the scales and redshifts asked.
# It can be done in two ways: 
#   - they can be loaded by a Boltzmann solver (CAMB, Class of Eisenstein-Hu). In this case the arguments 
#     'code' and 'nonlinear' become important.
#
#     e.g. S.load_power_spectra(z = np.linspace(0., 5., 51), k = np.logspace(-4., 2., 101), code = 'camb', nonlinear = True)
#
#   - they can be computed using any other tool (e.g. with the 'halofit_operator' class in 'nonlinear.py' file;
#     calling CAMB/Class externally; reading them from file, if they are taken from simulations).
#     In this case, the arguments 'code' and 'nonlinear' are useless and one must set as arguments the regular grid in scales and redshifts
#
#     e.g. CALL CAMB EXTERNALLY
#          kk = np.geomspace(1e-4, 1e2, 301)
#          zz = np.linspace(0., 5., 51)
#          _, pk_array = C.camb_Pk(z = zz, k = kk, nonlinear = True, halofit = 'mead2020')	# Call CAMB externally with 'mead2020' halofit version
#          S.load_power_spectra(z = zz, k = kk, power_spectra = pk_array)	                # Interpolate, pk_array must be a 2D array of shape (51,151)
#
#     e.g. READ FROM SIMULATIONS
#          kk = np.geomspace(1e-4, 1e2, 301)
#          zz = np.linspace(0., 5., 51)
#          pk_array = np.loadtxt('pk_array.txt', unpack = True)				# Read file containing power spectra
#          S.load_power_spectra(z = zz, k = kk, power_spectra = pk_array)	# Interpolate, pk_array must be a 2D array of shape (51,151)
#
kk = np.geomspace(1e-4, 1e2, 301)
zz = np.linspace(0., 5., 51)
_, pkz = C.camb_Pk(z = zz, k = kk, nonlinear = True, halofit = 'mead2020')
S.load_power_spectra(z = zz, k = kk, power_spectra = pkz)
print("> Power spectra loaded")
#-----------------

#-----------------
# 4) Window functions
#-----------------
# The function 'load_window_function' computes the W(z) for all the bins that are required.
# It can take as argument a thing called 'galaxy_distributions'.
#
#   S.load_window_functions(galaxy_distributions = n_z)
#
# This is a nested list, in which every element has two entries:
#   - the first is a function whose first argument must be redshift
#   - the second is a dictionary of remaining arguments to pass to the above function.
# As an example, here we define 4 bin edges, which will give rise to 3 redshift bins.
# We use the Euclid distribution which scales as z^a e^(-(z/z0)^b).
# The function self.euclid_distribution is already defined in the angular_spectra class
# in the following way
# 
# def euclid_distribution(self, z, zmin, zmax, a = 2.0, b = 1.5, z_med = 0.9):
#     z_0 = z_med/sqrt(2.)
#     step = 1e-4
#     lower = 0.5*(1.+np.tanh((z-zmin)/step))				# Heaviside-like step function
#     upper = 0.5*(1.+np.tanh((zmax-z)/step))				# Heaviside-like step function
#     n = (z/z_0)**a*np.exp(-(z/z_0)**b)*lower*upper
#     return n
# 
# Also a function that contains photometric errors (see self.euclid_distribution_with_photo_error),
# a Gaussian (see self.gaussian_distribution) and a constant distributions (see self.constant) are defined
# in the class but one can create a custom distribution on his/her own!
# The only important thing is that the first argument MUST be redshift!
# 
# Alternatively, one can leave 'galaxy_distributions' undefined and call e.g.
#   z_w  = np.linspace(0., 6., 1001)        # Redshift range must be wider than z_limits!!
#   nz_w = [S.euclid_distribution(z = z_w, a = 2.0, b = 1.5, zmin = bin_edges[i], zmax = bin_edges[i+1], step = 1e-4) for i in range(nbins)]
#   S.load_window_functions(z = z_w, nz = nz_w)
# 
# where 'z' is 1-D array of redshifts and 'nz_w' is a 2-D array of shape (number of bins x len(z))

if number_of_bins == 3:
    bin_edges = [0.00, 0.72, 1.11, 5.00]	            # Bin edges
elif number_of_bins == 4:
    bin_edges = [0.00, 0.62, 0.90, 1.23, 5.00]	        # Bin edges
elif number_of_bins == 5:
    bin_edges = [0.00, 0.56, 0.79, 1.02, 1.32, 5.00]	# Bin edges
else:
    raise ValueError("Choose among 3,4 or 5 bins (or implement your own set of galaxy distributions).")
nbins     = len(bin_edges)-1			            # Number of bins
if photometric_errors:
    n_z = [[S.euclid_distribution_with_photo_error, {'zmin': bin_edges[i], 'zmax': bin_edges[i+1], 'a': 2.0, 'b': 1.5}] for i in range(nbins)]
else:
    n_z = [[S.euclid_distribution, {'a': 2.0, 'b': 1.5, 'zmin': bin_edges[i], 'zmax': bin_edges[i+1], 'step': 1e-4}] for i in range(nbins)]
print("> Galaxy distribution functions:")
for i in range(len(n_z)):
	print("    Bin %i: using function '%s' with parameters %s" %(i+1, n_z[i][0].__name__, n_z[i][1]))
S.load_window_functions(galaxy_distributions = n_z)
#z_w  = np.linspace(0., 6., 1001)        # Redshift range must be wider than z_limits!!
#nz_w = [S.euclid_distribution_with_photo_error(z = z_w, a = 2.0, b = 1.5, zmin = bin_edges[i], zmax = bin_edges[i+1]) for i in range(nbins)]
#S.load_window_functions(z = z_w, nz = nz_w)
print("> Window functions loaded")
#-----------------

#-----------------
# 5) Galaxy bias
#-----------------
# Only needed if galaxy clustering must be computed, it takes as argument a function 
# whose first 2 arguments MUST be the scale k [in Mpc/h] and the redshift z.
# Further keyword arguments can be added as **kwargs
# The function 'load_galaxy_bias' returns a 2D interpolator in k and z.
# For how the code is built, this function must be called after 'load_power_spectrum'
#-----------------
def _bias(k,z,k_damp = 0.5):
    return (1. + (k/10.)**2.)*np.exp(-k/k_damp)*np.sqrt(1.+z)
S.load_galaxy_bias(bias_function = _bias, k_damp = 4.)
#-----------------


#-----------------
# 6) Angular power spectrum
#-----------------
# Compute the shear spectra at the given multipoles, with the power spectra computed above.
# If one wants to use power spectra from simulation, he/she can skip the load_power_spectra part
# and set 'power_spectra' equal to a 2D interpolation taken from the simulations.
# If power_spectra == None, they will be computed on the fly with 'kwargs_power_spectra' to specify
# which code to use and whether to use linear or non-linear spectra.
# 'IA' can take values of 'LA' for 'linear alignment' or 'NLA' for nonlinear alignment, and
# 'kwargs_IA' are the parameters the function 'intrinsic_alignment_kernel' takes, depending
# on the model chosen.
# The output of the function is a dictionary with:
#   - 'GG': cosmological signal
#   - 'GI': cross term of cosmological signal and intrinsic alignment
#   - 'II': pure intrinsic alignment signal
#   - 'gG': galaxy-galaxy lensing angular power spectrum
#   - 'gg': angular galaxy clustering
# Each of these keys has a shape (n_bins, n_bins, len(multipoles))
# 
# N.B. Willingly, this function can also be called by skipping the 'load_power_spectrum' line
#      and by adding to it the dictionary 'kwargs_power_spectra' containing all the relevant things.
ll     = np.geomspace(2., 1e4, 51)
Cl     = S.compute_angular_power_spectra(l            = ll,
                                         do_WL        = True,
                                         do_IA        = True,
                                         do_GC        = True,
                                         IA_model     = 'NLA',
                                         kwargs_IA    = {'A_IA': -1.3, 'beta_IA': 0., 'eta_IA': 0.})
print("> Spectra loaded")
#-----------------


#-----------------
# 7) Plot
#-----------------
# Multiplication constant for plotting
c = ll*(ll+1.)/(2.*np.pi)
# Colors
colors = ['r', 'b','g','goldenrod','m', 'k', 'springgreen', 'darkorange', 'pink', 'darkcyan', 'salmon']
# Linewidth
LW = 3
# Plot shear spectra
hf, axarr = plt.subplots(nbins, nbins, sharex = True, sharey = True, figsize=(30,20))

for j in range(1, nbins):
	for i in range(j):
		axarr[i,j].axis('off')
	plt.setp([a.get_xticklabels() for a in axarr[i, :]], visible=False)
	plt.setp([a.get_yticklabels() for a in axarr[:, j]], visible=False)
	plt.subplots_adjust(wspace=0, hspace=0)

for i in range(nbins):
	for j in range(i, nbins):
		# Plotting Cls and systematics
		axarr[j,i].loglog(ll, c*Cl['gg'][i,j],         'blue'     , ls='-' , lw=LW, label='$C_\mathrm{\gamma\gamma}^{(ij)}(\ell)$')
		axarr[j,i].loglog(ll, np.abs(c*Cl['gI'][i,j]), 'magenta'  , ls='-' , lw=LW, label='$C_\mathrm{\gamma I}^{(ij)}(\ell)$')
		axarr[j,i].loglog(ll, c*Cl['II'][i,j],         'red'      , ls='-' , lw=LW, label='$C_\mathrm{II}^{(ij)}(\ell)$')
		axarr[j,i].loglog(ll, c*Cl['LL'][i,j],         'black'    , ls='-' , lw=LW, label='$C_\mathrm{LL}^{(ij)}(\ell)$')
		axarr[j,i].loglog(ll, c*Cl['GL'][i,j],         'green'    , ls='-' , lw=LW, label='$C_\mathrm{GL}^{(ij)}(\ell)$')
		axarr[j,i].loglog(ll, c*Cl['GL'][j,i],         'limegreen', ls='--', lw=LW, label='$C_\mathrm{GL}^{(ji)}(\ell)$')
		axarr[j,i].loglog(ll, c*Cl['GG'][i,j],         'goldenrod', ls='-' , lw=LW, label='$C_\mathrm{GG}^{(ij)}(\ell)$')
		# Coloured box
		if i != j: color = 'grey'
		else:      color = colors[i]
		axarr[j,i].text(0.1, 0.85, '$%i \\times %i$' %(i+1,j+1),
						transform=axarr[j,i].transAxes,
						style='italic',
						fontsize = 30,
						horizontalalignment='center',
						bbox={'facecolor': color, 'alpha':0.5, 'pad':10})
		axarr[j,i].set_xlim(ll.min(), ll.max())
		axarr[i,j].set_ylim(5e-9, 1e-1)
plt.legend(bbox_to_anchor=(0.9, nbins))

# Single label
hf.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("$\ell$")
plt.ylabel("$\ell(\ell+1) \ C_\ell \ / \ (2\pi)$", labelpad = 35)
plt.show()

# Plot galaxy distributions and window functions
hf, axarr = plt.subplots(2, 1, sharex=True, figsize=(30,20))
plt.subplots_adjust(hspace = 0.)
zz = np.linspace(0.1, 5., 1000)
for i in range(nbins):
	axarr[0].plot(zz, S.window_function   [i](zz)*1e5, colors[i],            lw = 3.0, label = 'Bin %i' %(i+1))
	axarr[1].plot(zz, S.window_function_IA[i](zz)*1e3, colors[i], ls = '--', lw = 3.0)
axarr[1].set_xlabel('$z$')
axarr[0].set_xlim(zz.min(), zz.max())
axarr[0].set_ylabel('$10^5 \\times W_\gamma     (z) \ [h/\mathrm{Mpc}]$')
axarr[1].set_ylabel('$10^3 \\times W_\mathrm{IA}(z) \ [h/\mathrm{Mpc}]$')
axarr[0].legend()
plt.show()
#-----------------
