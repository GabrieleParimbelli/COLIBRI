import colibri.cosmology as cc
import numpy as np
import matplotlib.pyplot as plt
import colibri.nonlinear as NL

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size = 25)
colors = ['b','r','g','m','gray','c']

#=================
# This code computes the non-linear matter power spectrum with different methods
# 1) the Halofit operator defined in CAMB
# 2) the Halofit operator defined in the `HMcode2016' class in the `nonlinear.py' module
# 3) the Halofit operator defined in the `nonlinear_pk' class in the `nonlinear.py' module
#=================

# Cosmology, redshifts and scales
C = cc.cosmo(Omega_m = 0.3089, M_nu = 0.0, Omega_b = 0.0486, As = 2.14e-9, ns = 0.9667, h = 0.6774)

# Arguments for the code to compute power spectrum
code_arguments = {}

# Redshifts
zz = np.linspace(0., 5., 6)

# Scales
kk = np.logspace(-4., 2., 101)

#=================
# 1) Compute the non-linear power spectrum with CAMB (use Mead halofit)
#=================
k_camb, pk_camb = C.camb_Pk(k = kk, z = zz, nonlinear = True, halofit = 'mead')
print(">> Non-linear power spectrum computed with CAMB halofit")

#=================
# 2) Use the `HMcode2016' class, which takes as arguments
#    - an array of scales
#    - an array of redshifts
#    - a 2D array of linear power spectra (the shape must coincide with len(z) x len(k) )
#    - the field upon which the non-linear spectra must be computed ('cb' for CDM + baryons, 'tot' for total matter')
#    - a boolean value to smooth or not the BAO feature in non-linear power spectrum
#    - a cosmology instance
#=================

# Compute at first the linear power spectrum (in LCDM 'cb' and 'tot' is the same)
k_l, pk_l    = C.camb_Pk(z = zz, k = kk, var_1 = 'cb', var_2 = 'cb')
# Compute non-linear power spectrum
do_nonlinear = NL.HMcode2016(z = zz, k = k_l, pk = pk_l, field = 'cb', BAO_smearing = False, cosmology = C)
#do_nonlinear = NL.HMcode2020(z = zz, k = k_l, pk = pk_l, field = 'cb', BAO_smearing = False, cosmology = C)
# Take spectra
pk_hf        = do_nonlinear.pk_nl
print(">> Non-linear power spectrum computed with 'HMcode2016' module")

#=================
# 3) Use the `nonlinear_pk' module, which inherits the class `cosmo'
#=================

# `nonlinear_pk' instance
HF = NL.nonlinear_pk(k = kk, z = zz, code = 'camb', BAO_smearing = False, kwargs_code = code_arguments, cosmology = C)

# Take all possible quantities
k          = HF.k          # Scales
pk_nl      = HF.pk_nl      # Non-linear power spectrum
pk_l       = HF.pk_l       # Linear power spectrum
pk_nl_cbcb = HF.pk_nl_cbcb # cdm+b power spectrum
pk_cbcb    = HF.pk_cbcb    # cdm+b power spectrum
pk_cbnu    = HF.pk_cbnu    # cross power spectrum
pk_nunu    = HF.pk_nunu    # neutrino power spectrum
pk_nw      = HF.pk_nw      # no wiggles power spectrum
pk_dw      = HF.pk_dw      # de-wiggled power spectrum
print(">> Non-linear power spectrum computed with 'nonlinear_pk' module")

#=================
# Subplot & plot
#=================
plt.figure(figsize=(25,25))
ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
ax2 = plt.subplot2grid((4,4), (3,0), colspan=4)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.subplots_adjust(wspace=0, hspace=0)

for i in range(len(zz)):

	# Plot spectra
	ax1.loglog(k_camb, pk_camb[i], colors[i], ls = '-',  lw = 2.0, label = '$z=%.1f$' %zz[i]) 	# Plot CAMB halofit
	ax1.loglog(k,      pk_hf[i],   colors[i], ls = '--', lw = 2.0)								# Plot halofit module
	ax1.loglog(k,      pk_nl[i],   colors[i], marker = 'o', ls = '', ms = 5)					# Plot nonlinear module

	# Plot ratios
	ax2.semilogx(k_camb, (pk_nl[i]/pk_camb[i]-1.)*100., colors[i], ls = '--',    lw = 2.0)
	ax2.semilogx(k_camb, (pk_hf[i]/pk_camb[i]-1.)*100., colors[i], marker = 'o', ls = '', lw = 2.0, ms = 5)


ax1.loglog(0,0,'k', ls = '-', lw = 2.0, label = "CAMB halofit")
ax1.loglog(0,0,'k', ls = '--', lw = 2.0, label = "`halofit\_operator' module")
ax1.loglog(0,0,'k', marker = 'o', ls = '', ms = 5, label = "`nonlinear' module")
ax1.set_ylabel('$P(k) \ [(\mathrm{Mpc}/h)^3]$')
ax1.set_xlim(kk.min(), kk.max())
ax1.set_ylim(5e-2, 1e5)
ax1.grid(True)

ax2.fill_between(k_camb, -1., 1., color = 'k', alpha = 0.2)
ax2.set_xlabel('$k$ $[h/\mathrm{Mpc}]$')
ax2.set_ylabel(r'$\left(\frac{P(k)}{P_\mathrm{CAMB}(k)}-1\right)\times 100 \ [\%]$', fontsize = 20)
ax2.set_xlim(kk.min(), kk.max())
ax2.set_ylim(-3., 3.)
ax2.grid(True)

ax1.legend(loc='lower left', ncol = 3, fontsize = 20)
plt.show()

