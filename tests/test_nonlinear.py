import colibri.cosmology as cc
import numpy as np
import matplotlib.pyplot as plt
import colibri.nonlinear as NL

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size = 25)
colors = ['b','r','g','m','gray','c']

#########################
# Choose among
# 'mead'        (not good for neutrinos)
# 'mead2020'    (good for neutrinos)
# 'takahashi'   (not good for neutrinos)
# 'bird'        (good for neutrinos)
set_halofit = 'mead2020'
#########################

#=================
# This code computes the non-linear matter power spectrum with different methods
# 1) the Halofit operator defined in CAMB
# 2) the Halofit operator defined in the `nonlinear.py` module
#=================

# Cosmology, redshifts and scales
C  = cc.cosmo(Omega_m = 0.3089, Omega_b = 0.0486, As = 2.14e-9, ns = 0.9667, h = 0.6774, M_nu = 0.15)
zz = np.linspace(0., 5., 6)
kk = np.logspace(-4., 2., 201)

#=================
# 1) Compute the non-linear power spectrum with CAMB
#=================
k_camb, pk_nl_camb = C.camb_Pk(k = kk, z = zz, nonlinear = True, halofit = set_halofit)
print(">> Non-linear power spectrum computed with CAMB halofit '%s'" %set_halofit)

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
k_l, pk_l = C.camb_XPk(z = zz, k = kk, var_1 =  ['cb','tot'], var_2 = ['cb','tot'])
# Compute non-linear power spectrum
if set_halofit == 'mead':
    set_class = 'HMcode2016'
    pk_mm = pk_l['tot-tot']
    do_nonlinear = NL.HMcode2016(z            = zz,
                                 k            = k_l,
                                 pk           = pk_mm,
                                 field        = 'cb',
                                 BAO_smearing = False,
                                 cosmology    = C)
elif set_halofit == 'mead2020':
    set_class = 'HMcode2020'
    pk_cc = pk_l['cb-cb']
    pk_mm = pk_l['tot-tot']
    do_nonlinear = NL.HMcode2020(z         = zz,
                                 k         = k_l,
                                 pk_cc     = pk_cc,
                                 pk_mm     = pk_mm,
                                 cosmology = C)
elif set_halofit == 'takahashi':
    set_class = 'Takahashi'
    pk_mm = pk_l['tot-tot']
    do_nonlinear = NL.Takahashi (z            = zz,
                                 k            = k_l,
                                 pk           = pk_mm,
                                 cosmology    = C)
elif set_halofit == 'bird':
    set_class = 'TakaBird'
    pk_mm = pk_l['tot-tot']
    do_nonlinear = NL.TakaBird  (z = zz,
                                 k = k_l,
                                 pk = pk_mm,
                                 cosmology = C)
else:
    raise ValueError("Non-linear method not recognized")
# Retrieve the non-linear power spectrum
pk_nl_colibri = do_nonlinear.pk_nl
print(">> Non-linear power spectrum computed with '%s' class in 'nonlinear' module" %(set_class))

# 2bis) One can also use the `nonlinear_pk' module
#HF = NL.nonlinear_pk(k = kk, z = zz, code = 'camb', BAO_smearing = False, kwargs_code = {}, cosmology = C)
#pk_nl_full = HF.pk_nl
#print(">> Non-linear power spectrum computed with 'nonlinear_pk' module")


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
	ax1.loglog(kk,pk_nl_camb[i]   ,colors[i],ls='-', lw=2.0,label='$z=%.1f$'%zz[i]) # Plot CAMB halofit
	ax1.loglog(kk,pk_nl_colibri[i],colors[i],ls='' , marker='o',ms=3)               # Plot nonlinear module

	# Plot ratios
	ax2.semilogx(kk,(pk_nl_colibri[i]/pk_nl_camb[i]-1.)*100.,colors[i],ls='-',lw=2.0)


ax1.loglog(0,0,'k', ls = '', marker='o',lw = 2.0, label = "CAMB halofit %s" %set_halofit)
ax1.loglog(0,0,'k', ls = '-', lw = 2.0, label = "`nonlinear' module %s" %set_class)
ax1.set_ylabel('$P(k) \ [(\mathrm{Mpc}/h)^3]$')
ax1.set_xlim(kk.min(), 10.)
ax1.set_ylim(5e-2, 1e5)
ax1.grid(True)

ax2.fill_between(k_camb, -.5, .5, color = 'k', alpha = 0.1)
ax2.fill_between(k_camb, -1., 1., color = 'k', alpha = 0.05)
ax2.set_xlabel('$k$ $[h/\mathrm{Mpc}]$')
ax2.set_ylabel(r'$\left(\frac{P(k)}{P_\mathrm{CAMB}(k)}-1\right)\times 100 \ [\%]$', fontsize = 20)
ax2.set_xlim(kk.min(), 10.)
ax2.set_ylim(-2.25, 2.25)
ax2.grid(True)

ax1.legend(loc='lower left', ncol = 3, fontsize = 20)
plt.show()

