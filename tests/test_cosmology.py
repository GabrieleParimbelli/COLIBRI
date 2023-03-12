import colibri.cosmology as cc
import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size = 20)

#========================
# Test of 'cosmology'
#========================

# Define cosmology instance
# We report as example all cosmological parameters
# but the syntax cc.cosmo() is sufficient to have all
# parameters set to default value.
C = cc.cosmo(Omega_m = 0.32,        # total matter (CDM + baryons + neutrinos) density parameter today
             Omega_b = 0.05,        # baryon density parameter today
             Omega_K = 0.,          # Curvature density parameter (Omega_lambda will be set to sum to 1)
             ns      = 0.96,        # Scalar spectral index of primordial perturbation
             As      = 2.12605e-9,  # Scalar amplitude of primordial perturbation
             sigma_8 = None,        # Power spectrum normalization (set to None as As is defined)
             h       = 0.67,        # Hubble parameter in units of 100 km/s/Mpc
             w0      = -1.,         # Dark energy parameter of state today
             wa      = 0.,          # Evolution of dark energy parameter of state
             tau     = 0.06,        # Optical depth to reionization
             T_cmb   = 2.7255,      # CMB temperature today (fixes Omega_gamma)
             M_nu    = [0.05, 0.01],# Neutrino masses in eV: in this case we have 2 massive neutrinos
             N_nu    = 3,           # Total number of neutrino species
             N_eff   = 3.046)       # Effective number of neutrinos


print("Omega matter:         %.4f" %(C.Omega_m))
print("Omega CDM:            %.4f" %(C.Omega_cdm))
print("Omega baryons:        %.4f" %(C.Omega_b))
print("Omega curvature:      %.4f" %(C.Omega_K))
print("Omega Lambda:         %.3e" %(C.Omega_lambda))
print("Omega photons:        %.3e" %(C.Omega_gamma))
print("Neutrino masses (eV):", C.M_nu)
print("Omega neutrino:      ", C.Omega_nu)
print("Total omega neutrino: %.3e" %(np.sum(C.Omega_nu)))
print("Massive neutrinos:    %i" %(C.massive_nu))
print("Massless neutrinos:   %.3f" %(C.massless_nu))
print("Total neutrinos:      %i" %(C.N_nu))
print("Effective neutrinos:  %.3f" %(C.N_eff))
print("Primordial amplitude: %.3e" %(C.As))
print("Spectral index:       %.4f" %(C.ns))


# Scale factors and redshifts
aa = np.logspace(-7., 1., 101)    # Scale factor
zz = C.redshift(aa)               # Corresponding redshifts

# Omegas as functions of z
onz0 = C.Omega_nu_z(zz)
ocz0 = C.Omega_cdm_z(zz)
obz0 = C.Omega_b_z(zz)
olz0 = C.Omega_lambda_z(zz)
ogz0 = C.Omega_gamma_z(zz)
okz0 = C.Omega_K_z(zz)
otz0 = np.sum(onz0, axis=0)+olz0+ocz0+obz0+ogz0+okz0

plt.figure(figsize=(12,8))
plt.subplot(211)
L,B,R,T = 0.1, 0.12, 0.95, 0.95
plt.subplots_adjust(L,B,R,T,0.3,0.3)
LS = ['-','--',':']
for i in range(len(np.atleast_1d(onz0))):
    plt.semilogx(aa, onz0[i],'m', ls = LS[i], lw = 2.0, label ='$\\nu_%i$' %(i+1))
plt.semilogx(aa, olz0,    'k',   lw = 2.0, label ='$\Lambda$')
plt.semilogx(aa, ogz0,    'g',   lw = 2.0, label = '$\\gamma$')
plt.semilogx(aa, ocz0,    'b',   lw = 2.0, label = 'cdm')
plt.semilogx(aa, obz0,    'r',   lw = 2.0, label = 'b')
plt.semilogx(aa, otz0,    'k:',  lw = 4.0, label = 'total')
plt.axvline(1., c = 'k', ls = '-.', lw = 4.0, label = 'today')
plt.yscale('log')
plt.xlabel('$a$')
plt.ylabel('$\Omega_i(a)$')
plt.xlim(aa.min(), aa.max())
plt.ylim(1e-6, 5.)
plt.legend(loc = 'lower left', ncol = 3, fontsize = 16)

# Distances and Hubble parameter as function of redshift
# massive_nu_approx = True is a flag that approximate neutrinos as matter
# (it is faster, but less accurate; anyway the error is much smaller than 0.1% at z < 10.
zzz   = np.linspace(0., 10., 101)
d_com = C.comoving_distance(zzz, massive_nu_approx = True)
d_ang = C.angular_diameter_distance(zzz, massive_nu_approx = True)
d_lum = C.luminosity_distance(zzz, massive_nu_approx = True)
d_vol = C.isotropic_volume_distance(zzz, massive_nu_approx = True)
H_z   = C.H(zzz)

plt.subplot(212) 
plt.semilogy(zzz, d_com, 'k', lw = 2.0, label = '$\chi(z) \ [\mathrm{Mpc}/h]$')
plt.semilogy(zzz, d_ang, 'g', lw = 2.0, label = '$D_A(z) \ [\mathrm{Mpc}/h]$')
plt.semilogy(zzz, d_lum, 'b', lw = 2.0, label = '$D_L(z) \ [\mathrm{Mpc}/h]$')
plt.semilogy(zzz, d_vol, 'm', lw = 2.0, label = '$D_V(z) \ [\mathrm{Mpc}/h]$')
plt.semilogy(zzz, H_z,   'r', lw = 2.0, label = '$H(z) \ [\mathrm{km/s/Mpc}]$')
plt.yscale('log')
plt.xlabel('$z$')
plt.xlim(zzz.min(), zzz.max())
plt.legend(loc = 'lower right', ncol = 3, fontsize = 16)

# Show!
plt.show()


