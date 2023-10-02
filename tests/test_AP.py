import colibri.cosmology as cc
import numpy as np
import matplotlib.pyplot as plt
import colibri.useful_functions as UU

plt.rc('text',usetex=True)
plt.rc('font',size=14,family='serif')

# Redshift
z_eff = 1.0

# Scales and polar angles
ss_fid = np.linspace(10., 200.,51)
kk_fid = np.linspace(1e-3,0.5,51)
mu_fid = np.linspace(-1., 1., 20)

# Cartesian scales
spar_fid  = np.linspace(40.,200., 31)
sperp_fid = np.linspace(45.,180., 51)
kpar_fid  = np.linspace(1e-3,0.5, 31)
kperp_fid = np.linspace(1e-2,0.4, 51)


# Cosmologies
C_euclid = cc.cosmo()
C_planck = cc.cosmo(Omega_m=0.3089,h=0.6774)

# Polar transformation, Fourier space
k_prime,muk_prime = UU.AP_polar_coordinates_fourier_space(z_eff,kk_fid,mu_fid,
                                                          C_planck,C_euclid)

# Polar transformation, configuration space
s_prime,mus_prime = UU.AP_polar_coordinates_configuration_space(z_eff,ss_fid,mu_fid,
                                                                C_planck,C_euclid)
fig,ax=plt.subplots(2,2,figsize=(12,7))
L,B,R,T=0.13,0.13,0.95,0.90
plt.subplots_adjust(L,B,R,T,0.3,0.3)
ax[0,0].set_title('$(k,\mu)$')
ax[0,1].set_title('$(s,\mu)$')
for im in range(len(mu_fid)):
    ax[0,0].plot(kk_fid,k_prime[:,im]/kk_fid)
    ax[1,0].plot(mu_fid,muk_prime[im]/mu_fid)
ax[0,0].set_xlabel('$k_\mathrm{fid} \ [h \ \mathrm{Mpc}^{-1}]$')
ax[0,0].set_ylabel('$k/k_\mathrm{fid}$')
ax[1,0].set_xlabel('$\mu_\mathrm{fid}$')
ax[1,0].set_ylabel('$\mu/\mu_\mathrm{fid}$')
for im in range(len(mu_fid)):
    ax[0,1].plot(ss_fid,s_prime[:,im]/ss_fid)
    ax[1,1].plot(mu_fid,mus_prime[im]/mu_fid)
ax[0,1].set_xlabel('$s_\mathrm{fid} \ [h^{-1} \ \mathrm{Mpc}]$')
ax[0,1].set_ylabel('$s/s_\mathrm{fid}$')
ax[1,1].set_xlabel('$\mu_\mathrm{fid}$')
ax[1,1].set_ylabel('$\mu/\mu_\mathrm{fid}$')
plt.show()

# Cartesian transformation, Fourier space
kpar_prime, kperp_prime = UU.AP_cartesian_coordinates_fourier_space(z_eff,kpar_fid,kperp_fid,
                                                                    C_planck,C_euclid)
spar_prime,sperp_prime  = UU.AP_cartesian_coordinates_configuration_space(z_eff,spar_fid,sperp_fid,
                                                                          C_planck,C_euclid)

fig,ax=plt.subplots(1,2,figsize=(12,7))
ax[0].plot(kperp_fid,kperp_prime/kperp_fid,label='$k_\perp$')
ax[0].plot(kpar_fid ,kpar_prime /kpar_fid ,label='$k_\parallel$')
ax[1].plot(sperp_fid,sperp_prime/sperp_fid,label='$s_\perp$')
ax[1].plot(spar_fid ,spar_prime /spar_fid ,label='$s_\parallel$')
ax[0].set_xlabel('$k_\mathrm{X,fid} \ [h \ \mathrm{Mpc}^{-1}]$')
ax[0].set_ylabel('$k_\mathrm{X}/k_\mathrm{X,fid}$')
ax[1].set_xlabel('$s_\mathrm{X,fid} \ [h^{-1} \ \mathrm{Mpc}]$')
ax[1].set_ylabel('$s_\mathrm{X}/s_\mathrm{X,fid}$')
for a in ax: a.legend()
plt.show()
