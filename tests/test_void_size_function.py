import numpy as np
import colibri.cosmology as cc
import matplotlib.pyplot as plt
import scipy.interpolate as si

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size = 14)
plt.rcParams['lines.linewidth'] = 1.5

# This test file shows how to use the void size function routines in the 'cosmo' class.
# For more information on the quantities involved here, check out https://arxiv.org/abs/1206.3506

#==============
# Fixed parameters
#==============
zz      = [0.,1.,2.]                  # Redshifts
nz      = len(np.atleast_1d(zz))      # Number of redshifts
logM    = np.linspace(5., 17., 121)   # Masses
R_Eul   = np.linspace(1.,35.,69)      # Eulerian radii at which to compute voids
delta_v = -1.76                       # Linear extrapolation of underdensity for voids at "collapse"

#==============
# Colors and lines
#==============
ls     = ['-','--',':','-.']
colors = ['b','r','g','goldenrod','m','k']


#==============
# Define cosmological parameters
#==============
C = cc.cosmo()

#==============
# Load linear power spectra
#==============
k, pk = C.camb_Pk(z = zz)

#==============
# Mass variances multipoles, further smoothed for a Gaussian (useful for voids)
#==============
s0 = C.mass_variance_multipoles(logM=logM,k=k,pk=pk,j=0,smooth=True ,window='th',R_sm=5.5)
s1 = C.mass_variance_multipoles(logM=logM,k=k,pk=pk,j=1,smooth=True ,window='th',R_sm=5.5)
s2 = C.mass_variance_multipoles(logM=logM,k=k,pk=pk,j=2,smooth=True ,window='th',R_sm=5.5)

# Compute sigma8
s8 = C.compute_sigma_8(k=k,pk=pk)[0]
M8 = C.mass_in_radius(8.)

#==============
# Useful quantities
#==============
# Transform Eulerian to Lagrangian coordinates
R_Lag     = np.outer(C.lagrange_to_euler(z = zz),R_Eul)
# gamma_p and R* parameters
gamma_p   = s1/np.sqrt(s0*s2)
R_star    = np.sqrt(3.*s1/s2)
# Peak (i.e. Trough) height in voids
dv        = np.abs(delta_v)
nu        = dv/s0**.5
# Excursion Set Troughs functions
G1        = np.array([C.G_n_BBKS(1, gamma_p[iz], nu[iz]) for iz in range(nz)])
# Press-Schechter formalism for voids
R_of_M    = C.radius_of_mass(10.**logM) # Radii corresponding to masses
f_nu      = C.volume_of_radius(R_of_M, 'th')/(2.*np.pi*R_star**2.)**(3./2.)*C.PressSchechter_mass_function(s0**.5, delta_th = dv)/(2.*nu)*G1/(gamma_p*nu)
# High-mass approximation for Press-Schechter formalism for voids
f_high_nu = np.exp(-nu**2./2.)/np.sqrt(2.*np.pi)*C.volume_of_radius(R_of_M, 'th')/(2.*np.pi*R_star**2.)**1.5*(nu**3.-3*nu)*gamma_p**3.
# Mass at which peak height = 1
M_of_nu1 = 10.**np.array([si.interp1d(nu[iz], logM, 'cubic')(1.) for iz in range(nz)])



#==============
# Void size function
#==============
# a,p are Sheth-Tormen parameters, delta_v is the linear underdensity for "collapse" of voids
RL,VSF = C.void_size_function(R=R_Eul,z=zz,k=k,pk=pk,delta_v=-1.76,a=1.,p=0.)

#==============
# Plot
#==============
plt.figure(figsize = (20,20))
plt.subplots_adjust(hspace = 0.3, wspace = 0.2, right = 0.95, bottom = 0.09, top = 0.95)

ax1 = plt.subplot2grid((3,2), (0,0), colspan = 1, rowspan = 1)
ax2 = plt.subplot2grid((3,2), (0,1), colspan = 1, rowspan = 1)
ax3 = plt.subplot2grid((3,2), (1,0), colspan = 1, rowspan = 1)
ax4 = plt.subplot2grid((3,2), (1,1), colspan = 1, rowspan = 1)
ax5 = plt.subplot2grid((3,2), (2,0), colspan = 1, rowspan = 1)
ax6 = plt.subplot2grid((3,2), (2,1), colspan = 1, rowspan = 1)

# Mass variances and multipoles
ax1.loglog(10.**logM, s0[0], 'b'        ,ls=ls[0],label = '$s_0$')
ax1.loglog(10.**logM, s1[0], 'g'        ,ls=ls[0],label = '$s_1$')
ax1.loglog(10.**logM, s2[0], 'goldenrod',ls=ls[0],label = '$s_2$')
ax1.loglog(M8, s8**2., 'rP', label = '$\\sigma_8^2$', ms = 10)
ax1.text(0.85, 0.85, '$z=%i$'%zz[0],
         transform=ax1.transAxes,
         style='italic',
         fontsize = 15,
         horizontalalignment='center',
         bbox={'facecolor': 'white', 'alpha':0.5, 'pad':0.5, 'boxstyle':'round'})
ax1.set_xlabel('$M \ [M_\odot/h]$')
ax1.set_xlim(3e9,3e16)
ax1.set_ylim(1e-8,1e5)
ax1.legend(fontsize = 12,ncol=2,loc='lower left')

# Useful quantities for voids
ax2.loglog(10.**logM, nu[0],           'b'        ,ls=ls[0],label = '$\\nu$')
ax2.loglog(10.**logM, R_star[0],       'r'        ,ls=ls[0],label = '$R_* \ [\mathrm{Mpc}/h]$')
ax2.loglog(10.**logM, gamma_p[0],      'g'        ,ls=ls[0],label = '$\gamma_p$')
ax2.loglog(10.**logM, gamma_p[0]*nu[0],'goldenrod',ls=ls[0],label = '$\gamma_p\\nu$')
ax2.text(0.85, 0.85, '$z=%i$'%zz[0],
         transform=ax2.transAxes,
         style='italic',
         fontsize = 15,
         horizontalalignment='center',
         bbox={'facecolor': 'white', 'alpha':0.5, 'pad':0.5, 'boxstyle':'round'})
ax2.set_xlabel('$M \ [M_\odot/h]$')
ax2.set_xlim(3e9,3e16)
ax2.set_ylim(1e-1,1e2)
ax2.legend(fontsize = 12)

# Lagrangian radius
for iz in range(nz):
    ax3.plot(R_Eul, R_Lag[iz],c=colors[iz],label='$z=%i$'%(zz[iz]))
ax3.set_xlabel('$R_\mathrm{Eul} \ [\mathrm{Mpc}/h]$')
ax3.set_ylabel('$R_\mathrm{Lag} \ [\mathrm{Mpc}/h]$')
ax3.set_xlim(R_Eul.min(), R_Eul.max())
ax3.legend()

# BBKS function G1
for iz in range(nz):
    ax4.loglog(10.**logM, G1[iz],c=colors[iz])
    ax4.loglog(10.**logM, (gamma_p*nu*((nu**3.-3*nu)*gamma_p**3.))[iz],c=colors[iz],ls='--')
    ax4.axvline(M_of_nu1[iz], c=colors[iz],ls=':',lw=1.)
ax4.loglog(np.nan,np.nan, 'k:', label = 'approximation for $\\nu \gg 1$')
ax4.axvline(np.nan, c='k',ls=':',lw=0.5,label='$M(\\nu=1)$')
ax4.set_xlabel('$M \ [M_\odot/h]$')
ax4.set_ylabel('$G_1(\gamma_p, \gamma_p\\nu)$')
ax4.set_xlim(3e9,3e16)
ax4.set_ylim(1e-1,1e5)
ax4.legend()


# Excursion set of Troughs
for iz in range(nz):
    ax5.loglog(10.**logM, nu[iz]*f_nu[iz],     c=colors[iz])
    ax5.loglog(10.**logM, nu[iz]*f_high_nu[iz],c=colors[iz],ls=':')
    ax5.axvline(M_of_nu1[iz], c='k',ls=':',lw=1.)
ax5.loglog(np.nan,np.nan, 'k:', label = 'approximation for $\\nu \gg 1$')
ax5.axvline(np.nan, c='k',ls=':',lw=0.5,label='$M(\\nu=1)$')
ax5.set_xlabel('$M \ [M_\odot/h]$')
ax5.set_ylabel('$\\nu f(\\nu)$')
ax5.set_xlim(3e9,3e16)
ax5.set_ylim(1e-4,1e1)
ax5.legend()

# Void size function
for iz in range(nz):
    ax6.loglog(R_Lag[iz], VSF[iz], 'k',c=colors[iz])
ax6.set_xticks([5,10,20,50])
ax6.set_xlabel('$R_\mathrm{Lag} \ [\\mathrm{Mpc}/h]$')
ax6.set_ylabel('$\\frac{dn}{dR} \ [(h/\\mathrm{Mpc})^4]$')
ax6.set_xlim(5,40)
ax6.set_ylim(1e-11,1e-3)


plt.show()


