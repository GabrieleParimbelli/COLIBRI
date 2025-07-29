import colibri.cosmology as cc
import colibri.limber as LL
import numpy as np
import matplotlib.pyplot as plt
import colibri.constants as const

plt.rc('text',usetex=True)
plt.rc('font',size=14,family='serif')

# Colors
colors = ['r', 'b','g','goldenrod','m', 'k', 'springgreen', 'darkorange', 'pink', 'darkcyan', 'salmon']
LW = 3

# This scripts quantifies the impact of a scale-dependent intrinsic alignment (IA) effect which is not taken
# into account by default in the 'limber' class.
# The scale dependence enters in particular in the growth factor at denominator of the equations of IA.
# Here we show the case of neutrinos, where the effect should be of order % or smaller for reasonable values
# of the neutrino mass.
# Select neutrino mass here
M_nu = 0.06

#-----------------
# Define a cosmology instance (with default values)
C = cc.cosmo(M_nu=M_nu)
S = LL.limber(cosmology = C, z_limits = [0.01, 5.])
print("> Limber instance loaded")
#-----------------

#-----------------
# Load power spectra
kk = np.geomspace(1e-4, 1e2, 301)
zz = np.linspace(0., 5., 51)
_, pkz = C.camb_Pk(z = zz, k = kk, nonlinear = True, halofit = 'mead2020')
S.load_power_spectra(z = zz, k = kk, power_spectra = pkz)
print("> Power spectra loaded")
#-----------------

#-----------------
# Bins and galaxy distributions
bin_edges = [0.01,0.42,0.56,0.68,0.79,0.90,1.02,1.15,1.32,1.57,5.00]
nbins     = len(bin_edges)-1
z_gal     = np.linspace(S.z_min, S.z_max, 201)
nz_gal    = [S.euclid_distribution_with_photo_error(z=z_gal,zmin=bin_edges[i],zmax=bin_edges[i+1]) for i in range(nbins)]
#-----------------

#-----------------
# Load IA window function
a_ia    = 1.72
eta_ia  = -0.41
beta_ia = 2.17
lum_ia  = lambda z: (1+z)**-0.5
S.load_IA_window_functions(z       = z_gal,
                           nz      = nz_gal,
                           A_IA    = a_ia,
                           eta_IA  = eta_ia,
                           beta_IA = beta_ia,
                           lum_IA  = lum_ia,
                           name    = 'IA')
# Load cosmic shear
S.load_shear_window_functions(z       = z_gal,
                              nz      = nz_gal,
                              name    = 'shear')
print("> Window functions loaded")
#-----------------

#-----------------
# Angular spectra or correlation functions
ll    = np.geomspace(2., 5e3, 51)
Cl    = S.limber_angular_power_spectra(l = ll, windows = None)
Cl_ss = Cl['shear-shear']
Cl_sI = Cl['shear-IA']+Cl['IA-shear']
Cl_II = Cl['IA-IA']
Cl_LL = Cl_ss+Cl_sI+Cl_II
print("> Spectra loaded")
#-----------------

#-----------------
# Alternative IA window function
Hz_over_c   = C.H_massive(z_gal)/C.h/const.c
nz_gal_norm = nz_gal/np.expand_dims(np.trapz(nz_gal,x=z_gal,axis=1),1)
S.load_custom_window_functions(z=z_gal,window=nz_gal_norm*Hz_over_c,name='alternative_IA')
# Alternative IA-IA power spectrum
c1_ia    = 0.0134
front    = -a_ia*c1_ia*S.cosmology.Omega_m
growth   = S.cosmology.growth_factor_CDM_baryons_neutrinos(k=kk,z=zz)
pkz_iaia = (front/growth*np.expand_dims((1.+zz)**eta_ia*lum_ia(zz)**beta_ia,1))**2.*pkz
S.load_power_spectra(z = zz, k = kk, power_spectra = pkz_iaia)
Cl_alt    = S.limber_angular_power_spectra(l = ll, windows = None)
Cl_II_alt = Cl_alt['alternative_IA-alternative_IA']
print("> Alternative IA-IA spectra loaded")
# Alternative shear-IA power spectrum
pkz_sia  =  front/growth*np.expand_dims((1.+zz)**eta_ia*lum_ia(zz)**beta_ia,1)     *pkz
S.load_power_spectra(z = zz, k = kk, power_spectra = pkz_sia)
Cl_alt    = S.limber_angular_power_spectra(l = ll, windows = None)
Cl_sI_alt = Cl_alt['shear-alternative_IA']+Cl_alt['alternative_IA-shear']
Cl_LL_alt = Cl_ss+Cl_sI_alt+Cl_II_alt
print("> Alternative shear-IA spectra loaded")
#-----------------


#-----------------
# Plot total
hf, axarr = plt.subplots(nbins, nbins, sharex = True, sharey = True, figsize=(12,8))
L,R,T,B=0.1, 0.95, 0.95, 0.1
plt.subplots_adjust(left=L,right=R,top=T,bottom=B)
# Triangle plot
for j in range(1, nbins):
    for i in range(j):
        axarr[i,j].axis('off')
    plt.setp([a.get_xticklabels() for a in axarr[i, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, j]], visible=False)
    plt.subplots_adjust(wspace=0, hspace=0)

for i in range(nbins):
        for j in range(i, nbins):

            # Plotting Cls
            axarr[j,i].semilogx(ll,(Cl_II_alt[i,j]/Cl_II[i,j]-1)*100,'r',ls=':' ,lw=2,label='IA only')
            axarr[j,i].semilogx(ll,(Cl_LL_alt[i,j]/Cl_LL[i,j]-1)*100,'b',ls='--',lw=2,label='full lensing')
            # Coloured box
            if i != j: color = 'grey'
            else:      color = colors[i]
            axarr[j,i].text(0.25, 0.25, '$%i \\times %i$' %(i+1,j+1),
                                transform=axarr[j,i].transAxes,
                                style='italic',
                                fontsize = 8,
                                horizontalalignment='center',
                                bbox={'facecolor': color, 'alpha':0.5, 'pad':2})
            axarr[j,i].set_xlim(ll.min(), ll.max())
            #axarr[j,i].set_ylim(-7.5, 4.5)
            #axarr[j,i].set_yticks([-6.,-3,0,3])
            axarr[j,i].fill_between(ll,-1,1,color='k',alpha=0.1)
            axarr[j,i].fill_between(ll,-2,2,color='k',alpha=0.1)
plt.legend(bbox_to_anchor=(0.93, 0.98), fontsize = 12, bbox_transform=hf.transFigure)
plt.text((L+R)*0.5, B*0.4, "$\ell$", ha='center', transform=hf.transFigure)
plt.text(L*0.4,(T+B)*0.5, "diff. scale-independent vs scale-dependent IA (\%)", ha='center', va = 'center', rotation = 90, transform=hf.transFigure)
plt.show()
#-----------------
