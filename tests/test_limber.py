import numpy as np
import colibri.limber as LL
import colibri.cosmology as cc
import matplotlib.pyplot as plt

plt.rc('text',usetex=True)
plt.rc('font',size=20,family='serif')

# Colors
colors = ['r', 'b','g','goldenrod','m', 'k','springgreen','darkorange','pink','darkcyan','salmon',
          'c','y','grey']
# Linewidth
LW = 3

#########################
# Test of limber spectra class
#########################
nbins         = 3      # Number of bins to use 2->10
fourier       = True   # Compute power spectra (True) or correlation functions (False)

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
S = LL.limber(cosmology = C, z_limits = [0.01, 5.])
print("> Limber instance loaded")
#-----------------

#-----------------
# 3) Load power spectra
#-----------------
# The routine 'load_power_spectra' interpolates the power spectra at the scales and redshifts asked.
# It takes as inputs scales, redshifts and a table of power spectra. The shape of the latter must be 
# (number of scales, number of redshifts)
kk = np.geomspace(1e-4, 1e2, 301)
zz = np.linspace(0., 5., 51)
_, pkz = C.camb_Pk(z = zz, k = kk, nonlinear = True, halofit = 'mead2020')
S.load_power_spectra(z = zz, k = kk, power_spectra = pkz)
print("> Power spectra loaded")
#-----------------

#-----------------
# 4) Bins
#-----------------
# Select number of redshift bins
# In this case we chose to assume that each redshift bin has the same number of galaxies
# (according to the galaxy distribution we want to use)
if   nbins == 2 : bin_edges = [0.01,0.90,5.00]
elif nbins == 3 : bin_edges = [0.01,0.71,1.11,5.00]
elif nbins == 4 : bin_edges = [0.01,0.62,0.90,1.23,5.00]
elif nbins == 5 : bin_edges = [0.01,0.56,0.79,1.02,1.32,5.00]
elif nbins == 6 : bin_edges = [0.01,0.52,0.71,0.90,1.11,1.39,5.00]
elif nbins == 7 : bin_edges = [0.01,0.48,0.66,0.82,0.98,1.17,1.45,5.00]
elif nbins == 8 : bin_edges = [0.01,0.48,0.62,0.76,0.90,1.05,1.23,1.50,5.00]
elif nbins == 9 : bin_edges = [0.01,0.44,0.59,0.71,0.84,0.96,1.11,1.28,1.54,5.00]
elif nbins == 10: bin_edges = [0.01,0.42,0.56,0.68,0.79,0.90,1.02,1.15,1.32,1.57,5.00]
#elif nbins == 13: bin_edges = [0.01,0.15,0.31,0.46,0.62,0.77,0.92,1.08,1.23,1.38,1.54,1.69,1.85,2.00]
else: raise ValueError("Choose among 2->10 bins (or implement your own set of galaxy distributions).")

# The lines below find the bin edges with another method
# (they should return the same result as above)
"""
import scipy.integrate as sint
import scipy.optimize as so

def integral(a,b):
    denominator = sint.quad(lambda z: S.euclid_distribution_with_photo_error(z=z,zmin=0,zmax=np.inf),0.,5.)[0]
    numerator   = sint.quad(lambda z: S.euclid_distribution_with_photo_error(z=z,zmin=0,zmax=np.inf),a,b)[0]
    return numerator/denominator
bin_edges    = np.zeros(nbins+1)
bin_edges[0]  = 0.01
bin_edges[-1] = 5.00
for i in range(nbins-1):
    bin_edges[i+1] = so.root(lambda x: integral(bin_edges[i], x)-1./nbins, bin_edges[i])['x']
"""

#-----------------
# 5) Galaxy distributions (can be different for different observables!)
#-----------------
# Compute galaxy distribution in each redshift bin (they can be different for different probes!)
# 'z_gal' is an array of redshift (sample it with dz<0.0625, otherwise you get an error)
# 'nz_gal' is a 2-D array of shape (number of bins, number of redshifts)
z_gal     = np.linspace(S.z_min, S.z_max, 201)
nz_gal    = [S.euclid_distribution_with_photo_error(z=z_gal,zmin=bin_edges[i],zmax=bin_edges[i+1]) for i in range(nbins)]

#-----------------
# 6) Load window functions
#-----------------
# Compute the window functions for the Limber power spectra
# Cosmic shear
S.load_shear_window_functions  (z       = z_gal,
                                nz      = nz_gal,
                                name    = 'shear')
# Intrinsic alignment alone
S.load_IA_window_functions     (z       = z_gal,
                                nz      = nz_gal,
                                A_IA    = 1.72,
                                eta_IA  = -0.41,
                                beta_IA = 2.17,
                                lum_IA  = lambda z: (1+z)**-0.5,
                                name    = 'IA')
# Lensing (shear + intrinsic alignment)
# (Notice that the sum of the previous two should give the same result of the following,
# so the three of them are all computed here for didactic purposes.)
S.load_lensing_window_functions(z       = z_gal,
                                nz      = nz_gal,
                                A_IA    = 1.72,
                                eta_IA  = -0.41,
                                beta_IA = 2.17,
                                lum_IA  = lambda z: (1+z)**-0.5,
                                name    = 'lensing')
# Galaxy clustering
z_mean = (np.array(bin_edges[:-1])+np.array(bin_edges[1:]))*0.5
bias   = (1.+z_mean)**0.5
S.load_galaxy_clustering_window_functions(z = z_gal, nz = nz_gal, bias = bias, name = 'galaxy')

# Other window functions are implemented and custom window functions can also be used!
# e.g. the HI brightness temperature, the CMB lensing and the galaxy number counts
#S.load_HI_window_functions         (z=z_gal,nz=nz_gal,bias=1,Omega_HI=0.000625,name='HI')
#S.load_CMB_lensing_window_functions(z=z_gal,nz=nz_gal,z_LSS=1089,name='CMB')
#S.load_custom_window_functions     (z=z_gal,window=nz_gal,name='counts')
print("> Window functions loaded")
#-----------------

#-----------------
# 7) Angular spectra or correlation functions
#-----------------
# Compute the Limber power spectra for all the windows loaded above
# (if none has been loaded or if the 'windows' argument is an empty list,
# nothing will be returned)
if fourier:
    ll    = np.geomspace(2., 1e4, 51)
    Cl    = S.limber_angular_power_spectra(l = ll, windows = None)
    # Multiplication constant for plotting
    c = ll*(ll+1.)/(2.*np.pi)
    # Single components
    Cl_ss = Cl['shear-shear']
    Cl_sI = Cl['shear-IA']+Cl['IA-shear']
    Cl_II = Cl['IA-IA']
    Cl_LL = Cl['lensing-lensing']
    Cl_GL = Cl['galaxy-lensing']
    Cl_GG = Cl['galaxy-galaxy']
    print("> Spectra loaded")
# Compute the Limber correlation functions for pairs of windows
# (unfortunately different windows require different orders for Hankel transform,
# so a for loop with fine-tuned 'order' parameter must be performed)
else:
    ll    = np.geomspace(2., 1e4, 128)
    Cl    = S.limber_angular_power_spectra(l = ll)
    theta = np.geomspace(10., 800., 51) 
    xi    = {}
    for key in Cl.keys():
        if   key in ['lensing-lensing', 'shear-shear', 'shear-IA', 'IA-shear', 'IA-IA']:
            order_plus, order_minus = 0, 4
            xi[key+' +'] = S.limber_angular_correlation_functions(theta, ll, Cl[key], order_plus)
            xi[key+' -'] = S.limber_angular_correlation_functions(theta, ll, Cl[key], order_minus)
        elif key in ['lensing-galaxy', 'galaxy-lensing']:
            order = 2
            xi[key] = S.limber_angular_correlation_functions(theta, ll, Cl[key], order)
        elif key == 'galaxy-galaxy':
            order = 0
            xi[key] = S.limber_angular_correlation_functions(theta, ll, Cl[key], order)
    # Single components
    xi_ss_p = xi['shear-shear +']
    xi_sI_p = xi['shear-IA +']+xi['IA-shear +']
    xi_II_p = xi['IA-IA +']
    xi_LL_p = xi['lensing-lensing +']
    xi_ss_m = xi['shear-shear -']
    xi_sI_m = xi['shear-IA -']+xi['IA-shear -']
    xi_II_m = xi['IA-IA -']
    xi_LL_m = xi['lensing-lensing -']
    xi_GL   = xi['galaxy-lensing']
    xi_GG   = xi['galaxy-galaxy']
    print("> Correlation functions loaded")
#-----------------

#-----------------
# 8) Plot
#-----------------
# Plot shear spectra
hf, axarr = plt.subplots(nbins, nbins, sharex = True, sharey = True, figsize=(12,8))
L,R,T,B=0.1, 0.95, 0.95, 0.15
plt.subplots_adjust(left=L,right=R,top=T,bottom=B)
# Triangle plot
for j in range(1, nbins):
    for i in range(j):
        axarr[i,j].axis('off')
    plt.setp([a.get_xticklabels() for a in axarr[i, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, j]], visible=False)
    plt.subplots_adjust(wspace=0, hspace=0)

if fourier:
    for i in range(nbins):
        for j in range(i, nbins):

            # Plotting Cls
            axarr[j,i].loglog(ll, c*Cl_ss[i,j],'blue'     ,ls='-' ,lw=LW,label='$C_\mathrm{\gamma\gamma}(\ell)$')
            axarr[j,i].loglog(ll, c*Cl_II[i,j],'red'      ,ls='-' ,lw=LW,label='$C_\mathrm{II}(\ell)$')
            axarr[j,i].loglog(ll, c*Cl_sI[i,j],'magenta'  ,ls='-' ,lw=LW,label='$C_\mathrm{\gamma I}(\ell)$')
            axarr[j,i].loglog(ll,-c*Cl_sI[i,j],'magenta'  ,ls='--',lw=LW)
            axarr[j,i].loglog(ll, c*Cl_LL[i,j],'black'    ,ls='-' ,lw=LW,label='$C_\mathrm{LL}(\ell)$')
            axarr[j,i].loglog(ll, c*Cl_GL[i,j],'green'    ,ls='-' ,lw=LW,label='$C_\mathrm{GL}(\ell)$')
            axarr[j,i].loglog(ll, c*Cl_GL[j,i],'limegreen',ls=':' ,lw=LW,label='$C_\mathrm{LG}(\ell)$')
            axarr[j,i].loglog(ll, c*Cl_GG[i,j],'goldenrod',ls='-' ,lw=LW,label='$C_\mathrm{GG}(\ell)$')
            # Coloured box
            if i != j: color = 'grey'
            else:      color = colors[i]
            axarr[j,i].text(0.15, 0.85, '$%i \\times %i$' %(i+1,j+1),
                                transform=axarr[j,i].transAxes,
                                style='italic',
                                fontsize = 12*(1.-nbins/10.),
                                horizontalalignment='center',
                                bbox={'facecolor': color, 'alpha':0.5, 'pad':5})
            axarr[j,i].set_xlim(ll.min(), ll.max())
            axarr[j,i].set_ylim(5e-10, 1e0)
            axarr[j,i].set_yticks([1e-8,1e-5,1e-2])
    plt.legend(bbox_to_anchor=(0.93, 0.98), fontsize = 12, bbox_transform=hf.transFigure)
    plt.text((L+R)*0.5, B*0.4, "$\ell$", ha='center', transform=hf.transFigure)
    plt.text(L*0.4,(T+B)*0.5, "$\ell(\ell+1) \ C_\ell \ / \ (2\pi)$", ha='center', va = 'center', rotation = 90, transform=hf.transFigure)
else:
    for i in range(nbins):
        for j in range(i, nbins):

            # Plotting correlation functions
            axarr[j,i].loglog(theta, xi_ss_p[i,j],'blue',
                              ls='-' ,lw=LW,label='$\\xi^{+/-}_\mathrm{\gamma\gamma}(\\theta)$')
            axarr[j,i].loglog(theta, xi_II_p[i,j],'red',
                              ls='-' ,lw=LW,label='$\\xi^{+/-}_\mathrm{II}(\\theta)$')
            axarr[j,i].loglog(theta, xi_sI_p[i,j],'magenta',
                              ls='-' ,lw=LW,label='$\\xi^{+/-}_\mathrm{\gamma I}(\\theta)$')
            axarr[j,i].loglog(theta, xi_LL_p[i,j],'black',
                              ls='-' ,lw=LW,label='$\\xi^{+/-}_\mathrm{LL}(\\theta)$')
            axarr[j,i].loglog(theta, xi_ss_m[i,j],'blue',
                              ls='--',lw=LW)
            axarr[j,i].loglog(theta, xi_II_m[i,j],'red',
                              ls='--',lw=LW)
            axarr[j,i].loglog(theta, xi_sI_m[i,j],'magenta',
                              ls='--',lw=LW)
            axarr[j,i].loglog(theta, xi_LL_m[i,j],'black',
                              ls='--',lw=LW)
            axarr[j,i].loglog(theta, xi_GL  [i,j],'green',
                              ls='-' ,lw=LW,label='$\\xi_\mathrm{GL}(\\theta)$')
            axarr[j,i].loglog(theta, xi_GL  [j,i],'limegreen',
                              ls=':' ,lw=LW,label='$\\xi_\mathrm{LG}(\\theta)$')
            axarr[j,i].loglog(theta, xi_GG  [i,j],'goldenrod',
                              ls='-' ,lw=LW,label='$\\xi_\mathrm{GG}(\\theta)$')
            # Coloured box
            if i != j: color = 'grey'
            else:      color = colors[i]
            axarr[j,i].text(0.15, 0.85, '$%i \\times %i$' %(i+1,j+1),
                                transform=axarr[j,i].transAxes,
                                style='italic',
                                fontsize = 12*(1.-nbins/10.),
                                horizontalalignment='center',
                                bbox={'facecolor': color, 'alpha':0.5, 'pad':5})
            axarr[j,i].set_xlim(theta.min(), theta.max())
            axarr[j,i].set_ylim(1e-8, 1e-2)
    plt.legend(bbox_to_anchor=(0.93, 0.98), fontsize = 12, bbox_transform=hf.transFigure)
    plt.text((L+R)*0.5, B*0.4, "$\\theta \ [\mathrm{arcmin}]$", ha='center', transform=hf.transFigure)
    plt.text(L*0.4,(T+B)*0.5, "$\\xi(\\theta)$", ha='center', va = 'center', rotation = 90, transform=hf.transFigure)


# Plot galaxy distributions and window functions
plt.figure(figsize=(12,8))
plt.subplots_adjust(hspace = 0.)
zz = np.linspace(0.1, 3.5, 1000)
for i in range(nbins):
    plt.plot(zz, S.window_function['lensing'][i](zz)*1e5,colors[i],ls='-' ,lw=LW,label='Bin %i' %(i+1))
    plt.plot(zz, S.window_function['shear'  ][i](zz)*1e5,colors[i],ls='--',lw=LW)
    plt.plot(zz, S.window_function['IA'     ][i](zz)*1e5,colors[i],ls=':' ,lw=LW)
    plt.plot(zz, S.window_function['galaxy' ][i](zz)*1e3,colors[i],ls='-.',lw=LW)
    #plt.plot(zz, S.window_function['HI'     ][i](zz)*1e1,colors[i],ls=':' ,lw=LW)
    #plt.plot(zz, S.window_function['CMB'    ][i](zz)*1e1,colors[i],ls=':' ,lw=LW)
    #plt.plot(zz, S.window_function['counts' ][i](zz)*1e1,colors[i],ls=':' ,lw=LW)
plt.plot(np.nan,np.nan,'k-' ,lw=LW,label='$10^5\\times W_\mathrm{L}(z)$')
plt.plot(np.nan,np.nan,'k--',lw=LW,label='$10^5\\times W_\gamma(z)$')
plt.plot(np.nan,np.nan,'k:' ,lw=LW,label='$10^5\\times W_\mathrm{IA}(z)$')
plt.plot(np.nan,np.nan,'k-.',lw=LW,label='$10^3\\times W_\mathrm{G}(z)$')
plt.xlabel('$z$')
plt.xlim(zz.min(), zz.max())
plt.ylabel('$W_\mathrm{X}(z) \ [h/\mathrm{Mpc}]$')
plt.legend(ncol=2)
plt.show()
#-----------------

