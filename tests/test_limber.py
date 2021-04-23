import numpy as np
import colibri.limber as LL
import colibri.cosmology as cc
import matplotlib.pyplot as plt

plt.rc('text',usetex=True)
plt.rc('font',size=25,family='serif')

# Colors
colors = ['r', 'b','g','goldenrod','m', 'k', 'springgreen', 'darkorange', 'pink', 'darkcyan', 'salmon']
# Linewidth
LW = 2

#########################
# Test of limber spectra class
#########################
nbins         = 5       # Number of bins to use (choose among 3,4,5)
fourier_space = True    # If True compute the spectra in Fourier space; if False, compute the correlation functions


#-----------------
# 1) Define a cosmology instance (with default values)
#-----------------
C = cc.cosmo()
print("> Cosmology loaded")
#-----------------

#-----------------
# 2) Define an angular_spectra instance.
#-----------------
S = LL.limber_3x2pt(cosmology = C, z_limits = [0.01, 5.])
print("> Limber instance loaded")
#-----------------

#-----------------
# 3) Load power spectra
#-----------------
kk = np.geomspace(1e-4, 1e2, 301)
zz = np.linspace(0., 5., 51)
ZZ,KK = np.meshgrid(zz,kk,indexing='ij')
_, pkz = C.camb_Pk(z = zz, k = kk, nonlinear = True, halofit = 'mead2020')
bkz = (1.+ZZ)**0.5
S.load_power_spectra(z = zz, k = kk, power_spectra = pkz, galaxy_bias = bkz)
print("> Power spectra loaded")
#-----------------

#-----------------
# 4) Window functions
#-----------------
if   nbins == 2:
    bin_edges = [0.01, 0.9, 5.00]
elif nbins == 3:
    bin_edges = [0.01, 0.72, 1.11, 5.00]
elif nbins == 4:
    bin_edges = [0.01, 0.62, 0.90, 1.23, 5.00]
elif nbins == 5:
    bin_edges = [0.01, 0.56, 0.79, 1.02, 1.32, 5.00]
else:
    raise ValueError("Choose among 3,4 or 5 bins (or implement your own set of galaxy distributions).")
# Load galaxy distributions
z_tmp     = np.linspace(S.z_min, S.z_max, 201)
nz_tmp    = [S.euclid_distribution_with_photo_error(z=z_tmp,zmin = bin_edges[i], zmax = bin_edges[i+1]) for i in range(nbins)]
# Window functions
S.load_window_functions(z = z_tmp, nz = nz_tmp)
print("> Window functions loaded")
#-----------------

#-----------------
# 6) Angular spectra or correlation functions
#-----------------
if fourier_space: 
    ll     = np.geomspace(2., 1e4, 51)
    Cl     = S.limber_angular_power_spectra(l            = ll,
                                            do_WL        = True,
                                            do_IA        = True,
                                            do_GC        = True,
                                            A_IA = -1.3, beta_IA = 0., eta_IA = 0., lum_IA = 1.)
    print("> Spectra loaded")
else:
    theta = np.geomspace(1., 300., 51)      # in arcmin
    xi    = S.limber_angular_correlation_functions(theta        = theta,
                                                   do_WL        = True,
                                                   do_IA        = True,
                                                   do_GC        = True,
                                                   A_IA = -1.3, beta_IA = 0., eta_IA = 0., lum_IA = 1.)
    print("> Correlation functions loaded")
#-----------------


#-----------------
# 7) Plot
#-----------------
if fourier_space:
    # Plot shear spectra
    hf, axarr = plt.subplots(nbins, nbins, sharex = True, sharey = True, figsize=(15,10))
    L,R,T,B=0.1, 0.95, 0.95, 0.15
    plt.subplots_adjust(left=L,right=R,top=T,bottom=B)
    # Multiplication constant for plotting
    c = ll*(ll+1.)/(2.*np.pi)
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
            axarr[j,i].text(0.15, 0.85, '$%i \\times %i$' %(i+1,j+1),
                            transform=axarr[j,i].transAxes,
                            style='italic',
                            fontsize = 15*(1.-nbins/10.),
                            horizontalalignment='center',
                            bbox={'facecolor': color, 'alpha':0.5, 'pad':5})
            axarr[j,i].set_xlim(ll.min(), ll.max())
            axarr[j,i].set_ylim(5e-10, 1e0)
            axarr[j,i].set_yticks([1e-8,1e-5,1e-2])
    plt.legend(bbox_to_anchor=(0.93, 0.98), fontsize = 20, bbox_transform=hf.transFigure)
    plt.text((L+R)*0.5, B*0.4, "$\ell$", ha='center', transform=hf.transFigure)
    plt.text(L*0.4,(T+B)*0.5, "$\ell(\ell+1) \ C_\ell \ / \ (2\pi)$", ha='center', va = 'center', rotation = 90, transform=hf.transFigure)


else:
    # Plot correlation functions
    hf, axarr = plt.subplots(nbins, nbins, sharex = True, sharey = True, figsize=(15,10))
    L,R,T,B=0.1, 0.95, 0.95, 0.15
    plt.subplots_adjust(left=L,right=R,top=T,bottom=B)
    for j in range(1, nbins):
        for i in range(j):
            axarr[i,j].axis('off')
        plt.setp([a.get_xticklabels() for a in axarr[i, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in axarr[:, j]], visible=False)
        plt.subplots_adjust(wspace=0, hspace=0)

    for i in range(nbins):
        for j in range(i, nbins):
            # Plotting Cls and systematics
            axarr[j,i].loglog(theta, xi['gg+'][i,j],         'blue'     , ls='-' , lw=LW, label='$\\xi_\mathrm{\gamma\gamma}^{+,(ij)}(\\theta)$')
            axarr[j,i].loglog(theta, xi['gg-'][i,j],         'blue'     , ls='--', lw=LW, label='$\\xi_\mathrm{\gamma\gamma}^{-,(ij)}(\\theta)$')
            axarr[j,i].loglog(theta, np.abs(xi['gI+'][i,j]), 'magenta'  , ls='-' , lw=LW, label='$\\xi_\mathrm{\gamma I}^{+,(ij)}(\\theta)$')
            axarr[j,i].loglog(theta, np.abs(xi['gI-'][i,j]), 'magenta'  , ls='--', lw=LW, label='$\\xi_\mathrm{\gamma I}^{-,(ij)}(\\theta)$')
            axarr[j,i].loglog(theta, xi['II+'][i,j],         'red'      , ls='-' , lw=LW, label='$\\xi_\mathrm{II}^{+,(ij)}(\\theta)$')
            axarr[j,i].loglog(theta, xi['II-'][i,j],         'red'      , ls='--', lw=LW, label='$\\xi_\mathrm{II}^{-,(ij)}(\\theta)$')
            axarr[j,i].loglog(theta, xi['LL+'][i,j],         'black'    , ls='-' , lw=LW, label='$\\xi_\mathrm{LL}^{+,(ij)}(\\theta)$')
            axarr[j,i].loglog(theta, xi['LL-'][i,j],         'black'    , ls='--', lw=LW, label='$\\xi_\mathrm{LL}^{-,(ij)}(\\theta)$')
            axarr[j,i].loglog(theta, xi['GL'] [i,j],         'green'    , ls='-' , lw=LW, label='$\\xi_\mathrm{GL}^{(ij)}(\\theta)$')
            axarr[j,i].loglog(theta, xi['GG'] [i,j],         'goldenrod', ls='-' , lw=LW, label='$\\xi_\mathrm{GG}^{(ij)}(\\theta)$')


            # Coloured box
            if i != j: color = 'grey'
            else:      color = colors[i]
            axarr[j,i].text(0.75, 0.15, '$%i \\times %i$' %(i+1,j+1),
                            transform=axarr[j,i].transAxes,
                            style='italic',
                            fontsize = 15*(1.-nbins/10.),
                            horizontalalignment='center',
                            bbox={'facecolor': color, 'alpha':0.5, 'pad':5})
            axarr[j,i].set_xlim(theta.min(), theta.max())
            axarr[j,i].set_ylim(1e-8, 3e-3)
    plt.legend(bbox_to_anchor=(0.93, 0.98), fontsize = 20, bbox_transform=hf.transFigure)

    # Single label
    hf.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.text((L+R)*0.5, B*0.4, "$\\theta [\mathrm{arcmin}]$", ha='center', transform=hf.transFigure)
    plt.text(L*0.4,(T+B)*0.5, "$\\xi(\\theta)$", ha='center', va = 'center', rotation = 90, transform=hf.transFigure)


# Plot galaxy distributions and window functions
hf, axarr = plt.subplots(2, 1, sharex=True, figsize=(30,20))
plt.subplots_adjust(hspace = 0.)
zz = np.linspace(0.1, 5., 1000)
for i in range(nbins):
    axarr[0].plot(zz, S.window_function['g'][i](zz)*1e5, colors[i],            lw = LW, label = 'Bin %i' %(i+1))
    axarr[1].plot(zz, S.window_function['I'][i](zz)*1e3, colors[i], ls = '--', lw = LW)
axarr[1].set_xlabel('$z$')
axarr[0].set_xlim(zz.min(), zz.max())
axarr[0].set_ylabel('$10^5 \\times W_\gamma     (z) \ [h/\mathrm{Mpc}]$', fontsize = 20)
axarr[1].set_ylabel('$10^3 \\times W_\mathrm{IA}(z) \ [h/\mathrm{Mpc}]$', fontsize = 20)
axarr[0].legend()
plt.show()
#-----------------

