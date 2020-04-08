import colibri.cosmology as cc
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size = 40)


#==================
# This code computes the CMB Cls using the Python wrapper of class
#==================

# Define cosmology instance with default values
C = cc.cosmo()

# Compute Cls
l, Cl = C.class_Cl(l_max = 3000,
				   do_tensors = True,
				   lensing = True,
				   l_max_tensors = 1000)

# Separate components (T = temperature, E = scalar polarization modes, B = tensor polarization modes
# '-lensed' contain lensing effect)
ClTT   = Cl['TT']
ClTE   = Cl['TE']
ClEE   = Cl['EE']
ClBB   = Cl['BB']
ClTT_l = Cl['TT-lensed']
ClTE_l = Cl['TE-lensed']
ClEE_l = Cl['EE-lensed']
ClBB_l = Cl['BB-lensed']

# Plot
plt.figure(figsize = (25,17))
plt.loglog(l, l*(l+1)/(2.*np.pi)*ClTT, 'b', label = 'TT')
plt.loglog(l, l*(l+1)/(2.*np.pi)*ClTE, 'r', label = 'TE')
plt.loglog(l, l*(l+1)/(2.*np.pi)*ClEE, 'g', label = 'EE')
plt.loglog(l, l*(l+1)/(2.*np.pi)*ClBB, 'k', label = 'BB')
plt.loglog(l, l*(l+1)/(2.*np.pi)*ClTT_l, 'b--')
plt.loglog(l, l*(l+1)/(2.*np.pi)*ClTE_l, 'r--')
plt.loglog(l, l*(l+1)/(2.*np.pi)*ClEE_l, 'g--')
plt.loglog(l, l*(l+1)/(2.*np.pi)*ClBB_l, 'k--')
plt.xlim(2, l.max())
plt.xlabel('$\ell$')
plt.ylabel('$\\frac{\ell (\ell+1)}{2\pi} \ C_\ell$')
plt.legend()
plt.show()



