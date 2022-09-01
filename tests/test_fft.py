import numpy as np
import scipy
import colibri.fourier as FF
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rc('text', usetex=True)
plt.rc('font', family = 'serif', size = 15)

FS = 40

######################
# Test of FFT
######################
# This routine uses the methods of the file "functions.py"
# to test routines for 1D, 2D, 3D FFTs and their inverses.

#------------------------------
# Direct FFT 1D
# Gaussian --> Gaussian
#------------------------------

# x, f(x) arrays
x = np.linspace(-10., 10., 100001)
y = np.exp(-x**2.)

# do FFT, set frequencies and normalize all
xf, yf = FF.FFT_1D(x,y)

# plot theory vs computation
plt.figure(figsize = (15,8))
plt.title('1D Gaussian $\\rightarrow$ 1D Gaussian')
plt.plot(x,  np.sqrt(np.pi)*np.exp(-x**2./4.), 'r--',         label = 'theory')
plt.plot(xf, yf,                               'bo',  ms = 5, label = 'computation')
plt.xlabel('$k$')
plt.ylabel('$f_k$')
plt.xlim(-10.,10.)
plt.legend()
plt.show()

#------------------------------
# Direct FFT 1D
# Lorentzian --> Exponential
#------------------------------

# x, f(x) arrays
x = np.linspace(-10., 10., 100001)
y = 1./(1.+x**2.)

# do FFT, set frequencies and normalize all
xf, yf = FF.FFT_1D(x,y)

# plot theory vs computation
plt.figure(figsize = (15,8))
plt.title('Lorentzian $\\rightarrow$ Exponential')
plt.plot(x,  np.pi*np.exp(-np.abs(x)), 'r--',         label = 'theory')
plt.plot(xf, yf,                       'bo',  ms = 5, label = 'computation')
plt.xlabel('$k$')
plt.ylabel('$f_k$')
plt.xlim(-2.,2.)
plt.legend()
plt.show()

#------------------------------
# Direct FFT 1D
# Sine --> Delta
#------------------------------

# x, f(x) arrays
x    = np.linspace(0., 100., 100001)
peak = 20.								# Set frequency of oscillation
y    = np.sin(peak*x)


# do FFT, set frequencies and normalize all
xf, yf = FF.FFT_1D(x,y)

# plot theory vs computation
plt.figure(figsize = (15,8))
plt.title('Sine $\\rightarrow$ Dirac Delta')
plt.axvline(peak, c='r', ls = '--', label = 'theory')
plt.plot(xf, yf, 'bo', ms = 5, label = 'computation')
plt.xlabel('$k$')
plt.ylabel('$f_k$')
plt.xlim(0,100)
plt.legend()
plt.show()

#------------------------------
# Direct and inverse FT 1D
#------------------------------

# x, f(x) arrays with length and step to set FFT normalization
x = np.linspace(-10., 10., 100001)
y = 1./(1.+x**2.)

# do FFT, set frequencies and normalize all
xf, yf = FF.FFT_1D(x,y)

# go back
x2, y2 = FF.iFFT_1D(xf,np.abs(yf))

# plot theory vs computation
plt.figure(figsize = (15,8))
plt.title('Lorentzian, back and forth')
plt.plot(x,  y,  'r--',        label = 'initial')
plt.plot(x2, y2, 'bo', ms = 5, label = 'final')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.xlim(-2.,2.)
plt.legend()
plt.show()


#------------------------------
# Direct FFT 2D
# Gaussian x Lorentzian --> Gaussian x exponential
#------------------------------
# Define 2D Gaussian
x = np.linspace(-5., 5., 26)
y = np.linspace(-5., 5., 21)
X,Y = np.meshgrid(x,y)
F_G = np.exp(-X**2.)*1./(1.+Y**2.)


# Do FFT
kx, ky, fft_G = FF.FFT_2D(x, y, F_G)

# Theory
KX,KY  = np.meshgrid(kx,ky)
FFT_TH = np.sqrt(np.pi)*np.exp(-KX**2./4.)*np.pi*np.exp(-np.abs(KY))

plt.figure(figsize = (15,8))
plt.title('Gaussian $\\times$ Lorentzian $\\rightarrow$ Gaussian $\\times$ exponential')
plt.contourf(kx, ky, fft_G.T, cmap = 'Greys_r', extent=(np.amin(kx), np.amax(kx), np.amin(ky), np.amax(ky)))
#plt.contourf(kx, ky, FFT_TH, cmap = 'Greys_r', extent=(np.amin(kx), np.amax(kx), np.amin(ky), np.amax(ky)))
plt.xlabel('$k_x$')
plt.ylabel('$k_y$')
plt.axvline(0., c='r', ls = '--')
plt.axhline(0., c='r', ls = '--')
cax = plt.colorbar()
cax.ax.tick_params(labelsize=FS) 
plt.show()


#------------------------------
# Inverse FFT 2D
# Rectangular window --> 2D Sampling function
#------------------------------

# Define window
x = np.linspace(-10., 10., 1001)
y = np.linspace(-10., 10., 1001)
X,Y = np.meshgrid(x,y)
F_W = 1./(X**10.+8)**2./(Y**10.+8)**2.

# Do FFT
kx, ky, fft_W = FF.iFFT_2D(x, y, F_W)

plt.figure(figsize = (15,8))
plt.title('Square window $\\rightarrow$ Sampling function')
plt.contourf(kx, ky, fft_W, cmap = 'Greys_r', extent=(np.amin(kx), np.amax(kx), np.amin(ky), np.amax(ky)))
plt.xlim(-10., 10.)
plt.ylim(-10., 10.)
plt.xlabel('$k_x$')
plt.ylabel('$k_y$')
cax = plt.colorbar()
cax.ax.tick_params(labelsize=FS) 
plt.show()


###############################
# 3D FFT
###############################

#------------------------------
# Direct FFT 3D
# Gaussian x exponential x Lorentzian
#------------------------------
# Define 3D Gaussian
x = np.linspace(-10., 10., 104)
y = np.linspace(-10., 10., 106)
z = np.linspace(-10., 10., 108)
X,Y,Z = np.meshgrid(x,y,z)
F_G = np.exp(-X**2.)*np.exp(-np.abs(Y))*1./(1.+Z**2.)

#print(F_G.shape)
# Do FFT
kx, ky, kz, fft_G = FF.FFT_3D(x, y, z, F_G)


KX,KY,KZ = np.meshgrid(kx,ky,kz, indexing = 'ij')
FFT_TH = np.pi**0.5*np.exp(-KX**2./4.)*2./(1.+KY**2.)*np.pi*np.exp(-np.abs(KZ))


ind_x = np.linspace(1,len(x)-1,7).astype(int)[1:-1]
ind_y = np.linspace(1,len(y)-1,7).astype(int)[1:-1]


plt.figure(figsize=(12,12))
index = 0
for ix in ind_x:
	for iy in ind_y:
		index +=1
		plt.subplot(len(ind_x), len(ind_y), index)
		plt.plot(kz, fft_G[ix,iy], 'bo')
		plt.plot(kz, FFT_TH[ix,iy], 'r--')
		plt.xticks(fontsize = 12)
		plt.yticks(fontsize = 12)
plt.show()



#------------------------------
# Inverse FFT 3D
# Gaussian x exponential x Lorentzian
#------------------------------
# Define 3D Gaussian
x = np.linspace(-10., 10., 104)
y = np.linspace(-10., 10., 106)
z = np.linspace(-10., 10., 108)
X,Y,Z = np.meshgrid(x,y,z)
F_G = np.exp(-X**2.)*np.exp(-np.abs(Y))*1./(1.+Z**2.)

# Do FFT
kx, ky, kz, fft_G = FF.iFFT_3D(x, y, z, F_G)


KX,KY,KZ = np.meshgrid(kx,ky,kz, indexing = 'ij')
FFT_TH = np.pi**0.5*np.exp(-KX**2./4.)*2./(1.+KY**2.)*np.pi*np.exp(-np.abs(KZ))/(2.*np.pi)**3.


ind_x = np.linspace(1,len(x)-1,7).astype(int)[1:-1]
ind_y = np.linspace(1,len(y)-1,7).astype(int)[1:-1]



plt.figure(figsize=(12,12))
index = 0
for ix in ind_x:
	for iy in ind_y:
		index +=1
		plt.subplot(len(ind_x), len(ind_y), index)
		#plt.title('$x=%.1f, y=%.1f$' %(kx[ix],ky[iy]), fontsize = 15)
		plt.plot(kz, fft_G[ix,iy], 'bo')
		plt.plot(kz, FFT_TH[ix,iy], 'r--')
		plt.xticks(fontsize = 12)
		plt.yticks(fontsize = 12)
plt.show()



