import numpy as np
import scipy.interpolate as si
import scipy.fftpack as sfft
import scipy.integrate as sint
import scipy.optimize
import sys
import colibri.constants as const
try:
    import fftlog
except ImportError:
    try:
        import pyfftlog as fftlog
    except ImportError:
        pass


#-----------------------------------------------------------------------------------------
# FFT_1D
#-----------------------------------------------------------------------------------------
def FFT_1D(x, f_x, N = None, compl = False):
    """
    This routine returns the 1D FFT of a linearly-spaced array.
    It returns the radian frequencies and the Fourier transform.
    N.B.: the frequencies and FT are sorted.
    Note on normalizations: the returned frequencies are :math:`2\pi\\nu`;
    the FT is already normalized.

    Parameters
    ----------

    `x`: array
        Abscissae.

    `f_x`: array
        Function sampled at `x`.

    `N`: integer, default = None
        Number of output points. If None, the same length of `x` is used.

    `compl`: boolean, default = False
        Return complex values.

    Returns
    ----------

    xf: array
        Frequencies in k_x (already sorted).

    yf: array
        Fourier transform (sorted by frequencies).
    """
    if len(x) != len(f_x): raise IndexError("x and f_x must have same length")


    # N is the length of the output array.
    # If N is not specified, use the length of input x array
    if N == None:
        N = len(x)
    # Linear spacing
    D = x[1]-x[0]

    # do FFT, set frequencies and normalize all
    xf_tmp = sfft.fftfreq(N, D)*2.*np.pi
    yf_tmp = sfft.fft(f_x, N)*D
    
    # sort
    xf, yf = (list(t) for t in zip(*sorted(zip(xf_tmp, yf_tmp))))
    xf = np.array(xf)
    yf = np.array(yf)
    
    if compl == False:
        yf = np.abs(yf)
        
    return xf, yf

#-----------------------------------------------------------------------------------------
# iFFT_1D
#-----------------------------------------------------------------------------------------
def iFFT_1D(k, f_k, N = None, compl = False):
    """
    This routine returns the 1D FFT of a linearly-spaced array.
    It returns the radian frequencies and the Fourier transform.
    N.B.: the frequencies and FT are sorted.
    Note on normalizations: the returned frequencies are :math:`2\pi\\nu`;
    the inverse FT is already normalized.

    Parameters
    ----------

    `k`: array
        Abscissae.

    `f_k`: array
        Function sampled at `x`.

    `N`: integer, default = None
        Number of output points. If None, the same length of `x` is used.

    `compl`: boolean, default = False
        Return complex values.

    Returns
    ----------

    xf: array
        Frequencies in x (already sorted).

    yf: array
        Inverse Fourier transform (sorted by frequencies).
    """
    if len(k) != len(f_k): raise IndexError("k and f_k must have same length")

    # N is the length of the output array.
    # If N is not specified, use the length of input x array
    if N == None:
        N = len(k)
    # Linear spacing
    D = k[1]-k[0]
    
    # do FFT, set frequencies and normalize all
    xf_tmp = sfft.fftfreq(N, D)*2.*np.pi
    yf_tmp = sfft.ifft(f_k, N)*D*N/(2.*np.pi)
    
    # sort
    xf, yf = (list(t) for t in zip(*sorted(zip(xf_tmp, yf_tmp))))
    xf = np.array(xf)
    yf = np.array(yf)

    # Take modulus if real is required
    if compl == False:
        yf = np.abs(yf)
    
    return xf, yf

#-----------------------------------------------------------------------------------------
# FFT_2D
#-----------------------------------------------------------------------------------------
def FFT_2D(x, y, f_xy, compl = False, sort = True):
    """
    This routine returns the 2D FFT of a linearly-spaced array.
    It returns the radian frequencies and the Fourier transform.
    N.B.: the frequencies and FT are sorted.
    Note on normalizations: the returned frequencies are :math:`2\pi\\nu`; the FT is already normalized.

    Parameters
    ----------

    `x`: array
        Abscissa 1.

    `y`: array
        Abscissa 2.

    `f_xy`: 2D array
        Function sampled at (x,y).

    `compl`: boolean, default = False
        Return complex values.

    `sort`: boolean, dafault = True
        Return sorted frequencies and Fourier transform. If False, the NumPy order is used.

    Returns
    ----------

    KX: array
        Frequencies in k_x.

    KY: array
        Frequencies in k_y.

    F_KXKY: 2D array
        Fourier transform.
    """

    # Check if shape is correct and in case change it
    if f_xy.shape == (y.shape[0],x.shape[0]):    f_xy = np.transpose(f_xy)

    # Get mapping from unsorted to sorted FFT
    def get_map(N):
        if N%2 == 0:
            H = N//2
            S = H
        else:
            H = (N+1)//2
            S = H-1
        return [i+S if i<H else i-H for i in range(N)]


    # Linear spacing and useful stuff
    Dx = x[1]-x[0]
    Dy = y[1]-y[0]
    Nx = len(x)
    Ny = len(y)

    # Frequencies x and y
    KX_unsorted = sfft.fftfreq(Nx, Dx)*2.*np.pi
    KY_unsorted = sfft.fftfreq(Ny, Dy)*2.*np.pi

    # Transform
    f_kxky = np.fft.fft2(f_xy, axes = (-2,-1))*Dx*Dy

    # Sort the transform
    if sort:
        F_KXKY = np.zeros(f_kxky.shape, dtype = complex)
        # Sort values of X,Y
        mapX = get_map(Nx)
        mapY = get_map(Ny)
        for i, j in np.ndindex((Nx,Ny)):
            F_KXKY[mapX[i], mapY[j]] = f_kxky[i,j]
        # Sort frequencies
        KX = np.sort(KX_unsorted)
        KY = np.sort(KY_unsorted)
    else:
        F_KXKY = f_kxky
        # Sort frequencies
        KX = KX_unsorted
        KY = KY_unsorted



    # Take modulus if required
    if compl == False:
        F_KXKY = np.abs(F_KXKY)
    
    
    return KX, KY, F_KXKY

#-----------------------------------------------------------------------------------------
# iFFT_2D
#-----------------------------------------------------------------------------------------
def iFFT_2D(kx, ky, f_kxky, compl = False, sort = True):
    """
    This routine returns the inverse 2D FFT of a linearly-spaced array.
    It returns the radian frequencies and the Fourier transform.
    N.B.: the frequencies and FT are sorted.
    Note on normalizations: the returned frequencies are :math:`2\pi\\nu`; the FT is already normalized.

    Parameters
    ----------

    `kx`: array
        Abscissa 1.

    `ky`: array
        Abscissa 2.

    `f_kxky`: 2D array
        Function sampled at (kx,ky).

    `compl`: boolean, default = False
        Return complex values.

    `sort`: boolean, dafault = True
        Return sorted frequencies and Fourier transform. If False, the NumPy order is used.

    Returns
    ----------

    X: array
        Frequencies in x.

    Y: array
        Frequencies in y.

    F_XY: 2D array
        Fourier transform.
    """
    # Check if shape is correct and in case change it
    if f_kxky.shape == (ky.shape[0],kx.shape[0]):    f_kxky = np.transpose(f_kxky)

    # Get mapping from unsorted to sorted FFT
    def get_map(N):
        if N%2 == 0:
            H = N//2
            S = H
        else:
            H = (N+1)//2
            S = H-1
        return [i+S if i<H else i-H for i in range(N)]


    # Linear spacing and useful stuff
    Dx = kx[1]-kx[0]
    Dy = ky[1]-ky[0]
    Nx = len(kx)
    Ny = len(ky)

    # Frequencies x and y
    X_unsorted = sfft.fftfreq(Nx, Dx)*2.*np.pi
    Y_unsorted = sfft.fftfreq(Ny, Dy)*2.*np.pi

    # Transform
    f_xy = np.fft.ifft2(f_kxky, axes = (-2,-1))*Dx*Dy*Nx*Ny/(2.*np.pi)**2.

    # Sort the transform
    if sort:
        F_XY = np.zeros(f_xy.shape, dtype = complex)
        # Sort values of X,Y
        mapX = get_map(Nx)
        mapY = get_map(Ny)
        for i, j in np.ndindex((Nx,Ny)):
            F_XY[mapX[i], mapY[j]] = f_xy[i, j]
        # Sort frequencies
        X = np.sort(X_unsorted)
        Y = np.sort(Y_unsorted)
    else:
        F_XY = f_xy
        # Sort frequencies
        X = X_unsorted
        Y = Y_unsorted

    # Take modulus if required
    if compl == False:
        F_XY = np.abs(F_XY)
    
    
    return X,Y,F_XY


#-----------------------------------------------------------------------------------------
# FFT_3D
#-----------------------------------------------------------------------------------------
def FFT_3D(x, y, z, f_xyz, compl = False, sort = True):
    """
    This routine returns the 3D FFT of a linearly-spaced array.
    It returns the radian frequencies and the Fourier transform.
    N.B.: the frequencies and FT are sorted.
    Note on normalizations: the returned frequencies are :math:`2\pi\\nu`; the FT is already normalized.

    Parameters
    ----------

    `x`: array
        Abscissa 1.

    `y`: array
        Abscissa 2.

    `z`: array
        Abscissa 3.

    `f_xyz`: 3D array
        Function sampled at (x,y,z).

    `compl`: boolean, default = False
        Return complex values.

    `sort`: boolean, dafault = True
        Return sorted frequencies and Fourier transform. If False, the NumPy order is used.

    Returns
    ----------

    KX: array
        Frequencies in k_x.

    KY: array
        Frequencies in k_y.

    KZ: array
        Frequencies in k_z.

    F_KXKYKZ: 3D array
        Fourier transform.
    """

    # Check if shape is correct and in case change it
    if f_xyz.shape == (y.shape[0],x.shape[0],z.shape[0]):    f_xyz = np.transpose(f_xyz, (1,0,2))

    # Get mapping from unsorted to sorted FFT
    def get_map(N):
        if N%2 == 0:
            H = N//2
            S = H
        else:
            H = (N+1)//2
            S = H-1
        return [i+S if i<H else i-H for i in range(N)]

    # Linear spacing and useful stuff
    Dx = x[1]-x[0]
    Dy = y[1]-y[0]
    Dz = z[1]-z[0]
    Nx = len(x)
    Ny = len(y)
    Nz = len(z)
    
    # Transform
    f_kxkykz = sfft.fftn(f_xyz, axes = (-3,-2,-1))*Dx*Dy*Dz

    if sort:
        # Sort the transform
        F_KXKYKZ = np.zeros(f_kxkykz.shape, dtype = complex)
        mapX = get_map(Nx)
        mapY = get_map(Ny)
        mapZ = get_map(Nz)
        for i, j, k in np.ndindex((Nx,Ny,Nz)):
            F_KXKYKZ[mapX[i], mapY[j], mapZ[k]] = f_kxkykz[i,j,k]
        # Frequencies x and y
        KX = np.sort(sfft.fftfreq(Nx, Dx)*2.*np.pi)
        KY = np.sort(sfft.fftfreq(Ny, Dy)*2.*np.pi)
        KZ = np.sort(sfft.fftfreq(Nz, Dz)*2.*np.pi)
    else:
        F_KXKYKZ = f_kxkykz
        KX = sfft.fftfreq(Nx, Dx)*2.*np.pi
        KY = sfft.fftfreq(Ny, Dy)*2.*np.pi
        KZ = sfft.fftfreq(Nz, Dz)*2.*np.pi
    
    # Take modulus if required
    if compl == False:
        F_KXKYKZ = np.abs(F_KXKYKZ)
    

    
    return KX, KY, KZ, F_KXKYKZ

#-----------------------------------------------------------------------------------------
# iFFT_3D
#-----------------------------------------------------------------------------------------
def iFFT_3D(kx, ky, kz, f_kxkykz, compl = False, sort = True):
    """
    This routine returns the inverse 3D FFT of a linearly-spaced array.
    It returns the radian frequencies and the Fourier transform.
    N.B.: the frequencies and FT are sorted.
    Note on normalizations: the returned frequencies are :math:`2\pi\\nu`; the FT is already normalized.

    Parameters
    ----------

    `kx`: array
        Abscissa 1.

    `ky`: array
        Abscissa 2.

    `kz`: array
        Abscissa 3.

    `f_kxkykz`: 3D array
        Function sampled at (kx,ky,kz).

    `compl`: boolean, default = False
        Return complex values.

    `sort`: boolean, dafault = True
        Return sorted frequencies and Fourier transform. If False, the NumPy order is used.

    Returns
    ----------

    X: array
        Frequencies in x.

    Y: array
        Frequencies in y.

    Z: array
        Frequencies in z.

    F_XYZ: 3D array
        Fourier transform.
    """

    # Check if shape is correct and in case change it
    if f_kxkykz.shape == (ky.shape[0],kx.shape[0],kz.shape[0]):    f_kxkykz = np.transpose(f_kxkykz, (1,0,2))

    # Get mapping from unsorted to sorted FFT
    def get_map(N):
        if N%2 == 0:
            H = N//2
            S = H
        else:
            H = (N+1)//2
            S = H-1
        return [i+S if i<H else i-H for i in range(N)]


    # Linear spacing and useful stuff
    Dx = kx[1]-kx[0]
    Dy = ky[1]-ky[0]
    Dz = kz[1]-kz[0]
    Nx = len(kx)
    Ny = len(ky)
    Nz = len(kz)
    
    # Transform
    f_xyz = sfft.ifftn(f_kxkykz, axes = (-3,-2,-1))*Dx*Dy*Dz*Nx*Ny*Nz/(2.*np.pi)**3.

    # Sort the transform
    if sort:
        F_XYZ = np.zeros(f_xyz.shape, dtype = complex)
        # Sort values of X,Y
        mapX = get_map(Nx)
        mapY = get_map(Ny)
        mapZ = get_map(Nz)
        for i, j, k in np.ndindex((Nx,Ny,Nz)):
            F_XYZ[mapX[i], mapY[j], mapZ[k]] = f_xyz[i,j,k]
        # Frequencies x and y (already sorted)
        X = np.sort(sfft.fftfreq(Nx, Dx)*2.*np.pi)
        Y = np.sort(sfft.fftfreq(Ny, Dy)*2.*np.pi)
        Z = np.sort(sfft.fftfreq(Nz, Dz)*2.*np.pi)
    else:
        F_XYZ = f_xyz
        X = sfft.fftfreq(Nx, Dx)*2.*np.pi
        Y = sfft.fftfreq(Ny, Dy)*2.*np.pi
        Z = sfft.fftfreq(Nz, Dz)*2.*np.pi
    
    # Take modulus if required
    if compl == False:
        F_XYZ = np.abs(F_XYZ)
    
    return X, Y, Z, F_XYZ




#-----------------------------------------------------------------------------------------
# FFTLOG 3D (ISOTROPIC)
#-----------------------------------------------------------------------------------------
def FFT_iso_3D(r, f , N = 4096):
    """
    This routine returns the FFT of a radially symmetric function.
    It employs the ``FFTlog`` module, which in turn makes use of the Hankel transform: therefore the function
    that will be actually transformed is :math:`f(r) \ (2\pi r)^{1.5}/k^{1.5}`.
    N.B. Since the integral is performed in log-space, the exponent of `r` is 1.5 instead of 0.5.
    The computation is

    .. math ::

      f(k) = \int_0^\infty dr \ 4\pi r^2 \ f(r) \ j_0(kr)

    Parameters
    ----------

    `r`: array
        Abscissae of function, log-spaced.

    `f`: array
        ordinates of function.

    `N`: int, default = 4096
        Number of output points

    Returns
    ----------

    kk: array
        Frequencies.

    Fk: array
        Transformed array.
    """
    mu      = 0.5  # Order mu of Bessel function
    q       = 0    # Bias exponent: q = 0 is unbiased
    kr      = 1    # Sensible approximate choice of k_c r_c
    kropt   = 1    # Tell fhti to change kr to low-ringing value
    tdir    = 1    # Forward transform
    logrc   = (np.max(np.log10(r))+np.min(np.log10(r)))/2.
    dlogr   = (np.max(np.log10(r))-np.min(np.log10(r)))/N
    dlnr    = dlogr*np.log(10.)
    if   N%2 == 0: nc = N/2
    else         : nc = (N+1)/2
    r_t = 10.**(logrc + (np.arange(1, N+1) - nc)*dlogr)

    # Initialise function
    funct = si.interp1d(r, f, kind = "cubic", bounds_error = False, fill_value = 0.)
    ar    = funct(r_t)*(2.*np.pi*r_t)**1.5

    # Initialization of transform
    #kr, xsave, ok = fftlog.fhti(N, mu, dlnr, q, kr, kropt)
    fft_obj = fftlog.fhti(N, mu, dlnr, q, kr, kropt)
    kr, xsave = fft_obj[0], fft_obj[1]
    logkc = np.log10(kr) - logrc

    # Transform
    ak = fftlog.fht(ar.copy(), xsave, tdir)
    if   N%2 == 0: kk = 10.**(logkc + (np.arange(N) - nc)*dlogr)
    else         : kk = 10.**(logkc + (np.arange(1,N+1) - nc)*dlogr)
    Fk = ak/kk**1.5

    return kk, Fk


#-----------------------------------------------------------------------------------------
# INVERSE FFTLOG 3D (ISOTROPIC)
#-----------------------------------------------------------------------------------------
def iFFT_iso_3D(k, f, N = 4096):
    """
    This routine returns the inverse FFT of a radially symmetric function.
    It employs the ``FFTlog`` module, which in turn makes use of the Hankel transform: therefore the function
    that will be actually transformed is :math:`f(k) \ k^{1.5}/(2\pi r)^{1.5}`.
    N.B. Since the integral is performed in log-space, the exponent of `r` is 1.5 instead of 0.5.
    The computation is

    .. math ::

      f(r) = \int_0^\infty \\frac{dk \ k^2}{2\pi^2} \ f(k) \ j_0(kr)

    Parameters
    ----------

    `k`: array
        Abscissae of function, log-spaced.

    `f`: array
        ordinates of function.

    `N`: int, default = 4096
        Number of output points

    Returns
    ----------

    rr: array
        Frequencies.

    Fr: array
        Transformed array.
    """

    # FFT specifics
    mu      = 0.5   # Order mu of Bessel function
    q       = 0     # Bias exponent: q = 0 is unbiased
    kr      = 1     # Sensible approximate choice of k_c r_c
    kropt   = 1     # Tell fhti to change kr to low-ringing value
    tdir    = -1    # Backward transform
    logkc   = (np.max(np.log10(k))+np.min(np.log10(k)))/2.
    dlogk   = (np.max(np.log10(k))-np.min(np.log10(k)))/N
    dlnk    = dlogk*np.log(10.)
    if   N%2 == 0: nc = N/2
    else         : nc = (N+1)/2
    k_t   = 10.**(logkc + (np.arange(1, N+1) - nc)*dlogk)

    # Initialise function
    funct = si.interp1d(k, f, kind = "cubic", bounds_error = False, fill_value = 0.)
    ak    = funct(k_t)*k_t**1.5
    
    # Initialization of transform
    #kr, xsave, ok = fftlog.fhti(N, mu, dlnk, q, kr, kropt)
    fft_obj = fftlog.fhti(N, mu, dlnk, q, kr, kropt)
    kr, xsave = fft_obj[0], fft_obj[1]
    logrc = np.log10(kr) - logkc

    # Transform
    ar = fftlog.fht(ak.copy(), xsave, tdir)
    if   N%2 == 0: rr = 10.**(logrc + (np.arange(N) - nc)*dlogk)
    else         : rr = 10.**(logrc + (np.arange(1,N+1) - nc)*dlogk)
    Fr = ar/(2*np.pi*rr)**1.5

    return rr, Fr

#-----------------------------------------------------------------------------------------
# 3D Hankel
#-----------------------------------------------------------------------------------------
def Hankel_spherical_Bessel(r, f , N = 4096, order = 0):
    """
    This routine is analogous to the Fourier transform in 3D but it can return any order of the Bessel function

    .. math ::

      f_\ell(k) = \int_0^\infty dr \ 4\pi r^2 \ f(r) \ j_\ell(kr)

    Parameters
    ----------

    `r`: array
        Abscissae of function, log-spaced.

    `f`: array
        ordinates of function.

    `N`: int, default = 4096
        Number of output points

    `order`: float, default = 0
        Order of the Bessel spherical polynomial.

    Returns
    ----------

    kk: array
        Frequencies.

    Fk: array
        Transformed array.
    """
    mu      = order+0.5  # Order mu of Bessel function
    q       = 0          # Bias exponent: q = 0 is unbiased
    kr      = 1          # Sensible approximate choice of k_c r_c
    kropt   = 1          # Tell fhti to change kr to low-ringing value
    tdir    = 1          # Forward transform
    logrc   = (np.max(np.log10(r))+np.min(np.log10(r)))/2.
    dlogr   = (np.max(np.log10(r))-np.min(np.log10(r)))/N
    dlnr    = dlogr*np.log(10.)
    if   N%2 == 0: nc = N/2
    else         : nc = (N+1)/2
    r_t = 10.**(logrc + (np.arange(1, N+1) - nc)*dlogr)

    # Initialise function
    funct = si.interp1d(r, f, kind = "cubic", bounds_error = False, fill_value = 0.)
    ar    = funct(r_t)*(2.*np.pi*r_t)**1.5

    # Initialization of transform
    #kr, xsave, ok = fftlog.fhti(N, mu, dlnr, q, kr, kropt)
    fft_obj = fftlog.fhti(N, mu, dlnr, q, kr, kropt)
    kr, xsave = fft_obj[0], fft_obj[1]
    logkc = np.log10(kr) - logrc

    # Transform
    ak = fftlog.fht(ar.copy(), xsave, tdir)
    if   N%2 == 0: kk = 10.**(logkc + (np.arange(N) - nc)*dlogr)
    else         : kk = 10.**(logkc + (np.arange(1,N+1) - nc)*dlogr)
    Fk = ak/kk**1.5

    return kk, Fk

#-----------------------------------------------------------------------------------------
# 3D inverse Hankel
#-----------------------------------------------------------------------------------------
def iHankel_spherical_Bessel(k, f, N = 4096, order = 0):
    """
    This routine is similar to the Fourier transform in 3D but it can return any order of the Bessel function

    .. math ::

      f_\ell(r) = \int_0^\infty \\frac{dk \ k^2}{2\pi^2} \ f(k) \ j_\ell(kr)

    Parameters
    ----------

    `k`: array
        Abscissae of function, log-spaced.

    `f`: array
        ordinates of function.

    `N`: int, default = 4096
        Number of output points

    `order`: float, default = 0
        Order of the Bessel spherical polynomial.

    Returns
    ----------

    rr: array
        Frequencies.

    Fr: array
        Transformed array.
    """
    # FFT specifics
    mu      = order+0.5   # Order mu of Bessel function
    q       = 0           # Bias exponent: q = 0 is unbiased
    kr      = 1           # Sensible approximate choice of k_c r_c
    kropt   = 1           # Tell fhti to change kr to low-ringing value
    tdir    = -1          # Backward transform
    logkc   = (np.max(np.log10(k))+np.min(np.log10(k)))/2.
    dlogk   = (np.max(np.log10(k))-np.min(np.log10(k)))/N
    dlnk    = dlogk*np.log(10.)
    if   N%2 == 0: nc = N/2
    else         : nc = (N+1)/2
    k_t   = 10.**(logkc + (np.arange(1, N+1) - nc)*dlogk)

    # Initialise function
    funct = si.interp1d(k, f, kind = "cubic", bounds_error = False, fill_value = 0.)
    ak    = funct(k_t)*k_t**1.5
    
    # Initialization of transform
    #kr, xsave, ok = fftlog.fhti(N, mu, dlnk, q, kr, kropt)
    fft_obj = fftlog.fhti(N, mu, dlnk, q, kr, kropt)
    kr, xsave = fft_obj[0], fft_obj[1]
    logrc = np.log10(kr) - logkc

    # Transform
    ar = fftlog.fht(ak.copy(), xsave, tdir)
    if   N%2 == 0: rr = 10.**(logrc + (np.arange(N) - nc)*dlogk)
    else         : rr = 10.**(logrc + (np.arange(1,N+1) - nc)*dlogk)
    Fr = ar/(2.*np.pi*rr)**1.5

    return rr, Fr

#-----------------------------------------------------------------------------------------
# HANKEL TRANSFORM
#-----------------------------------------------------------------------------------------
def Hankel(r, f , N = 4096, order = 0.5):
    """
    This routine returns the Hankel transform of order :math:`\\nu` of a log-spaced function.
    The computation is

    .. math ::

      f_\\nu(k) = \int_0^\infty \ dr \ r \ f(r) \ J_\\nu(kr)

    .. warning::

     Because of log-spacing, an extra `r` factor has been already added in the code.

    .. warning::

     If ``order = 0.5`` it is similar to the 3D Fourier transform of a spherically symmetric function.

    Parameters
    ----------

    `r`: array
        Abscissae of function, log-spaced.

    `f`: array
        ordinates of function.

    `N`: int, default = 4096
        Number of output points

    `order`: float, default = 0.5
        Order of the transform (Bessel polynomial).

    Returns
    ----------

    kk: array
        Frequencies.

    Fk: array
        Transformed array.
    """
    mu      = order # Order mu of Bessel function
    q       = 0     # Bias exponent: q = 0 is unbiased
    kr      = 1     # Sensible approximate choice of k_c r_c
    kropt   = 1     # Tell fhti to change kr to low-ringing value
    tdir    = 1     # Forward transform
    logrc   = (np.max(np.log10(r))+np.min(np.log10(r)))/2.
    dlogr   = (np.max(np.log10(r))-np.min(np.log10(r)))/N
    dlnr    = dlogr*np.log(10.)
    if   N%2 == 0: nc = N/2
    else         : nc = (N+1)/2
    r_t = 10.**(logrc + (np.arange(1, N+1) - nc)*dlogr)


    # Initialise function (add r_t extra factor because of log-spacing)
    funct = si.interp1d(r, f, kind = "cubic", bounds_error = False, fill_value = 0.)
    ar    = funct(r_t)*r_t

    # Initialization of transform
    #kr, xsave, ok = fftlog.fhti(N, mu, dlnr, q, kr, kropt)
    fft_obj = fftlog.fhti(N, mu, dlnr, q, kr, kropt)
    kr, xsave = fft_obj[0], fft_obj[1]
    logkc = np.log10(kr) - logrc

    # Transform (dividing by a kk extra factor because of log-spacing)
    Fk = fftlog.fht(ar.copy(), xsave, tdir)
    if   N%2 == 0: kk = 10.**(logkc + (np.arange(N) - nc)*dlogr)
    else         : kk = 10.**(logkc + (np.arange(1,N+1) - nc)*dlogr)
    Fk /= kk

    return kk, Fk


#-----------------------------------------------------------------------------------------
# INVERSE HANKEL
#-----------------------------------------------------------------------------------------
def iHankel(k, f, N = 4096, order = 0.5):
    """
    This routine returns the inverse Hankel transform of order :math:`\\nu` of a log-spaced function.
    The computation is

    .. math ::

      f(r) = \int_0^\infty \ dk \ k \ f(k) \ J_\\nu(kr)

    .. warning::

     Because of log-spacing, an extra `k` factor has been already added in the code.

    .. warning::

     If ``order = 0.5`` it is similar to the 3D Fourier transform of a spherically symmetric function.


    Parameters
    ----------

    `k`: array
        Abscissae of function, log-spaced.

    `f`: array
        ordinates of function.

    `N`: int, default = 4096
        Number of output points

    `order`: float, default = 0.5
        Order of the transform (Bessel polynomial).

    Returns
    ----------

    rr: array
        Frequencies.

    Fr: array
        Transformed array.
    """

    # FFT specifics
    mu      = order # Order mu of Bessel function
    q       = 0     # Bias exponent: q = 0 is unbiased
    kr      = 1     # Sensible approximate choice of k_c r_c
    kropt   = 1     # Tell fhti to change kr to low-ringing value
    tdir    = -1    # Backward transform
    logkc   = (np.max(np.log10(k))+np.min(np.log10(k)))/2.
    dlogk   = (np.max(np.log10(k))-np.min(np.log10(k)))/N
    dlnk    = dlogk*np.log(10.)
    if   N%2 == 0: nc = N/2
    else         : nc = (N+1)/2
    k_t   = 10.**(logkc + (np.arange(1, N+1) - nc)*dlogk)

    # Initialise function (add k_t extra factor because of log-spacing)
    funct = si.interp1d(k, f, kind = "cubic", bounds_error = False, fill_value = 0.)
    ak    = funct(k_t)*k_t

    # Initialization of transform
    #kr, xsave, ok = fftlog.fhti(N, mu, dlnk, q, kr, kropt)
    fft_obj = fftlog.fhti(N, mu, dlnk, q, kr, kropt)
    kr, xsave = fft_obj[0], fft_obj[1]
    logrc = np.log10(kr) - logkc

    # Transform (dividing by a rr extra factor because of log-spacing)
    Fr = fftlog.fht(ak.copy(), xsave, tdir)
    if   N%2 == 0: rr = 10.**(logrc + (np.arange(N) - nc)*dlogk)
    else         : rr = 10.**(logrc + (np.arange(1,N+1) - nc)*dlogk)
    Fr /= rr

    return rr, Fr

#-----------------------------------------------------------------------------------------
# CORRELATION FUNCTION
#-----------------------------------------------------------------------------------------
def correlation_function(k, pk, N = 4096):
    """
    This routine computes the 2-point correlation function of a field given its power spectrum.


    Parameters
    ----------

    `k`: array
        Abscissae of function, log-spaced.

    `pk`: array
        Power spectrum

    `N`: int, default = 4096
        Number of output points

    Returns
    ----------

    r: array
        Radii.

    xi: array
        Correlation function.
    """
    return iFFT_iso_3D(k, pk, N)

#-----------------------------------------------------------------------------------------
# PROJECTED CORRELATION FUNCTION
#-----------------------------------------------------------------------------------------
def projected_correlation_function(k, pk, N = 4096):
    """
    This routine computes the projected 2-point correlation function of a field given its power spectrum.


    Parameters
    ----------

    `k`: array
        Abscissae of function, log-spaced.

    `pk`: array
        Power spectrum

    `N`: int, default = 4096
        Number of output points

    Returns
    ----------

    rp: array
        Radii.

    wp: array
        Projected correlation function.
    """
    rp,wp = iHankel(k, pk, N = 4096, order = 0.)
    wp /= 2.*np.pi
    return rp,wp


def angular_correlation_function(theta, k, pk, z, chi_z, N_z, H_z, N = 4096):
    """
    This routine computes the angular 2-point correlation function of a field given its power spectrum.


    Parameters
    ----------
    `theta`: array
        Angles in arcminutes.


    `k`: array
        Abscissae of function, log-spaced.

    `pk`: array
        Power spectrum

    `z`: array
        Redshifts at which to integrate.

    `chi_z`: array
        Comoving distances at redshifts of integration.

    `N_z`: array
        Source density at redshifts of integration.

    `H_z`: array
        Hubble parameters (in km/s/(Mpc/h)) at redshifts of integration.

    `N`: int, default = 4096
        Number of output points

    Returns
    ----------

    theta: array
        Angles.

    wt: array
        Angular correlation function.
    """
    # Angles in radians
    theta_rad  = theta/60.*np.pi/180.
    # Expand dims of quantities
    theta_2d   = np.expand_dims(theta_rad,0)
    chi_2d,N_2d,H_2d = np.expand_dims(chi_z,1),np.expand_dims(N_z,1),np.expand_dims(H_z,1)
    # Projected 2PCF
    rt,wt      = iHankel(k,pk,N)
    # Interpolation
    wt_int     = si.interp1d(rt,wt)
    # angular correlation function
    wt         = sint.simps(H_2d/const.c/(2.*np.pi)*N_2d**2.*wt_int(theta_2d*chi_2d),x=z,axis=0)
    return theta, wt



