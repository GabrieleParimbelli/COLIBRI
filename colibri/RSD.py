import colibri.constants as const
import colibri.cosmology as cc
import colibri.halo as hc
import colibri.galaxy as gc
import numpy as np
import scipy
import colibri.useful_functions as UU
import colibri.nonlinear as NL



class RSD(gc.galaxy):
    """
    The ``RSD`` class inherits all the functions from the :func:`colibri.galaxy.galaxy` class and
    therefore from the :func:`colibri.halo.halo` one.
    It computes the galaxy redshift-space power spectrum in the Halo Occupation Distribution (HOD)
    prescription through the Kaiser effect and the dispersion model (both Gaussian and Lorentzian damping
    functions are provided).
    This can be done in multipole space (with Legendre expansion), in the (:math:`k_\parallel, k_\perp`)
    space and also in the (:math:`k,\mu`) space.

    There are 4 models which can be used to compute the power spectrum, tarting from the (:math:`k,\mu`) case

     - `linear`: the linear matter power spectrum is used as a base

     .. math:: 

       P_g^{(s)}(k,\mu) = \left(b+f\mu^2\\right)^2 \ P_{lin}(k) \ \mathcal D(k\mu\sigma_v)

     - `non-linear`:  the non-linear matter power spectrum is used (with Halofit) as a base

     .. math:: 

       P_g^{(s)}(k,\mu) = \left(b+f\mu^2\\right)^2 \ P_{nl}(k) \ \mathcal D(k\mu\sigma_v)

     - `HOD`: the galaxy power spectrum in real space is used (with HOD) as a base

     .. math:: 

       P_g^{(s)}(k,\mu) = \left(1+\\beta\mu^2\\right)^2 \ P_g^{(HOD)}(k) \ \mathcal D(k\mu\sigma_v)

     - `halo model`: the galaxy power spectrum in redshift space is computed directly from HOD, with the damping functions which are applied directly to halo profiles.


    :type z: array
    :param z: Redshifts.

    :param k: Array of scales in :math:`h/\mathrm{Mpc}`.
    :type k: array

    :param mu: Cosines of angle between `k` and the line-of-sight.
    :type mu: array

    :param k_par: Array of scales in :math:`h/\mathrm{Mpc}`.
    :type k_par: array

    :param k_perp: Array of scales in :math:`h/\mathrm{Mpc}`.
    :type k_perp: array

    :param code: Boltzmann solver to compute the linear power spectrum. Choose among `'camb'`, `'class'`, `'eh'` (for Eisenstein-Hu). N.B. If Eisenstein-Hu is selected, effects of massive neutrinos and evolving dark energy cannot be accounted for, as such spectrum is a good approximation for LCDM cosmologies only.
    :type code: string, default = `'camb'`

    :param BAO_smearing: Whether to damp the BAO feature due to non-linearities.
    :type BAO_smearing: boolean, default = True

    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``

    :return: The initialization automatically calls a ``galaxy`` instance (see :func:`colibri.galaxy.galaxy`) and therefore all the quantities described there are also available here. Also, the key ``['galaxies']['redshift space']`` is added to the ``self.Pk`` dictionary.
    """
    #-----------------------------------------------------------------------------------------
    # INITIALIZATION FUNCTION
    #-----------------------------------------------------------------------------------------
    def __init__(self,
                 z,
                 k,
                 mu,
                 k_par,
                 k_perp,
                 code = 'camb',
                 BAO_smearing = False,
                 cosmology = cc.cosmo()):

        # Initialize galaxy parameters
        gc.galaxy.__init__(self,
                           z = z,
                           k = k,
                           code = code,
                           BAO_smearing = BAO_smearing,
                           cosmology = cosmology)


        # Initialize RSD parameters
        self.mu      = mu
        self.k_par   = k_par
        self.k_perp  = k_perp

        self.nmu     = len(np.atleast_1d(mu))
        self.nk_par  = len(np.atleast_1d(k_par))
        self.nk_perp = len(np.atleast_1d(k_perp))

        # Initialize RSD galaxy power spectrum
        self.Pk['galaxies']['redshift space'] = {}

    #-----------------------------------------------------------------------------------------
    # DAMPING FUNCTIONS
    #-----------------------------------------------------------------------------------------
    def damping(self, x, kind = 'Lorentzian'):
        """
        Damping in power spectrum due to random motions of galaxies inside halos.

        :param x: Abscissa.
        :type x: array

        :param kind: Kind of damping function, choose between `Gaussian` and `Lorentzian`.
        :type kind: string, default = `Lorentzian`

        :return: array
        """
        if kind == 'Lorentzian': return 1./(1.+x**2./2.)
        elif kind == 'Gaussian': return np.exp(-x**2./2.)
        else:                    raise ValueError("Damping function for Fingers of God not known. Choose between 'Gaussian' and 'Lorentzian'.")

    #-----------------------------------------------------------------------------------------
    # GALAXY REDSHIFT-SPACE POWER SPECTRUM
    #-----------------------------------------------------------------------------------------
    def galaxy_RSD_Pk(self,
                      model = 'nonlinear',
                      bias = 1.,
                      growth_rate = 0.5,
                      velocity_dispersion = 300.,
                      fingers_of_god = 'Lorentzian',
                      kind_central = None,
                      kwargs_central = {},
                      kind_satellite = None,
                      kwargs_satellite = {},
                      kwargs_mass_function = {},
                      kwargs_concentration = {}):
        """
        Galaxy power spectrum in redshift-space as function of (:math:`k,\mu`)


        :param model: Model of computation of the power spectrum. Accepted values are:

            - [`linear`, `Linear`, `lin`, `l`, `L`] for the linear power spectrum
            - [`non-linear', `nonlinear`, `NL`, `nl`] for the non-linear power spectrum (with Halofit)
            - [`HOD`, `hod`] to use the HOD for the galaxy real-space power spectrum, 
            - [`halomodel`, `halo model`, `halo`, `H`] to compute directly the halo model redshift-space
              power spectrum.
        :type model: string, default = `nonlinear`

        :type bias: array of size ``len(z)``, default = 1.
        :param bias: Array of galaxy biases (used only if linear or non-linear models are used).

        :type growth_rate: array of size ``len(z)``, default = 0.5
        :param growth_rate: Array of growth rates, :math:`f = \\frac{d\ln D}{d\ln a}`.

        :type velocity_dispersion: array of size ``len(z)``, default = 300.
        :param velocity_dispersion: Array of average velocity dispersion of galaxies in halos, in :math:`km/s`.

        :type fingers_of_god: string, default = 'Lorentzian'
        :param fingers_of_god: Kind of damping function, choose between 'Gaussian' and 'Lorentzian'.

        :type kind_central: callable, default = None
        :param kind_central: First argument must be mass (in :math:`M_\odot/h`), other specified by kwargs_central.

        :type kind_satellite: callable, default = None
        :param kind_satellite: First argument must be mass (in :math:`M_\odot/h`), other specified by kwargs_satellite.

        :type kwargs_central: dictionary
        :param  kwargs_central: Every key is a keyword parameter for the galaxy distribution and each value is a list of length equal to the size of the redshift required.

        :type kwargs_satellite: dictionary
        :param kwargs_satellite: Every key is a keyword parameter for the galaxy distribution and each value is a list of length equal to the size of the redshift required.

        :param kwargs_mass_function: Keyword arguments to pass to :func:`colibri.halo.halo.mass_fun_ST`.
        :type kwargs_mass_function: dictionary, default = {}

        :param kwargs_concentration: Keyword arguments to pass to :func:`colibri.halo.halo.conc`.
        :type kwargs_concentration: dictionary, default = {}

        :return: Nothing, but the following keys are added to the ``self.Pk`` dictionary

         - ``['galaxies']['redshift space']['k-mu']`` (`3D array of shape` ``(len(z), len(k), len(mu))`` ) - 1-halo term of the matter power spectrum
        """

        # List different models:
        linear_model    = ['linear', 'Linear', 'lin', 'l', 'L']
        nonlinear_model = ['non-linear', 'nonlinear', 'NL', 'nl']
        hod_model       = ['HOD', 'hod']
        halo_model      = ['halomodel', 'halo model', 'halo', 'H']

        # Vectorize quantities
        bias        = np.atleast_1d(bias)
        growth_rate = np.atleast_1d(growth_rate)
        sigma       = np.atleast_1d(velocity_dispersion)*(1.+self.z)/self.cosmology.H(self.z)/self.cosmology.h

        # Assertions for shapes
        if model in hod_model + halo_model:
            things_to_check = [growth_rate, sigma]
        else:
            things_to_check = [bias, growth_rate, sigma]
        assert all(X.shape == np.atleast_1d(self.nz) for X in things_to_check), "Parameters of wrong shape"

        # Create matrices of bias, growth_rate, fingers_of_god damping factor
        ZZ, KK, MU   = np.array(np.meshgrid(self.z, self.k, self.mu, indexing = 'ij'))
        GROWTH_RATE  = np.transpose(np.tile(growth_rate, (self.nk, self.nmu)).reshape((self.nmu, self.nk, self.nz)), (2,1,0))
        SIGMA        = np.transpose(np.tile(sigma      , (self.nk, self.nmu)).reshape((self.nmu, self.nk, self.nz)), (2,1,0))
        FoG          = self.damping(x = (1.+ZZ)*SIGMA*KK*MU, kind = fingers_of_god)
        P_K_MU       = np.zeros_like(ZZ)

        # Differentiate models
        # 'linear': Kaiser + linear power spectrum + FoG
        if model in linear_model:
            # Compute galaxy bias
            BIAS = np.transpose(np.tile(bias, (self.nk, self.nmu)).reshape((self.nmu, self.nk, self.nz)), (2,1,0))
            # Take linear matter power spectrum
            PK_BASE = np.transpose(np.tile(self.Pk['matter']['linear'],self.nmu).reshape((self.nz, self.nmu, self.nk)), (0,2,1))
            # Redshift-space galaxy power spectrum 
            P_K_MU = (BIAS + GROWTH_RATE*MU**2.)**2.*PK_BASE*FoG

        # 'nonlinear': Kaiser + non-linear power spectrum (Halofit) + FoG
        elif model in nonlinear_model:
            if self.k.min() > 0.001 or self.k.max() < 10. or len(self.k)<100:
                raise ValueError("To use this model you should set a self.k array such that min(k) <= 0.001, max(k) >= 10 and with length greater than 100")
            # Galaxy bias
            BIAS = np.transpose(np.tile(bias, (self.nk, self.nmu)).reshape((self.nmu, self.nk, self.nz)), (2,1,0))
            # Non-linear galaxy power spectrum
            pk_ext = []
            for iz in range(len(np.atleast_1d(self.z))):
                k_ext, pk_tmp = UU.extrapolate_log(self.k,self.Pk['matter']['linear'][iz],self.k.min(),1e4)
                pk_ext.append(pk_tmp)
            pk_ext = np.array(pk_ext)
            do_nonlinear = NL.HMcode2016(z = self.z, k = k_ext, pk = pk_ext, cosmology = self.cosmology)
            pk_hf        = do_nonlinear.pk_nl
            pk_hf_int    = scipy.interpolate.interp1d(k_ext,pk_hf,'cubic')
            pk_hf        = pk_hf_int(self.k)
            PK_BASE = np.transpose(np.tile(pk_hf,self.nmu).reshape((self.nz, self.nmu, self.nk)), (0,2,1))
            # Redshift-space galaxy power spectrum
            P_K_MU = (BIAS + GROWTH_RATE*MU**2.)**2.*FoG*PK_BASE

        # 'HOD': compute galaxy real space power spectrum with HOD, then add dispersion model
        elif model in hod_model:
            if self.k.min() > 0.05: raise ValueError("To use this model you should set a self.k array such that min(k) <= 0.05 for a correct determination of galaxy bias.")
            # Load HOD
            self.load_HOD(kind_central     = kind_central,
                          kwargs_central   = kwargs_central,
                          kind_satellite   = kind_satellite,
                          kwargs_satellite = kwargs_satellite)
            # Load galaxy and halo power spectra
            self.galaxy_Pk(kwargs_mass_function = kwargs_mass_function, kwargs_concentration = kwargs_concentration)
            self.halo_Pk(kwargs_mass_function = kwargs_mass_function, kwargs_concentration = kwargs_concentration)
            # Compute galaxy bias as function of scale
            bias = (self.Pk['galaxies']['real space']['total halo']/self.Pk['matter']['total halo'])**.5
            # Take first value (at k < 0.05 h/Mpc is almost a constant) and tile it
            BIAS = np.transpose(np.tile(bias[:,0], (self.nk, self.nmu)).reshape((self.nmu, self.nk, self.nz)), (2,1,0))
            # Non-linear galaxy power spectrum
            PK_BASE = np.transpose(np.tile(self.Pk['galaxies']['real space']['total halo'],self.nmu).reshape((self.nz, self.nmu, self.nk)), (0,2,1))
            # Redshift-space galaxy power spectrum (here, since bias is already included in P_g(k), I do (1+beta*mu^2)
            P_K_MU = (1. + GROWTH_RATE/BIAS*MU**2.)**2.*FoG*PK_BASE

        # 'HOD': compute galaxy real space power spectrum with HOD, then add dispersion model
        elif model in halo_model:
            if self.k.min() > 0.05: raise ValueError("To use this model you should set a self.k array such that min(k) <= 0.05 for a correct determination of galaxy bias.")
            # Load HOD
            self.load_HOD(kind_central     = kind_central,
                          kwargs_central   = kwargs_central,
                          kind_satellite   = kind_satellite,
                          kwargs_satellite = kwargs_satellite)
            # Masses
            M       = self.mass
            dlnM    = np.log(M[1]/M[0])
            # Scale radii
            R_s     = self.R_s(M)
            # Halo mass function
            dndM    = self.load_halo_mass_function(**kwargs_mass_function)
            # HOD galaxies
            n_avg   = self.average_galaxy_density(**kwargs_mass_function)
            Nc      = self.Ncen
            Ns      = self.Nsat
            # Compute galaxy bias
            nu      = self.peak_height    
            bias    = self.halo_bias_ST(nu, **kwargs_mass_function)
            BIAS    = np.transpose(np.tile(bias[:,0], (self.nk, self.nmu)).reshape((self.nmu, self.nk, self.nz)), (2,1,0))
            # Concentratiom
            CONC    = self.conc(M, **kwargs_concentration)
            # NFW transforms
            UNFW = np.zeros((self.nz, self.nm, self.nk))
            for ik in range(self.nk):
                UNFW[:, :, ik] = self.u_NFW(CONC, self.k[ik]*R_s)
            # FoG terms for central-satellite and satellite-satellite
            PCS  = np.zeros((self.nz, self.nm, self.nk, self.nmu))
            PSS  = np.zeros((self.nz, self.nm, self.nk, self.nmu))
            if fingers_of_god == 'Gaussian':
                for im in range(self.nm):
                    for imu in range(self.nmu):
                        PCS[:, im, :, imu] = UNFW[:, im, :]*self.damping(x = ((1.+ZZ)*SIGMA*KK*MU)[:,:,imu], kind = fingers_of_god)
                PSS = PCS**2.
            elif fingers_of_god == 'Lorentzian':
                for im in range(self.nm):
                    for imu in range(self.nmu):
                        PCS[:, im, :, imu] = UNFW[:, im, :]*self.damping(x =             ((1.+ZZ)*SIGMA*KK*MU)[:,:,imu], kind = fingers_of_god)
                        PSS[:, im, :, imu] = UNFW[:, im, :]*self.damping(x = np.sqrt(2.)*((1.+ZZ)*SIGMA*KK*MU)[:,:,imu], kind = fingers_of_god)
            else:
                raise ValueError("Damping function for Fingers of God not known.")

            # Linear matter power spectrum
            PK_BASE = np.transpose(np.tile(self.Pk['matter']['linear'],self.nmu).reshape((self.nz, self.nmu, self.nk)), (0,2,1))

            # normalization and scale of cutoff
            normalization = self.norm_2h()
            k_star        = 0.01*(1.+self.z)

            # Filling power spectrum array
            P_K_MU = np.zeros_like(PK_BASE)

            for iz in range(self.nz):
                for ik in range(self.nk):
                    for im in range(self.nmu):
                        integrand_1h       = (1./n_avg[iz]**2.*dndM[iz]*(2.*Nc[iz]*Ns[iz]*PCS[iz,:,ik,im]+Ns[iz]**2.*PSS[iz,:,ik,im]))*M
                        integrand_2h       = (1./n_avg[iz]*dndM[iz]*(bias[iz] + growth_rate[iz]*self.mu[im]**2.)*(Nc[iz]+Ns[iz]*PCS[iz,:,ik,im]))*M
                        P_g_1h             = np.trapz(integrand_1h, dx = dlnM)*(1-np.exp(-(self.k[ik]/k_star[iz])**2.))
                        P_g_2h             = np.trapz(integrand_2h, dx = dlnM)**2.*normalization[iz]**2*PK_BASE[iz,ik,im]
                        P_K_MU[iz, ik, im] = P_g_1h + P_g_2h
            del PCS, PSS, CONC
        else:
            raise ValueError("Model for RSD not known")

        # Fill the dictionary
        self.Pk['galaxies']['redshift space']['k-mu'] = P_K_MU    

        # Delete quantities that occupy memory
        del P_K_MU, BIAS, GROWTH_RATE, SIGMA, FoG, PK_BASE



    #-----------------------------------------------------------------------------------------
    # GALAXY REDSHIFT-SPACE POWER SPECTRUM MULTIPOLES
    #-----------------------------------------------------------------------------------------
    def galaxy_RSD_Pk_multipoles(self,
                                 l = [0,2,4],
                                 model = 'nonlinear',
                                 bias = 1.,
                                 growth_rate = 0.5,
                                 velocity_dispersion = 300.,
                                 fingers_of_god = 'Lorentzian',
                                 kind_central = None,
                                 kwargs_central = {},
                                 kind_satellite = None,
                                 kwargs_satellite = {},
                                 kwargs_mass_function = {},
                                 kwargs_concentration = {}):
        """
        Galaxy power spectrum multipoles in redshift-space as function of k

        :param l: List of multipoles to compute.
        :type l: list/array of integers, default = [0,2,4]

        :param model: Model of computation of the power spectrum. Accepted values are:

            - [`linear`, `Linear`, `lin`, `l`, `L`] for the linear power spectrum
            - [`non-linear', `nonlinear`, `NL`, `nl`] for the non-linear power spectrum (with Halofit)
            - [`HOD`, `hod`] to use the HOD for the galaxy real-space power spectrum, 
            - [`halomodel`, `halo model`, `halo`, `H`] to compute directly the halo model redshift-space power spectrum.
        :type model: string, default = `nonlinear`

        :type bias: array of size ``len(z)``, default = 1.
        :param bias: Array of galaxy biases (used only if linear or non-linear models are used).

        :type growth_rate: array of size ``len(z)``, default = 0.5
        :param growth_rate: Array of growth rates, :math:`f = \\frac{d\ln D}{d\ln a}`.

        :type velocity_dispersion: array of size ``len(z)``, default = 300.
        :param velocity_dispersion: Array of average velocity dispersion of galaxies in halos, in :math:`km/s`.

        :type fingers_of_god: string, default = 'Lorentzian'
        :param fingers_of_god: Kind of damping function, choose between 'Gaussian' and 'Lorentzian'.

        :type kind_central: callable, default = None
        :param kind_central: First argument must be mass (in :math:`M_\odot/h`), other specified by kwargs_central.

        :type kind_satellite: callable, default = None
        :param kind_satellite: First argument must be mass (in :math:`M_\odot/h`), other specified by kwargs_satellite.

        :type kwargs_central: dictionary
        :param  kwargs_central: Every key is a keyword parameter for the galaxy distribution and each value is a list of length equal to the size of the redshift required.

        :type kwargs_satellite: dictionary
        :param kwargs_satellite: Every key is a keyword parameter for the galaxy distribution and each value is a list of length equal to the size of the redshift required.

        :param kwargs_mass_function: Keyword arguments to pass to :func:`colibri.halo.halo.mass_fun_ST`.
        :type kwargs_mass_function: dictionary, default = {}

        :param kwargs_concentration: Keyword arguments to pass to :func:`colibri.halo.halo.conc`.
        :type kwargs_concentration: dictionary, default = {}

        :return: Nothing, but the key ``['galaxies']['redshift space']['multipoles']`` is added to the ``self.Pk`` dictionary. This in turn is a new dictionary whose keys are ``['number of multiple']``. Each of these new keys is a 2D array of shape ``(len(z), len(k))``

        """
        # List different models:
        linear_model    = ['linear', 'Linear', 'lin', 'l', 'L']
        nonlinear_model = ['non-linear', 'nonlinear', 'NL', 'nl']
        hod_model       = ['HOD', 'hod']
        halo_model      = ['halomodel', 'halo model', 'halo', 'H']

        # Vectorize quantities
        bias        = np.atleast_1d(bias)
        growth_rate = np.atleast_1d(growth_rate)
        sigma       = np.atleast_1d(velocity_dispersion)*(1.+self.z)/self.cosmology.H(self.z)/self.cosmology.h

        # Assertions for shapes
        if model in hod_model + halo_model:
            things_to_check = [growth_rate, sigma]
        else:
            things_to_check = [bias, growth_rate, sigma]
        assert all(X.shape == np.atleast_1d(self.nz) for X in things_to_check), "Parameters of wrong shape"

        # Create matrices of bias, growth_rate, fingers_of_god damping factor
        mu           = np.linspace(-1., 1., 201)        # Dont' put from 0 to 1, because if e.g. l=1 is asked it must come = 0.
        nmu          = len(mu)
        ZZ, KK, MU   = np.array(np.meshgrid(self.z, self.k, mu, indexing = 'ij'))
        GROWTH_RATE  = np.transpose(np.tile(growth_rate, (self.nk, nmu)).reshape((nmu, self.nk, self.nz)), (2,1,0))
        SIGMA        = np.transpose(np.tile(sigma      , (self.nk, nmu)).reshape((nmu, self.nk, self.nz)), (2,1,0))
        FoG          = self.damping(x = (1.+ZZ)*SIGMA*KK*MU, kind = fingers_of_god)
        P_K_MU       = np.zeros_like(ZZ)

        # Differentiate models
        # 'linear': Kaiser + linear power spectrum + FoG
        if model in linear_model:
            # Compute galaxy bias
            BIAS = np.transpose(np.tile(bias, (self.nk, nmu)).reshape((nmu, self.nk, self.nz)), (2,1,0))
            # Take linear matter power spectrum
            PK_BASE = np.transpose(np.tile(self.Pk['matter']['linear'],nmu).reshape((self.nz, nmu, self.nk)), (0,2,1))
            # Redshift-space galaxy power spectrum 
            P_K_MU = (BIAS + GROWTH_RATE*MU**2.)**2.*FoG*PK_BASE

        # 'nonlinear': Kaiser + non-linear power spectrum (Halofit) + FoG
        elif model in nonlinear_model:
            if self.k.min() > 0.001 or self.k.max() < 10. or len(self.k)<100:
                raise ValueError("To use this model you should set a self.k array such that min(k) <= 0.001, max(k) >= 10 and with length greater than 100")
            # Galaxy bias
            BIAS = np.transpose(np.tile(bias, (self.nk, nmu)).reshape((nmu, self.nk, self.nz)), (2,1,0))
            # Non-linear galaxy power spectrum
            pk_ext = []
            for iz in range(len(np.atleast_1d(self.z))):
                k_ext, pk_tmp = UU.extrapolate_log(self.k,self.Pk['matter']['linear'][iz],self.k.min(),1e4)
                pk_ext.append(pk_tmp)
            pk_ext = np.array(pk_ext)
            do_nonlinear = NL.HMcode2016(z = self.z, k = k_ext, pk = pk_ext, cosmology = self.cosmology)
            pk_hf        = do_nonlinear.pk_nl
            pk_hf_int    = scipy.interpolate.interp1d(k_ext,pk_hf,'cubic')
            pk_hf        = pk_hf_int(self.k)
            PK_BASE = np.transpose(np.tile(pk_hf,nmu).reshape((self.nz, nmu, self.nk)), (0,2,1))
            # Redshift-space galaxy power spectrum
            P_K_MU = (BIAS + GROWTH_RATE*MU**2.)**2.*FoG*PK_BASE

        # 'HOD': compute galaxy real space power spectrum with HOD, then add dispersion model
        elif model in hod_model:
            if self.k.min() > 0.05: raise ValueError("To use this model you should set a self.k array such that min(k) <= 0.05 for a correct determination of galaxy bias.")
            # Load HOD
            self.load_HOD(kind_central     = kind_central,
                          kwargs_central   = kwargs_central,
                          kind_satellite   = kind_satellite,
                          kwargs_satellite = kwargs_satellite)
            # Load galaxy and halo power spectra
            self.galaxy_Pk(kwargs_mass_function = kwargs_mass_function, kwargs_concentration = kwargs_concentration)
            self.halo_Pk(kwargs_mass_function = kwargs_mass_function, kwargs_concentration = kwargs_concentration)
            # Compute galaxy bias as function of scale
            bias = (self.Pk['galaxies']['real space']['total halo']/self.Pk['matter']['total halo'])**.5
            # Take first value (at k < 0.05 h/Mpc is almost a constant) and tile it
            BIAS = np.transpose(np.tile(bias[:,0], (self.nk, nmu)).reshape((nmu, self.nk, self.nz)), (2,1,0))
            # Non-linear galaxy power spectrum
            PK_BASE = np.transpose(np.tile(self.Pk['galaxies']['real space']['total halo'],nmu).reshape((self.nz, nmu, self.nk)), (0,2,1))
            # Redshift-space galaxy power spectrum (here, since bias is already included in P_g(k), I do (1+beta*mu^2)
            P_K_MU = (1. + GROWTH_RATE/BIAS*MU**2.)**2.*FoG*PK_BASE

        # 'HOD': compute galaxy real space power spectrum with HOD, then add dispersion model
        elif model in halo_model:
            if self.k.min() > 0.05: raise ValueError("To use this model you should set a self.k array such that min(k) <= 0.05 for a correct determination of galaxy bias.")
            # Load HOD
            self.load_HOD(kind_central     = kind_central,
                          kwargs_central   = kwargs_central,
                          kind_satellite   = kind_satellite,
                          kwargs_satellite = kwargs_satellite)
            # Masses
            M       = self.mass
            dlnM    = np.log(M[1]/M[0])
            # Scale radii
            R_s     = self.R_s(M)
            # Halo mass function
            dndM    = self.load_halo_mass_function(**kwargs_mass_function)
            # HOD galaxies
            n_avg   = self.average_galaxy_density(**kwargs_mass_function)
            Nc      = self.Ncen
            Ns      = self.Nsat
            # Compute galaxy bias
            nu      = self.peak_height    
            bias    = self.halo_bias_ST(nu, **kwargs_mass_function)
            BIAS    = np.transpose(np.tile(bias[:,0], (self.nk, nmu)).reshape((nmu, self.nk, self.nz)), (2,1,0))
            # Concentratiom
            CONC    = self.conc(M, **kwargs_concentration)
            # NFW transforms
            UNFW = np.zeros((self.nz, self.nm, self.nk))
            for ik in range(self.nk):
                UNFW[:, :, ik] = self.u_NFW(CONC, self.k[ik]*R_s)
            # FoG terms for central-satellite and satellite-satellite
            PCS     = np.zeros((self.nz, self.nm, self.nk, nmu))
            PSS     = np.zeros((self.nz, self.nm, self.nk, nmu))
            DAMP_CS = self.damping(x = ((1.+ZZ)*SIGMA*KK*MU), kind = fingers_of_god)
            if fingers_of_god == 'Gaussian':
                for im in range(self.nm):
                    for imu in range(nmu):
                        PCS[:, im, :, imu] = UNFW[:, im, :]*DAMP_CS[:,:,imu]
                PSS = PCS**2.
            elif fingers_of_god == 'Lorentzian':
                DAMP_SS = self.damping(x = np.sqrt(2.)*((1.+ZZ)*SIGMA*KK*MU), kind = fingers_of_god)
                for im in range(self.nm):
                    for imu in range(nmu):
                        PCS[:, im, :, imu] = UNFW[:, im, :]*DAMP_CS[:,:,imu]
                        PSS[:, im, :, imu] = UNFW[:, im, :]*DAMP_SS[:,:,imu]
            else:
                raise ValueError("Damping function for Fingers of God not known.")

            # Linear matter power spectrum
            PK_BASE = np.transpose(np.tile(self.Pk['matter']['linear'],nmu).reshape((self.nz, nmu, self.nk)), (0,2,1))

            # normalization and scale of cutoff
            normalization = self.norm_2h()
            k_star        = 0.01*(1.+self.z)

            # Filling power spectrum array
            P_K_MU = np.zeros_like(PK_BASE)

            for iz in range(self.nz):
                for ik in range(self.nk):
                    for imu in range(nmu):
                        integrand_1h        = (1./n_avg[iz]**2.*dndM[iz]*(2.*Nc[iz]*Ns[iz]*PCS[iz,:,ik,imu]+Ns[iz]**2.*PSS[iz,:,ik,imu]))*M
                        integrand_2h        = (1./n_avg[iz]*dndM[iz]*(bias[iz] + growth_rate[iz]*mu[imu]**2.)*(Nc[iz]+Ns[iz]*PCS[iz,:,ik,imu]))*M
                        P_g_1h              = np.trapz(integrand_1h, dx = dlnM)*(1-np.exp(-(self.k[ik]/k_star[iz])**2.))
                        P_g_2h              = np.trapz(integrand_2h, dx = dlnM)**2.*normalization[iz]**2*PK_BASE[iz,ik,imu]
                        P_K_MU[iz, ik, imu] = P_g_1h + P_g_2h
            del PCS, PSS, CONC
        else:
            raise ValueError("Model for RSD not known")

        # Set multipoles        
        l = np.atleast_1d(l)
        nl = len(l)

        # Use P(k, mu) as a proxy
        k    = self.k
    
        P_ell = np.zeros((self.nz, nl, self.nk))
        self.Pk['galaxies']['redshift space']['multipoles'] = {}

        for il in range(nl):
            leg = scipy.special.legendre(l[il])(mu)
            self.Pk['galaxies']['redshift space']['multipoles'][str(l[il])] = np.zeros((self.nz, self.nk))
            for iz in range(self.nz):
                for ik in range(self.nk):
                    P_ell[iz,il,ik] = (2.*l[il]+1.)/2.*scipy.integrate.simps(P_K_MU[iz,ik,:]*leg, x = mu)
                self.Pk['galaxies']['redshift space']['multipoles'][str(l[il])][iz] = P_ell[iz,il]
        del P_K_MU, BIAS, GROWTH_RATE, SIGMA, FoG, PK_BASE, P_ell



    #-----------------------------------------------------------------------------------------
    # GALAXY REDSHIFT-SPACE POWER SPECTRUM PARALLEL AND PERPENDICULAR
    #-----------------------------------------------------------------------------------------
    def galaxy_RSD_Pk_2D(self,
                         model = 'nonlinear',
                         bias = 1.,
                         growth_rate = 0.5,
                         velocity_dispersion = 300.,
                          fingers_of_god = 'Lorentzian',
                         kind_central = None,
                         kwargs_central = {},
                         kind_satellite = None,
                         kwargs_satellite = {},
                         kwargs_mass_function = {},
                         kwargs_concentration = {}):
        """
        Galaxy power spectrum multipoles in redshift-space as function of wavevectors
        parallel and perpendicular to the line of sight.

        :param model: Model of computation of the power spectrum. Accepted values are:

            - [`linear`, `Linear`, `lin`, `l`, `L`] for the linear power spectrum
            - [`non-linear', `nonlinear`, `NL`, `nl`] for the non-linear power spectrum (with Halofit)
            - [`HOD`, `hod`] to use the HOD for the galaxy real-space power spectrum, 
            - [`halomodel`, `halo model`, `halo`, `H`] to compute directly the halo model redshift-space power spectrum.
        :type model: string, default = `nonlinear`

        :type bias: array of size ``len(z)``, default = 1.
        :param bias: Array of galaxy biases (used only if linear or non-linear models are used).

        :type growth_rate: array of size ``len(z)``, default = 0.5
        :param growth_rate: Array of growth rates, :math:`f = \\frac{d\ln D}{d\ln a}`.

        :type velocity_dispersion: array of size ``len(z)``, default = 300.
        :param velocity_dispersion: Array of average velocity dispersion of galaxies in halos, in :math:`km/s`.

        :type fingers_of_god: string, default = 'Lorentzian'
        :param fingers_of_god: Kind of damping function, choose between 'Gaussian' and 'Lorentzian'.

        :type kind_central: callable, default = None
        :param kind_central: First argument must be mass (in :math:`M_\odot/h`), other specified by kwargs_central.

        :type kind_satellite: callable, default = None
        :param kind_satellite: First argument must be mass (in :math:`M_\odot/h`), other specified by kwargs_satellite.

        :type kwargs_central: dictionary
        :param  kwargs_central: Every key is a keyword parameter for the galaxy distribution and each value is a list of length equal to the size of the redshift required.

        :type kwargs_satellite: dictionary
        :param kwargs_satellite: Every key is a keyword parameter for the galaxy distribution and each value is a list of length equal to the size of the redshift required.

        :param kwargs_mass_function: Keyword arguments to pass to :func:`colibri.halo.halo.mass_fun_ST`.
        :type kwargs_mass_function: dictionary, default = {}

        :param kwargs_concentration: Keyword arguments to pass to :func:`colibri.halo.halo.conc`.
        :type kwargs_concentration: dictionary, default = {}

        :return: Nothing, but the following keys are added to the ``self.Pk`` dictionary

         - ``['galaxies']['redshift space']['k_par-k_perp']`` (`3D array of shape` ``(len(z), len(k_par), len(k_perp))`` ) - 1-halo term of the matter power spectrum

        """


        # List different models:
        linear_model    = ['linear', 'Linear', 'lin', 'l', 'L']
        nonlinear_model = ['non-linear', 'nonlinear', 'NL', 'nl']
        hod_model       = ['HOD', 'hod']
        halo_model      = ['halomodel', 'halo model', 'halo', 'H']

        # Vectorize quantities
        bias        = np.atleast_1d(bias)
        growth_rate = np.atleast_1d(growth_rate)
        sigma       = np.atleast_1d(velocity_dispersion)*(1.+self.z)/self.cosmology.H(self.z)/self.cosmology.h


        # Assertions for shapes
        if model in hod_model + halo_model:
            things_to_check = [growth_rate, sigma]
        else:
            things_to_check = [bias, growth_rate, sigma]
        assert all(X.shape == np.atleast_1d(self.nz) for X in things_to_check), "Parameters of wrong shape"

        # Create matrices of bias, growth_rate, fingers_of_god damping factor
        ZZ, KPAR, KPERP = np.array(np.meshgrid(self.z, self.k_par, self.k_perp, indexing = 'ij'))
        KK              = np.sqrt(KPAR**2.+KPERP**2.)
        MU              = KPAR*1./KK
        GROWTH_RATE     = np.transpose(np.tile(growth_rate, (self.nk_par, self.nk_perp)).reshape((self.nk_perp, self.nk_par, self.nz)), (2,1,0))
        SIGMA           = np.transpose(np.tile(sigma      , (self.nk_par, self.nk_perp)).reshape((self.nk_perp, self.nk_par, self.nz)), (2,1,0))
        FoG             = self.damping(x = (1.+ZZ)*SIGMA*KPAR, kind = fingers_of_god)
        P_KPAR_KPERP    = np.zeros_like(ZZ)

        # Differentiate models
        # 'linear': Kaiser + linear power spectrum + FoG
        if model in linear_model:
            # Compute galaxy bias
            BIAS = np.transpose(np.tile(bias, (self.nk_par, self.nk_perp)).reshape((self.nk_perp, self.nk_par, self.nz)), (2,1,0))
            # Take linear matter power spectrum
            PK_BASE = np.zeros_like(ZZ)
            for iz in range(self.nz):
                power_interp = scipy.interpolate.interp1d(self.k, self.Pk['matter']['linear'][iz], kind = 'cubic')
                PK_BASE[iz] = power_interp(KK[iz])
            # Redshift-space galaxy power spectrum 
            P_KPAR_KPERP = (BIAS + GROWTH_RATE*MU**2.)**2.*FoG*PK_BASE


        # 'nonlinear': Kaiser + non-linear power spectrum (Halofit) + FoG
        elif model in nonlinear_model:
            if self.k.min() > 0.001 or self.k.max() < 10. or len(self.k)<100:
                raise ValueError("To use this model you should set a self.k array such that min(k) <= 0.001, max(k) >= 10 and with length greater than 100")
            # Galaxy bias
            BIAS = np.transpose(np.tile(bias, (self.nk_par, self.nk_perp)).reshape((self.nk_perp, self.nk_par, self.nz)), (2,1,0))
            # Non-linear galaxy power spectrum
            pk_ext = []
            for iz in range(len(np.atleast_1d(self.z))):
                k_ext, pk_tmp = UU.extrapolate_log(self.k,self.Pk['matter']['linear'][iz],self.k.min(),1e4)
                pk_ext.append(pk_tmp)
            pk_ext = np.array(pk_ext)
            do_nonlinear = NL.HMcode2016(z = self.z, k = k_ext, pk = pk_ext, cosmology = self.cosmology)
            pk_hf        = do_nonlinear.pk_nl
            pk_hf_int    = scipy.interpolate.interp1d(k_ext,pk_hf,'cubic')
            pk_hf        = pk_hf_int(self.k)
            PK_BASE      = np.zeros_like(ZZ)
            for iz in range(self.nz):
                power_interp = scipy.interpolate.interp1d(self.k, pk_hf[iz], kind = 'cubic')
                PK_BASE[iz]  = power_interp(KK[iz])
            # Redshift-space galaxy power spectrum
            P_KPAR_KPERP = (BIAS + GROWTH_RATE*MU**2.)**2.*FoG*PK_BASE

        # 'HOD': compute galaxy real space power spectrum with HOD, then add dispersion model
        elif model in hod_model:
            if self.k.min() > 0.05: raise ValueError("To use this model you should set a self.k array such that min(k) <= 0.05 for a correct determination of galaxy bias.")
            # Load HOD
            self.load_HOD(kind_central     = kind_central,
                          kwargs_central   = kwargs_central,
                          kind_satellite   = kind_satellite,
                          kwargs_satellite = kwargs_satellite)
            # Load galaxy and halo power spectra
            self.galaxy_Pk(kwargs_mass_function = kwargs_mass_function, kwargs_concentration = kwargs_concentration)
            self.halo_Pk(kwargs_mass_function = kwargs_mass_function, kwargs_concentration = kwargs_concentration)
            # Compute galaxy bias as function of scale
            bias = (self.Pk['galaxies']['real space']['total halo']/self.Pk['matter']['total halo'])**.5
            # Take first value (at k < 0.05 h/Mpc is almost a constant) and tile it
            BIAS = np.transpose(np.tile(bias[:,0], (self.nk_par, self.nk_perp)).reshape((self.nk_perp, self.nk_par, self.nz)), (2,1,0))
            # Non-linear galaxy power spectrum
            PK_BASE = np.zeros_like(ZZ)
            for iz in range(self.nz):
                power_interp = scipy.interpolate.interp1d(self.k, self.Pk['galaxies']['real space']['total halo'][iz], kind = 'cubic')
                PK_BASE[iz]  = power_interp(KK[iz])
            # Redshift-space galaxy power spectrum (here, since bias is already included in P_g(k), I do (1+beta*mu^2)
            P_KPAR_KPERP = (1. + GROWTH_RATE/BIAS*MU**2.)**2.*FoG*PK_BASE

        # 'HOD': compute galaxy real space power spectrum with HOD, then add dispersion model
        elif model in halo_model:
            if self.k.min() > 0.05: raise ValueError("To use this model you should set a self.k array such that min(k) <= 0.05 for a correct determination of galaxy bias.")
            # Load HOD
            self.load_HOD(kind_central     = kind_central,
                          kwargs_central   = kwargs_central,
                          kind_satellite   = kind_satellite,
                          kwargs_satellite = kwargs_satellite)
            # Masses
            M       = self.mass
            dlnM    = np.log(M[1]/M[0])
            # Scale radii
            R_s     = self.R_s(M)
            # Halo mass function
            dndM    = self.load_halo_mass_function(**kwargs_mass_function)
            # HOD galaxies
            n_avg   = self.average_galaxy_density(**kwargs_mass_function)
            Nc      = self.Ncen
            Ns      = self.Nsat
            # Compute galaxy bias
            nu      = self.peak_height    
            bias    = self.halo_bias_ST(nu, **kwargs_mass_function)
            BIAS    = np.transpose(np.tile(bias[:,0], (self.nk_par, self.nk_perp)).reshape((self.nk_perp, self.nk_par, self.nz)), (2,1,0))
            # Concentratiom
            CONC    = self.conc(M, **kwargs_concentration)
            # NFW transforms
            UNFW = np.zeros((self.nz, self.nm, self.nk_par))
            for ik in range(self.nk_par):
                UNFW[:, :, ik] = self.u_NFW(CONC, self.k[ik]*R_s)
            # FoG terms for central-satellite and satellite-satellite
            PCS  = np.zeros((self.nz, self.nm, self.nk_par, self.nk_perp))
            PSS  = np.zeros((self.nz, self.nm, self.nk_par, self.nk_perp))
            if fingers_of_god == 'Gaussian':
                for im in range(self.nm):
                    for imu in range(self.nk_perp):
                        PCS[:, im, :, imu] = UNFW[:, im, :]*self.damping(x = ((1.+ZZ)*SIGMA*KK*MU)[:,:,imu], kind = fingers_of_god)
                PSS = PCS**2.
            elif fingers_of_god == 'Lorentzian':
                for im in range(self.nm):
                    for imu in range(self.nk_perp):
                        PCS[:, im, :, imu] = UNFW[:, im, :]*self.damping(x =             ((1.+ZZ)*SIGMA*KK*MU)[:,:,imu], kind = fingers_of_god)
                        PSS[:, im, :, imu] = UNFW[:, im, :]*self.damping(x = np.sqrt(2.)*((1.+ZZ)*SIGMA*KK*MU)[:,:,imu], kind = fingers_of_god)
            else:
                raise ValueError("Damping function for Fingers of God not known.")

            # Linear matter power spectrum
            PK_BASE = np.zeros_like(ZZ)
            for iz in range(self.nz):
                power_interp = scipy.interpolate.interp1d(self.k, self.Pk['matter']['linear'][iz], kind = 'cubic')
                PK_BASE[iz]  = power_interp(KK[iz])

            # normalization and scale of cutoff
            normalization = self.norm_2h(bias, **kwargs_mass_function)
            k_star        = 0.01*(1.+self.z)

            # Filling power spectrum array
            P_KPAR_KPERP = np.zeros_like(PK_BASE)

            for iz in range(self.nz):
                for ipar in range(self.nk_par):
                    for iperp in range(self.nk_perp):
                        integrand_1h  = (1./n_avg[iz]**2.*dndM[iz]*(2.*Nc[iz]*Ns[iz]*PCS[iz,:,ipar,iperp]+Ns[iz]**2.*PSS[iz,:,ipar,iperp]))*M
                        integrand_2h  = (1./n_avg[iz]*dndM[iz]*(bias[iz] + growth_rate[iz]*MU[iz,ipar,iperp]**2.)*(Nc[iz]+Ns[iz]*PCS[iz,:,ipar,iperp]))*M
                        P_g_1h        = np.trapz(integrand_1h, dx = dlnM)*(1-np.exp(-(self.k[ik]/k_star[iz])**2.))
                        P_g_2h        = np.trapz(integrand_2h, dx = dlnM)**2.*normalization[iz]**2*PK_BASE[iz,ipar,iperp]
                        P_KPAR_KPERP[iz, ipar, iperp] = P_g_1h + P_g_2h
            del PCS, PSS, CONC
        else:
            raise ValueError("Model for RSD not known")

        # Fill the dictionary
        self.Pk['galaxies']['redshift space']['k_par-k_perp'] = P_KPAR_KPERP    

        # Delete quantities that occupy memory
        del P_KPAR_KPERP, BIAS, GROWTH_RATE, SIGMA, FoG, PK_BASE





