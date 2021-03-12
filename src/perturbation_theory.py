import colibri.constants as const
import numpy as np
import colibri.cosmology as cc
import scipy.special as ss
import scipy.integrate as integrate
import scipy.interpolate as si
import colibri.useful_functions as UF
import itertools
from six.moves import xrange


class spt:
	"""
	The 'spt' class contains routines relative to cosmological Standard Perturbation Theory such as
	vertices and kernels. Currently it contains routines to compute the 1-loop matter power spectrum
	ad the tree-level bispectrum.
	The instance to call it takes the following arguments.

	Parameters
	----------

	'z': array, default = 0
	    Redshifts.

	'k_p': array, default = np.logspace(-4., 1., 31)
	    Scales for P(k) in h/Mpc.

	'k_b': array, default = np.logspace(-4., 1., 31)
	    Scales for bispectrum in h/Mpc.

	'code': what, default = 'camb'
	    Boltzmann solver to compute linear power spectrum: choose between 'camb', 'class'.

	'fundamental': what, default = 'cb'
	    Density field to use: choose between 'cb', 'tot' for CDM+baryons and total matter respectively.

	'cosmology': 'cosmo' instance, default = see 'cosmology.py'
	    A 'cosmo' instance containing cosmological parameters.

	When called, the class immediately loads the linear power spectrum according to 'code' and  'fundamental'. This quantities are stored in 'k_load' and 'pk_load'. Also the linear power spectrum at 'k_p' is returned ('Pk_L') and an interpolation of it at all scales ('power_l_q')

	'k_load': array
	    Array of scales in h/Mpc, np.logspace(-4,2.5,501).

	'pk_load': 2D array
	    Power spectrum evaluated at ('z', 'k_load').

	'Pk': dictionary
	    Only one key is initialized, Pk['1-1'], with the linear power spectrum evaluated at 'z', 'k_p'.

	'power_l_q': interpolated object
	    interpolation for P(k,z).

	"""
	#-----------------------------------------------------------------------------------------
	# INITIALIZATION FUNCTION
	#-----------------------------------------------------------------------------------------
	def __init__(self,
			z = 0.,								# Redshift
			k_p = np.logspace(-4., -1., 31),	# Scales for power spectrum
			k_b = np.logspace(-4., -1., 31),	# Scales for bispectrum
			code = 'camb',						# CAMB or Class to compute linear P(k)
			fundamental = 'cb',					# Use 'tot' or 'cb', 'cdm' for computing bias
			cosmology = cc.cosmo()):


		# Scales and redshifts
		self.z   = z
		self.k_p = k_p
		self.k_b = k_b
		self.n_k_p = len(self.k_p)
		self.n_k_b = len(self.k_b)

		# Reading all cosmological parameters
		self.cosmology    = cosmology

		# Prescription and code
		self.fundamental = fundamental
		self.code        = code

		# Load linear power spectrum
		NPOINTS = 501
		LOGKMIN = -4.05
		LOGKMAX = 2.55
		if code == 'camb':
			# with CAMB
			self.k_load, self.pk_load = self.cosmology.camb_Pk(z         = self.z,
									  		  				   k         = np.logspace(LOGKMIN, LOGKMAX, NPOINTS),
															   nonlinear = False,
															   var_1     = self.fundamental,
											 				   var_2     = self.fundamental)
			# Normalization of PT (sigh!)
			self.pk_load = self.pk_load/(2.*np.pi)**3.

		else:
			# with Class
			self.k_load, self.pk_load = self.cosmology.class_XPk(z         = self.z,
																 k         = np.logspace(LOGKMIN, LOGKMAX, NPOINTS),
																 nonlinear = False,
																 var_1     = [self.fundamental],
																 var_2     = [self.fundamental])
			# Normalization of PT (sigh!)
			self.pk_load = self.pk_load[0,0]/(2.*np.pi)**3.

		# Initialize power spectrum
		self.Pk = {}

		self.nz = np.size(self.z)
		if self.nz == 1:
			# Evaluate P linear at k_p's
			power_linear   = si.interp1d(self.k_load, self.pk_load[0], kind = 'cubic')
			self.Pk['lin'] = np.array([power_linear(self.k_p)])
			self.Pk['1-1'] = self.Pk['lin']
			# Extrapolate to all scales
			k_q, Pk_q = UF.extrapolate_log(self.k_load, self.pk_load[0], 1e-8, 1e8)
			self.power_l_q = si.interp1d(k_q, Pk_q, kind = 'linear')
		else:
			power_linear = si.interp2d(self.k_load, self.z, self.pk_load, kind = 'linear')
			self.Pk['lin']    = power_linear(self.k_p, self.z)
			# Extrapolate to all scales
			Pk_q = []
			for i in range(self.nz):
				k_q, P_tmp = UF.extrapolate_log(self.k_load, self.pk_load[i], 1e-8, 1e8)
				Pk_q.append(P_tmp)
			Pk_q = np.array(Pk_q)
			self.power_l_q = si.interp2d(k_q, self.z, Pk_q, kind = 'linear')

	#-----------------------------------------------------------------------------------------
	# SPT VERTICES
	#-----------------------------------------------------------------------------------------
	def alpha(self, k_1, k_2):
		"""
		Vertex in SPT given two 3-vectors.

		Parameters
		----------

		'k_1': array of size 3
		    First 3-vector.

		'k_2': array of size 3
		    Second 3-vector.

		Returns
		----------

		float
		"""
		# Add an 'epsilon' to avoid infinities
		k_1 += np.finfo(float).eps
		k_2 += np.finfo(float).eps

		# Sum of k_1 and k_2
		K = np.sum((k_1, k_2), axis = 0)

		# modules of k_1, k_2 and K
		k_1_m = np.linalg.norm(k_1)
		k_2_m = np.linalg.norm(k_2)
		K_m   = np.linalg.norm(K)

		return np.dot(K, k_1)/k_1_m**2.

	def beta(self, k_1, k_2):
		"""
		Vertex in SPT given two 3-vectors.

		Parameters
		----------

		'k_1': array of size 3
		    First 3-vector.

		'k_2': array of size 3
		    Second 3-vector.

		Returns
		----------

		float
		"""
		# Add an 'epsilon' to avoid infinities
		k_1 += np.finfo(float).eps
		k_2 += np.finfo(float).eps

		# Sum of k_1 and k_2
		K = np.sum((k_1, k_2), axis = 0)

		# modules of k_1, k_2 and K
		k_1_m = np.linalg.norm(k_1)
		k_2_m = np.linalg.norm(k_2)
		K_m   = np.linalg.norm(K)

		return np.dot(k_1, k_2)*K_m**2./(2.*k_1_m**2.*k_2_m**2.)

	#-----------------------------------------------------------------------------------------
	# SPT KERNELS (FOR VECTORS)
	#-----------------------------------------------------------------------------------------

	# F kernels to be symmetrized
	def F(self, n, kk = []):
		"""
		F kernel of any order, given a set of 3-vectors. The number of vectors
        must match the order of the kernel.
        N.B. This kernel is not symmetrized!

		Parameters
		----------

		'n': int
		    Kernel order.

		'kk': list, default  = []
		    List of 3-vectors at which to compute the kernel.

		Returns
		----------

		float
		"""
		# Check number of arguments and order
		if n != len(kk):
			raise ValueError("Error! Number of arguments must match order of kernel")

		else:
			for i in range(len(kk)):
				kk = np.array(kk)+np.finfo(float).eps

		# Recursive formula
		if n == 1.: return 1.
		else:
			factor = 1./((2.*n+3.)*(n-1.))
			result = 0.
			for m in range(1,n):
				arg_1st = kk[:m]
				arg_2nd = kk[m:n]
				K1  = np.sum(arg_1st, axis = 0)
				K2  = np.sum(arg_2nd, axis = 0)
				alp = self.alpha(K1, K2)
				bet = self.beta (K1, K2)
				result += factor*self.G(m, kk = arg_1st)*((2.*n+1.)*alp*self.F(n-m, arg_2nd) + 2.*bet*self.G(n-m, arg_2nd))
			return result

	# Symmetrized F kernels
	def F_symm(self, n, kk = []):
		"""
		Symmetrized F kernel of any order, given a set of 3-vectors. The number of vectors
        must match the order of the kernel.

		Parameters
		----------

		'n': int
		    Kernel order.

		'kk': list, default  = []
		    List of 3-vectors at which to compute the kernel.

		Returns
		----------

		float
		"""
		# number of combinations
		comb = np.math.factorial(n)
		# take all the permutations
		perm = np.array(list(itertools.permutations(kk))).tolist()
		# sum all the permutations
		tot = 0.
		for p in perm:
			tot += self.F(n,p)
		return tot/comb

	# G kernels to be symmetrized
	def G(self, n, kk):
		"""
		G kernel of any order, given a set of 3-vectors. The number of vectors
        must match the order of the kernel.
        N.B. This kernel is not symmetrized!

		Parameters
		----------

		'n': int
		    Kernel order.

		'kk': list, default  = []
		    List of 3-vectors at which to compute the kernel.

		Returns
		----------

		float
		"""
		# Check number of arguments and order
		if n != len(kk):
			raise ValueError("Error! Number of arguments must match order of kernel")
		else:
			kk = np.array(kk)+np.finfo(float).eps
		# Recursive formula
		if n == 1.: return 1.
		else:
			factor = 1./((2.*n+3.)*(n-1.))
			result = 0.
			for m in range(1,n):
				arg_1st = kk[:m]
				arg_2nd = kk[m:n]
				K1  = np.sum(arg_1st, axis = 0)
				K2  = np.sum(arg_2nd, axis = 0)
				alp = self.alpha(K1, K2)
				bet = self.beta (K1, K2)
				result += factor*self.G(m, kk = arg_1st)*(3.*alp*self.F(n-m, arg_2nd) + 2.*n*bet*self.G(n-m, arg_2nd))
			return result


	# Symmetrized G kernels
	def G_symm(self, n, kk = []):
		"""
		Symmetrized G kernel of any order, given a set of 3-vectors. The number of vectors
        must match the order of the kernel.

		Parameters
		----------

		'n': int
		    Kernel order.

		'kk': list, default  = []
		    List of 3-vectors at which to compute the kernel.

		Returns
		----------

		float
		"""
		# number of combinations
		comb = np.math.factorial(n)
		# take all the permutations
		perm = np.array(list(itertools.permutations(kk))).tolist()
		# sum all the permutations
		tot = 0.
		for p in perm:
			tot += self.G(n,p)
		return tot/comb

	#-----------------------------------------------------------------------------------------
	# SPT KERNELS (FOR SCALARS)
	#-----------------------------------------------------------------------------------------
	def F_2_brute(self, k1, k2, mu):
		"""
		Brute formula for the symmetrized 'F_2' kernel given the amplitude of the two
		vectors and the cosine of the angle between them.

		Parameters
		----------

		'k1': float
		    Length of first vector.

		'k2': float
		    Length of second vector.

		'mu': float
		    cosine of the angle between k1 and k2.

		Returns
		----------

		float
		"""
		return 5./7. + 1./2.*mu*(k1/k2 + k2/k1) + 2./7.*mu**2.

	def G_2_brute(self, k1, k2, mu):
		"""
		Brute formula for the symmetrized 'G_2' kernel given the amplitude of the two
		vectors and the cosine of the angle between them.

		Parameters
		----------

		'k1': float
		    Length of first vector.

		'k2': float
		    Length of second vector.

		'mu': float
		    cosine of the angle between k1 and k2.

		Returns
		----------

		float
		"""
		return 3./7. + 1./2.*mu*(k1/k2 + k2/k1) + 4./7.*mu**2.

	# Uses recursivity (for scalars)
	def F_2(self, k1, k2, mu):
		"""
		Symmetrized 'F_2' kernel given the amplitude of the two vectors and the cosine of the angle
		between them.

		Parameters
		----------

		'k1': float
		    Length of first vector.

		'k2': float
		    Length of second vector.

		'mu': float
		    cosine of the angle between k1 and k2.

		Returns
		----------

		float
		"""
		k1_vec = np.array([0., 0., 1.])*k1
		k2_vec = np.array([np.sqrt(1.-mu**2.), 0., mu])*k2
		return self.F_symm(2, kk = [k1_vec, k2_vec])

	def G_2(self, k1, k2, mu):
		"""
		Symmetrized 'G_2' kernel given the amplitude of the two vectors and the cosine of the angle
		between them.

		Parameters
		----------

		'k1': float
		    Length of first vector.

		'k2': float
		    Length of second vector.

		'mu': float
		    cosine of the angle between k1 and k2.

		Returns
		----------

		float
		"""
		k1_vec = np.array([0., 0., 1.])*k1
		k2_vec = np.array([np.sqrt(1.-mu**2.), 0., mu])*k2
		return self.G_symm(2, kk = [k1_vec, k2_vec])

	def F_3(self, k1, k2, k3, mu12, mu23, mu13):
		"""
		Symmetrized 'F_3' kernel given the amplitude of the two vectors and the cosine of the angles
		among them.

		Parameters
		----------

		'k1': float
		    Length of first vector.

		'k2': float
		    Length of second vector.

		'k3': float
		    Length of third vector.

		'mu12': float
		    cosine of the angle between k1 and k2.

		'mu23': float
		    cosine of the angle between k2 and k3.

		'mu13': float
		    cosine of the angle between k1 and k3.

		Returns
		----------

		float
		"""

		# Add an 'epsilon' to avoid infinities
		mu12 -= np.finfo(float).eps*np.sign(mu12)
		mu23 -= np.finfo(float).eps*np.sign(mu23)
		mu13 -= np.finfo(float).eps*np.sign(mu13)
		
		sin_theta_12 = (1.-mu12**2.)**0.5
		sin_theta_13 = (1.-mu13**2.)**0.5
		sin_phi_13   = (mu23-mu12*mu13)/(sin_theta_12*sin_theta_13)
		cos_phi_13   = (1.-sin_phi_13**2.)**.5
		

		k1_vec = np.array([0., 0., 1.])*k1
		k2_vec = np.array([sin_theta_12, 0., mu12])*k2
		k3_vec = np.array([sin_phi_13*sin_theta_13, cos_phi_13*sin_theta_13, mu13])*k3
		return self.F_symm(3, kk = [k1_vec, k2_vec, k3_vec])

	def G_3(self, k1, k2, k3, mu12, mu23, mu13):
		"""
		Symmetrized 'G_3' kernel given the amplitude of the two vectors and the cosine of the angles
		among them.

		Parameters
		----------

		'k1': float
		    Length of first vector.

		'k2': float
		    Length of second vector.

		'k3': float
		    Length of third vector.

		'mu12': float
		    cosine of the angle between k1 and k2.

		'mu23': float
		    cosine of the angle between k2 and k3.

		'mu13': float
		    cosine of the angle between k1 and k3.

		Returns
		----------

		float
		"""
		# Add an 'epsilon' to avoid infinities
		mu12 -= np.finfo(float).eps*np.sign(mu12)
		mu23 -= np.finfo(float).eps*np.sign(mu23)
		mu13 -= np.finfo(float).eps*np.sign(mu13)

		sin_theta_12 = (1.-mu12**2.)**0.5
		sin_theta_13 = (1.-mu13**2.)**0.5
		sin_phi_13   = (mu23-mu12*mu13)/(sin_theta_12*sin_theta_13)
		cos_phi_13   = (1.-sin_phi_13**2.)**.5

		k1_vec = np.array([0., 0., 1.])*k1
		k2_vec = np.array([sin_theta_12, 0., mu12])*k2
		k3_vec = np.array([sin_phi_13*sin_theta_13, cos_phi_13*sin_theta_13, mu13])*k3
		return self.G_symm(3, kk = [k1_vec, k2_vec, k3_vec])

	#-----------------------------------------------------------------------------------------
	# 1-LOOP POWER SPECTRUM
	#-----------------------------------------------------------------------------------------
	def Pk_1_loop(self):
		"""
		1-loop power spectrum in SPT, i.e the sum P_22 and P_13.

		Returns
		----------

		Nothing, but adds self.Pk['2-2'], self.Pk['1-3'] and Pk['1 loop'] to the self.Pk dictionary
		"""

		kL = self.k_load
		PL = self.pk_load

		# Powers of kL and PL, and combinations needed
		kkL, kL2, kL3, kL4, kL5, kL6, kL7, kL8 = 2*kL, kL*kL, kL*kL*kL, kL*kL*kL*kL, kL*kL*kL*kL*kL, kL*kL*kL*kL*kL*kL, kL*kL*kL*kL*kL*kL*kL, kL*kL*kL*kL*kL*kL*kL*kL
		PL_kL2  = PL/kL2
		PL2_kL4 = PL_kL2*PL_kL2
		PI      = np.pi

		# Define the logarithmically spaced k's to be interpolated, and compute useful quantities		
		step = 6
		eps  = 2.6912e-3#0.5*np.log10(kL[1]/kL[0])
		kout = kL[::step]*(1.+eps)         #Displacement in initial k's can better reproduce FAST-PT results

		ko2, ko3, ko4, ko5, ko6, ko7, ko8 = kout*kout, kout*kout*kout, kout*kout*kout*kout, kout*kout*kout*kout*kout, kout*kout*kout*kout*kout*kout, kout*kout*kout*kout*kout*kout*kout, kout*kout*kout*kout*kout*kout*kout*kout
		if self.nz == 1:
			PLout = np.array([self.power_l_q(kout)])
		else:
			PLout = self.power_l_q(kout, self.z)
		dk = np.log(10)*np.log10(kL[1]/kL[0])
		dk2 = dk*dk

		ID = np.ones([len(kL)])
		IDo = ID[::step]
		
		# Initialize lists of P_13 and P_22
		P13f = []
		P22f = []
		P13 = np.zeros((self.nz, len(self.k_p)))
		P22 = np.zeros((self.nz, len(self.k_p)))
		
		#######
		# P13 #
		#######
		# Compute the integrand as combination of matrices
		for i in xrange(self.nz):
			A1, A2, A3, A4, A5, A6, A7 = 6*np.outer(ID,ko8), 3*np.outer(kL2,ko6), -45*np.outer(kL4,ko4), 57*np.outer(kL6,ko2), -21*np.outer(kL8,IDo), np.arctanh(2*np.outer(kL,kout)/np.add.outer(kL2,ko2)), np.outer(PL_kL2[i],IDo)
			B1, B2, B3, B4 = 12*np.outer(PL[i]/kL,ko7), -158*np.outer(PL[i]*kL,ko5), 100*np.outer(PL[i]*kL3,ko3), -42*np.outer(PL[i]*kL5,kout)
			integrand    = B1+B2+B3+B4-(A1+A2+A3+A4+A5)*A6*A7
			P13f.append(np.sum(integrand, axis=0))
			P13f[i]     *= PI*dk*PLout[i]/126./ko3
			P13g         = si.interp1d(kout,P13f[i], kind="cubic")
			P13[i]  = P13g(self.k_p)

		#######
		# P22 #
		#######
		# Set grid for P22 and compute needed terms
		p, q = np.meshgrid(kL,kL)
		for i in xrange(self.nz):
			Pp, Pq = np.meshgrid(PL[i],PL[i])
			pg, qg, Ppg, Pqg = p[(p>q)], q[(p>q)], Pp[(p>q)], Pq[(p>q)]
			pg2, qg2, p2mq22, p2pq2, PpPq_2p2q2, pgpqg, pgmqg = pg*pg, qg*qg, (pg*pg-qg*qg)*(pg*pg-qg*qg), pg*pg+qg*qg, Ppg*Pqg/(2*pg*pg*qg*qg), pg+qg, pg-qg

			# Compute integral with two contributions: p = q and p > q (multiplied by two, the integrand is symmetric in p and q)
			P22f.append([np.sum(PL2_kL4[i,(kkL>=K)]*(K*K+3.*kL2[(kkL>=K)])**2)+np.sum((2.*K*K-5.*p2mq22[(K<=pgpqg)&(pgmqg<=K)]/(K*K)+3.*p2pq2[(K<=pgpqg)&(pgmqg<=K)])**2*PpPq_2p2q2[(K<=pgpqg)&(pgmqg<=K)]) for K in kout])
			P22f[i] *= PI*dk2*ko3/49.
			P22g        = si.interp1d(kout,P22f[i], kind="cubic")
			P22[i] = P22g(self.k_p)

		self.Pk['1-3'] = P13
		self.Pk['2-2'] = P22
		
		self.Pk['1 loop'] = P13+P22

		

	"""
	#-----------------------------------------------------------------------------------------
	#-----------------------------------------------------------------------------------------
	#-----------------------------------------------------------------------------------------
	# OTHER USEFUL FUNCTIONS TO IMPLEMENT
	#-----------------------------------------------------------------------------------------
	#-----------------------------------------------------------------------------------------
	#-----------------------------------------------------------------------------------------

	#-------------------------
	# TREE-LEVEL BISPECTRUM
	#-------------------------
	def Bk_TL(self, k_1, k_2, k_3):

		# Transform into arrays
		k_1 = np.array(k_1)
		k_2 = np.array(k_2)
		k_3 = np.array(k_3)

		# Check if closed triangle
		assert np.all(k_1 + k_2 + k_3 == 0.), "Bispectrum can only be computed for a closed triangular configuration."

		# Compute the norms of vectors
		k_1_m = np.linalg.norm(k_1)
		k_2_m = np.linalg.norm(k_2)
		k_3_m = np.linalg.norm(k_3)

		# Compute tree-level bispectrum
		b1 = np.array([2.*self.F_symm(2, kk = [k_1, k_2])*self.pk_linear_interpolation[iz](k_1_m)*self.pk_linear_interpolation[iz](k_2_m) for iz in range(self.nz)])
		b2 = np.array([2.*self.F_symm(2, kk = [k_2, k_3])*self.pk_linear_interpolation[iz](k_2_m)*self.pk_linear_interpolation[iz](k_3_m) for iz in range(self.nz)])
		b3 = np.array([2.*self.F_symm(2, kk = [k_3, k_1])*self.pk_linear_interpolation[iz](k_3_m)*self.pk_linear_interpolation[iz](k_1_m) for iz in range(self.nz)])

		return b1+b2+b3

	#-------------------------
	# TREE-LEVEL TRISPECTRUM
	#-------------------------
	def Tk_TL(self, k_1, k_2, k_3, k_4):

		# Transform into arrays
		k_1 = np.array(k_1)
		k_2 = np.array(k_2)
		k_3 = np.array(k_3)
		k_4 = np.array(k_4)

		# Check if closed quadrilateral
		assert np.all(k_1 + k_2 + k_3 + k_4 == 0.), "Trispectrum can only be computed for a closed quadrilateral configuration."

		# Compute the norms of vectors
		k_1_m = np.linalg.norm(k_1)
		k_2_m = np.linalg.norm(k_2)
		k_3_m = np.linalg.norm(k_3)
		k_4_m = np.linalg.norm(k_4)

		# Compute sums of arrays and norms
		k_12, k_12_m = k_1 + k_2, np.linalg.norm(k_1 + k_2)
		k_13, k_13_m = k_1 + k_3, np.linalg.norm(k_1 + k_3)
		k_14, k_14_m = k_1 + k_4, np.linalg.norm(k_1 + k_4)
		k_23, k_23_m = k_2 + k_3, np.linalg.norm(k_2 + k_3)
		k_24, k_24_m = k_2 + k_4, np.linalg.norm(k_2 + k_4)
		k_34, k_34_m = k_3 + k_4, np.linalg.norm(k_3 + k_4)

		# Short-cut notation
		pk1  = np.array([self.pk_linear_interpolation[iz](k_1_m)  for iz in range(self.nz)])
		pk2  = np.array([self.pk_linear_interpolation[iz](k_2_m)  for iz in range(self.nz)])
		pk3  = np.array([self.pk_linear_interpolation[iz](k_3_m)  for iz in range(self.nz)])
		pk4  = np.array([self.pk_linear_interpolation[iz](k_4_m)  for iz in range(self.nz)])
		pk12 = np.array([self.pk_linear_interpolation[iz](k_12_m) for iz in range(self.nz)])
		pk13 = np.array([self.pk_linear_interpolation[iz](k_13_m) for iz in range(self.nz)])
		pk14 = np.array([self.pk_linear_interpolation[iz](k_14_m) for iz in range(self.nz)])
		F2   = lambda k1, k2:     self.F_symm(2, kk = [k1, k2])
		F3   = lambda k1, k2, k3: self.F_symm(3, kk = [k1, k2, k3])

		# First part
		Ta = 0.
		Ta += pk1 * pk12 * pk3 * F2(k_1, -k_12) * F2(k_3, k_12)
		Ta += pk1 * pk12 * pk4 * F2(k_1, -k_12) * F2(k_4, k_12)
		Ta += pk2 * pk12 * pk3 * F2(k_2, -k_12) * F2(k_3, k_12)
		Ta += pk2 * pk12 * pk4 * F2(k_2, -k_12) * F2(k_4, k_12)
		Ta += pk1 * pk13 * pk2 * F2(k_1, -k_13) * F2(k_2, k_13)
		Ta += pk1 * pk13 * pk4 * F2(k_1, -k_13) * F2(k_4, k_13)
		Ta += pk3 * pk13 * pk2 * F2(k_3, -k_13) * F2(k_2, k_13)
		Ta += pk3 * pk13 * pk4 * F2(k_3, -k_13) * F2(k_4, k_13)
		Ta += pk1 * pk14 * pk2 * F2(k_1, -k_14) * F2(k_2, k_14)
		Ta += pk1 * pk14 * pk3 * F2(k_1, -k_14) * F2(k_3, k_14)
		Ta += pk4 * pk14 * pk2 * F2(k_4, -k_14) * F2(k_2, k_14)
		Ta += pk4 * pk14 * pk3 * F2(k_4, -k_14) * F2(k_3, k_14)

		# Second part
		Tb  = 0.
		Tb += pk1 * pk2 * pk3 * F3(k_1, k_2, k_3)
		Tb += pk2 * pk3 * pk4 * F3(k_2, k_3, k_4)
		Tb += pk3 * pk4 * pk1 * F3(k_3, k_4, k_1)
		Tb += pk4 * pk1 * pk2 * F3(k_4, k_1, k_2)

		return 4.*Ta + 6.*Tb
	"""


