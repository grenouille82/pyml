'''
Created on Aug 2, 2011

@author: marcel

@todo: - remove precomputed distance matrix R
	   - implement deepth copy method
		
'''

import numpy as np
import scipy.optimize as spopt

from abc import ABCMeta, abstractmethod
from upgeo.util.metric import distance_matrix
from scipy.linalg.decomp_cholesky import cho_solve

class Kernel(object):
	'''
	Abstract base class for all gaussian process kernels. The hyperparmater are 
	defined in log space so that all implemented subclasses must consider this.
	
	@todo: - allow the definition of priors on hyperparameters
		   - naming hyperparameters (access by name)
		   - which dtype should be the hyperparameter array?
		   - kernel should return a diag of covariance matrix, optionally
	'''
	
	__metaclass__ = ABCMeta
	
	__slots__ = ('_params',   #an array of the kernel hyperparameters
				 '_n',		#the number of hyperparameters
				 )
	
	def __init__(self, params):
		'''
		'''
		params = np.ravel(np.atleast_1d(np.asarray(params, dtype=np.float64)))
		
		self._params = params
		self._n = len(params)
	
	def __mul__(self, rhs):
		'''
		Composite two kernels by the product kernel.
		
		@return: The product of self and rhs. 
		'''
		return ProductKernel(self, rhs)
	
	def __add__(self, rhs):
		'''
		Composite two kernels by the sum kernel.
		
		@return: The sum of self and rhs. 
		'''
		return SumKernel(self, rhs)
	
	def __str__(self):
		"""
		String representation of kernel
		"""
		raise RuntimeError( """%s.__str__() Should have been implemented """
			"""in base class""" % str(self.__class__) )
		
	@abstractmethod
	def __call__(self, X, Z=None, diag=False):
		"""
		Returns value of kernel for the specified data points
		"""
		pass
	
	@abstractmethod
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		'''
		self._range_check(i)
	
	def get_parameter(self, i):
		self._range_check(i)
		return self._params[i]
	
	def set_parameter(self, i, value):
		self._range_check(i)
		self._params[i] = value
	
	def _get_params(self):
		'''
		Return the hyperparameters of the kernel. Note, the returned array is 
		not immutable and a reference to the original, so that specific paramaters
		could be changed by the returned reference. 
		
		@return:  an array of hyperparameters 
		'''
		return self._params
	
	def _set_params(self, params):
		'''
		Sets the specified hyperparameters of the kernel. The method raise an 
		exception if the size of given parameter array not the same with the
		number of parameters needed by the kernel.
		'''
		params = np.ravel(np.array(params))
		if len(params) != self.n_params:
			raise TypeError('''wrong number of parameters.
							{0} parameters are needed.'''.format(self.n_params))
			
		for i in xrange(len(params)):
			self._params[i] = params[i]
		
	params = property(fget=_get_params, fset=_set_params)
	
	def _number_of_params(self):
		return self._n
	
	n_params = property(fget=_number_of_params) #todo: remove
	
	nparams = property(fget=_number_of_params)
		
	def _range_check(self, index):
		'''
		'''
		if index < 0 or index > self._n-1:
			raise IndexError("Index %s out of bound." % index)
		
	def copy(self):
		params = np.copy(self._params)
		new_kernel = self.__class__(params)
		return new_kernel
	
	class _IDerivativeFun(object):

		__metaclass__ = ABCMeta

		@abstractmethod
		def __call__(self, X, Z=None, diag=False):
			return None

	class _IDerivativeFunX(object):

		__metaclass__ = ABCMeta

		@abstractmethod
		def __call__(self, x, Z):
			return None
	
	
class ConstantKernel(Kernel):
	'''
	Covariance Kernel for a constant function. The covariance kernel is parametrized
	as:
	
	k(x_p,x_q) = c
	
	'''
	def __init__(self, const=0.0):
		'''
		Initialize the constant kernel.
		
		@arg const: the constant parameter.
		'''
		Kernel.__init__(self, const)
	
	def __call__(self, X, Z=None, diag=False):
		'''
		'''
		c = np.exp(self.params[0])
		
		xeqz = (Z == None)
		m = np.size(X, 0)
		if xeqz:
			if diag:
				K = np.ones(m)
			else:
				K = np.ones((m,m))
		else:
			n = np.size(Z, 0)
			K = np.ones((m,n))
		
		K *= c
		return K
	
	def __str__( self ):
		return "ConstantKernel({0})".format(self.params[0])
	
	def gradient(self, covGrad, X, Z=None, diag=False):
		c = np.exp(self.kernel.params[0])
		xeqz = (Z == None)
		m = np.size(X, 0)
		if xeqz:
			if diag:
				dK = np.ones(m)
			else:
				dK = np.ones((m,m))
		else:
			n = np.size(Z, 0)
			dK = np.ones((m,n))
			
		dK *= c
		grad = np.array([np.sum(covGrad*dK)])
		return grad
	
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				c = np.exp(self.kernel.params[0])
				
				xeqz = (Z == None)
				m = np.size(X, 0)
				
				if xeqz:
					if diag:
						dK = np.ones(m)
					else:
						dK = np.ones((m,m))
				else:
					n = np.size(Z, 0)
					dK = np.ones((m,n))
				
				dK *= c
				return dK
			
		fun = _DerivativeFun(self)
		return fun
	
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					dK = np.zeros((m,n,d))
				return dK
				
		return _DerivativeFun(self)

	
class SqConstantKernel(Kernel):
	'''
	Covariance Kernel for a squared constant function. The covariance kernel is parametrized
	as:
	
	k(x_p,x_q) = c**2
	
	'''
	def __init__(self, const=0.0):
		'''
		'''
		Kernel.__init__(self, const)
		
	def __call__(self, X, Z=None, diag=False):
		'''
		'''
		c = np.exp(2*self.params[0])
		
		xeqz = (Z == None)
		m = np.size(X, 0)
		
		if xeqz:
			if diag:
				K = np.ones(m)
			else:
				K = np.ones((m,m))
		else:
			n = np.size(Z, 0)
			K = np.ones((m,n))
		
		K *= c
		return K
	
	def __str__( self ):
		'''
		'''
		return "SqConstantKernel({0})".format(self.params[0]**2.0)
		
	def gradient(self, covGrad, X, Z=None, diag=False):
		c = np.exp(2.0*self.kernel.params[0])
		xeqz = (Z == None)
		m = np.size(X, 0)
		if xeqz:
			if diag:
				dK = np.ones(m)
			else:
				dK = np.ones((m,m))
		else:
			n = np.size(Z, 0)
			dK = np.ones((m,n))
			
		dK *= 2.0*c
		grad = np.array([np.sum(covGrad*dK)])
		return grad

	
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				c = np.exp(2.0*self.kernel.params[0])
				
				xeqz = (Z == None)
				m = np.size(X, 0)
				
				if xeqz:
					if diag:
						dK = np.ones(m)
					else:
						dK = np.ones((m,m))
				else:
					n = np.size(Z, 0)
					dK = np.ones((m,n))
			
				dK *= 2.0*c
				return dK
			
		fun = _DerivativeFun(self)
		return fun
		
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					dK = np.zeros((m,n,d))
				return dK
				
		return _DerivativeFun(self)
			 
	
		
class CompositeKernel(Kernel):
	'''
	'''
	
	__slots__ = ('_lhs',	#left hand side kernel of the compositum
				 '_rhs'	 #right hand side kernel of the compositum
				 )
	
	def __init__(self, lhs, rhs):
		params = CompositeKernel.MixedArray(lhs.params, rhs.params)
		#Kernel.__init__(self, params)
		
		self._params = params
		self._n = len(params)
		
		self._lhs = lhs
		self._rhs = rhs
		
	def _get_lhs(self):
		'''
		Return the kernel on the left hand side of the compositum.
		'''
		return self._lhs
		
	lhs = property(fget=_get_lhs)
	
	def _get_rhs(self):
		'''
		Return the kernel on the right hand side of the compositum.
		'''
		return self._rhs
		
	rhs = property(fget=_get_rhs)
	
	def kernel_by_parameter(self, i):
		self._range_check(i)
		return self.lhs if i < self.lhs.n_params else self.rhs
	
	def _lookup_kernel_and_param(self, i): 
		'''
		Returns a triplet of both kernels and the parameter index of the active 
		kernel as tuple. The first element of the tuple is the active kernel for
		which the parameter request is done. The second kernel is the passive 
		part. The returned tuple has the following form:
		
		(active kernel, passive kernel, param) 
		'''
		self._range_check(i)
		if i < self.lhs.nparams:
			return (self.lhs, self.rhs, i)
		else:
			return (self.rhs, self.lhs, i-self.lhs.nparams)
		
	def copy(self):
		new_lhs = self._lhs.copy()
		new_rhs = self._rhs.copy()
		new_kernel = self.__class__(new_lhs, new_rhs)
		return new_kernel
		
	class MixedArray(object):
		def __init__(self, a, b):
			self.a = a
			self.b = b
		
		def __len__(self):
			return len(self.a) + len(self.b)
		
		def __getitem__(self, i):
			array, idx = self.__array_idx_at(i)
			return array[idx]
		
		def __setitem__(self, i, value):
			array, idx = self.__array_idx_at(i)
			array[idx] = value

		def __array_idx_at(self, i):
			return (self.a, i) if i < len(self.a) else (self.b, i-len(self.a))
		
		def __str__(self):
			return str(np.r_[self.a, self.b]) 
		
class ProductKernel(CompositeKernel):
	
	def __init__(self, lhs, rhs):
		CompositeKernel.__init__(self, lhs, rhs)
		
	def __call__(self, X, Z=None, diag=False):
		return self.lhs(X,Z,diag=diag) * self.rhs(X,Z,diag=diag) #elementwise multiplication
	
	def __str__( self ):
		return "ProductKernel({0},{1})".format(str(self.lhs), str(self.rhs))

	def gradient(self, covGrad, X, Z=None, diag=False):
		lhs = self._lhs
		rhs = self._rhs
		
		Klhs = lhs(X,Z,diag)
		Krhs = rhs(X,Z,diag)
		
		#@todo: check if the gradient works correctly
		grad = np.empty(self.nparams)
		grad[:lhs.nparams] = lhs.gradient(covGrad*Krhs, X, Z, diag)
		grad[lhs.nparams:] = rhs.gradient(covGrad*Klhs, X, Z, diag)
		return grad
		

	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		'''
		u, v, i = self._lookup_kernel_and_param(i)
		u_deriv = u.derivate(i)
		
		fun = lambda X, Z=None, diag=False: v(X,Z,diag) * u_deriv(X,Z,diag)
		return fun

	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				kernel = self.kernel
				lhs = kernel.lhs
				rhs = kernel.rhs
				dX_lhs = kernel.lhs.derivateX()
				dX_rhs = kernel.rhs.derivateX()
				
				dK_lhs = dX_lhs(X,Z,diag)
				dK_rhs = dX_rhs(X,Z,diag)
				K_lhs = lhs(X,Z,diag)
				K_rhs = rhs(X,Z,diag)
				
				d = X.shape[1]
				dK = np.empty(dK_lhs.shape)
				for i in xrange(d):
					dK[:,:,i] = dK_lhs[:,:,i]*K_rhs.T + dK_rhs[:,:,i]*K_lhs.T
				
				return dK
					
		return _DerivativeFun(self)
	

class SumKernel(CompositeKernel):
	
	def __init__(self, lhs, rhs):
		CompositeKernel.__init__(self, lhs, rhs)
		
	def __call__(self, X, Z=None, diag=False):
		return self.lhs(X,Z,diag=diag) + self.rhs(X,Z,diag=diag)

	def __str__( self ):
		return "SumKernel({0},{1})".format(str(self.lhs), str(self.rhs))


	def gradient(self, covGrad, X, Z=None, diag=False):
		lhs = self._lhs
		rhs = self._rhs
		
		#@todo: check if the gradient works correctly
		grad = np.empty(self.nparams)
		grad[:lhs.nparams] = lhs.gradient(covGrad, X, Z, diag)
		grad[lhs.nparams:] = rhs.gradient(covGrad, X, Z, diag)
		return grad


	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		u, v, i = self._lookup_kernel_and_param(i)
		return u.derivate(i)	

	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				kernel = self.kernel
				dX_lhs = kernel.lhs.derivateX()
				dX_rhs = kernel.rhs.derivateX()
				
				dK = dX_lhs(X,Z,diag) + dX_rhs(X,Z,diag)
				return dK
					
		return _DerivativeFun(self)

class HiddenKernel(Kernel):
	'''
	A wrapper kernel that hides (i.e. fixes) the parameters of another kernel
	
	To users of this class it appears as the kernel has no parameters to optimise.
	This can be useful when you have a mixture kernel and you only want to learn 
	one child kernel's parameters.
	
	@todo: - we have no read access to the parameters of the hidden kernel (fix this)
	'''
	
	__slots__ = ('_hidden_kernel')
	
	def __init__(self, hidden_kernel):
		Kernel.__init__(self, np.asarray([]))
		self._hidden_kernel = hidden_kernel
		
	def __str__(self):
		return "HiddenKernel"

	def __call__(self, X, Z=None, diag=False):
		return self._hidden_kernel(X, Z, diag=diag)
	
	def gradient(self, covGrad, X, Z=None, diag=False):
		pass
	
	def derivate(self, i):
		pass
	
	def derivateX(self):
		return self._hidden_kernel.derivateX()
	
	def copy(self):
		cp_hidden_kernel = self._hidden_kernel.copy()
		return self.__class__(cp_hidden_kernel)
	

class MaskedFeatureKernel(Kernel):
	'''
	todo : param problems
	'''
	
	__slots__ = ('_kernel',
				 '_mask' 	#mask of the used features
				 )
	def __init__(self, kernel, mask):
		Kernel.__init__(self, np.array([]))
		
		self._params = kernel.params
		self._n = len(kernel.params)
		
		self._kernel = kernel
		self._mask = mask
	
	def __str__(self):
		return "MaskedFeatureKernel"

	def __call__(self, X, Z=None, diag=False):
		mask = self._mask
		if Z == None:
			Z = X
		return self._kernel(X[:,mask],Z[:,mask],diag=diag)
	
	def gradient(self, covGrad, X, Z=None, diag=False):
		''' 
		'''
		mask = self._mask
		if Z == None:
			Z = X
		return self._kernel.gradient(covGrad, X[:,mask], Z[:,mask], diag=diag)
		
	
	def derivate(self, i): 

		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, mask, i):
				self.kernel = kernel
				self.mask = mask
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):
				
				deriv = self.kernel.derivate(self.i)
				mask = self.mask
				
				if Z == None:
					Z = X
				
				return deriv(X[:,mask], Z[:,mask], diag=diag)
			
		fun = _DerivativeFun(self._kernel, self._mask, i)
		return fun
	
	def derivateX(self):
		return self._kernel.derivateX()
	
	def copy(self):
		'''
		@todo: .implement copy constructor
		'''
		pass
	


class FixedParameterKernel(Kernel):
	'''
	@todo: - 
	'''
	
	__slots__ = ('_kernel',	 #kernel with fixed parameters 
				 '_param_map'   #mapping to the flexible parameters of the kernel
				 ) 
	
	def __init__(self, kernel, fixed_params):
		'''
		'''
		fixed_params = np.ravel(np.asarray(fixed_params))
		
		params = kernel.params
		n = kernel.n_params
		
		mask = np.empty(n, dtype=np.bool)
		mask[fixed_params] = False
		
		param_map = np.where(mask)
		
		Kernel.__init__(self, params[param_map])
		self._kernel = kernel
		self._param_map = param_map
	
	def __str__(self):
		return "FixedParameterKernel"
	
	def __call__(self, X, Z=None, diag=False):
		return self._kernel(X,Z,diag=diag)

	def gradient(self, covGrad, X, Z=None, diag=False):
		'''
		I think its faster to compute the gradient for all parameters
		and then extract the unmasked features.
		'''
		grad = self._kernel.gradient(covGrad, X, Z, diag)
		return grad[self._param_map]
		
	def derivate(self, i): 
		return self._kernel.derivate(self._param_map[i])
	
	def derivateX(self):
		return self._kernel.derivateX()
	
	def copy(self):
		'''
		@todo: .implement copy constructor
		'''
		pass

class NoiseKernel(Kernel):
	'''
	Independent covariance kernel, i.e. 'white noise', with specified variance.
	The covariance function is specified as:
	
	k(x_q, x_p) = s**2 * \delta(p,q)
	
	where s is the noise variance and \delta(p,q) is a Kronecker delta function
	which is 1 iff p==q and zero otherwise.
	'''
	
	def __init__(self, s=0.0):
		Kernel.__init__(self, np.asarray([s]))
	
	def __str__( self ):
		return "NoiseKernel({0})".format(self.params[0])
	
	def __call__(self, X, Z=None, diag=False):
		s = np.exp(2.0*self.params[0])
		
		xeqz = (Z == None)
		m = np.size(X, 0)
		
		if xeqz:
			if diag:
				K = np.ones(m)*s
			else:
				K = np.diag(np.ones(m)*s)
		else:
			n = np.size(Z, 0)
			K = np.zeros((m,n))

		return K
	
	def gradient(self, covGrad, X, Z=None, diag=False):
		s = np.exp(2.0*self.kernel.params[0])
				
		xeqz = (Z == None)
		m = np.size(X, 0)
				
		if xeqz:
			if diag:
				dK = 2.0*s*np.ones(m)
			else:
				dK = np.diag(2.0*s*np.ones(m)) 
		else:
			n = np.size(Z, 0)
			dK = np.zeros((m,n))
			
		grad = np.array([np.sum(covGrad*dK)])
		return grad

	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				s = np.exp(2.0*self.kernel.params[0])
				
				xeqz = (Z == None)
				m = np.size(X, 0)
				
				if xeqz:
					if diag:
						dK = 2.0*s*np.ones(m)
					else:
						dK = np.diag(2.0*s*np.ones(m)) 
				else:
					n = np.size(Z, 0)
					dK = np.zeros((m,n))
					
					#K *= 2.0#*params[0]
				return dK
			
		fun = _DerivativeFun(self)
		return fun
	
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					dK = np.zeros((m,n,d))
				return dK
				
		return _DerivativeFun(self)


class GroupNoiseKernel(Kernel):
	'''
	Independent covariance kernel, i.e. 'white noise', with specified variance.
	The covariance function is specified as:
	
	k(x_q, x_p) = s**2 * \delta(p_i,q_i)
	
	where s is the noise variance and \delta(p,q) is a Kronecker delta function
	which is 1 iff p==q and zero otherwise.
	
	TODO: check efficient of kernel computation, maybe its iterate by itself or sorting the vectors before
	'''
	
	__slots__ = ('_group_idx')
	
	def __init__(self, group_idx, s=0.0):
		Kernel.__init__(self, np.asarray([s]))
		self._group_idx = group_idx
	
	def __str__( self ):
		return "GroupNoiseKernel({0})".format(self.params[0])
	
	def __call__(self, X, Z=None, diag=False):
		s = np.exp(2.0*self.params[0])
		i = self._group_idx
		
		xeqz = (Z == None)
		if xeqz:
			if diag:
				K = np.diag(np.array((np.equal.outer(X[:,i],X[:,i])),dtype=int)) * s
			else:
				K = np.array((np.equal.outer(X[:,i],X[:,i])),dtype=int) * s
		else:
			K = np.array((np.equal.outer(X[:,i],Z[:,i])),dtype=int) * s
	
		return K

	def gradient(self, covGrad, X, Z=None, diag=False):
		s = np.exp(2.0*self.kernel.params[0])
		i = self.kernel._group_idx
		
		
		xeqz = (Z == None)
		if xeqz:
			if diag:
				dK = np.diag(np.array((np.equal.outer(X[:,i],X[:,i])),dtype=int)) * 2.0*s
			else:
				dK = np.array((np.equal.outer(X[:,i],X[:,i])),dtype=int) * 2.0*s
		else:
			dK = np.array((np.equal.outer(X[:,i],Z[:,i])),dtype=int) * 2.0*s
	
		grad = np.array([np.sum(covGrad*dK)])
		return grad


	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False, K=None):
				s = np.exp(2.0*self.kernel.params[0])
				i = self.kernel._group_idx
				
				
				xeqz = (Z == None)
				if xeqz:
					if diag:
						dK = np.diag(np.array((np.equal.outer(X[:,i],X[:,i])),dtype=int)) * 2.0*s
					else:
						dK = np.array((np.equal.outer(X[:,i],X[:,i])),dtype=int) * 2.0*s
				else:
					dK = np.array((np.equal.outer(X[:,i],Z[:,i])),dtype=int) * 2.0*s
			
				return dK
			
		fun = _DerivativeFun(self)
		return fun
	

	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					dK = np.zeros((m,n,d))
				return dK
				
		return _DerivativeFun(self)


class RBFKernel(Kernel):
	'''
	Exponential covariance kernel with isotropic distance measure. 
	The kernel is parametrized as:
	
	k(x_p, x_q) = s**2 * exp(-|x_p-x_q| / l) 
	
	'''
	
	def __init__(self, l, s=0.0):
		Kernel.__init__(self, np.asarray([l,s]))
		
	def __str__( self ):
		return "RbfKernel({0},{1})".format(self.params[0],self.params[1])
	
	def __call__(self, X, Z=None, diag=False):
		
		l = np.exp(self.params[0])
		s = np.exp(2.0*self.params[1])
		xeqz = (Z == None)
		
		if xeqz and diag:
			R = np.zeros(X.shape[0])
		else:
			R = distance_matrix(X, Z, metric='euclidean')
			
		K = s * np.exp(-R/l)
		return K


	def gradient(self, covGrad, X, Z=None, diag=False):
		'''
		'''
		l = np.exp(self.params[0])
		s = np.exp(2.0*self.params[1])
		xeqz = (Z == None)
		
		grad = np.zeros(2)


		if xeqz and diag:
			R = np.zeros(X.shape[0])
		else:
			R = distance_matrix(X, Z, metric='euclidean')
		K = s * np.exp(-R/l)
		
		#gradient of the length scale l
		dKl = K * R/l
		grad[0] = np.sum(covGrad, dKl)
		
		#gradient of the signal variance
		dKs = 2.0 * K
		grad[1] = np.sum(covGrad, dKs)

		return grad

	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):
				l = np.exp(self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
				xeqz = (Z == None)
				
				if xeqz and diag:
					R = np.zeros(X.shape[0])
				else:
					R = distance_matrix(X, Z, metric='euclidean')
			
				if self.i == 0:
					#gradient of scale parameter l
					dK = s * np.exp(-R/l) * R / l 
				elif self.i == 1:
					#gradient of the variance parameter s
					dK = 2.0*s * np.exp(-R/l)
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
				
				return dK
			
		fun = _DerivativeFun(self, i)
		return fun

	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				l = np.exp(self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
		
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					R = distance_matrix(X, Z, metric='euclidean')
					K = s * np.exp(-R/l)
					dK = np.zeros((m,n,d))
					R = R + 1e-16 #prevent division by zeros
					for i in xrange(d):
						dK[:,:,i] = np.add.outer(Z[:,i],-X[:,i]) * K.T / (R.T*l)
				return dK
				
		return _DerivativeFun(self)

	
	def copy(self):
		cp_kernel = RBFKernel(self._params[0], self._params[1])
		return cp_kernel

class ARDRBFKernel(Kernel):
	'''
	RBF covariance kernel with automatic relevance determinition
	distance measure. The kernel is parametrized as:
	
	k(x_p, x_q) = s**2 * exp(-(x_p-x_q)' * inv(L) *  (x_p-x_q)/ 2) 
	'''
	
	__slots__ = ('_d')
	
	def __init__(self, l, s=0.0):
		l = np.asarray(l).ravel()
		params = np.r_[l,s]
		Kernel.__init__(self, params)
		
		self._d = len(l)
		
	def __str__( self ):
		return "ARDSEKernel()"
	
	def __call__(self, X, Z=None, diag=False):
		'''
		@todo: - optimize the distance computation of the diagonal dot prod
		'''
		
		xeqz = (Z == None)
			
		d = self._d
		l = np.exp(self.params[0:d])
		s = np.exp(2.0*self.params[d])
			
		P = np.diag(1/l)
		
		if xeqz:	
			if diag:
				K = np.zeros(X.shape[0])
			else:
				K = distance_matrix(np.dot(P, X.T).T,  metric='euclidean') 
		else:
			K = distance_matrix(np.dot(P, X.T).T, np.dot(P, Z.T).T, metric='euclidean')
			
		
		K = s*np.exp(-K)
		return K
			
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - check the gradient of the length scale vector (gradient check failed, 
				 but its identical to matlab gpml implementation)
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):	 
				xeqz = (Z == None)
					
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				s = np.exp(2.0*self.kernel.params[d])
					
				P = np.diag(1/l)
				if xeqz:
					if diag:
						R = np.zeros(X.shape[0])
					else:
						R = distance_matrix(np.dot(P, X.T).T,  metric='euclidean') 
				else:
					R = distance_matrix(np.dot(P, X.T).T, np.dot(P, Z.T).T, 
										metric='euclidean')
				
				K = s*np.exp(-R)
				if self.i < d:
					if xeqz:
						if diag:
							Kprime = 0.0
						else:
							Kprime = distance_matrix(X[:,i,np.newaxis]/l[i], metric='sqeuclidean')
					else:
						Kprime = distance_matrix(X[:,i,np.newaxis]/l[i], 
												 Z[:,i,np.newaxis]/l[i], metric='sqeuclidean')
						
					R  = R+1e-16
					dK = K*Kprime/R
				elif self.i == d:
					#gradient of the variance parameter s
					dK = 2.0*K
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
					
				return dK
					
		fun = _DerivativeFun(self, i)
		return fun
	
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				s = np.exp(2.0*self.kernel.params[d])
		
				P = np.diag(1/l)
	
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					R = distance_matrix(np.dot(P, X.T).T, np.dot(P, Z.T).T, metric='euclidean')
					K = s*np.exp(-R)
					dK = np.zeros((m,n,d))
					R = R + 1e-16 #prevent division by zeros
					for i in xrange(d):
						dK[:,:,i] = 1.0/l[i]**2 * np.add.outer(Z[:,i],-X[:,i]) * K.T / R.T
				return dK
				
		return _DerivativeFun(self)
	
	def copy(self):
		d = self._d
		cp_kernel = ARDRBFKernel(self._params[0:d], self._params[d])
		return cp_kernel


class SEKernel(Kernel):
	'''
	Squared Exponential covariance kernel with isotropic distance measure. 
	The kernel is parametrized as:
	
	k(x_p, x_q) = s**2 * exp(-|x_p-x_q|^2 / l^2) 
	'''
	
	def __init__(self, l, s=0.0):
		Kernel.__init__(self, np.asarray([l,s]))
		
	def __str__( self ):
		return "SqExpKernel({0},{1})".format(self.params[0],self.params[1])
	
	def __call__(self, X, Z=None, diag=False):
		
		l = np.exp(2.0*self.params[0])
		s = np.exp(2.0*self.params[1])
		xeqz = (Z == None)
		
		if xeqz and diag:
			R = np.zeros(X.shape[0])
		else:
			R = distance_matrix(X, Z, metric='sqeuclidean')
			
		K = s * np.exp(-R/(2.0*l))
		return K
	
	def gradient(self, covGrad, X, Z=None, diag=False):
		'''
		'''
		l = np.exp(2.0*self.params[0])
		s = np.exp(2.0*self.params[1])
		xeqz = (Z == None)
		
		grad = np.zeros(2)

		if xeqz and diag:
			R = np.zeros(X.shape[0])
		else:
			R = distance_matrix(X, Z, metric='sqeuclidean')
		K = s * np.exp(-R/(2.0*l))
		
		#gradient of the length scale l
		dKl = K * R/l
		grad[0] = np.sum(covGrad, dKl)
		
		#gradient of the signal variance
		dKs = 2.0 * K
		grad[1] = np.sum(covGrad, dKs)

		return grad

		
	
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):
				l = np.exp(2.0*self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
		
				xeqz = (Z == None)
		
				if xeqz and diag:
					R = np.zeros(X.shape[0])
				else:
					R = distance_matrix(X, Z, metric='sqeuclidean')
				K = s * np.exp(-R/(2.0*l))
				
				if self.i == 0:
					#gradient of scale parameter l
					dK = K * R/l 
				elif self.i == 1:
					#gradient of the variance parameter s
					dK = 2.0 * K
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
				
				return dK
					
		fun = _DerivativeFun(self, i)
		return fun
	
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				l = np.exp(2.0*self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
		
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					R = distance_matrix(X, Z, metric='sqeuclidean')
					K = s * np.exp(-R/(2.0*l))
					dK = np.zeros((m,n,d))
					for i in xrange(d):
						dK[:,:,i] = 1.0/l * np.add.outer(Z[:,i],-X[:,i]) * K.T
						
				print 'shape'
				print dK.shape
				return dK
				
		return _DerivativeFun(self)
	
	def derivateXOld(self):
		class _DerivativeFunX(Kernel._IDerivativeFunX):
			def __init__(self, kernel):
				self.kernel = kernel
			
			def __call__(self, x, Z):
				x = np.squeeze(x)

				l = np.exp(self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
				
				R = distance_matrix(x, Z, metric='sqeuclidean')
				K = s * np.exp(-R/(2.0*(l**2)))
				
				d = len(x)
				G = np.zeros(Z.shape)
				for i in xrange(d):
					G[:,i] = 1.0/l**2 * (Z[:,i] - x[i]) * K
					
				return G
		fun = _DerivativeFunX(self)
		return fun


	
	def copy(self):
		cp_kernel = SEKernel(np.copy(self._params[0]), self._params[1])
		return cp_kernel
	
class ARDSEKernel(Kernel):
	'''
	Squared Exponential covariance kernel with automatic relevance determinition
	distance measure. The kernel is parametrized as:
	
	k(x_p, x_q) = s**2 * exp(-(x_p-x_q)' * inv(L) *  (x_p-x_q)/ 2) 
	'''
	
	__slots__ = ('_d')
	
	def __init__(self, l, s=0.0):
		l = np.asarray(l).ravel()
		params = np.r_[l,s]
		Kernel.__init__(self, params)
		
		self._d = len(l)
		
	def __str__( self ):
		return "ARDSEKernel()"
	
	def __call__(self, X, Z=None, diag=False):
		'''
		@todo: - optimize the distance computation of the diagonal dot prod
		'''
		
		xeqz = (Z == None)
			
		d = self._d
		l = np.exp(self.params[0:d])
		s = np.exp(2.0*self.params[d])
			
		P = np.diag(1/l)
		
		if xeqz:	
			if diag:
				K = np.zeros(X.shape[0])
			else:
				K = distance_matrix(np.dot(P, X.T).T,  metric='sqeuclidean') 
		else:
			K = distance_matrix(np.dot(P, X.T).T, np.dot(P, Z.T).T, metric='sqeuclidean')
			
		
		K = s*np.exp(-K/2.0)
		return K
			
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - check the gradient of the length scale vector (gradient check failed, 
				 but its identical to matlab gpml implementation)
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):	 
				xeqz = (Z == None)
					
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				s = np.exp(2.0*self.kernel.params[d])
					
				P = np.diag(1/l)
				if xeqz:
					if diag:
						K = np.zeros(X.shape[0])
					else:
						K = distance_matrix(np.dot(P, X.T).T,  metric='sqeuclidean') 
				else:
					K = distance_matrix(np.dot(P, X.T).T, np.dot(P, Z.T).T, 
										metric='sqeuclidean')
				
				K = s*np.exp(-K/2)
				if self.i < d:
					if xeqz:
						if diag:
							Kprime = 0.0
						else:
							Kprime = distance_matrix(X[:,i,np.newaxis]/l[i], metric='sqeuclidean')
					else:
						Kprime = distance_matrix(X[:,i,np.newaxis]/l[i], 
												 Z[:,i,np.newaxis]/l[i], metric='sqeuclidean')
					dK = K*Kprime
				elif self.i == d:
					#gradient of the variance parameter s
					dK = 2.0*K
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
					
				return dK
					
		fun = _DerivativeFun(self, i)
		return fun
	
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				s = np.exp(2.0*self.kernel.params[d])
		
				P = np.diag(1/l)
	
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					R = distance_matrix(np.dot(P, X.T).T, np.dot(P, Z.T).T, metric='sqeuclidean')
					K = s*np.exp(-R/2)
					dK = np.zeros((m,n,d))
					for i in xrange(d):
						dK[:,:,i] = 1.0/l[i]**2 * np.add.outer(Z[:,i],-X[:,i]) * K.T
				return dK
				
		return _DerivativeFun(self)
	
	def copy(self):
		d = self._d
		cp_kernel = ARDSEKernel(self._params[0:d], self._params[d])
		return cp_kernel

class ARDSELinKernel(Kernel):
	'''
	Squared Exponential + linear covariance kernel with automatic relevance determinition
	distance measure. The kernel is parametrized as:
	
	k(x_p, x_q) = s**2 * exp(-(x_p-x_q)' * inv(L) *  (x_p-x_q)/ 2) 
	'''
	
	__slots__ = ('_d')
	
	def __init__(self, l, se=0.0, sl=0.0):
		l = np.asarray(l).ravel()
		params = np.r_[l,se,sl]
		Kernel.__init__(self, params)
		
		self._d = len(l)
		
	def __str__( self ):
		return "ARDSELinKernel()"
	
	def __call__(self, X, Z=None, diag=False):
		'''
		@todo: - optimize the distance computation of the diagonal dot prod
		'''
		
		xeqz = (Z == None)
			
		d = self._d
		l = np.exp(self.params[0:d])
		se = np.exp(2.0*self.params[d])
		sl = np.exp(2.0*self.params[d+1])
			
		X = np.dot(X, np.diag(1/l))
		if xeqz:	
			if diag:
				Ke = np.zeros(X.shape[0])
				Kl = np.sum(X*X,1)
			else:
				Ke = distance_matrix(X,  metric='sqeuclidean')
				Kl = np.dot(X,X.T) 
		else:
			Z = np.dot(Z, np.diag(1.0/l))
			Ke = distance_matrix(X, Z, metric='sqeuclidean')
			Kl = np.dot(X,Z.T)
			
		
		Ke = np.exp(-Ke/2.0)
		K = se*Ke + sl*Kl
		return K
			
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - check the gradient of the length scale vector (gradient check failed, 
				 but its identical to matlab gpml implementation)
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):	 
				xeqz = (Z == None)
					
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				se = np.exp(2.0*self.kernel.params[d])
				sl = np.exp(2.0*self.kernel.params[d+1])
					
				Xl = np.dot(X, np.diag(1/l))
				if xeqz:
					if diag:
						Ke = np.zeros(Xl.shape[0])
						Kl = np.sum(Xl*Xl,1)
					else:
						Ke = distance_matrix(X, metric='sqeuclidean')
						Kl = np.dot(Xl,Xl.T) 
				else:
					Zl = np.dot(Z, np.diag(1.0/l))
					Kl = distance_matrix(Xl,Zl,metric='sqeuclidean')
					Kl = np.dot(Xl,Zl.T)
				
				Ke = se*np.exp(-Ke/2)
				Kl = sl*Kl
				if self.i < d:
					xl = Xl[:,i]
					if xeqz:
						if diag:
							Keprime = 0.0
							dKl = xl*xl
						else:
							Keprime = distance_matrix(X[:,i,np.newaxis]/l[i], metric='sqeuclidean')
							dKl = np.outer(xl,xl.T)
					else:
						Keprime = distance_matrix(X[:,i,np.newaxis]/l[i], 
												 Z[:,i,np.newaxis]/l[i], metric='sqeuclidean')
						zl = Zl[:,i]
						dKl = np.outer(xl,zl.T)
					dKe = Ke*Keprime
					dKl = -2.0*sl*dKl
					dK = dKe+dKl
				elif self.i == d:
					#gradient of the variance parameter s
					dK = 2.0*Ke
				elif self.i == d+1:
					dK = 2.0*Kl
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
					
				return dK
					
		fun = _DerivativeFun(self, i)
		return fun
	
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				s = np.exp(2.0*self.kernel.params[d])
		
				P = np.diag(1/l)
	
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					R = distance_matrix(np.dot(P, X.T).T, np.dot(P, Z.T).T, metric='sqeuclidean')
					K = s*np.exp(-R/2)
					dK = np.zeros((m,n,d))
					for i in xrange(d):
						dK[:,:,i] = 1.0/l[i]**2 * np.add.outer(Z[:,i],-X[:,i]) * K.T
				return dK
				
		return _DerivativeFun(self)
	
	def copy(self):
		d = self._d
		cp_kernel = ARDSELinKernel(self._params[0:d], self._params[d], self._params[d+1])
		return cp_kernel


class GaussianKernel(Kernel):
	'''
	Squared Exponential covariance kernel with automatic relevance determinition
	distance measure. The kernel is parametrized as:
	
	k(x_p, x_q) = exp(-(x_p-x_q)' * inv(L) *  (x_p-x_q)/ 2) 
	'''
	
	__slots__ = ('_d')
	
	def __init__(self, l):
		params = np.asarray(l).ravel()
		Kernel.__init__(self, params)
		
		self._d = len(l)
		
	def __str__( self ):
		return "ARDSEKernel()"
	
	def __call__(self, X, Z=None, diag=False):
		'''
		@todo: - optimize the distance computation of the diagonal dot prod
		'''
		
		xeqz = (Z == None)
			
		d = self._d
		l = np.exp(self.params)
			
		P = np.diag(1/l)
		
		if xeqz:
			
			if diag:
				K = np.zeros(X.shape[0])
			else:
				K = distance_matrix(np.dot(P, X.T).T,  metric='sqeuclidean') 
		else:
			K = distance_matrix(np.dot(P, X.T).T, np.dot(P, Z.T).T, metric='sqeuclidean')
			
		K = np.exp(-K/2.0)
		detP = np.prod(1/(l**2.0))
		K = (2*np.pi)**(d/2.0) * np.sqrt(detP) * K
		#K = np.sqrt(detP) * K
		return K
			
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - check the gradient of the length scale vector (gradient check failed, 
				 but its identical to matlab gpml implementation)
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):	 
				xeqz = (Z == None)
					
				d = self.kernel._d
				l = np.exp(self.kernel.params)
				detP = np.prod(1/(l**2.0))
				
				P = np.diag(1/l)
				if xeqz:
					if diag:
						K = np.zeros(X.shape[0])
					else:
						K = distance_matrix(np.dot(P, X.T).T,  metric='sqeuclidean') 
				else:
					K = distance_matrix(np.dot(P, X.T).T, np.dot(P, Z.T).T, 
										metric='sqeuclidean')
			
				K = np.exp(-K/2)
				K = (2*np.pi)**(d/2.0) * np.sqrt(detP) * K
				
				if self.i < d:
					#gradient of scale parameter l
					if xeqz:
						if diag:
							Kprime = 0.0
						else:
							Kprime = distance_matrix(X[:,i,np.newaxis]/l[i], metric='sqeuclidean')
					else:
						Kprime = distance_matrix(X[:,i,np.newaxis]/l[i], 
												 Z[:,i,np.newaxis]/l[i], metric='sqeuclidean')
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
				
				dK = K*Kprime
				dK = dK - K*np.sqrt(detP)
				dK = dK
				
				return dK
					
		fun = _DerivativeFun(self, i)
		return fun
	
	def derivateX(self):
		'''
		'''
		
		class _DerivativeFunX(Kernel._IDerivativeFunX):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, x, Z):
				x = np.squeeze(x)
				
				d = self.kernel._d
				l = np.exp(self.kernel.params)
		
				P = np.diag(1/l)
				detP = np.prod(1/(l**2.0))
				R = distance_matrix(np.dot(P, x), np.dot(P, Z.T).T, metric='sqeuclidean')
				K = np.exp(-R/2.0)
				K = (2*np.pi)**(d/2.0) * np.sqrt(detP) * K
				
				
				d = len(x)
				G = np.zeros(Z.shape)
				for i in xrange(d):
					G[:,i] = 1.0/l[i]**2 * (Z[:,i] - x[i]) * K
				
				#G = (1.0/l**2 * (Z - x).T * K).T
				return G
							
		fun = _DerivativeFunX(self)
		return fun

	
	def copy(self):
		cp_kernel = GaussianKernel(self._params)
		return cp_kernel

	
class LinearKernel(Kernel):
	'''
	todo: add a variance term
	'''
	def __init__(self):
		Kernel.__init__(self, np.array([]))
	
	def __str__(self):
		return 'Linear Kernel'
	
	def __call__(self, X, Z=None, diag=False):
		xeqz = (Z==None)
		if xeqz:
			if diag:
				K = np.sum(X*X,1)
			else:
				K = np.dot(X,X.T)
		else:
			K = np.dot(X,Z.T)
		return K
	
	def derivate(self, i):
		'''
		@todo: return zero array
		'''
		pass
	
	def derivateXOld(self):
		'''
		'''
		class _DerivativeFunX(Kernel._IDerivativeFunX):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, x, Z):
				G = Z.copy()
				n = G.shape[0]
				for i in xrange(n):
					#todo: make diag part more clearly
					if np.all(np.equal(x, G[i])):
						G[i] = G[i]*2.0
				return G
							
		fun = _DerivativeFunX(self)
		return fun
	
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					print m
					dK = np.tile(Z[:,np.newaxis,:], (1,n,1))
					if xeqz:
						dK[np.diag(np.ones(n,dtype=np.bool)),:] *= 2.0
				print 'shape'
				print dK.shape
				return dK
				
		return _DerivativeFun(self)
	
	def copy(self):
		cp_kernel = LinearKernel()
		return cp_kernel

class BiasedLinearKernel(Kernel):
	
	def __init__(self, bias):
		Kernel.__init__(self, np.array([bias]))
	
	def __str__(self):
		return 'BiasedLinearKernel({0})'.format(self.params[0])
	
	def __call__(self, X, Z=None, diag=False):
		bias = np.exp(2.0*self.params[0])
		xeqz = (Z==None)
		
		if xeqz:
			if diag:
				K = np.sum(X*X,1)
			else:
				K = np.dot(X,X.T)
		else:
			K = np.dot(X,Z.T)
			
		K = (K+1.0)/bias 
		return K
	
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				bias = np.exp(2.0*self.kernel.params[0])
				xeqz = (Z==None)
						
				if xeqz:
					if diag:
						K = np.sum(X*X,1)
					else:
						K = np.dot(X,X.T)
				else:
					K = np.dot(X,Z.T)
				
				(K+1.0)/bias
				dK = -2.0*K
				return dK
			
		fun = _DerivativeFun(self)
		return fun
	
	def derivateX(self):
		'''
		'''
		class _DerivativeFunX(Kernel._IDerivativeFunX):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, x, Z):
				bias = np.exp(2.0*self.kernel.params[0])
				
				G = Z.copy()
				n = G.shape[0]
				for i in xrange(n):
					#todo: make diag part more clearly
					if np.all(np.equal(x, G[i])):
						G[i] = G[i]*2.0
				G = G/bias
				return G
							
		fun = _DerivativeFunX(self)
		return fun
	
	def copy(self):
		cp_kernel = BiasedLinearKernel(self._params[0])
		return cp_kernel
	
class ARDLinearKernel(Kernel):
	
	__slots__ = ('_d')
	
	def __init__(self, l, s=0.0):
		l = np.asarray(l).ravel()
		params = np.r_[l,s]
		Kernel.__init__(self, params)
		
		self._d = len(l)
		
	def __str__(self):
		return 'ARDLinearKernel()'
	
	def __call__(self, X, Z=None, diag=False):
		d = self._d
		l = np.exp(self.params[0:d])
		s = np.exp(2.0*self.params[d])
		
		xeqz = (Z==None)
		
		X = np.dot(X, np.diag(1.0/l))
		if xeqz:
			if diag:
				K = np.sum(X*X,1)
			else:
				K = np.dot(X,X.T)
		else:
			Z = np.dot(Z, np.diag(1.0/l))
			K = np.dot(X,Z.T)
		
		K *= s
		return K
	
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False, K=None):
				i = self.i
				
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				s = np.exp(2.0*self.kernel.params[d])
				
				xeqz = (Z == None)
				
				X = np.dot(X, np.diag(1.0/l))
				if i < d:
					x = X[:,i]
					if xeqz:
						if diag:
							dK = x*x
						else:
							dK = np.outer(x,x.T)
					else:
						z = Z[:,i]
						dK = np.outer(z,z.T)
						
					dK = -2.0*s*dK
				elif i == d:
					if K == None:
						if xeqz:
							if diag:
								K = np.sum(X*X,1)
							else:
								K = np.dot(X,X.T)
						else:
							Z = np.dot(Z, np.diag(1.0/l))
							K = np.dot(X,Z.T)
						K = s*K
						
					dK = 2.0*K
				else:
					raise TypeError('Unknown hyperparameter')
				
				#K = -2*K
				return dK
			
		fun = _DerivativeFun(self, i)
		return fun

	def derivateX(self):
		'''
		'''
		class _DerivativeFunX(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, x, Z):
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				s = np.exp(2.0*self.kernel.params[d])
				
				G = Z.copy()
				n = G.shape[0]
				for i in xrange(n):
					#todo: make diag part more clearly
					if np.all(np.equal(x, G[i])):
						G[i] = G[i]*2.0
				G = s * G*l
				return G
							
		fun = _DerivativeFunX(self)
		return fun
	
	def copy(self):
		d = self._d
		cp_kernel = ARDLinearKernel(np.copy(self._params[0:d]), self._params[d])
		return cp_kernel
	

class PolynomialKernel(Kernel):
	
	__slots__ = ('_degree')
	
	def __init__(self, degree, c, s):
		Kernel.__init__(self, np.array([c,s]))
		self._degree = degree
		
	def __str__(self):
		return 'PolynomialKernel({0},{1})'.format(self.params[0],self.params[1])
	
	def __call__(self, X, Z=None, diag=False):
		d = self._degree
		c = np.exp(self.params[0])
		s = np.exp(2.0*self.params[1])
		xeqz = (Z==None)
		
		if xeqz: 
			if diag:
				K = np.sum(X*X,1)			
			else:
				K = np.dot(X,X.T)
		else:
			K = np.dot(X,Z.T)
			
		K = s * (K + c)**d
		return K

	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):
				d = self.kernel._degree
				c = np.exp(self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
				
				xeqz = (Z==None)
		
				if xeqz: 
					if diag:
						K = np.sum(X*X,1)
					else:
						K = np.dot(X,X.T)
				else:
					K = np.dot(X,Z.T)
								
				if self.i == 0:
					#gradient of bias parameter c
					K = s * c * d * (K + c)**(d-1)
		
				elif self.i == 1:
					K = 2.0*s * (K + c)**d
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
				
				return K
		
		fun = _DerivativeFun(self, i)
		return fun
	
	def derivateX(self):
		'''
		'''
		class _DerivativeFunX(Kernel._IDerivativeFunX):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, x, Z):
				d = self.kernel._degree
				c = np.exp(self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
				
				K = (np.dot(Z, x) + c)**(d-1)
				
				m = len(x)
				G = np.zeros(Z.shape)
				for i in xrange(m):
					G[:,i] = d*s*Z[:,i]*K

				n = len(Z)
				for i in xrange(n):
					#todo: make diag part more clearly
					if np.all(np.equal(x, Z[i])):
						G[i] = G[i]*2.0
				
				return G
							
		fun = _DerivativeFunX(self)
		return fun

	
	def copy(self):
		cp_kernel = PolynomialKernel(self._degree, self._params[0], self._params[1])
		return cp_kernel

class ARDPolynomialKernel(Kernel):
	
	__slots__ = ('_degree',
				 '_d')
	
	def __init__(self, degree, l, c, s):
		l = np.asarray(l).ravel()
		params = np.r_[l,c,s]
		Kernel.__init__(self, params)
		
		self._d = len(l)
		self._degree = degree
		
	def __str__(self):
		return 'PolynomialKernel({0},{1})'.format(self.params[0],self.params[1])
	
	def __call__(self, X, Z=None, diag=False):
		deg = self._degree
		d = self._d
		l = np.exp(self.params[0:d])
		c = np.exp(self.params[d])
		s = np.exp(2.0*self.params[d+1])
		
		xeqz = (Z==None)
		
		X = np.dot(X, np.diag(1.0/l))
		if xeqz: 
			if diag:
				K = np.sum(X*X,1)			
			else:
				K = np.dot(X,X.T)
		else:
			Z = np.dot(Z, np.diag(1.0/l))
			K = np.dot(X,Z.T)
			
		K = s * (K + c)**deg
		return K

	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):
				deg = self.kernel._degree
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				c = np.exp(self.kernel.params[d])
				s = np.exp(2.0*self.kernel.params[d+1])
				
				xeqz = (Z==None)
		
				X = np.dot(X, np.diag(1.0/l))
				if xeqz: 
					if diag:
						K = np.sum(X*X,1)
					else:
						K = np.dot(X,X.T)
				else:
					K = np.dot(X,np.dot(Z, np.diag(1.0/l)).T)
					
				if self.i < d:
					x = X[:,i]
					if xeqz:
						if diag:
							dK = x*x
						else:
							dK = np.outer(x,x.T)
					else:
						z = Z[:,i]
						dK = np.outer(z,z.T)
						
					dK = np.dot(s*deg*dK, (K+c)**(deg-1))

				elif self.i == d:
					#gradient of bias parameter c
					K = s * c * deg * (K + c)**(deg-1)
		
				elif self.i == d+1:
					K = 2.0*s * (K + c)**deg
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
				
				return K
		
		fun = _DerivativeFun(self, i)
		return fun
	
	def derivateX(self):
		'''
		'''
		class _DerivativeFunX(Kernel._IDerivativeFunX):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, x, Z):
				d = self.kernel._degree
				c = np.exp(self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
				
				K = (np.dot(Z, x) + c)**(d-1)
				
				m = len(x)
				G = np.zeros(Z.shape)
				for i in xrange(m):
					G[:,i] = d*s*Z[:,i]*K

				n = len(Z)
				for i in xrange(n):
					#todo: make diag part more clearly
					if np.all(np.equal(x, Z[i])):
						G[i] = G[i]*2.0
				
				return G
							
		fun = _DerivativeFunX(self)
		return fun

	
	def copy(self):
		cp_kernel = PolynomialKernel(self._degree, self._params[0], self._params[1])
		return cp_kernel

	
class PiecewisePolyKernel(Kernel):
	
	__slots__ = ('v'	#degree of the polynom
				 )
	pass

class RQKernel(Kernel):
	pass

class MaternKernel(Kernel):
	'''
	@todo: -check if euclidean distance is correct
	'''
	__slots__ = ('_degree',
				 '_m',
				 '_dm')
	
	def __init__(self, degree, l, s):
		Kernel.__init__(self, np.array([l,s]))
		self._degree = degree
		
		if degree == 1:
			f = lambda t: 1
			df = lambda t: 1
			pass
		elif degree == 3:
			f = lambda t: 1 + t
			df = lambda t: t
			pass
		elif degree == 5:
			f = lambda t: 1 + t*(1.0+t/3.0)
			df = lambda t: t*(1.0+t)/3.0
		else:
			raise ValueError('degree must be 1, 3 or 5.')
		
		m = lambda t: f(t)*np.exp(-t)
		dm = lambda t: df(t)*t*np.exp(-t)
		
		self._m = m
		self._dm = dm 
		
		
	def __str__(self):
		return 'MaternKernel()'
	
	def __call__(self, X, Z=None, diag=False):
		
		d = self._degree
		m = self._m
		l = np.exp(self.params[0])
		s = np.exp(2.0*self.params[1])
		
		xeqz = (Z==None)
		if xeqz:
			if diag:
				R = np.zeros(X.shape[0])
			else:
				R = distance_matrix(X, metric='euclidean')
		else:
			R = distance_matrix(X, Z, metric='euclidean')
		
		K = np.sqrt(d)*R/l   
		K = s*m(K)
		return K

	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):
				
				d = self.kernel._degree
				m = self.kernel._m
				dm = self.kernel._dm
				
				l = np.exp(self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
				
				xeqz = (Z==None)
				if xeqz:
					if diag:
						R = np.zeros(X.shape[0])
					else:
						R = distance_matrix(X, metric='euclidean')
				else:
					R = distance_matrix(X, Z, metric='euclidean')
					
				K = np.sqrt(d)*R/l	   
				if self.i == 0:
					#todo: has big error for high variance or length scale
					K = s*dm(K)
				elif self.i == 1:
					K = 2.0*s * m(K)
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
				
				return K
		
		fun = _DerivativeFun(self, i)
		return fun

	def copy(self):
		cp_kernel = MaternKernel(self._degree, self._params[0], self._params[1])
		return cp_kernel


class NeuralNetKernel(Kernel):
	pass

class PeriodicKernel(Kernel):

	def __init__(self, l, p, s=0.0):
		Kernel.__init__(self, np.asarray([l,p,s]))

	def __str__( self ):
		return "PeriodicKernel()"
	
	def __call__(self, X, Z=None, diag=False):
		
		l = np.exp(self.params[0])
		p = np.exp(self.params[1])
		s = np.exp(2.0*self.params[2])

		xeqz = (Z==None)
		if xeqz:
			if diag:
				R = np.zeros(X.shape[0])
			else:
				R = distance_matrix(X, metric='euclidean')
		else:
			R = distance_matrix(X, Z, metric='euclidean')
		
		K = np.pi*R/p
		K = (np.sin(K)/l)**2
		K = s * np.exp(-2.0*K)
		return K
			
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):
				l = np.exp(self.kernel.params[0])
				p = np.exp(self.kernel.params[1])
				s = np.exp(2.0*self.kernel.params[2])
				
				xeqz = (Z==None)
				if xeqz:
					if diag:
						R = np.zeros(X.shape[0])
					else:
						R = distance_matrix(X, metric='euclidean')
				else:
					R = distance_matrix(X, Z, metric='euclidean')
				
				K = np.pi*R/p
				if self.i == 0:
					K = np.sin(K)**2
					K = 4.0*s * K * np.exp(-2.0*K/l**2) / l**2
				elif self.i == 1:
					R = 4.0*s * np.pi*R
					K = R * np.cos(K) * np.sin(K) * np.exp(-2.0*(np.sin(K)/l)**2)
					K = K/(l**2 * p)
				elif self.i == 2:
					K = (np.sin(K)/l)**2
					K = 2.0*s * np.exp(-2.0*K)
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
				
				return K
					
		fun = _DerivativeFun(self, i)
		return fun

	def copy(self):
		cp_kernel = PeriodicKernel(self._params[0], self._params[1], self._params[2])
		return cp_kernel

class ConvolvedKernel(Kernel):
	
	__slots__ = ('_ntheta'
				)
	
	def __init__(self, params, ntheta=0):
		'''
		'''
		Kernel.__init__(self, params)
		self._ntheta = ntheta
	
		
	def __call__(self, X, Z=None, thetaP=None, thetaQ=None, diag=False, latent=False):
		'''
		Computes the covariance matrix cov[f_p(x), f_p(x')] of the convolved process f_p
		Computes the cross covariance matrix cov[f_p(x), f_q(x')] of the convolved process f_p and f_q
		Computes the cross covariance matrix cov[f_p(x), u(x')] of the convolved process f_p and the latent process u
		'''
		xeqz = (Z==None)
		
		if xeqz:
			if latent:
				K = self.latent_cov(X, diag=diag)
			else:
				K = self.cov(X, thetaP, diag)
		else:
			if latent and thetaP==None and thetaQ==None:
				K = self.latent_cov(X, Z, diag)
			else:
				K = self.cross_cov(X, Z, thetaP, thetaQ, latent)
			
		return K
	
	@abstractmethod
	def cov(self, X, theta, diag=False):
		'''
		'''

	@abstractmethod
	def cross_cov(self, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		Computes the cross covariance matrix cov[f_p(x), f_q(x)]
		Computes the cross covariance matrix cov[f_p(x), f_q(x)]
		'''
		
	@abstractmethod
	def latent_cov(self, X, Z=None, diag=False):
		'''
		Computes the covariance matrix cov[u(x), u(x')] uof the latent process u
		'''
	
	@abstractmethod
	def derivateTheta(self, i):
		'''
		Returns the derivateve of the smoothing kernel parameters
		'''
	
	@abstractmethod
	def derivateX(self):
		'''
		'''
		
	def _number_of_theta(self):
		return self._ntheta
	
	ntheta = property(fget=_number_of_theta)
	
	
	def copy(self):
		params = np.copy(self._params)
		new_kernel = self.__class__(params)
		new_kernel._ntheta = self._ntheta
		return new_kernel
			
	class _IDerivativeFun(object):

		def __call__(self, X, Z=None, thetaP=None, thetaQ=None, diag=False, latent=False):
		
			xeqz = (Z==None)
			
			if xeqz:
				if latent:
					K = self.latent_cov(X, diag)
				else:
					K = self.cov(X, thetaP, diag)
			else:
				if latent and thetaP==None and thetaQ==None:
					K = self.latent_cov(X, Z, diag)
				else:
					K = self.cross_cov(X, Z, thetaP, thetaQ, latent)
				
				
			return K


		@abstractmethod
		def cov(self, X, theta, diag=False):
			'''
			'''
		
		@abstractmethod
		def cross_cov(self, X, Z, thetaP, thetaQ=None, latent=False):
			'''
			'''
		
		@abstractmethod
		def latent_cov(self, X, Z=None, diag=False):
			'''
			'''

	class _IDerivativeFunX(object):

		def __call__(self, x, Z, thetaP=None, thetaQ=None, latent=False):
		
			xeqz = (thetaP==None or (thetaQ==None and latent==False))

			if xeqz:
				if latent:
					K = self.latent_cov(x, Z)
				else:
					K = self.cov(x, Z, thetaP)
			else:
				K = self.cross_cov(x, Z, thetaP, thetaQ, latent)
				
			return K


		@abstractmethod
		def cov(self, x, Z, theta):
			'''
			'''
		
		@abstractmethod
		def cross_cov(self, x, Z, thetaP, thetaQ=None, latent=False):
			'''
			'''
		
		@abstractmethod
		def latent_cov(self, x, Z):
			'''
			'''

class DiracConvolvedKernel(ConvolvedKernel):

	__slots__ = ('_kernel'	#latent kernel
				)

	def __init__(self, kernel):
		'''
		'''
		params = kernel.params
		self._kernel = kernel
		self._params = params
		self._n = len(params)
		self._ntheta = 1


	def cov(self, X, theta, diag=False):
		'''
		'''
		kernel = self._kernel
		s = np.exp(2*theta[0])
		K = kernel(X, diag=diag)
		K = s*K
		return K
	
	def cross_cov(self, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		Computes the cross covariance matrix cov[f_p(x), f_q(x')]
		Computes the cross covariance matrix cov[f_p(x), u(x')]
		'''
		kernel = self._kernel
		K = kernel(X,Z)
		sp = np.exp(thetaP[0])
		K = sp*K
		
		if not latent:
			sq = np.exp(thetaQ[0])
			K = sq*K
			
		return K
		
	def latent_cov(self, X, Z=None, diag=False):
		'''
		Computes the covariance matrix cov[u(x), u(x')] uof the latent process u
		'''	
		kernel = self._kernel
		K = kernel(X, Z, diag=diag)
		return K
	
	def derivate(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.dKernel = kernel.derivate(i)
			
			def cov(self, X, theta, diag=False):
				dKernel = self.dKernel
				s = np.exp(2*theta[0])
				dK = s*dKernel(X, diag=diag)
				return dK
			
			def cross_cov(self, X, Z, thetaP, thetaQ=None, latent=False):
				dKernel = self.dKernel
				
				dK = dKernel(X,Z)
				sp = np.exp(thetaP[0])
				dK = sp*dK
				
				if not latent:
					sq = np.exp(thetaQ[0])
					dK = sq*dK
					
				return dK
			
			def latent_cov(self, X, Z=None, diag=False):	
				dKernel = self.dKernel
				dK = dKernel(X,Z,diag=diag)
				return dK

								
		fun = _DerivativeFun(self._kernel, i)
		return fun

	def derivateTheta(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.dKernel = kernel.derivate(i)
			
			def cov(self, X, theta, diag=False):
				kernel = self.kernel
				s = np.exp(2*theta[0])
				dK = 2*s*kernel(X, diag=diag)
				return dK
			
			def cross_cov(self, X, Z, thetaP, thetaQ=None, latent=False):
				kernel = self.kernel
				
				sp = np.exp(thetaP[0])
				K = kernel(X,Z)
				dK = sp*K
				if not latent:
					sq = np.exp(thetaQ[0])
					dKp = sq*dK
					dKq = sq*dK
					return dKp, dKq
					
				return dK
			
			
			def latent_cov(self, X, Z=None, diag=False):
				raise NotImplementedError('latent cov has no hyperparameter theta')

		if i != 0:
			raise TypeError('Unknown hyperparameter')
			
								
		fun = _DerivativeFun(self._kernel, i)
		return fun

	def derivateX(self):
		'''
		'''
		
		class _DerivativeFunX(object):
			def __init__(self, kernel):
				self.dKernelX = kernel.derivateX()
				
			def cov(self, x, Z, theta):
				dKernelX = self.dKernelX
				s = np.exp(2*theta[0])
				dK = 2*s*dKernelX(x, Z)
				return dK

			def cross_cov(self, x, Z, thetaP, thetaQ=None, latent=False):
				dKernelX = self.dKernelX
				
				sp = np.exp(thetaP[0])
				dK = sp*dKernelX(x,Z)
				
				if not latent:
					sq = np.exp(thetaQ[0])
					dK = sq*dK
					
				return dK
			
			def latent_cov(self, x, Z):
				dKernelX = self.dKernelX
				dK = dKernelX(x,Z)
				return dK
							
		fun = _DerivativeFunX(self)
		return fun
	
	def copy(self):
		cp_kernel = DiracConvolvedKernel(self._kernel.copy())
		return cp_kernel
		
class ExpGaussianKernel(ConvolvedKernel):

	__slots__ = ('_d')

	def __init__(self, l):
		'''
		'''
		l = np.asarray(l).ravel()
		d = len(l)
		ConvolvedKernel.__init__(self, l, d+1)
		self._d = d


	def cov(self, X, theta, diag=False):
		'''
		'''
		d = self._d
		l = np.exp(2*self.params[0:d])
		lp = np.exp(4.0*theta[0:d])
		sp = np.exp(2.0*theta[d])
		
		lpu = l + lp
		detL = np.prod(l)
		detLpu = np.prod(lpu)
		
		K = self._compute_gausskern(lpu, X, diag=diag)
		K = sp * np.sqrt(detLpu/detL) * K
		
		return K
	
	def cross_cov(self, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		Computes the cross covariance matrix cov[f_p(x), f_q(x')]
		Computes the cross covariance matrix cov[f_p(x), u(x')]
		'''
		d = self._d
		l = np.exp(2*self.params[0:d])
		lp = np.exp(2*thetaP[0:d])
		sp = np.exp(thetaP[d])
		
		if latent:
			'''
			Compute cov[f_p(x), u(x')]
			'''
			lpu = l + lp
			detL = np.prod(l)
			detLpu = np.prod(lpu)
		
			K = self._compute_gausskern(lpu, X, Z)
			K = sp * np.sqrt(detLpu/detL) * K

		else:
			'''
			Compute cov[f_p(x), f_q(x')]
			'''			
			lq = np.exp(2*thetaQ[0:d])
			sq = np.exp(thetaQ[d])
			
			lpq = l+lp+lq
			detL = np.prod(l)
			detLpq = np.prod(lpq)

			K = self._compute_gausskern(lpq, X, Z)
			K = sp * sq * np.sqrt(detLpq/detL) * K
		
		return K			
		
	def latent_cov(self, X, Z=None, diag=False):
		'''
		Computes the covariance matrix cov[u(x), u(x')] uof the latent process u
		'''			
		d = self._d
		l = np.exp(2*self.params[0:d])
		K = self._compute_gausskern(l, X, Z, diag=diag)
		return K

	def derivate(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
			
			def cov(self, X, theta, diag=False):
				d = self.kernel._d
				l = np.exp(2*self.kernel.params[0:d])
				lp = np.exp(4.0*theta[0:d])
				sp = np.exp(2.0*theta[d])
				
				lpu = l + lp
				detL = np.prod(l)
				detLpu = np.prod(lpu)
				detLpui = l[i]*detLpu/lpu[i]
				dl = np.sqrt(l[i])/lpu[i]
				
				K = self.kernel._compute_gausskern(lpu, X, diag=diag)
				if diag:
					Kprime = 0
				else:
					Kprime = distance_matrix(X[:,i,np.newaxis]*dl, metric='sqeuclidean')
					
				dK = K*Kprime
				dnorm = detLpui/np.sqrt(detL*detLpu) - np.sqrt(detLpu/detL) 
				dK = np.sqrt(detLpu/detL) * dK + dnorm * K
				dK = sp * dK
				
				return dK
			
			def cross_cov(self, X, Z, thetaP, thetaQ=None, latent=False):
				d = self.kernel._d
				l = np.exp(2*self.kernel.params[0:d])
				lp = np.exp(2*thetaP[0:d])
				sp = np.exp(thetaP[d])
				
				if latent:
					'''
					Compute cov[f_p(x), u(x')]
					'''
					lpu = l + lp
					detL = np.prod(l)
					detLpu = np.prod(lpu)
					detLpui = l[i]*detLpu/lpu[i]
					dl = np.sqrt(l[i])/lpu[i]
					
					K = self.kernel._compute_gausskern(lpu, X, Z)
					Kprime = distance_matrix(X[:,i,np.newaxis]*dl, Z[:,i,np.newaxis]*dl, metric='sqeuclidean')
					dK = K*Kprime
					dnorm = detLpui/np.sqrt(detL*detLpu) - np.sqrt(detLpu/detL) 
					dK = np.sqrt(detLpu/detL) * dK + dnorm * K
					dK = sp * dK
		
				else:
					'''
					Compute cov[f_p(x), f_q(x')]
					'''
					lq = np.exp(2*thetaQ[0:d])
					sq = np.exp(thetaQ[d])
					
					lpq = l+lp+lq
					detL = np.prod(l)
					detLpq = np.prod(lpq)
					detLpqi = l[i]*detLpq/lpq[i]
					dl = np.sqrt(l[i])/lpq[i]
					
					K = self.kernel._compute_gausskern(lpq, X)
					Kprime = distance_matrix(X[:,i,np.newaxis]*dl, Z[:,i,np.newaxis]*dl, metric='sqeuclidean')
					dK = K*Kprime
					dnorm = detLpqi/np.sqrt(detL*detLpq) - np.sqrt(detLpq/detL) 
					dK = np.sqrt(detLpq/detL) * dK + dnorm * K
					dK = sp * sq * dK
			
				return dK			
			
			def latent_cov(self, X, Z=None, diag=False):
				d = self.kernel._d
				xeqz = Z == None
				if self.i < d:	
					l = np.exp(2*self.kernel.params[0:d])
					K = self.kernel._compute_gausskern(l, X, Z, diag=diag)
					if xeqz:
						if diag:
							Kprime = 0
						else:
							Kprime = distance_matrix(X[:,i,np.newaxis]/np.sqrt(l[i]), metric='sqeuclidean')
					else:
						Kprime = distance_matrix(X[:,i,np.newaxis]/np.sqrt(l[i]), 
												 Z[:,i,np.newaxis]/np.sqrt(l[i]), metric='sqeuclidean')
					dK = K*Kprime
				else:
					raise TypeError('Unknown hyperparameter')
				return dK

		if i >= self._d:	
			raise TypeError('Unknown hyperparameter')
								
		fun = _DerivativeFun(self, i)
		return fun

	def derivateTheta(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
			
			def cov(self, X, theta, diag=False):
				d = self.kernel._d
				l = np.exp(2*self.kernel.params[0:d])
				lp = np.exp(4.0*theta[0:d])
				sp = np.exp(2.0*theta[d])
				
				
				lpu = l + lp
				detL = np.prod(l)
				detLpu = np.prod(lpu)
				
				K = self.kernel._compute_gausskern(lpu, X, diag=diag)
				if i < d:
					detLpui = lp[i]*detLpu/lpu[i]
					dlp = np.sqrt(lp[i])/lpu[i]
					
					if diag:
						Kprime = 0
					else:
						Kprime = distance_matrix(X[:,i,np.newaxis]*dlp, metric='sqeuclidean')
						
					dK = K*Kprime
					dnorm = detLpui/np.sqrt(detL*detLpu) 
					dK = np.sqrt(detLpu/detL) * dK + dnorm  * K
					dK = 2.0*sp*dK
				elif self.i == d:
					dK = 2.0*sp*np.sqrt(detLpu/detL)*K
				else:
					raise ValueError('Unknown hyperparameter')
				
				return dK
			
			def cross_cov(self, X, Z,  thetaP, thetaQ=None, latent=False):
				d = self.kernel._d
				l = np.exp(2*self.kernel.params[0:d])
				lp = np.exp(2*thetaP[0:d])
				sp = np.exp(thetaP[d])
				
				if latent:
					'''
					Compute cov[f_p(x), u(x')]
					'''
					lpu = l + lp
					detL = np.prod(l)
					detLpu = np.prod(lpu)
					
					K = self.kernel._compute_gausskern(lpu, X, Z)
					if i < d:
						detLpui = lp[i]*detLpu/lpu[i]
						dlp = np.sqrt(lp[i])/lpu[i]	
						
						Kprime = distance_matrix(X[:,i,np.newaxis]*dlp, Z[:,i,np.newaxis]*dlp, metric='sqeuclidean')
						dK = K*Kprime
						dnorm = detLpui/np.sqrt(detL*detLpu) 
						dK = np.sqrt(detLpu/detL) * dK + dnorm * K
						dK = sp * dK
					elif self.i == d:
						dK = sp*np.sqrt(detLpu/detL)*K
					else:
						raise ValueError('Unknown hyperparameter')
		
				else:
					'''
					Compute cov[f_p(x), f_q(x')]
					'''
					
					lq = np.exp(2*thetaQ[0:d])
					sq = np.exp(thetaQ[d])
						
					lpq = l+lp+lq
					detL = np.prod(l)
					detLpq = np.prod(lpq)
					K = self.kernel._compute_gausskern(lpq, X, Z)
					if i < d:
						detLpi = lp[i]*detLpq/lpq[i]
						detLqi = lq[i]*detLpq/lpq[i]
						dlp = np.sqrt(lp[i])/lpq[i]
						dlq = np.sqrt(lq[i])/lpq[i]
						
						Kprime_p = distance_matrix(X[:,i,np.newaxis]*dlp, Z[:,i,np.newaxis]*dlp, metric='sqeuclidean')
						Kprime_q = distance_matrix(X[:,i,np.newaxis]*dlq, Z[:,i,np.newaxis]*dlq, metric='sqeuclidean')
						dKp = K*Kprime_p
						dKq = K*Kprime_q
						dnorm_p = detLpi/np.sqrt(detL*detLpq) 
						dnorm_q = detLqi/np.sqrt(detL*detLpq)
						dKp = np.sqrt(detLpq/detL) * dKp + dnorm_p * K
						dKp = sp * sq * dKp
						dKq = np.sqrt(detLpq/detL) * dKq + dnorm_q * K
						dKq = sp * sq * dKq
					elif self.i == d:
						dKp = sq*sp*np.sqrt(detLpq/detL)*K
						dKq = sp*sq*np.sqrt(detLpq/detL)*K
					else:
						raise ValueError('Unknown hyperparameter')
					
					return dKp, dKq
				
				return dK			
			
			def latent_cov(self, X, Z=None, diag=False):
				raise NotImplementedError('latent cov has no hyperparameter theta')
			
		fun = _DerivativeFun(self, i)
		return fun

	
	def derivateX(self):
		raise NotImplementedError()
	
	def _compute_gausskern(self, l, X, Z=None, diag=False):
		P = np.diag(np.sqrt(1.0/l))
		xeqz = Z==None
		
		if xeqz:	
			if diag:
				K = np.zeros(X.shape[0])
			else:
				K = distance_matrix(np.dot(P, X.T).T,  metric='sqeuclidean') 
		else:
			K = distance_matrix(np.dot(P, X.T).T, np.dot(P, Z.T).T, metric='sqeuclidean')
		K = np.exp(-K/2)
		return K
	
class ExpARDSEKernel(ExpGaussianKernel):

	'''
		todo: maybe inherit from ExpGaussianKernel
	'''

	__slots__ = ()

	def __init__(self, l, s=0.0):
		l = np.asarray(l).ravel()
		d = len(l)
		super(ExpARDSEKernel, self).__init__(l)
		ConvolvedKernel.__init__(self, l, d+1)
		
		params = np.r_[l,s]
		self._params = params
		self._n = len(params)

	def cov(self, X, theta, diag=False):
		'''
		'''
		d = self._d
		s = np.exp(self._params[d])
		K = super(ExpARDSEKernel, self).cov(X, theta, diag)
		K = s*K
		return K
	
	def cross_cov(self, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		Computes the cross covariance matrix cov[f_p(x), f_q(x')]
		Computes the cross covariance matrix cov[f_p(x), u(x')]
		'''
		d = self._d
		s = np.exp(self._params[d])
		K = super(ExpARDSEKernel, self).cross_cov(X, Z, thetaP, thetaQ, latent)
		K = s*K
		return K
			
	def latent_cov(self, X, Z=None, diag=False):
		'''
		Computes the covariance matrix cov[u(x), u(x')] uof the latent process u
		'''			
		d = self._d 
		s = np.exp(2*self._params[d])
		K = super(ExpARDSEKernel, self).latent_cov(X, Z, diag)
		K = s*K
		return K
	
		
	def derivate(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
			
			def cov(self, X, theta, diag=False):
				kernel = self.kernel
				i = self.i
				d = self.kernel._d 
				s = np.exp(self.kernel._params[d])
				
				if i<d:
					dkernel = super(ExpARDSEKernel, kernel).derivate(i)
					dK = dkernel.cov(X, theta, diag)
					dK = s*dK
				elif i==d:
					dK = self.kernel.cov(X, theta, diag)
				else:
					raise ValueError("unknown hyperparameter")
				return dK
			
			def cross_cov(self, X, Z, thetaP, thetaQ=None, latent=False):
				kernel = self.kernel
				i = self.i
				d = self.kernel._d 
				s = np.exp(self.kernel._params[d])
				
				if i<d:
					dkernel = super(ExpARDSEKernel, kernel).derivate(i)
					dK = dkernel.cross_cov(X, Z, thetaP, thetaQ, latent)
					dK = s*dK
				elif i==d:
					dK = self.kernel.cross_cov(X, Z, thetaP, thetaQ, latent)
				else:
					raise ValueError("unknown hyperparameter")
				return dK
						
			
			def latent_cov(self, X, Z=None, diag=False):
				kernel = self.kernel
				i = self.i
				d = self.kernel._d 
				s = np.exp(2*self.kernel._params[d])
				
				if i<d:
					dkernel = super(ExpARDSEKernel, kernel).derivate(i)
					dK = dkernel.latent_cov(X, Z, diag)
					dK = s*dK
				elif i==d:
					dK = self.kernel.latent_cov(X, Z, diag)
					dK = 2*dK
				else:
					raise ValueError("unknown hyperparameter")
				return dK

		if i >= self.nparams:	
			raise TypeError('Unknown hyperparameter')
								
		fun = _DerivativeFun(self, i)
		return fun

	def derivateTheta(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.dkernel = super(ExpARDSEKernel, kernel).derivateTheta(i)
				self.i = i
			
			def cov(self, X, theta, diag=False):
				d = self.kernel._d
				s = np.exp(self.kernel._params[d])
				
				dK = self.dkernel.cov(X, theta, diag)
				dK = s*dK
				return dK
						
			def cross_cov(self, X, Z, thetaP, thetaQ=None, latent=False):
				d = self.kernel._d
				s = np.exp(self.kernel._params[d])
				
				if not latent:
					dKp, dKq = self.dkernel.cross_cov(X, Z, thetaP, thetaQ, latent)
					dKp = s*dKp
					dKq = s*dKq
					return dKp, dKq
				else:
					dK = self.dkernel.cross_cov(X, Z, thetaP, thetaQ, latent)
					dK = s*dK
					return dK
							
			def latent_cov(self, X, diag=False):
				raise NotImplementedError('latent cov has no hyperparameter theta')
			
		fun = _DerivativeFun(self, i)
		return fun

	
	def derivateX(self):
		raise NotImplementedError()


class CompoundKernel(ConvolvedKernel):
	
	__slots__ = ('_kernels',
				 '_q',
				 '_theta_idx')
	
	def __init__(self, kernels):
		q = len(kernels)
		array = np.empty(q, dtype='object')
		#array = ()
		theta_idx = np.zeros(q+1, dtype='int')
		for i in xrange(q):
			array[i] = kernels[i].params
			theta_idx[i+1] = theta_idx[i]+kernels[i].ntheta
		params = CompoundKernel.MixedArray(array)
		
		self._params = params
		self._n = len(params)
		self._q = q
		self._kernels = kernels
		self._theta_idx = theta_idx
		self._ntheta = theta_idx[q]

	def cov(self, X, theta, diag=False):
		'''
		'''
		q = self._q
		kernels = self._kernels
		theta_idx = self._theta_idx
		
		n = len(X)
		K = np.zeros((n,n)) if diag == False else np.zeros(n)
		for i in xrange(q):
			start = theta_idx[i]
			end = theta_idx[i+1]
			theta_i = theta[start:end]
			K += kernels[i].cov(X, theta_i, diag)
		return K

	def cross_cov(self, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		Computes the cross covariance matrix cov[f_p(x), f_q(x)]
		Computes the cross covariance matrix cov[f_p(x), f_q(x)]
		'''
		q = self._q
		kernels = self._kernels
		theta_idx = self._theta_idx
				
		n = len(X)
		m = len(Z)
		K = np.zeros((n,m))
		for i in xrange(q):
			start = theta_idx[i]
			end = theta_idx[i+1]
			thetaPi = thetaP[start:end]
			if latent:
				K += kernels[i].cross_cov(X, Z, thetaPi, latent=latent)
			else:
				thetaQi = thetaQ[start:end]
				K += kernels[i].cross_cov(X, Z, thetaPi, thetaQi, latent=latent)
		return K

	def latent_cov(self, X, Z=None, diag=False):
		'''
		Computes the covariance matrix cov[u(x), u(x')] uof the latent process u
		'''
		q = self._q
		kernels = self._kernels
		xeqz = Z == None
		n = len(X)
		if xeqz:
			K = np.zeros((n,n)) if diag == False else np.zeros(n)
		else:
			m = len(Z)
			K = np.zeros((n,m))
		for i in xrange(q):
			K += kernels[i].latent_cov(X, Z, diag)
		return K
	
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		'''
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, dkernel, start_idx, end_idx):
				self.dkernel = dkernel
				self.start_idx = start_idx
				self.end_idx = end_idx
			
			def cov(self, X, theta, diag=False):
				dkernel = self.dkernel
				start_idx = self.start_idx
				end_idx = self.end_idx
				dK = dkernel.cov(X, theta[start_idx:end_idx], diag)
				return dK
						
			def cross_cov(self, X, Z, thetaP, thetaQ=None, latent=False):
				dkernel = self.dkernel
				start_idx = self.start_idx
				end_idx = self.end_idx
				if thetaQ == None:
					dK = dkernel.cross_cov(X, Z, thetaP[start_idx:end_idx], latent=latent)
					return dK
				else:
					dKp, dKq = dkernel.cross_cov(X, Z, thetaP[start_idx:end_idx], thetaQ[start_idx:end_idx], latent=latent)
					return dKp, dKq
							
			def latent_cov(self, X, Z=None, diag=False):
				dKernel = self.dkernel
				dK = dKernel.latent_cov(X, Z, diag)
				return dK 

		
		u, i, k = self._lookup_kernel_and_param(i)
		dKernel = u.derivate(i)
		return _DerivativeFun(dKernel, self._theta_idx[k], self._theta_idx[k+1])	

	def derivateTheta(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, dkernel, start_idx, end_idx):
				self.dkernel = dkernel
				self.start_idx = start_idx
				self.end_idx = end_idx
			
			def cov(self, X, theta, diag=False):
				dkernel = self.dkernel
				start_idx = self.start_idx
				end_idx = self.end_idx
				dK = dkernel.cov(X, theta[start_idx:end_idx], diag)
				return dK
						
			def cross_cov(self, X, Z, thetaP, thetaQ=None, latent=False):
				dkernel = self.dkernel
				start_idx = self.start_idx
				end_idx = self.end_idx
				if thetaQ == None:
					dK = dkernel.cross_cov(X, Z, thetaP[start_idx:end_idx], latent=latent)
					return dK
				else:
					dKp, dKq = dkernel.cross_cov(X, Z, thetaP[start_idx:end_idx], thetaQ[start_idx:end_idx], latent=latent)
					return dKp, dKq
							
			def latent_cov(self, X, Z, diag=False):
				raise NotImplementedError('latent cov has no hyperparameter theta')

		u, i, k = self._lookup_kernel_and_theta(i)
		dKernel = u.derivateTheta(i)
		return _DerivativeFun(dKernel, self._theta_idx[k], self._theta_idx[k+1])
	
	def derivateX(self):
		raise NotImplementedError()
	
	def _lookup_kernel_and_param(self, i): 
		''' 
		'''
		q = self._q
		kernels = self._kernels
		offset = 0
		for j in xrange(q):
			if i < offset+kernels[j].nparams:
				return kernels[j], i-offset, j
			offset += kernels[j].nparams
		raise IndexError('Unknown hyperparameter')

	def _lookup_kernel_and_theta(self, i): 
		''' 
		'''
		q = self._q
		kernels = self._kernels
		offset = 0
		for j in xrange(q):
			if i < offset+kernels[j].ntheta:
				return kernels[j], i-offset, j
			offset += kernels[j].ntheta
		raise IndexError('Unknown hyperparameter')
	
	def copy(self):
		q = self._q
		kernels = self._kernels
		cp_kernels = np.empty(q, dtype='object')
		for i in xrange(q):
			cp_kernels[i] = kernels[i].copy()
		
		return CompoundKernel(cp_kernels)
			
	class MixedArray(object):
		def __init__(self, array):
			self.array = array
		
		def __len__(self):
			m = len(self.array)
			n = 0
			for i in xrange(m):
				n += len(self.array[i])
			return n
		
		def __getitem__(self, i):
			array, idx = self.__array_idx_at(i)
			return array[idx]
		
		def __setitem__(self, i, value):
			array, idx = self.__array_idx_at(i)
			array[idx] = value

		def __array_idx_at(self, i):
			m = len(self.array)
			offset = 0
			for j in xrange(m):
				if i < offset+len(self.array[j]):
					return self.array[j], i-offset
				offset += len(self.array[j])
			raise IndexError('Unknown hyperparameter')

			return (self.a, i) if i < len(self.a) else (self.b, i-len(self.a))
		
		def __str__(self):
			a = np.empty(0)
			m = len(self.array)
			for i in xrange(m):
				a = np.r_[a, self.array[i]]
			return str(a) 



def check_kernel_gradient(kernel, X, Z=None):
	n = kernel.n_params
	
	err = np.zeros(n, dtype=np.float)
	gradK_tilde = approx_Kprime(kernel, X, Z)
	for i in xrange(n):
		deriv = kernel.derivate(i)
		err[i] = np.sum((gradK_tilde[i]-deriv(X,Z))**2)
		
	return err
	
def approx_Kprime(kernel, X, Z=None, epsilon=np.sqrt(np.finfo(float).eps)):
	params = np.copy(kernel.params)
	n = kernel.n_params
	grad = np.empty(n, dtype=np.object)
	
	K0 = kernel(X,Z)
	#print K0
	#kernel.params = np.zeros(n)
	for i in xrange(n):
		kernel.params[i] = params[i]+epsilon
		grad[i] = (kernel(X,Z)-K0) / epsilon
		kernel.params[i] = params[i]
	
	kernel.params = params
	return grad		


def _check_kernel_gradX(kernel, X, Y):
	[n,m] = X.shape
	YY = np.dot(Y,Y.T)
	
	def _likel_fun(x):
		t = time.time()
		Z = x.reshape(n,m)
		K = kernel(Z)
		
		likel = 0.5*np.trace(np.dot(K, YY))
		print 'likel_gradX={0}'.format(time.time()-t)
		return likel
	
	def _grad_fun(x):
		t = time.time()
		Z = x.reshape(n,m)
		dX_kernel = kernel.derivateX()
		
		gradX = np.empty((n,m))
		for i in xrange(n):
			#t = time.time()
			gradXn = dX_kernel(Z[i], Z)
			#print 'grad_gradX1={0}'.format(time.time()-t)
			#print 'gu'
			#print dX_kernel(Z[i], Z)
			#t = time.time()
			for j in xrange(m):
				#G = np.zeros((n,n))
				#G[:,i] = G[i,:] = gradXn[:,j]
				#gradX[i,j] = np.trace(np.dot(YY,G))*0.5
				gradX[i,j] = 2*np.dot(YY[:,i], gradXn[:,j]) - YY[i,i]*gradXn[i,j]
				gradX[i,j] = gradX[i,j]*0.5
				
			#print 'grad_gradX2={0}'.format(time.time()-t)
		
			
		print 'grad_gradX={0}'.format(time.time()-t)
		#print gradX
		return gradX

	_likel_fun(X.ravel())
	_grad_fun(X.ravel())

	#print _likel_fun(X.ravel())
	#print spopt.approx_fprime(X.ravel(), _likel_fun, np.sqrt(np.finfo(float).eps)).reshape(n,m)
	#print _grad_fun(X.ravel())
	#print nig_grad_a(np.log(2.0))

def _check_kernel_grad(kernel, X, Y):
	YY = np.dot(Y,Y.T)
	params = np.copy(kernel.params)
	
	def _likel_fun(p):
		
		t = time.time()
		kernel.params = p
		K = kernel(X)
		#iK = np.linalg.inv(K)
		likel = np.trace(np.dot(K, YY))
		#likel = -0.5*np.log(np.linalg.det(K)) #- 0.5*np.dot(np.dot(Y[:,0], iK), Y[:,0])
		#likel = -0.5 * np.dot(np.dot(Y[:,0], iK), Y[:,0])
		
		print 'likel_grad={0}'.format(time.time()-t)
		
		return likel
	
	def _grad_fun(p):
		kernel.params = p
		d = len(p)
		
		#K = kernel(X)
		#iK = np.linalg.inv(K)
		t = time.time()
		grad = np.zeros(d)
		for j in xrange(d):
			d_kernel = kernel.derivate(j)
			dK = d_kernel(X)

			grad[j] = np.trace(np.dot(YY,dK))
			#grad[i] = np.trace(np.dot(iK,dK))
			#grad[j] = -0.5*np.trace(np.dot(iK,dK))# - 0.5*np.dot(np.dot(np.dot(Y[:,0], iK), dK), np.dot(iK, Y[:,0]))
			#grad[j] = 0.5*np.dot(np.dot(np.dot(Y[:,0], iK), dK), np.dot(iK, Y[:,0]))
			#grad[j] = -0.5*np.dot(np.dot(Y[:,0], dK), Y[:,0])
		
		print 'grad_grad={0}'.format(time.time()-t)
		return grad

	_likel_fun(params)
	_grad_fun(params)
	#print _likel_fun(params)
	#print spopt.approx_fprime(params, _likel_fun, np.sqrt(np.finfo(float).eps))
	#print _grad_fun(params)
	#print nig_grad_a(np.log(2.0))


if __name__ == '__main__':
	import numpy as np	
	import scipy as sp
	import time
	
	
	from upgeo.ml.regression.np.gp import GPRegression
	
	
	P = np.asarray([1,1])
	
	#kernel = SEKernel(np.log(1), np.log(0.1))# + NoiseKernel(np.log(0.5))
	kernel = ARDSEKernel(np.log(1)*np.ones(3), np.log(5)) #+ NoiseKernel(np.log(0.5))
	#kernel = LinearKernel()
	#kernel = BiasedLinearKernel(np.log(2))
	#kernel = ARDLinearKernel(np.log(1)*np.ones(3), np.log(12)) #+ NoiseKernel(np.log(0.5))
	#kernel = PolynomialKernel(1, np.log(14.2), np.log(20))
	#kernel = SEKernel(np.log(1), np.log(0.1)) + LinearKernel() + NoiseKernel(np.log(2))
	#kernel = SEKernel(np.log(1), np.log(0.1)) * SEKernel(np.log(2), np.log(4)) 
	#kernel = SEKernel(np.log(1), np.log(0.1)) + SqConstantKernel(np.log(2))*LinearKernel() + NoiseKernel(np.log(2))
	kernel_deriv = kernel.derivate(1)
	kernel_derivX = kernel.derivateX()
	
	X =  np.array( [[-0.5046,	0.3999,   -0.5607],
					[-1.2706,   -0.9300,	2.1778],
					[-0.3826,   -0.1768,	1.1385],
					[0.6487,   -2.1321,   -2.4969],
					[0.8257,	1.1454,	0.4413],
					[-1.0149,   -0.6291,   -1.3981],
					[-0.4711,   -1.2038,   -0.2551],
					[0.1370,   -0.2539,	0.1644],
					[-0.2919,   -1.4286,	0.7477],
					[0.3018,   -0.0209,   -0.2730]])
	x1 = np.array([-0.5046,	0.3999, -0.5607])
	Y = np.random.randn(10,8)
	
	print 'grad'
	_check_kernel_grad(kernel, X, Y)
	print 'gradX'
	_check_kernel_gradX(kernel, X, Y)
	
	Z = np.random.randn(2000,3)
	Y = np.random.randn(2000,8)
	_check_kernel_grad(kernel, Z, Y)
	_check_kernel_gradX(kernel, Z, Y)
	
	print 't'
	print kernel(X,X)
	print 't1'
	print kernel_deriv(X)
	print 't2'
	print kernel_derivX(x1, X)
	
	#kernel = ARDSEKernel(np.log(1)*np.ones(10), np.log(5))
	
	Z = np.random.randn(2000,3)
	z1 = np.random.randn(3)
	
	t = time.time()
	kernel(Z)
	print 't1={0}'.format(time.time()-t)
	
	t = time.time()
	kernel_deriv(Z)
	print 't2={0}'.format(time.time()-t)
	
	t = time.time()
	kernel_derivX(z1, Z)
	print 't3a={0}'.format(time.time()-t)
	
	t = time.time()
	for i in xrange(2000):
		kernel_derivX(z1, Z)
	print 't3b={0}'.format(time.time()-t)
	
	t = time.time()
	R = distance_matrix(Z, Z, metric='sqeuclidean')
	print 't5={0}'.format(time.time()-t)

	t = time.time()
	0.2 * np.exp(-R/(2.0*1.5))
	print 't6={0}'.format(time.time()-t)
	
	
#	K = fitc_kernel(X)
#	print np.linalg.inv(K+np.eye(10))
#	L = np.linalg.cholesky(K+np.eye(10))
#	print np.dot(np.linalg.inv(L).T, np.linalg.inv(L))
#	t = time.time()
#	#cho_solve((L, 1), np.eye(2000))
#	print 't1={0}'.format(time.time()-t)
#	K = kernel(X)
#	t = time.time()
#	np.linalg.inv(K)
#	print 't2={0}'.format(time.time()-t)
#	
#	
#	
#	print kernel
#	l = 4
#	X = np.asarray([[2,3],[1,2],[4,5],[0.2,1],[0.8,3]])
#	y = np.asarray([2,3,4,5,0.2])
#	print kernel(X) 
#	print kernel_deriv(X)
#	print check_kernel_gradient(kernel, X)
#	
#	gp = GPRegression(kernel)
#	gp.fit(X, y)
#	print gp.log_likel
#	print gp.likel_fun.gradient()
#	
#	#print X.shape
#	#print np.dot(np.diag(1/P),X.T)
#	#K = distance_matrix(np.dot(np.diag(1/P),X.T).T,metric='sqeuclidean')
#	#print np.exp(-K/2)
#	#print kernel(X)
#	
#	#X = X[:,np.newaxis]
#	#gp = GPRegression(kernel)
#	#gp.fit(X, y)
#	#print gp.likel_fun.gradient()
#	#print gp.log_likel
#	
#	i = 0
#	kernel_deriv = kernel.derivate(i)
#	#print kernel_deriv(X)
#	
#	K = distance_matrix(np.dot(np.diag(1/P),X.T).T,metric='cityblock')
#	
#	#print check_kernel_gradient(kernel, X)
#	
#	#print distance_matrix(X[:,i,np.newaxis]/2, metric='sqeuclidean')
#	
	
#	X = np.random.randn(5,2)
#	Y = np.random.randn(10,2)
#	Z = np.r_[X,Y]
#	k = SEKernel(2) * NoiseKernel(4)
#	
#	print k
#	print k.params
#	print k.params[0]
#	#k.params = np.array([3,1,4,4])
#	print k
#	print k.params
#	
#	
#	print 'hallo'
#	print check_kernel_gradient(k, X)
#	
#	
#	
#	
#	
#	kprime = k.derivate(1)
#	
#	
#	approxK = approx_Kprime(k, X)
##	print approxK[1]
##	print kprime(X)
##	print approxK[1]-kprime(X)
##	print sum(sum((approxK[1]-kprime(X))**2))
#	kprime = k.derivate(1)
	
	
#	fprime = lambda x: kprime(np.atleast_2d(x[0]), np.atleast_2d(x[1]))
	
#	f = lambda hyp: k._set_params(hyp); k(X[0], X[1])
	
#	print f([1,2])
	 

	
	
#	from scipy.spatial.distance import cityblock, euclidean, sqeuclidean, minkowski
#	
#	k = SEKernel(8)+NoiseKernel()
#	
#	x = np.random.randn(6)
#	y = np.random.randn(10000)
#	
#	X = np.random.randn(10000,20)
#	K = k(X)
#	
#	
#	t = time.time()
#	K_inv = np.linalg.inv(K)
#	#print K_inv
#	print time.time()-t
#	
#	t = time.time()
#	L = np.linalg.cholesky(K)
#	print time.time()-t
#	K_inv = cho_solve((L,1), np.eye(10000))
#	#print K_inv
#	print time.time()-t
#	
#	a = np.linalg.solve(L, y)
#	a = np.linalg.solve(L.T, a)
#	print a
#	print cho_solve((L, 1), y)
#	
#	Q = np.dot(a,a.T) - K_inv
#	print np.trace(Q)
#	
#	Q = K_inv - np.dot(a,a.T)
#	print np.sum(np.sum(Q))
#	