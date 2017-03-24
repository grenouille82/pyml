'''
Created on Jun 24, 2011

@author: marcel
'''
import numpy as np
import scipy.special as sps

from upgeo.prior.base import ConjugatePrior
from scipy.linalg.decomp_cholesky import cho_solve
from upgeo.util.stats import mvtlnpdf

class NormalInvWishart(ConjugatePrior):
    
    __slots__ = ('_m',      # (1,d)-dimensional prior mean vector
                 '_mn',     # (1,d)-dimensional posterior mean vector
                 '_R',      # (d,d)-dimensional prior precision matrix
                 '_Rn',     # (d,d)-dimensional posterior precision matrix 
                 '_v',      # prior degree of freedom, (hyperparameter) of inverse Wishart covariance prior.
                 '_vn',
                 '_r',       # prior on relative precision
                 '_rn',
                 '_L',
                 '_Ln',
                 '_S',
                 '_mu'
                 )      
        
    def __init__(self, X, m, R, r, v):
        ConjugatePrior.__init__(self, X)
        
        if len(m) != self.ndim:
            raise ValueError('dim of mean must be the same as data dim.')
        if R.ndim != 2 or R.shape[0] != R.shape[1] or R.shape[0] != self.ndim:
            raise ValueError('R must be quadratic and dimension must be equal to data dim.')
        if r <= 0.0:
            raise ValueError('r must be greater than 0')
        if v <= 0.0:
            raise ValueError('v must be greater than 0')
        
        self._m = m
        self._R = R
        self._r = r
        self._v = v
        self.__update_posterior_params(X, m, R, r, v)
        
    @property
    def hyperparams(self):
        return (self._m, self._R, self._r, self._v)
    
    @property
    def posterior_hyperparams(self):
        return (self._mn, self._Rn, self._rn, self._vn)
    
    @property
    def n_hyperparams(self):
        return 4
    
    @property
    def n_params(self):
        return 2
    
    def posterior(self, params):
        pass
    
    def marginal_posterior(self, i):
        pass
    
    def predictive(self, X):
        '''
        '''
        d = self._d
        vn = self._vn
        mn = self._mn
        rn = self._rn
        Rn = self._Rn
        
        vp = vn-d+1.0
        Rp = Rn*(rn+1) / (rn*vp)
                
        likels = mvtlnpdf(X, vp, mn, Rp)
        return likels
    
    def likelihood(self):
        '''
        Returns the marginal log likeliohood of the model.
        '''
        n,d = self._n, self._d
        
        r,rn = self._r, self._rn
        v,vn = self._v, self._vn
        
        L,Ln = self._L, self._Ln
        
        detR = 2.0*np.sum(np.log(np.diag(L)))
        detRn = 2.0*np.sum(np.log(np.diag(Ln)))
        
        A = -n*d*np.log(np.pi)/2.0
        B = (d*np.log(r) + v*detR - d*np.log(rn) - vn*detRn) / 2.0
    
        #print 'v={0}'.format(v)
        #print 'vn={0}'.format(vn)
    
        C = 0.0
        for i in xrange(d): 
            C += sps.gammaln((vn+1.0-i)/2.0) - sps.gammaln((v+1.0-i)/2.0) 
        
        #print 'A={0}, B={1}, C={2}'.format(A,B,C)
        likel = A+B+C
        return likel
    
    log_likel = property(fget=likelihood)
    
    def map(self, i=None):
        '''
        Returns the map estimates of the latent variables of the distribution.
        '''
        pass
    
    @property
    def likel_fun(self):
        fun = NormalInvWishart.LikelihoodFunction(self)
        return fun
    
    def sample(self, n):
        pass
    
    def update(self, X):
        pass
    
    def __update_posterior_params(self, X, m, R, r, v):
        n = X.shape[0]
        
        #pre-computation
        mu = np.mean(X,0)
        Xdiff = X-mu
        S = np.dot(Xdiff.T, Xdiff)
        
        self._mu = mu
        self._S = S
        
        #update hyperparams
        rn = r+n
        vn = v+n
        mn = (r*m + n*mu) / rn
        Rn = R + S + r*n*np.outer(mu-m, mu-m)/rn
        
        self._rn = rn
        self._vn = vn
        self._mn = mn
        self._Rn = Rn
        
        #cholesky decomposition of precision matrices
        L = np.linalg.cholesky(R)
        Ln = np.linalg.cholesky(Rn)
        
        self._L = L
        self._Ln = Ln
        
    class LikelihoodFunction(object):
        
        __slots__ = ('_prior')
        
        def __init__(self, prior):
            self._prior = prior
        
        def __call__(self, m, L, r, v):
            '''
            @todo: L could have negative diagonal elements, to compute the determinant of L, 
                   we have to compute the product of diagonals at first, because to permit
                   invalid values in the log  
            '''
            n,d = self._prior._n, self._prior._d
            #vn = self._prior._v, self._prior._vn
            
            mu = self._prior._mu
            S = self._prior._S
            
            L += 1e-6*np.eye(d) #add some jitter
            #if we have an cholupdate function we haven't to transform L back to
            #the precision matrix
            R = np.dot(L, L.T)
 
            vn = v+n
            rn = r+n
            Rn = R + S + r*n*np.outer(mu-m, mu-m)/rn 
            Rn = Rn + 1e-6*np.eye(np.size(Rn, 0)) #add some jitter
            
            Ln = np.linalg.cholesky(Rn)
            
            if not np.any(np.diag(L) < 0):
                detR = 2.0*np.sum(np.log(np.diag(L)))  #determinant of prior precision matrix
            else:
                detR = np.linalg.det(R)
            
            detRn = 2.0*np.sum(np.log(np.diag(Ln)))#determinant of posterior precision matrix
            A = -n*d*np.log(np.pi)/2.0
            B = (d*np.log(r) + v*detR - d*np.log(rn) - vn*detRn) / 2.0
        
            C = 0.0
            for i in xrange(d):
                C += sps.gammaln((vn+1.0-i)/2.0) - sps.gammaln((v+1.0-i)/2.0) 
        
        
            likel = A+B+C
            return likel
        
        def gradient(self, m, L, r, v):
            L = np.tril(L)
            
            
            n,d = self._prior._n, self._prior._d
            v, vn = self._prior._v, self._prior._vn
            
            mu = self._prior._mu
            S = self._prior._S
            
            L += 1e-6*np.eye(d) #add some jitter
            #if we have an cholupdate function we haven't to transform L back to
            #the precision matrix
            R = np.dot(L, L.T)
            
            rn = r+n
            #mn = (r*m + n*mu) / rn
            Rn = R + S + r*n*np.outer(mu-m, mu-m)/rn
            
            Ln = np.linalg.cholesky(Rn + 1e-6*np.eye(len(Rn))) #add some jitter
            
            detR = 2.0*np.sum(np.log(np.diag(L)))  #determinant of prior precision matrix
            detRn = 2.0*np.sum(np.log(np.diag(Ln)))#determinant of posterior precision matrix
            
            #compute the gradient of mean m
            m_prime = cho_solve((Ln,1), (r*n)*(mu-m)/rn)*vn  #inverse of cholesky decomposition
            
            #compute the gradient of cholesky decomposiot of the precision matrix
            L_prime = np.tril(cho_solve((L,1), L)*v - cho_solve((Ln,1), L)*vn)
            
            #compute the gradient of the relevance precision
            diff = mu-m
            Sn_prime = n*np.outer(diff, diff)/rn - n*r*np.outer(diff, diff)/rn**2.0
            r_prime = d/(2*r) - d/(2*(n+r))- vn/2* np.trace(cho_solve((Ln,1), Sn_prime))
            
            #compute the gradient of dof
            C_prime = 0
            for i in xrange(d):
                C_prime += (sps.digamma((vn+1.0-i)/2.0) - sps.digamma((v+1.0-i)/2.0))/2.0
            v_prime = (detR - detRn)/2.0 + C_prime
            return (m_prime, L_prime, r_prime, v_prime)
        
if __name__  == '__main__':
    
    import scipy.optimize as spopt
    #np.random.seed(20)
    
    X = np.random.randn(200,10)
    m = np.mean(X,0)
    S = np.cov(X.T)
    #R = np.linalg.inv(S)
    
    x = np.random.randn(2,10)
    
    prior = NormalInvWishart(X, m, np.eye(10), 1.5, 10)
    print prior.likelihood()
    likel_fun = prior.likel_fun
    
    L = np.linalg.cholesky(np.eye(10))
    z = np.random.randn(10)
    r = 1.5
    
    f = lambda x: likel_fun(z, L, x)
    fprime =lambda x: likel_fun.gradient(z, L, x)[2]
    
    print likel_fun(m+1, L, 1.5)
    print likel_fun.gradient(z, L, 1.5)
    print f(r)
    print fprime(r)
    print spopt.approx_fprime(np.atleast_1d([r]), f, np.sqrt(np.finfo(float).eps))
    
    #print mvtlnpdf(x, 105, m, np.eye(10))
