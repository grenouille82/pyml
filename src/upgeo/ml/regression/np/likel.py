'''
Created on Aug 4, 2011

@author: marcel
'''

import numpy as np
import time

from abc import ABCMeta, abstractmethod
from scipy.linalg.decomp_cholesky import cho_solve
from upgeo.ml.regression.np.kernel import SEKernel, NoiseKernel

class LikelihoodFunction(object):
    
    __metaclass__ = ABCMeta
    
    __slots__ = ( '_gp'    #the gp model for which the likelihood should be determined
                )
    
    def __init__(self, gp_model):
        self._gp = gp_model
    
    @abstractmethod
    def __call__(self):
        pass
    
    @abstractmethod
    def gradient(self):
        pass
    
class GaussianLogLikel(LikelihoodFunction):
    '''
    @todo: - consider to define the input noise implicit in the likelihood function. 
             at the moment input noise can be formulated by the covariance function 
             as hyperparameter. 
    '''
    
    def __init__(self, gp_model):
        LikelihoodFunction.__init__(self, gp_model)
        
    def __call__(self):
        gp = self._gp
        
        meanfct = gp._meanfct
        X = gp._X
        y = gp._y           #targets of the training cases
        L = gp._L           #cholesky decomposition of the covariance matrix
        n = gp._n           #number of training samples
        alpha = gp._alpha   #weight vector for each kernel component???
        
        
        params = gp.hyperparams
        priors = gp._priors
        
        m = np.zeros(len(y))
        if meanfct != None:
            m = meanfct(X)
        likel = (-np.dot((y-m).T, alpha) - n*np.log(2.0*np.pi)) / 2.0 #check, if we net the transpose
        likel -= np.sum(np.log(np.diag(L)))
        
        if priors != None:
            for i in xrange(len(params)):
                pr = priors[i]
                if pr != None:
                    likel += pr.log_pdf(params[i])
                    
        print 'likel={0}'.format(likel)
        
        return likel
    
    def gradient(self):
        gp = self._gp
        
        #@todo look for a better way to store the proximity
        
        X = gp._X           #covariates of the training cases
        L = gp._L           #cholesky decomposition of the covariance matrix
        n = gp._n           #number of training samples
        alpha = gp._alpha   #weight vector for each kernel component???
        kernel = gp._kernel #used kernel in gp model
        k = kernel.nparams  #number of hyperparameters
        params = gp.hyperparams
        priors = gp._priors
        
        #print 'grad'
        K_inv = cho_solve((L,1), np.eye(n))
        Q = np.outer(alpha, alpha)-K_inv
        
        #determine the gradient of the kernel hyperparameters
        gradK = np.empty(k)
        for i in xrange(k):
            #compute the gradient of the i-th hyperparameter
            d_kernel = kernel.derivate(i)
            dK = d_kernel(X)
            gradK[i] = np.sum(Q*dK) / 2.0
            if priors != None:
                pr = priors[i]
                if pr != None:
                    gradK[i] += pr.log_gradient(params[i])
                    
        #determine the gradients of the mean fct hyperparameters if present
        meanfct = gp._meanfct
        if meanfct != None:
            offset = k #need to determine the correct prior
            k = meanfct.nparams
            gradM = np.empty(k)
            for i in xrange(k):
                d_meanfct = meanfct.derivate(i)
                dM = d_meanfct(X)
                gradM[i] = np.dot(dM, alpha) #check if the gradient is correct
                if priors != None:
                    pr = priors[offset+i]
                    gradM[i] += pr.log_gradient(params[i])
            grad = np.r_[gradK, gradM]
        else:
            grad = gradK
                    
        return grad

    
class SparseGaussianLogLikel(LikelihoodFunction):

    def __init__(self, gp_model):
        LikelihoodFunction.__init__(self, gp_model)
        
    def __call__(self):
        '''
        @todo: - optimize the quadratic forms
        '''
        gp = self._gp
        
        n = gp._n
        y = gp._y
        
        Lnm = gp._Lnm
        V = gp._V
        G = gp._G
        
        #1.part of the woodbury inverse (y*G*y)
        w1 = y/np.sqrt(G)
        #2.part of the woodbury inverse 
        w2 = np.linalg.solve(Lnm, np.dot(V,w1))
        
        l1 = np.dot(w1,w1) - np.dot(w2,w2)
        l2 = np.sum(np.log(G)) + 2.0*np.sum(np.log(np.diag(Lnm)))
        
        likel = (-l1 - l2 - n*np.log(2.0*np.pi)) / 2.0
        return likel
    
    def gradient(self):
        '''
        @todo: - mark as depricated because this method is slower than gradientFast and 
                 this method has not the ability to optimize Xu in a direct way
        '''
        gp = self._gp
        X = gp._X
        y = gp._y
        Xu = gp._Xu
        
        kernel = gp._kernel
        k = kernel.nparams
        
        Knm = gp._Knm
        
        iQ = gp._iQ
        iKm = gp._iKm
        G = gp._G
        
        uKnm = Knm.T/np.sqrt(G)
        uKnm = uKnm.T
        
        #t = time.time()
        B1 = np.dot(iKm, Knm.T)
        B2 = np.dot(iQ, uKnm.T)
        r = y/np.sqrt(G)
        Br = np.dot(B2,r)
  
        #print 'B1={0}'.format(time.time()-t)
        
        grad = np.empty(k)
        for i in xrange(k):
            #todo: optimize the quadratic terms
            dkernel = kernel.derivate(i)
            
            dKm = dkernel(Xu, Xu)
            dKn = dkernel(X, diag=True)
            dKnm = dkernel(X, Xu)
            
            
            
            duKnm = dKnm.T/np.sqrt(G)
            duKnm = duKnm.T
            
            
           
            #compute diag(dQn)
            #B = np.dot(iKm, Knm.T)
            R = 2*dKnm.T - np.dot(dKm, B1)
            ddQn = np.sum(R*B1,0)
            
            
            #compute gradient dG
            dG =  dKn - ddQn
            
            qfG = dG/(G**2)
            V = np.dot(duKnm.T, uKnm)
            dQ = dKm + V+V.T - np.dot(Knm.T*qfG, Knm)
           
            #B = np.dot(iQ, uKnm.T)
            #r = y/np.sqrt(G)
            #Br = np.dot(B,r)
            
            
            V = -2.0*Knm.T*qfG + 2.0*dKnm.T/G 
            V -= np.dot(B2.T, dQ).T/np.sqrt(G)
            V = np.dot(V.T, Br)  
            
            dl1 = -np.dot(y*qfG, y) - np.dot(y, V)
            dl2 = np.sum(1.0/G*dG) - np.sum(iKm*dKm) + np.sum(iQ*dQ)
            
            grad[i] = (-dl1-dl2) / 2.0
            
        return grad
    
    def gradientFast(self):
        '''
        Very Fast gradient computation of the kernel hyperparameters and inducing data points 
        by using chain rule of the likelihood derivative (see Alvarez 2011).
        
        '''
        gp = self._gp
        
        #data 
        X = gp._X
        y = gp._y
        Xu = gp._Xu
        
        #precomuted stuff
        iKm = gp._iKm
        Knm = gp._Knm
        G = gp._G
        alpha = gp._alpha
        iQ = gp._iQ
        kernel = gp._kernel
        
        #formulas uses identical notation to the Paper of Alvarez
        #so the diag matrices are computed in a efficient way
        yy = y*y    
        C = iQ + np.outer(alpha, alpha)
        H = G - yy + 2.0*np.dot(Knm, alpha)*y
        J = H-np.sum(np.dot(Knm, C)*Knm,1)
        Q = J/(G**2)
        
        B1 = np.dot(iKm, Knm.T)
        B2 = B1*Q

        dKn = -0.5*Q
        dKm = 0.5*(iKm - C - np.dot(B2,B1.T))
        dKmn = B2 - np.dot(C, Knm.T)/G + np.outer(alpha,y/G)
        dKmn = dKmn.T
        

        k = kernel.nparams
        grad = np.empty(k)
        for i in xrange(k):
            dkernel = kernel.derivate(i)
            
            dTm = dkernel(Xu, Xu)
            dTn = dkernel(X, diag=True)
            dTnm = dkernel(X, Xu)
            grad[i] = np.dot(dTn, dKn) 
            grad[i] += np.dot(dTm.flatten(), dKm.flatten()) 
            grad[i] += np.dot(dTnm.flatten(), dKmn.flatten())

        if not gp._fix_inducing:
            dkernelX = kernel.derivateX()
            dXm = dkernelX(Xu,Xu)*2.0
            dXmn = dkernelX(Xu,X)
            m,d = Xu.shape
            gradX = np.zeros((m,d))
            for i in xrange(m):
                for j in xrange(d):
                    gradX[i,j] = np.dot(dXm[:,i,j], dKm[:,i])
                    gradX[i,j] += np.dot(dXmn[:,i,j], dKmn[:,i])
            
            grad = np.r_[grad, gradX.flatten()]

        return grad 
    
def _update_model(gp):
    X = gp._X           #training cases
    y = gp._y
         
    kernel = gp._kernel #kernel function
        
    likel_fun = gp._likel_fun #log likelihood fun of the gp model
        
    #determine the cov matrix K and compute lower matrix L
    K = kernel(X)
    L = np.linalg.cholesky(K)
        
    #compute alpha = K^(-1)*y 
    alpha = cho_solve((L,1), y)
        
    gp._K = K
    gp._L = L
    gp._alpha = alpha
        
    #compute the model log likelihood (gp relevant parameters must be updated before)
    likel = likel_fun()
    gp._log_likel = likel

    
if __name__ == '__main__':
    import numpy as np    
    import scipy as sp
    import time
    import scipy.optimize as spopt
    
    from upgeo.ml.regression.np.gp import GPRegression
    
    
    P = np.asarray([1,1])
    
    kernel = SEKernel(np.log(2), np.log(21)) #+ NoiseKernel(np.log(0.5))
    #kernel = ARDSEKernel(np.log(1)*np.ones(3), np.log(5)) #+ NoiseKernel(np.log(0.5))
    
    X =  np.array( [[-0.5046,    0.3999,   -0.5607],
                    [-1.2706,   -0.9300,    2.1778],
                    [-0.3826,   -0.1768,    1.1385],
                    [0.6487,   -2.1321,   -2.4969],
                    [0.8257,    1.1454,    0.4413],
                    [-1.0149,   -0.6291,   -1.3981],
                    [-0.4711,   -1.2038,   -0.2551],
                    [0.1370,   -0.2539,    0.1644],
                    [-0.2919,   -1.4286,    0.7477],
                    [0.3018,   -0.0209,   -0.2730]])
    y = np.random.randn(10)
    
    gp = GPRegression(kernel)
    gp.fit(X,y)
    likel_fun = gp.likel_fun
    
    
    def _l(p):
        gp.kernel.params = p
        _update_model(gp)
        return likel_fun()
    
    def _g(p):
        gp.kernel.params = p
        _update_model(gp)
        return likel_fun.gradient()

    print _l(np.log(np.array([0.2, 20])))
    print _g(np.log(np.array([0.2, 20])))
    print spopt.approx_fprime(np.log(np.array([0.2, 20])), _l, np.sqrt(np.finfo(float).eps))