'''
Created on Aug 4, 2011

@author: marcel

@todo: - refactor duplicate code snippets (especially the full an sparse inference methods)
'''
import numpy as np
import scipy.optimize as spopt

from abc import ABCMeta, abstractmethod
from scipy.linalg.decomp_cholesky import cho_solve
from numpy.linalg.linalg import LinAlgError
from upgeo.util.math import fmin_ras


class InferenceAlgorithm(object):

    __metaclass__ = ABCMeta
    
    __slots__ = ()
    
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, gp):
        pass
    

class OnePassInference(InferenceAlgorithm):
    
    __slots__ = ()
    
    def apply(self, gp):
        
        #training cases
        X = gp._X
        y = gp._y
        
        kernel = gp._kernel #kernel function to use 
        meanfct = gp._meanfct
        likel_fun = gp._likel_fun #log likelihood fun of the gp model
        
        #determine the cov matrix K and compute lower matrix L
        K = kernel(X)
        L = np.linalg.cholesky(K)
        
        #determin the mean vector if the mean function is specified
        m = np.zeros(len(y))
        if meanfct != None:
            m = meanfct(X)
        
        #compute alpha = K^(-1)*y 
        alpha = cho_solve((L,1), y-m)
        
        gp._K = K
        gp._L = L
        gp._alpha = alpha
        
        
        #compute the model log likelihood (gp relevant parameters must be updated before)
        #todo: check if it logical to update the likel here        
        likel = likel_fun()
        gp._log_likel = likel

class ExactInference(InferenceAlgorithm):
    
    __slots__ = ()
    
    def apply(self, gp):
        '''
        '''
        ExactInference._update_model(gp)
        
        params = gp.hyperparams
        
        f = ExactInference.__TargetFunction(gp) #reference to fun to be optimized
        fprime = f.gradient                     #reference to the gradient
        cb = f.callback                         #reference to the callback fun
        x0 = np.copy(params)                    #initial guess of the hyperparameters
       
        #opt_result = spopt.fmin_bfgs(f, x0, fprime, callback=cb, gtol=1e-2, maxiter=100, full_output = True)
        #opt_result = spopt.fmin_bfgs(f, x0, fprime, callback=cb, full_output = True)
        opt_result = fmin_ras(x0, f, fprime, [], 100)
        #opt_result = spopt.fmin_cg(f, x0, fprime, gtol=1e-2, callback=cb, full_output = True)
        
        xopt = opt_result[0]    #optimal hyperparameters with respect to log likelihood
        
        #Debug output
        if False:
            fun_calls = opt_result[4]
            grad_calls = opt_result[5]
            print 'number of opt calls: fun={0}, grad={1}'.format(fun_calls, grad_calls)
            
            err = spopt.check_grad(f, fprime, x0)
            print 'gradient check at {0}: error={1}'.format(x0, err)
            print spopt.approx_fprime(x0, f, np.sqrt(np.finfo(float).eps))
            
        
        ExactInference._update_parameters(gp, xopt)
        ExactInference._update_model(gp)
            
    class __TargetFunction(object):
        
        __slots__ = ('_gp',
                     '_likel_fun'
                     )
        
        def __init__(self, gp):
            self._gp = gp
            self._likel_fun = gp._likel_fun
        
        def __call__(self, params):
            flag = np.logical_or(np.isnan(params), np.isinf(params))
            if np.any(flag):
                print 'bad params={0}'.format(params)
                return 1e+300

            
            gp = self._gp
            old_params = gp.hyperparams 
            
            try: 
                ExactInference._update_parameters(gp, params)
                ExactInference._update_model(gp)
            except LinAlgError:
                ExactInference._update_parameters(gp, old_params)
                return 1e+300
            
            #print 'gp likel={0}'.format(-gp._log_likel)
            #print 'gp likel params ={0}'.format(np.exp(params))
                
            return -gp._log_likel
        
        def gradient(self, params):
            #print 'params1={0}'.format(params)
            likel_fun = self._likel_fun
            #print 'gradient={0}'.format(-likel_fun.gradient()) 
            grad = -likel_fun.gradient()
            flag = np.logical_or(np.isnan(grad), np.isinf(grad))
            if np.any(flag):
                grad[flag] = 0
                print 'bad gradient={0}'.format(grad)
            return grad
                
        def callback(self, params):
            gp = self._gp
            
            ExactInference._update_parameters(gp, params)
            ExactInference._update_model(gp)
        
    
    @staticmethod
    def _update_parameters(gp, params):
        kernel = gp._kernel
        meanfct = gp._meanfct
        kernel.params = params[:kernel.nparams]
        if meanfct != None:
            offset = kernel.nparams
            meanfct.params = params[offset:]
        
    @staticmethod
    def _update_model(gp):
        X = gp._X           #training cases
        y = gp._y
         
        kernel = gp._kernel #kernel function to use 
        meanfct = gp._meanfct
        likel_fun = gp._likel_fun #log likelihood fun of the gp model
        
        #determine the cov matrix K and compute lower matrix L
        K = kernel(X)
        L = np.linalg.cholesky(K)
        
        #determin the mean vector if the mean function is specified
        m = np.zeros(len(y))
        if meanfct != None:
            m = meanfct(X)
        
        #compute alpha = K^(-1)*y 
        alpha = cho_solve((L,1), y-m)
        
        gp._K = K
        gp._L = L
        gp._alpha = alpha
        
        
        #compute the model log likelihood (gp relevant parameters must be updated before)
        #todo: check if it logical to update the likel here        
        likel = likel_fun()
        gp._log_likel = likel
        
class FITCOnePassInference(InferenceAlgorithm):
    
    __slots__ = ()
    
    def apply(self, gp):
        #inducing points
        Xu = gp._Xu

        #training cases
        X = gp._X
        y = gp._y
        
        m = len(Xu)
        n = len(X)
        
        kernel = gp._kernel #kernel function to use 
        
        likel_fun = gp._likel_fun #log likelihood fun of the gp model
        
        #determine the cov matrix K and compute lower matrix L
        Km = kernel(Xu, Xu)     #invoking with two parameters to prevent adding noise term
        Kn = kernel(X,diag=True)
        Knm = kernel(X, Xu)
        Lm = np.linalg.cholesky(Km + 1e-6*np.eye(m)) #add some jitter
        
        iKm = cho_solve((Lm,1), np.eye(m))
        V = np.linalg.solve(Lm, Knm.T) #Qn=V*V'; V = Lm^-1*Kmn
        
        G = Kn-np.sum(V*V, 0) #G = diag(K-Q)+noise, remember noise is implicitly in K
        V = V/np.sqrt(G) #V=Lm^-1*Kmn*G^(-1/2)
        Lnm = np.linalg.cholesky(np.eye(m)+np.dot(V, V.T)) #Lmn = chol(I + V*V)  
        Lq = np.dot(Lm, Lnm)
        Q = np.dot(Lq, Lq.T) #could be removed, if not necessary
        iQ = cho_solve((Lq, 1), np.eye(m))
        
        r = np.dot(Knm.T, y/G) #helper
        
        #compute alpha and B  
        alpha = np.dot(iQ, r)
        B = iKm-iQ #maybe reverse
        
        gp._Km = Km
        gp._Lm = Lm
        gp._iKm = iKm
        gp._Kn = Kn
        gp._Knm = Knm
        gp._G  = G
        gp._Q = Q
        gp._V = V
        gp._Lnm = Lnm
        gp._Lq = Lq
        gp._iQ = iQ
        
        gp._alpha = alpha
        gp._B = B
        
        #print 'alpha={0}, B={1}'.format(alpha, B)
        
        #compute the model log likelihood (gp relevant parameters must be updated before)
        #todo: check if it logical to update the likel here        
        likel = likel_fun()
        gp._log_likel = likel

class FITCExactInference(InferenceAlgorithm):
    
    __slots__ = ()
    
    def apply(self, gp):
        '''
        '''
        FITCExactInference._update_model(gp)
        
        params = gp.hyperparams
        if not gp._fix_inducing:
            params = np.r_[params, gp._Xu.flatten()]
        
        f = FITCExactInference._TargetFunction(gp) #reference to fun to be optimized
        fprime = f.gradient                     #reference to the gradient
        cb = f.callback                         #reference to the callback fun
        x0 = np.copy(params)                    #initial guess of the hyperparameters
        
        #opt_result = spopt.fmin_bfgs(f, x0, fprime, callback=cb, gtol=1e-2, maxiter=100, full_output=True, disp=1)
        opt_result = fmin_ras(x0, f, fprime, [], 100)
        #opt_result = spopt.fmin_bfgs(f, x0, fprime, callback=cb, full_output=True, disp=1)
        
        
        xopt = opt_result[0]    #optimal hyperparameters with respect to log likelihood
        #xopt = opt_result
        
        #Debug output
        if False:
            fun_calls = opt_result[4]
            grad_calls = opt_result[5]
            print 'number of opt calls: fun={0}, grad={1}'.format(fun_calls, grad_calls)
            
            err = spopt.check_grad(f, fprime, x0)
            print 'gradient check at {0}: error={1}'.format(x0, err)
            print spopt.approx_fprime(x0, f, np.sqrt(np.finfo(float).eps))
            
        
        FITCExactInference._update_parameters(gp, xopt)
        FITCExactInference._update_model(gp)

    
    class _TargetFunction(object):
        
        __slots__ = ('_gp',
                     '_likel_fun',
                     )
        
        def __init__(self, gp):
            self._gp = gp
            self._likel_fun = gp._likel_fun
        
        def __call__(self, params):
            flag = np.logical_or(np.isnan(params), np.isinf(params))
            if np.any(flag):
                print 'bad params={0}'.format(params)
                return 1e+300
            
            gp = self._gp
            old_params = gp.hyperparams 
            if not gp._fix_inducing:
                old_params = np.r_[old_params, gp._Xu.flatten()]
            
            #print 'params={0}'.format(params)
            try: 
                FITCExactInference._update_parameters(gp, params)
                FITCExactInference._update_model(gp)
            except LinAlgError:
                ExactInference._update_parameters(gp, old_params)
                return 1e+300
                
            return -gp._log_likel
        
        def gradient(self, params):
            #print 'params1={0}'.format(params)
            likel_fun = self._likel_fun
            #print 'gradient={0}'.format(-likel_fun.gradient()) 
            grad = -likel_fun.gradientFast()
            flag = np.logical_or(np.isnan(grad), np.isinf(grad))
            if np.any(flag):
                grad[flag] = 0
                print 'bad gradient={0}'.format(grad)
            return grad
        
        def callback(self, params):
            gp = self._gp
            
            FITCExactInference._update_parameters(gp, params)
            FITCExactInference._update_model(gp)
        
    
    @staticmethod
    def _update_parameters(gp, params):
        '''
        gp should be handle this decision by invoking hyperparams
        '''
        kernel = gp._kernel
        if gp._fix_inducing:
            kernel.params = params
        else:
            Xu = gp._Xu
            m,d = Xu.shape
            kernel.params = params[0:kernel.nparams]
            Xu = params[kernel.nparams:].reshape(m,d)
            gp._Xu = Xu
                    
    @staticmethod
    def _update_model(gp):
        #inducing points
        Xu = gp._Xu

        #training cases
        X = gp._X
        y = gp._y
        
        m = len(Xu)
        n = len(X)
        
        kernel = gp._kernel #kernel function to use 
        
        likel_fun = gp._likel_fun #log likelihood fun of the gp model
        
        #determine the cov matrix K and compute lower matrix L
        Km = kernel(Xu, Xu)     #invoking with two parameters to prevent adding noise term
        Kn = kernel(X,diag=True)
         
        Knm = kernel(X, Xu)
        Lm = np.linalg.cholesky(Km + 1e-6*np.eye(m)) #add some jitter
        
        iKm = cho_solve((Lm,1), np.eye(m))
        V = np.linalg.solve(Lm, Knm.T) #Qn=V*V'; V = Lm^-1*Kmn
        
        G = Kn-np.sum(V*V, 0) #G = diag(K-Q)+noise, remember noise is implicitly in K
        V = V/np.sqrt(G) #V=Lm^-1*Kmn*G^(-1/2)
        Lnm = np.linalg.cholesky(np.eye(m)+np.dot(V, V.T)) #Lmn = chol(I + V*V) + some jitter  
        Lq  = np.dot(Lm, Lnm)
        Q = np.dot(Lq, Lq.T) #could be removed, if not necessary
        iQ = cho_solve((Lq, 1), np.eye(m))
        
        r = np.dot(Knm.T, y/G) #helper
        
        #compute alpha and B  
        alpha = np.dot(iQ, r)
        B = iKm-iQ #maybe reverse
        
        gp._Km = Km
        gp._Lm = Lm
        gp._iKm = iKm
        gp._Kn = Kn
        gp._Knm = Knm
        gp._G  = G
        gp._Q = Q
        gp._V = V
        gp._Lnm = Lnm
        gp._Lq = Lq
        gp._iQ = iQ
        
        gp._alpha = alpha
        gp._B = B
        
        #print 'alpha={0}, B={1}'.format(alpha, B)
        
        #compute the model log likelihood (gp relevant parameters must be updated before)
        #todo: check if it logical to update the likel here        
        likel = likel_fun()
        gp._log_likel = likel

if __name__ == '__main__':            
    import time
    
    from upgeo.ml.regression.np.kernel import NoiseKernel, SEKernel
    from upgeo.ml.regression.np.gp import SparseGPRegression
    from upgeo.ml.regression.np.selector import FixedSelector
    
    X =  np.array([[ 0.5201,   -0.2938,   -1.3320],
                   [-0.0200,   -0.8479,   -2.3299],
                   [-0.0348,   -1.1201,   -1.4491],
                   [-0.7982,    2.5260,    0.3335],
                   [ 1.0187,    1.6555,    0.3914],
                   [-0.1332,    0.3075,    0.4517],
                   [-0.7145,   -1.2571,   -0.1303],
                   [ 1.3514,   -0.8655,    0.1837],
                   [-0.2248,   -0.1765,   -0.4762],
                   [-0.5890,    0.7914,    0.8620]]
                   )
    Xu = np.array([[-1.3617,    1.0391,   -0.1952],
                   [ 0.4550,   -1.1176,   -0.2176],
                   [-0.8487,    1.2607,   -0.3031],
                   [-0.3349,    0.6601,    0.0230],
                   [ 0.5528,   -0.0679,    0.0513]])
    
    y = np.array([0.8261, 1.5270, 0.4669, -0.2097, 0.6252, 0.1832, -1.0298, 0.9492, 0.3071, 0.1352])
    
    kernel = SEKernel(np.log(0.5), np.log(1)) + NoiseKernel(np.log(0.5))
    selector = FixedSelector(Xu)
    
    gp = SparseGPRegression(kernel, infer_method=FITCOnePassInference, selector=selector)
    likel = gp.fit(X, y)
    print 'likel={0}'.format(likel)  
    (yhat, var) = gp.predict(X, ret_var=True)
    (mean, cov) = gp.posterior(X)
    print 'result'
    print yhat
    print var
    #print cov
    
    likel_fun = gp.likel_fun
    print 'gradient'
    print likel_fun.gradient()
    
    
