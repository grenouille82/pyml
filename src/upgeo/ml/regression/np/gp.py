'''
Created on Jul 15, 2011

@author: marcel
'''

import numpy as np

from scikits.learn.base import RegressorMixin, BaseEstimator

from upgeo.ml.regression.np.likel import GaussianLogLikel,\
    SparseGaussianLogLikel
from upgeo.ml.regression.np.infer import OnePassInference, ExactInference,\
    FITCOnePassInference, FITCExactInference
from upgeo.exception import NotFittedError
from upgeo.ml.regression.np.util import plot1d_gp
from upgeo.ml.regression.np.kernel import SqConstantKernel,\
    check_kernel_gradient, ARDSEKernel
from upgeo.util.metric import distance_matrix
from upgeo.ml.regression.np.selector import RandomSubsetSelector, FixedSelector

class GPRegression(BaseEstimator, RegressorMixin):
    '''
    @todo: 
           - define priors on hyperparameters
           - make noise term explicit, that means likel_fun should include this term
           - check whether the gradient and the prediction with a specified mean fct works correctly
    '''
    
    __slots__ = ('_kernel',     #covariance kernel
                 '_meanfct',    #mean function
                 '_likel_fun',  #likelihood function of the model
                 '_infer',      #inference algorithm used to optimize hyperparameters
                 '_priors',     #prior distribution over the parameters
                 
                 '_X',          #covariates of training set
                 '_y',          #targets of training set
                 
                 '_K',          #covariance matrix of the training set
                 '_L',          #cholesky decomposition of the covariance matrix
                 '_alpha',      #weight vector for each data point???
                 
                 '_log_likel',  #log likelihood of the trained model
                           
                 '_d',          #dimension of each input vector
                 '_n',          #number of training samples
                 
                 '_is_init')

    def __init__(self, kernel, meanfct=None, likel_fun=GaussianLogLikel, 
                 infer_method=OnePassInference, priors=None):
        '''
        Constructor
        '''
        if priors != None and len(priors) != kernel.n_params:
            raise ValueError('''number of priors must be equal 
                             to the number of hyperparameters.''')
            
        self._kernel = kernel
        self._meanfct = meanfct
        self._likel_fun = likel_fun(self)
        self._infer = infer_method()
        self._priors = priors    
        
        self._is_init = False
        
    def fit(self, X, y):
        '''
        @todo: parameter check
        '''
        
        self._X = X
        self._y = y
        
        n,d = X.shape
        self._n = n
        self._d = d
         
        infer_method = self._infer
        infer_method.apply(self)
        
        self._is_init = True
        
        return self._log_likel
        
    def predict(self, X, ret_var=False):
        '''
        R contains distances to training and test set 
        @todo: - parameter check
               - check the indexing of R
        '''
        self._init_check()
        
        alpha = self._alpha
        kernel = self._kernel
        meanfct = self._meanfct
        X_train = self._X
        
        ms = np.zeros(len(X))
        if meanfct != None:
            ms = meanfct(X)
        
        Ks = kernel(X, X_train)
        yfit = ms+np.dot(Ks, alpha)
        
        if ret_var:
            L = self._L            
        
            kss = kernel(X, diag=True)
            V = np.linalg.solve(L, Ks.T)
            var = kss - np.sum(V*V,0)
            return (yfit, var)
        
        return yfit
    
    def posterior(self, X):
        '''
        check if this posterior p(f|X) or predictive p(y|X)
        '''
        self._init_check()
        
        L = self._L
        alpha = self._alpha
        kernel = self._kernel
        meanfct = self._meanfct
        X_train = self._X
        
        Ks = kernel(X, X_train)
        Kss = kernel(X)
        
        ms = np.zeros(len(X))
        if meanfct != None:
            ms = meanfct(X)
        
        mean = ms + np.dot(Ks, alpha)
            
        V = np.linalg.solve(L, Ks)
        cov = Kss - np.dot(V.T, V)
            
        return (mean, cov)
    
    
    def _get_log_likel(self):
        return self._log_likel
    
    log_likel = property(fget=_get_log_likel)
    
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        params = np.copy(self._kernel.params)
        if self._meanfct != None:
            params = np.r_[params, self._meanfct.params]
        return params
    
    def _set_hyperparams(self, params):
        '''
        '''
        kernel = self._kernel
        meanfct = self._meanfct
        kernel.params = np.copy(params[:kernel.nparams])
        if meanfct != None:
            offset = kernel.nparams
            meanfct = np.copy(params[offset:])
        
    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)
    
    def _get_kernel(self):
        return self._kernel
    
    kernel = property(fget=_get_kernel)
    
    def _get_meanfct(self):
        return self._meanfct
    
    meanfct = property(fget=_get_meanfct)
    
    def _get_infer_method(self):
        return self._infer
    
    infer_method = property(fget=_get_infer_method)
    
    def _get_likel_fun(self):
        return self._likel_fun
    
    likel_fun = property(fget=_get_likel_fun)
    
    def _get_training_set(self):
        return self._X, self._y
    
    training_set = property(fget=_get_training_set)
    
    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')

class SparseGPRegression(BaseEstimator, RegressorMixin):
    '''
    @todo: - allow to specify a mean function
           - define priors on hyperparameters
           - make noise term explicit, that means likel_fun should include this term
           - remove the parametrization by likelihood function, because its strongly
             depend on the inference method
           - should we modify the inducing points by hyperparameters update?
    '''
    
    __slots__ = ('_kernel',     #covariance kernel
                 '_likel_fun',  #likelihood function of the model
                 '_infer',      #inference algorithm used to optimize hyperparameters
                 '_selector',   #subset selection method
                 '_priors',     #prior distribution over the parameters
       
                 '_fix_inducing'#flag if inducing dp are optimized         
                 '_Xu',         #set of the induced datapoints
                 
                 '_X',          #covariates of training set
                 '_y',          #targets of training set
                
                 '_Km',         #covariance matrix of inducing points
                 '_iKm',        #inverse of cov matrix Km
                 '_Lm',         #cholesky decomposition of cov matrix Km
                 
                 '_Kn',         #covariance matrix of training points
                 '_Knm',        #covariance matrix of training and inducing points
                 '_Lnm',
                 
                 '_G',          #G = diag(K-Q)+noise
                 
                 '_V',
                 '_Q',          #symmetric matrix Q of the woodbury identity
                 '_Lq',         #cholesky decomposition of Q
                 '_iQ',         #inverse of Q
                  
                 '_alpha',      #weight vector for each data point???
                 '_B',          #B = iKm - iQ #for covariance prediction
                 
                 '_log_likel',  #log likelihood of the trained model
                           
                 '_d',          #dimension of each input vector
                 '_n',          #number of training samples
                 
                 '_is_init')

    def __init__(self, kernel, likel_fun=SparseGaussianLogLikel, 
                 infer_method=FITCOnePassInference, 
                 selector=RandomSubsetSelector(10),
                 priors=None, fix_inducing=True):
        '''
        Constructor
        '''
        if priors != None and len(priors) != kernel.n_params:
            raise ValueError('''number of priors must be equal 
                             to the number of hyperparameters.''')
            
        self._kernel = kernel
        self._likel_fun = likel_fun(self)
        self._infer = infer_method()
        self._selector = selector
        self._priors = priors    
        
        self._fix_inducing = fix_inducing
        self._is_init = False
        
    def fit(self, X, y):
        '''
        @todo: parameter check
        '''
        
        selector = self._selector
        
        #print 'gp_hyperparams={0}'.format(self.hyperparams)
        
        self._Xu = selector.apply(X,y)
        self._X = X
        self._y = y
        
        n,d = X.shape
        self._n = n
        self._d = d
         
        infer_method = self._infer
        infer_method.apply(self)
        
        self._is_init = True
        
        return self._log_likel
        
    def predict(self, X, ret_var=False):
        '''
        R contains distances to training and test set 
        @todo: - parameter check
               - check the indexing of R
        '''
        self._init_check()
        
        print 'gp_pred_hyperparams={0}'.format(self.hyperparams)
        
        alpha = self._alpha
        kernel = self._kernel
        Xu = self._Xu
        
        Ks = kernel(X, Xu)
        yfit = np.dot(Ks, alpha)
        
        if ret_var:
            B = self._B
            kss = np.diag(kernel(X))
            V = np.sum(Ks*np.dot(B,Ks.T).T,1) #V=diag(Ks*B*Ks')
            #V = np.dot(np.dot(Ks, B), Ks.T)
            var = kss - V
            return (yfit, var)
        
        return yfit
    
    def posterior(self, X, R=None):
        self._init_check()
        
        B = self._B
        alpha = self._alpha
        kernel = self._kernel
        Xu = self._Xu
        
        Ks = kernel(X, Xu)
        Kss = kernel(X)
        
        mean = np.dot(Ks, alpha)
            
        V = np.dot(np.dot(Ks, B), Ks.T)
        cov = Kss - V
            
        return (mean, cov)
    
    
    def _get_log_likel(self):
        return self._log_likel
    
    log_likel = property(fget=_get_log_likel)
    
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        return np.copy(self._kernel.params)
    
    def _set_hyperparams(self, params):
        '''
        '''
        self._kernel.params = np.copy(params)
    
    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)
    
    def _get_kernel(self):
        return self._kernel
    
    kernel = property(fget=_get_kernel)
    
    def _get_infer_method(self):
        return self._infer
    
    infer_method = property(fget=_get_infer_method)
    
    def _get_likel_fun(self):
        return self._likel_fun
    
    likel_fun = property(fget=_get_likel_fun)
    
    def _get_training_set(self):
        return self._X, self._y
    
    training_set = property(fget=_get_training_set)
    
    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')


if __name__ == '__main__':
    import time
    
    from upgeo.util.metric import mspe
    from upgeo.ml.regression.np.kernel import NoiseKernel, SEKernel
    from upgeo.ml.regression.np.util import plot1d_gp, gendata_1d, f1, f2
        
    (X,y) = gendata_1d(f2, 0, 3, 100, 0.5)
    X = X[:,np.newaxis]
    
    kernel = SEKernel(np.log(0.5), np.log(1)) + NoiseKernel(np.log(0.5))
    #print check_kernel_gradient(kernel, X)
    
#    gp = GPRegression(kernel, GaussianLogLikel, OnePassInference)
#    t = time.time()
#    gp.fit(X, y)
#    print 'fit_time: {0}'.format(time.time()-t)
#    #plot1d_gp(gp, 0, 3)
#    #print gp.log_likel
#    t = time.time()
#    gp.likel_fun()
#    print 'likel_time: {0}'.format(time.time()-t)
#    t = time.time()
#    gp.likel_fun.gradient()
#    print 'grad_time: {0}'.format(time.time()-t)
#    print np.exp(gp.hyperparams)
#    print gp.log_likel
#    print gp.likel_fun.gradient()
    
    #(X,y) = gendata_1d(f1, -3, 3, 50, 0.01)
    #X = X[:,np.newaxis]
    
    #kernel = SqConstantKernel(np.log(10)) + SEKernel(np.log(1.4), np.log(1.7)) + NoiseKernel(np.log(0.2))
    
    kernel = SEKernel(np.log(0.5), np.log(1)) + NoiseKernel(np.log(0.5))
    t = time.time()
    gp = GPRegression(kernel, GaussianLogLikel, ExactInference)
    gp.fit(X, y)
    print 'opt_time: {0}'.format(time.time()-t)
    plot1d_gp(gp, 0, 3)
    
    kernel = ARDSEKernel([np.log(0.5)], np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = SEKernel(np.log(0.58486524), np.log(0.90818889)) + NoiseKernel(np.log(0.4807206))
    Xu = np.linspace(0,3,20)
    Xu = Xu[:,np.newaxis]
    selector = RandomSubsetSelector(10)
    #selector = FixedSelector(Xu)
    gp = SparseGPRegression(kernel, infer_method=FITCOnePassInference, selector=selector)
    t = time.time()
    gp.fit(X, y)
    print 'fit_time: {0}'.format(time.time()-t)
    #plot1d_gp(gp, 0, 3)
    #print gp.log_likel
    t = time.time()
    gp.likel_fun()
    print 'likel_time: {0}'.format(time.time()-t)
    t = time.time()
    gp.likel_fun.gradient()
    print 'grad_time: {0}'.format(time.time()-t)
    print np.exp(gp.hyperparams)
    print gp.log_likel
    print gp.likel_fun.gradient()
    
    t = time.time()
    gp = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector)
    gp.fit(X, y)
    print 'opt_time: {0}'.format(time.time()-t)
    plot1d_gp(gp, 0, 3)
