'''
Created on Mar 14, 2011

@author: marcel
'''

import numpy as np

from scikits.learn.base import BaseEstimator, RegressorMixin

from upgeo.ml.cluster.mixture import MoG
from upgeo.util.metric import mspe

class MoGRegression(BaseEstimator, RegressorMixin):
    '''
    '''

    __slots__ = ('_gmm', '__pred_method', '__covariates', '__targets')

    def __init__(self, nstates, cov_type='full', pred_method='lcs'):
        '''
        '''
        if not pred_method in ['lcs', 'slcs', 'storch']:
            raise TypeError('bad prediction method: {0}'.format(pred_method))
        
        gmm = MoG(nstates, cov_type)
        self.__gmm = gmm
        self.__pred_method = pred_method
    
    def fit(self, X, y, init='kmeans', max_iter=100, min_covar=1e-3, thresh=1e-5):
        '''
        '''
        X = np.asarray(X)
        y = np.asarray(y)
        
        cn = X.shape[1]
        tn = 1 if y.ndim == 1 else y.shape[1]
        self.__covariates = np.arange(cn)
        self.__targets = np.arange(cn, tn)
        
        data = np.c_[X, y]
        self.__gmm.fit(data, init, max_iter, min_covar, thresh)
        
        yhat = self.predict(X)
        error = mspe(y, yhat)
        return error
    
    def predict(self, X):
        '''
        '''
        X = np.asarray(X)
        if self.__pred_method == 'lcs':
            yhat = self.__lcs_predict(X)
        elif self.__pred_method == 'slcs':
            yhat = self.__slcs_predict(X)
        elif self.__pred_method == 'storch':
            yhat = self.__storch_predict(X)
        else:
            raise ValueError()
        
        return yhat
    
    def _get_nstates(self):
        return self.__gmm.nstates
    
    nstates = property(fget=_get_nstates)
    
    def _get_cov_type(self):
        return self.gmm.cov_type
    
    cov_type = property(fget=_get_cov_type)
    
    def _get_covariates(self):
        return self.__covariates.copy()
    
    covariates = property(fget=_get_covariates())
    
    def _get_ncovariates(self):
        return self.__covariates.size
    
    ncovariates = property(fget=_get_ncovariates)
    
    def _get_targets(self):
        return self.__targets.copy()
    
    targets = property(fget=_get_targets)
    
    def _get_ntargets(self):
        return self.__targets.size
    
    ntargets = property(fget=_get_ntargets)
    
    def _get_gmm(self):
        return self.__gmm.copy()
    
    gmm = property(fget=_get_gmm)
    
    def __lcs_predict(self, X):
        '''
        '''
        yhat = self.__gmm.expectation(X, self.__covariates, self.__targets)
        return yhat

    def __slcs_predict(self, X):
        '''
        '''
        gmm = self.__gmm
        covariates = self.__covariates
        targets = self.__targets
        
        m = X.shape[0]
        d = targets.size
        
        yhat = np.squeeze(np.zeros((m,d)))
        
        #determine the mixture with the highest posterior for each case
        covar_gmm = gmm.marginalize(covariates)
        post = covar_gmm.posterior(X)
        max_components = np.argmax(post, 1)
        
        for j in xrange(self.ncomponents):
            i = np.where(max_components == j)
            yhat[i] = gmm.component_expectation(j, X, covariates, targets)
        
        return yhat
    
    def __storch_predict(self, X):
        raise NotImplementedError()