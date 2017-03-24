'''
Created on Aug 19, 2011

@author: marcel
'''
import numpy as np

from abc import ABCMeta, abstractmethod, abstractproperty

class ConjugatePrior(object):
    
    __metaclass__ = ABCMeta
    
    __slots__ = ('_X',      #(n,d)-dimensional data matrix
                 '_d',      #dimension of each data point
                 '_n'       #number of data points in X
                 )
    
    def __init__(self, X):
        X = np.atleast_2d(np.asarray(X))
        if X.ndim != 2:
            raise ValueError('X must be 2-dimensional.')
        
        self._X = X
        self._n, self._d = X.shape
    
    @property
    def data(self):
        '''
        @todo: - make immutable
        '''
        return self._X
    
    @property
    def ndim(self):
        return self._d
    
    @abstractproperty
    def hyperparams(self):
        pass
    
    @abstractproperty
    def posterior_hyperparams(self):
        pass    
    
    @abstractproperty
    def n_hyperparams(self):
        pass
    
    @abstractproperty
    def n_params(self):
        pass
    
    @abstractmethod
    def posterior(self, params):
        pass
    
    def marginal_posterior(self, i):
        pass
    
    @abstractmethod
    def predictive(self, X):
        '''
        '''
        pass
    
    @abstractmethod
    def likelihood(self):
        '''
        Returns the marginal log likeliohood of the model.
        '''
        pass
    
    @abstractmethod
    def map(self, i=None):
        '''
        Returns the map estimates of the latent variables of the distribution.
        '''
        pass
    
    @abstractmethod
    def likel_fun(self):
        pass
    
    @abstractmethod
    def sample(self, n):
        pass
    
    @abstractmethod
    def update(self, X):
        pass