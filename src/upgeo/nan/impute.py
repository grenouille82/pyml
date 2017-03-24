'''
Created on Mar 11, 2011

@author: marcel

@todo:
- handling discrete attributes
- extract abstract basis class
- Imputation classes should check whether the models are initialized before 
  the impute method can invoked. throw an exception in such case.
'''
import numpy as np

from scikits.learn.base import BaseEstimator
from upgeo.exception import NotFittedError
from upgeo.util.metric import distance_matrix
from upgeo.util.array import sqix_
from upgeo.util.stats import nanmean, nancov
from upgeo.ml.cluster.mixture import MoG

class ABCImputation(BaseEstimator):
    __slot__ = ('__nfeatures', '_is_init')
    
    def fit(self, x):
        pass
    
    def impute(self, x):
        self.__init_check()
        
            
    def __init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked')
    
class MoGImputation(ABCImputation):
    '''
    @todo:
    - implement model selection criterion for the gaussian mixture model
    '''
    __slots__ = ('__gmm')
    
    def __init__(self, k, cov_type='full'):
        self.__gmm = MoG(k, cov_type)
        
    def fit(self, x):
        x = np.asarray(x)
        
        self.__gmm.fit(x, init='kmeans')
        self._is_init = True
        self.__nfeatures = x.shape[1] 
    
    def impute(self, x):
        super.impute(self, x)
        
        x = np.asarray(x)
        complete_x = x.copy()
        nan_mask = np.isnan(x)
        nan_rows = np.any(nan_mask, 1)
        
        for i in np.flatnonzero(nan_rows):
            mv = nan_mask[i]
            ov = ~mv
            complete_x[i,mv] = self.__gmm.expectation(x, ov, mv)
        
        return complete_x
    

class KNNImputation(ABCImputation):
    '''
    @todo
    - parametrize by distance measure
    - validate the imputation method
    - swap the knn computation in an own class (using for example kd-trees)
    '''
    __slots__ = ('__k', '__data', '__method')
    
    def __init__(self, k=3, method='mean'):
        '''
        Constructur
        '''
        self.__k = k
        self.__method = method 
        
    def fit(self, x):
        '''
        '''
        
        x = np.asarray(x)
        nan_rows = np.any(np.isnan(x), 1)
        self.__data = x[~nan_rows].copy()
        
        if self.__data.shape[0] < self.__k:
            raise ValueError('complete cases is less than k')
        
        return self.impute(x)
        
    def impute(self, x):
        '''
        '''
        super.impute(x)
        
        x = np.asarray(x)
        complete_x = x.copy()
        nan_mask = np.isnan(x)
        nan_rows = np.any(nan_mask, 1)
        
        for i in np.flatnonzero(nan_rows):
            mv = nan_mask[i]
            ov = ~mv
            
            _, dist, kneighbors  = _knn_search(x[i,ov], self.__data[:,ov], self.__k)
            if self.__method == 'mean':
                complete_x[i, mv] = np.mean(self.__data[sqix_(kneighbors, mv)], 0)
            elif self.__method == 'wmean':
                complete_x[i, mv] = np.median(self.__data[sqix_(kneighbors, mv)], 0)
            elif self.__method == 'median':
                complete_x[i, mv] = np.average(self.__data[sqix_(kneighbors, mv)], 0, dist)
            else:
                raise ValueError('unknown imputation method: {0}'.format(self.__method))
                
        return complete_x
                
class MeanImputation(ABCImputation):
    '''
    '''
    __slots__ = ('__mean')
    
    def fit(self, x):
        x = np.asarray(x)
        
        self.__mean = np.mean(x, 0)
        self._is_init = True
        self.__nfeatures = x.shape[1]
        
        return self.impute(x)

    def impute(self, x):
        super.impute(x)
        
        complete_x = np.asarray(x).copy()
        nan_mask = np.isnan(x)
        nan_rows = np.any(nan_mask, 1)
        
        for i in np.flatnonzero(nan_rows):
            complete_x[i, nan_mask[i]] = self.__median[nan_mask[i]]
        
        return complete_x

class MedianImputation(ABCImputation):
    '''
    '''
    __slots__ = ('__median')
    
    def fit(self, x):
        x = np.asarray(x)
        
        self.__median = np.median(x, 0)
        self._is_init = True
        self.__nfeatures = x.shape[1]
        
        return self.impute(x)

    def impute(self, x):
        super.impute(x)
        
        complete_x = np.asarray(x).copy()
        nan_mask = np.isnan(x)
        nan_rows = np.any(nan_mask, 1)
        
        for i in np.flatnonzero(nan_rows):
            complete_x[i, nan_mask[i]] = self.__median[nan_mask[i]]
        
        return complete_x

class RandomNormalImputation(ABCImputation):
    '''
    '''
    __slots__ = ('__mean', '__var')
    
    def fit(self, x):
        x = np.asarray(x)
        
        self.__mean = np.mean(x, 0)
        self.__var = np.var(x, 0)
        self._is_init = True
        self.__nfeatures = x.shape[1]
        
        return self.impute(x)
    
    def impute(self, x):
        super.impute(x)
        
        complete_x = np.asarray(x).copy()
        nan_mask = np.isnan(x)
        nan_cols = np.any(nan_mask, 0)
        
        for i in np.flatnonzero(nan_cols):
            mu = self.__mean[i]
            sigma = self.__var[i]
            n = nan_mask[:,i]
            complete_x[nan_mask[:,i], i] = mu + np.random.rand(n) * sigma
        
        return complete_x
        pass
    
class RandomMVNImputation(ABCImputation):
    '''
    '''
    __slots__ = ('__mu', '__sigma', '__cov_type')
    
    def __init__(self, cov_type='full'):
        self.__cov_type = cov_type
    
    def fit(self, x):
        x = np.asarray(x)
        
        self.__mu = nanmean(x, 0)
        self.__sigma = nancov(x, 0)
        self._is_init = True
        self.__nfeatures = x.shape[1]
        
        return self.impute(x)
    
    def impute(self, x):
        raise NotImplementedError()
                
def _knn_search(v, X, k):
    '''
    '''
    dist = distance_matrix(X, v)
    perm = np.argsort(v)
    return X[perm[:k]], dist[perm[:k]]
