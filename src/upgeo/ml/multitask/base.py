'''
Created on May 2, 2011

@author: marcel
'''
import numpy as np

from scikits.learn.base import BaseEstimator, RegressorMixin
from upgeo.exception import NotFittedError
from upgeo.util.metric import mspe, NaiveMatrixSearch
from _abcoll import Container
from upgeo.ml.cluster.prototype import KMeans
from scikits.learn.cross_val import LeaveOneOut


class MultiISpaceLearner(BaseEstimator, RegressorMixin):
    '''
    Meta Learner
    '''
    __slots__ = ('_model',
                 '_use_meta_features'
                 '_is_init'
                 )
    
    def __init__(self, model, use_meta_features=False):
        '''
        @todo: - check the type of the model
        '''
        self._model = model
        self._use_meta_features = use_meta_features
        self._is_init = False
        
    def fit(self, X, Y, Z):
        '''
        '''
        #if len(X) != len(Y) != len(Z):
        #    raise ValueError('number of task data X, Y and task ' + 
        #                     'description Z must be the same')
        
        if self._use_meta_features == True:
            X = compound_data(X, Z)
        
        covars = flatten_data(X)
        targets = flatten_data(Y)
        self._model.fit(covars, targets)
        
        self._is_init = True
    
    def predict(self, X, Z):
        '''
        '''
        self._init_check()
        
        #if len(X) != len(Z):
        #    raise ValueError('number of task data X and task' + 
        #                     'description Z must be the same')
        
        X = compound_data(X,Z) if self._use_meta_features == True else asmtarray_object(X)
        n = len(X)

        Y = np.empty(n, dtype='object')
        for i in xrange(n):
            Y[i] = self._model.predict(X[i])
            
        return Y
    
    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')
        
    def _get_model(self):
        return self._model
    
    model = property(fget=_get_model)
    
    def _get_hyperparams(self):
        return self.model.hyperparams
    
    def _set_hyperparams(self, params):
        self.model.hyperparams = params
    
    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)

class MetaSpaceClusterLearner(BaseEstimator, RegressorMixin):
    '''
    @todo: - use metafeatures in prediction 
    '''
    __slots__ = ('_max_k',
                 '_model',
                 '_use_meta_features',  #flag for using meta features in prediction
                 '_k',
                 '_centres',
                 '_X',
                 '_Z',
                 '_Y',
                 '_labels',
                 '_is_init'
                 )
    
    def __init__(self, model, max_k=10, use_meta_features=False):
        '''
        @todo: - check the type of model
        '''
        if max_k < 1: 
            raise ValueError('max number of cluster must be greater than zero.')
        
        self._max_k = max_k
        self._model = model
        self._use_meta_features = use_meta_features
        
    def fit(self, X, Y, Z):
        '''
        @todo: - problems occur when Z is transformed to an mtarray
        '''
        X = asmtarray_object(X)
        Y = asmtarray_object(Y)
        #Z = asmtarray_object(Z)
        Z = np.asanyarray(Z.tolist(), dtype='float64')
        
        if self._use_meta_features:
            X = compound_data(X, Z)

        #leave one out cv method over task to determine the optimial model size
        mspe_opt = np.inf
        k_opt = np.nan
        for k in xrange(2, self._max_k+1):
            print 'step={0}'.format(k)
            mspe = self._validate_model(k, X, Y, Z)
            if mspe < mspe_opt:
                mspe_opt = mspe
                k_opt = k
        
        print 'opt_k={0}'.format(k_opt)
        #estimate the task cluster
        centres, labels = self._estimate_metaspace_model(k, Z)
        
        self._X = X
        self._Y = Y
        self._Z = Z
        
        self._centres = centres
        self._labels = labels
        self._k = k_opt
        
        self._is_init = True
        
    def predict(self, X, Z):
        '''
        '''
        self._init_check()
        
        X = asmtarray_object(X)
        Z = asmtarray_object(Z)
        
        if self._use_meta_features:
            X = compound_data(X, Z)
        
        n = len(X)

        Yhat = np.empty(n, dtype='object')
        for i in xrange(n):
            model = self._find_cluster_learner(Z[i], self._centres, self._X, 
                                               self._Y, self._Z, self._labels)
            Yhat[i] = model.predict(X[i])
            
        return Yhat

        
    def _validate_model(self, k, X, Y, Z):
        '''
        Leave-one-out model validation
        '''
        total_mspe = 0
        total_m = 0
        
        n = len(X)
        loo = LeaveOneOut(n)
        for train, test in loo:
            #todo: remove hack
            Xtest = X[test][0] 
            Ytest = Y[test][0]
            
            if Xtest.ndim != 2:
                Xtest = X[test]
                Ytest = Y[test]
            
            centres, labels = self._estimate_metaspace_model(k, Z[train])
            model = self._find_cluster_learner(Z[test], centres, X, Y, Z, labels)
            
            yhat = model.predict(Xtest)
            m = len(X[test])
            total_mspe += m * mspe(Ytest, yhat)
            total_m += m
            
        mse = total_mspe/total_m
        return mse
    
    def _find_cluster_learner(self, z, centres, X, Y, Z, labels):
        '''
        @todo: the member model is modified and returned, make a copy by using
               factory pattern
        '''
        neighbor_struct = NaiveMatrixSearch(centres)
        winner, _ = neighbor_struct.query_knn(z)
        
        cluster = np.flatnonzero(labels == winner)
        #@todo hack
        if len(cluster) > 1:
            covars = flatten_data(X[cluster])
            targets = flatten_data(Y[cluster])
        else:
            print cluster
            print cluster.shape
            print X[cluster]
            print X[cluster].shape
            covars = X[cluster][0]
            targets = Y[cluster][0]
        
        self._model.fit(covars, targets)
        return self._model
        
    def _estimate_metaspace_model(self, k, Z):
        '''
        '''
        kmeans = KMeans(k)
        labels,_ = kmeans.fit(Z, k)
        centres = kmeans.centers
        return centres, labels
    
    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')

class PseudoMultilevelRegression(BaseEstimator, RegressorMixin):
    '''
    Meta Learner
    '''
    __slots__ = ('_model',
                 '_is_init'
                 )

    def __init__(self, reg_model):
        '''
        @todo: - check the type of the reg_model
               - check whether the reg_model contains basis_functions 
                 (should not have)
        '''
        self._model = reg_model
        self._is_init = False
        
    def fit(self, X, Y, Z):
        '''
        '''
        #if len(X) != len(Y) != len(Z):
        #    raise ValueError('number of task data X, Y and task ' + 
        #                     'description Z must be the same')
        
        covars = flatten_data(mix_compound_data(X, Z))
        targets = flatten_data(Y)
        self._model.fit(covars, targets)
        
        self._is_init = True
    
    def predict(self, X, Z):
        '''
        '''
        self._init_check()
        
        #if len(X) != len(Z):
        #    raise ValueError('number of task data X and task' + 
        #                     'description Z must be the same')
        
        X = mix_compound_data(X, Z)
        n = len(X)
        
        Y = np.empty(n, dtype='object')
        for i in xrange(n):
            Y[i] = self._model.predict(X[i])
            
        return Y

    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')
        
    def _get_model(self):
        return self._model
    
    model = property(fget=_get_model)

def mix_compound_data(X, Z):
    '''
    '''
    X = asmtarray_object(X)
    Z = asmtarray_object(Z)
    
    n = len(X)
    Xres = np.empty(n, dtype='object')
    for i in xrange(n):
        x = np.asarray(X[i])
        z = np.squeeze(np.asarray(Z[i]))
        d = len(z)
        k = len(X[i])
        
        #create mixing coefficients
        for j in xrange(d):
            x = np.c_[x, x*z[j]]
            
        Xres[i] = np.c_[x, np.tile(z, [k,1])]
        
    return Xres
    

def compound_data(X, Z):
    '''
    '''
    X = asmtarray_object(X)
    Z = asmtarray_object(Z)
        
    n = len(X)
    Xres = np.empty(n, dtype='object')
    for i in xrange(n):
        k = len(X[i])
        Xres[i] = np.c_[X[i], np.tile(Z[i], [k,1])]
    
    return Xres 

def flatten_data(X):
    '''
    '''
    X = asmtarray_object(X)
    n = len(X)
    
    Xres = np.asarray(X[0]) if n > 0 else np.empty(0)
    for i in xrange(1,n):
        if len(X[i]) > 0:
            x = np.asarray(X[i])
            Xres = np.r_[Xres, x]
        
    return Xres

def asmtarray_object(a):
    if isinstance(a, Container):
        n = len(a)
        b = np.empty(n, dtype='object')
        for i in xrange(n):
            b[i] = np.asarray(a[i])
    elif isinstance(a, np.ndarray):
        if a.dtype != 'object':
            b = np.empty(1, dtype='object')
            b[0] = a
        else:
            b = a
    else:
        raise TypeError('a must be a type of Container or ndarray.')
    
    return b