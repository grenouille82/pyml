'''
Created on Mar 11, 2011

@author: marcel
'''
import numpy as np

from numpy.core.numeric import NaN

def mcar(x, ratio, features=None, exact=True):
    '''
    '''
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError('x must be 2 dimensional')
    
    m,d = x.shape
    features = np.arange(d) if features == None else __asarray_indices(features)
    use_all_features = features.size == d and features[0] == 0 and features[-1] == (d-1)
    
    if use_all_features and (ratio < 0.0 or ratio > (d-1.0)/d):
        raise ValueError('invalid ratio: {0}'.format(ratio))
    
    nan_mask = np.zeros((m,d), dtype='bool')
    if exact:
        restr_nan_mask = np.ravel(nan_mask[:, features])
        
        n = restr_nan_mask.size
        k = np.floor(n*ratio)
        perm = np.random.permutation(n)
        if k > 0:
            restr_nan_mask[perm[:k]] = True
            
        if use_all_features:
            full_nan_cases = np.all(nan_mask, 1)
            for i in np.nonzero(full_nan_cases):
                j = np.random.random_integers(0, d)
                nan_mask[i,j] = False
            
            
    else:
        rand_matrix = np.random.rand(m, features.size)
        nan_mask[rand_matrix <= ratio] = True
        
        if use_all_features:
            full_nan_cases = np.all(nan_mask, 1)
            for i in np.nonzero(full_nan_cases):
                j = np.argmax(rand_matrix[i])
                nan_mask[i,j] = False
        
    nan_x = x.copy
    np.putmask(nan_x, nan_mask, NaN)
    
    return nan_x

def mar(x, ratio):
    raise NotImplementedError()

def __asarray_indices(indices):
    '''
    
    '''
    indices = np.ravel(indices)
        
    if indices.dtype == 'bool':
        indices = np.flatnonzero(indices)
    else:
        indices = np.unique(indices);
        indices = np.asarray(indices, dtype='int')
        
    return indices
