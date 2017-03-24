'''
Created on Jul 13, 2011

@author: marcel
'''

import numpy as np

class RandKFold(object):
    '''
    classdocs
    '''

    __slots__ = ('_n',
                 '_k',
                 '_seed')

    def __init__(self, n, k, seed=None):
        '''
        Constructor
        '''
        if k < 1: 
            raise ValueError('k must be greater 0.')
        if k > n: 
            raise ValueError('k cannot be below the number of samples n') 
        
        self._n = n
        self._k = k
        self._seed = seed
        
    def __iter__(self):
        n = self._n
        k = self._k 
        
        rand = np.random.mtrand.RandomState(self._seed)    
        perm = rand.permutation(n)
        
        fold_size = np.ones(k) * np.ceil(n/k) #todo: check for casting to float
        fold_size[0:n%k] += 1
        
        cursor = 0
        for i in xrange(k):
            test_idx = np.zeros(n, dtype=bool)
            test_idx[perm[cursor:cursor+fold_size[i]]] = True
            train_idx = np.logical_not(test_idx)
            
            cursor += fold_size[i]
            
            yield train_idx, test_idx
            


class BagFold(object):
    
    __slots__ = ('_x')
    
    def __init__(self, x):
        '''
        @todo: check dimension
        '''
        self._x = np.asarray(x)
    
    def __iter__(self):
        '''
        '''
        x = self._x
        n = len(x)
        x_unique = np.unique(x) 
        
        for bag in x_unique:
            test_idx = np.zeros(n, dtype=bool)
            test_idx[x == bag] = True
            train_idx = np.logical_not(test_idx)
            
            yield train_idx, test_idx