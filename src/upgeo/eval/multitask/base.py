'''
Created on May 13, 2011

@author: marcel
'''
import numpy as np

from upgeo.data import MultitaskDataSource, dataset2array
from upgeo.ml.regression.function import GaussianBasis, RadialBasisFunction
from upgeo.ml.multitask.base import compound_data, flatten_data,\
    asmtarray_object

def load_multidataset(filename, task_key, task_fields, target_field):
    '''
    '''
    ds = MultitaskDataSource.create_datasource(filename)
    ds.task_key = task_key
    ds.task_id_field = task_key
    ds.task_fields = task_fields
    
    dataset = ds.load()
    dataset.targets = target_field
    return dataset
    

def load_data(filename, task_key, task_fields, target_field):
    '''
    Load a multitask dataset for a regression problem from a specified file
    and returns a triplet (X,Y,Z), where X are the covariate data matrix, 
    Y the target data and Z the task meta data for each task. 
    '''
    ds = MultitaskDataSource.create_datasource(filename)
    ds.task_key = task_key
    ds.task_id_field = task_key
    ds.task_fields = task_fields
    
    dataset = ds.load()
    dataset.targets = target_field
    X,Y,Z = dataset2array(dataset)
    return (X,Y,Z)

def make_rbf(X, kernel=GaussianBasis(0.25), ratio=0.05):
    '''
    Returns the 
    '''
    X = asmtarray_object(X)
    Xf = flatten_data(X)
    if ratio == 1.0:
        rbf = RadialBasisFunction.make_rbf_from_data(Xf, kernel)
    else:
        rbf = RadialBasisFunction.make_rbf_from_kmeans(Xf, kernel, ratio)
        
    n = len(X)
    Xrbf = np.empty(n, dtype='object')
    for i in xrange(n):
        Xrbf[i] = rbf(X[i])
    return rbf, Xrbf

def shrink_data(size, X, Y, Z):
    '''
    Shrink the multitask data to the given size for each task. If the task below
    the shrinkage size then the task is removed from the collection. The samples
    are choosen randomly for each task if the collection must be shrinked. 
    '''
    X = asmtarray_object(X)
    Z = asmtarray_object(Z)
    Y = asmtarray_object(Y)
    
    if len(X) != len(Z) != len(Y):
        raise ValueError('X, Y, Z must have the same length.')
    
    Xret = []
    Zret = []
    Yret = []
    
    n = len(X)
    for i in xrange(n):
        Xi = X[i]
        Yi = Y[i]
        Zi = Z[i]
        
        m = len(X[i])
        if m >= size:
            if m > size:
                perm = np.random.permutation(m)
                Xi = Xi[perm[size]]
                Yi = Yi[perm[size]]
            Xret.append(Xi)
            Yret.append(Yi)
            Zret.append(Zi)
        
    Xret = asmtarray_object(Xret)
    Yret = asmtarray_object(Yret)
    Zret = asmtarray_object(Zret)
    
    return Xret, Yret, Zret


def del_rare_tasks(min_size, X, Y, Z):
    X = asmtarray_object(X)
    Z = asmtarray_object(Z)
    Y = asmtarray_object(Y)
    
    if len(X) != len(Z) != len(Y):
        raise ValueError('X, Y, Z must have the same length.')
    
    Xret = []
    Zret = []
    Yret = []
    
    n = len(X)
    for i in xrange(n):
        if len(X[i]) >= min_size:
            Xret.append(X[i])
            Yret.append(Y[i])
            Zret.append(Z[i])
            
    Xret = asmtarray_object(Xret)
    Yret = asmtarray_object(Yret)
    Zret = asmtarray_object(Zret)
    
    return Xret, Yret, Zret