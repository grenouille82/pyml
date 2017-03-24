from scikits.learn.base import BaseEstimator

class GaussianBHC(BaseEstimator):
    
    '''
    
    _cluster_tree - 
    _alpha - prior
    _
    '''
    
    __slots__ = ('_cluster_tree', '_alpha', '_is_init', '')
    
    def __init__(self):
        pass
    
    def fit(self, x, m, S, ):
        pass

class GaussianPrior:
    __slots__ = ()

class _GaussianClusterData:
    __slots__ = ('dof', 'pi', )