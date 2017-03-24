'''
Created on Feb 14, 2012

@author: marcel
'''
import numpy as np
import upgeo.util.metric as metric
import upgeo.util.stats as stats

from upgeo.eval.cv import RandKFold

class CVRegressionExperiment(object):

    __slots__ = ('_X',
                 '_y',
                 '_nfolds',
                 '_filter',
                 '_filter_mask',
                 '_seed' 
                 )
    
    def __init__(self, X, y, nfolds=10, filter=None, filter_mask=None, seed=None):
        self._X = X
        self._y = y
        self._nfolds = nfolds
        self._filter = filter
        self._filter_mask = filter_mask
        self._seed = seed        
    
    def eval(self, algo):
        '''
        @todo: - check that the algo isnt a type of multitask learner
        '''
        hyperparams = algo.hyperparams
        
        X = self._X
        y = self._y
        
        n = X.shape[0]
        k = self._nfolds
        
        scores = np.empty(k)
        weights = np.empty(k)
        
        loo = RandKFold(n, k, self._seed)
        i = 0
        for train, test in loo:
            Xtrain, ytrain, Xtest, ytest = self.__prepare_eval_data(X, y, train, test)
            
            algo.hyperparams = hyperparams
            algo.fit(Xtrain, ytrain)
            print 'hyperparams = {0}'.format(np.exp(algo.hyperparams))
            yhat = algo.predict(Xtest)
            
            mse = metric.mspe(ytest, yhat)
            scores[i] = mse
            weights[i] = len(ytrain)
            i = i+1
            
        algo.hyperparams = hyperparams
        
        print scores
        mse = stats.mean(scores, weights)
        err = stats.stddev(scores, weights)
        
        return mse, err
    
    def __prepare_eval_data(self, X, y, train, test):
        Xtr, ytr = X[train], y[train]
        Xte, yte = X[test], y[test]
        
        if self._filter is not None:
            Ztr = np.c_[Xtr, ytr]
            Zte = np.c_[Xte, yte]
            
            k = Ztr.shape[1]
            mask = self._filter_mask
            if mask is None:
                mask = np.arange(k)
            
            Ztr[:,mask] = self._filter.apply(Ztr[:,mask])
            Zte[:,mask] = self._filter.apply(Zte[:,mask], True)
            
            Xtr = Ztr[:,0:(k-1)]
            ytr = Ztr[:,k-1]
            Xte = Zte[:,0:(k-1)]
            yte = Zte[:,k-1]
                
        return Xtr, ytr, Xte, yte
