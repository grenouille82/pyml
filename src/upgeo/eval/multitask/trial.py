'''
Created on May 26, 2011

@author: marcel

@todo: refactor, classes are not really implemented consistently
'''

import numpy as np
import upgeo.util.metric as metric
import upgeo.util.stats as stats

from upgeo.ml.multitask.base import asmtarray_object, MultiISpaceLearner,\
    compound_data, flatten_data, MetaSpaceClusterLearner
from scikits.learn.cross_val import LeaveOneOut, KFold
from upgeo.ml.regression.bayes import EMBayesRegression, MapBayesRegression,\
    EBChenRegression, FastBayesRegression
from upgeo.eval.multitask.base import load_data, make_rbf, shrink_data,\
    del_rare_tasks
from numpy.lib.npyio import savetxt
from upgeo.util.metric import mspe, rsquare
from numpy.core.numeric import array_str, asanyarray
from upgeo.ml.regression.linear import OptRBFRegression, LinearRegressionModel,\
    RidgeRegression
from upgeo.ml.regression.function import GaussianBasis
from upgeo.ml.multitask.regression.dpm import ItGradBHCRegression, BHCRegression
from upgeo.eval.cv import RandKFold
from abc import ABCMeta, abstractmethod
from upgeo.exception import NotFittedError

class MultitaskRegressionExperiment(object):
    '''
    Abstract class for multitask regression experiments.
    '''
    __metaclass__ = ABCMeta
    
    __slots__ = ('_task_result',
                 '_total_result',
                 '_Ypred',
                 '_is_init')
    
    def __init__(self):
        '''
        '''
        self._is_init = False
    
    @abstractmethod
    def eval(self, algo):
        pass
           
    def _get_task_result(self):
        '''
        '''
        self._init_check()
        return self._task_result
    
    task_result = property(fget=_get_task_result)
    
    def _get_total_result(self):
        '''
        '''
        self._init_check()
        return self._total_result
    
    total_result = property(fget=_get_total_result)
    
    def _get_overall_prediction(self):
        self._init_check()
        return self._Ypred
    
    overall_prediction = property(fget=_get_overall_prediction)
    
    def _calc_task_scores(self, test_err, test_w, train_err, train_w):
        '''
        '''
        mean_test_err = stats.mean(test_err, test_w)
        dev_test_err = stats.stddev(test_err, test_w)
        mean_train_err = stats.mean(train_err, test_w)
        dev_train_err = stats.stddev(train_err, test_w)
        
        scores = np.array([mean_test_err, dev_test_err, mean_train_err, dev_train_err]) 
        return scores    
    
    def _init_check(self):
        '''
        @todo: - replace exception class 
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')  
 

class TaskLearningExperiment(MultitaskRegressionExperiment):
    '''
    @todo: - parametrize the cross validation method
    '''
    __slots__ = ('_X',
                 '_Y',
                 '_Z',
                 '_nfolds',
                 '_use_background_data',
                 '_use_meta_features',
                 '_seeds' #for each task
                 )
    
    def __init__(self, X, Y, Z, nfolds=10, use_meta_features=False, use_background_data=False, seeds=None):
        MultitaskRegressionExperiment.__init__(self)
        
        self._X = asmtarray_object(X)
        self._Y = asmtarray_object(Y)
        self._Z = asmtarray_object(Z)
        
        if len(X) != len(Y) != len(Z):
            raise ValueError('Number of tasks must be equal in X,Y,Z.')
        
        self._nfolds = nfolds
        self._use_meta_features = use_meta_features
        self._use_background_data = use_background_data
        
        self._seeds = seeds
        
    def eval(self, algo):
        '''
        @todo: - check that the algo isnt a type of multitask learner
        '''
        X = self._X
        Y = self._Y
        Z = self._Z
        
        hyperparams = algo.hyperparams
           
        k = self._nfolds
        #preprocessing for transfer learning - remove all task with less than 10 samples
        rem = np.zeros(len(X),dtype=np.bool)
        for i in xrange(len(X)):
            if len(X[i]) < k:
                rem[i] = True
        X = X[np.nonzero(np.invert(rem))[0]]
        Y = Y[np.nonzero(np.invert(rem))[0]]
        Z = Z[np.nonzero(np.invert(rem))[0]]
 
        
        n = len(X)
        Ypred = np.empty(n, dtype='object')
        
        if self._use_meta_features == True:
            X = compound_data(X, Z)
       
        task_result = np.empty((n,4))
        weights = np.empty(n)
        #evaluation is performed for each task individually 
        for i in xrange(n):
            print 'task={0}'.format(i+1) 
            
            Xi = X[i]
            Yi = Y[i]
            
            m = len(Xi)
            if self._use_background_data == True:
                indices = np.ones(n, dtype=bool)
                indices[i] = False
                Xback = flatten_data(X[indices])
                Yback = flatten_data(Y[indices])
            
            
            if self._seeds != None:
                np.random.seed(self._seeds[i])
            loo = RandKFold(m, self._nfolds)
            Yhat = np.empty(m)
            
            train_err = []
            test_err = []
            train_w = []
            test_w = []
            #k-fold cv for eacht task
            for train, test in loo:
                Yi_hat = np.empty(m)
                Z = np.empty(m)
                
                Xtrain = Xi[train]
                Ytrain = Yi[train]
                print Xtrain.shape
                
                if self._use_background_data == True:
                    #todo: filter background data
                    Xtrain = np.r_[Xtrain, Xback]
                    Ytrain = np.r_[Ytrain, Yback]
                
                algo.hyperparams = hyperparams
                algo.fit(Xtrain, Ytrain)
                Yi_hat[test] = algo.predict(Xi[test])
                Yi_hat[train] = algo.predict(Xi[train])
                
                Yhat[test] = Yi_hat[test]
                
                train_err.append(metric.mspe(Yi[train], Yi_hat[train]))
                test_err.append(metric.mspe(Yi[test], Yi_hat[test]))
                
                train_w.append(np.sum(train))
                test_w.append(np.sum(test))
                
            task_result[i] = self._calc_task_scores(test_err, test_w, train_err, train_w)
            print task_result[i]
            
            Ypred[i] = Yhat
            weights[i] = m
        
        #compute the total error with its variance (check if its true)
        mean_test_mse = stats.mean(task_result[:,0], weights)
        dev_test_mse = stats.stddev(task_result[:,0], weights)
        mean_train_mse = stats.mean(task_result[:,2], weights)
        dev_train_mse = stats.stddev(task_result[:,2], weights)
        
        total_result = np.array([mean_test_mse, dev_test_mse, mean_train_mse, 
                                 dev_train_mse])
        
        np.random.seed(None)
        
        self._Ypred = Ypred
        self._task_result = task_result
        self._total_result = total_result
        self._is_init = True
        print 'total_result'
        print total_result
        print '------------'
        return total_result

class CVTaskLearningExperiment(MultitaskRegressionExperiment):
    '''
    @todo: - parametrize the cross validation method
    '''
    __slots__ = ('_X',
                 '_Y',
                 '_Z',
                 '_nfolds',
                 '_seed',
                 '_use_background_data',
                 '_use_meta_features')
    
    def __init__(self, X, Y, Z, nfolds=10, seed=None, use_meta_features=False, use_background_data=False):
        MultitaskRegressionExperiment.__init__(self)
        
        self._X = asmtarray_object(X)
        self._Y = asmtarray_object(Y)
        self._Z = asmtarray_object(Z)
        
        if len(X) != len(Y) != len(Z):
            raise ValueError('Number of tasks must be equal in X,Y,Z.')
        
        self._seed = seed
        self._nfolds = nfolds
        self._use_meta_features = use_meta_features
        self._use_background_data = use_background_data
        
    def eval(self, algo):
        '''
        @todo: - check that the algo isnt a type of multitask learner
        '''
        np.random.seed(self._seed)
        
        X = self._X
        Y = self._Y
        Z = self._Z
        
        hyperparams = algo.hyperparams
    
        k = self._nfolds
        #preprocessing for transfer learning - remove all task with less than 10 samples
        rem = np.zeros(len(X),dtype=np.bool)
        for i in xrange(len(X)):
            if len(X[i]) < k:
                rem[i] = True
        X = X[np.nonzero(np.invert(rem))[0]]
        Y = Y[np.nonzero(np.invert(rem))[0]]
        Z = Z[np.nonzero(np.invert(rem))[0]]
        
        n = len(Z)
        
        if self._use_meta_features == True:
            X = compound_data(X, Z)
            
        mse = np.empty(k)
        sse = 0
        
        task_folds = np.empty((n,k), dtype='object')
        for i in xrange(n):
            m = len(X[i])
            loo = RandKFold(m,k)
            j = 0
            for fold in loo:
                task_folds[i,j] = fold
                j += 1
            
            
        for i in xrange(k):
            Xtrain = np.empty(n, dtype='object')
            Xtest = np.empty(n, dtype='object')
            Ytrain = np.empty(n, dtype='object')
            Ytest = np.empty(n, dtype='object')
           
            for j in xrange(n):
                train, test = task_folds[j,i]
                Xtrain[j] = X[j][train]
                Xtest[j] = X[j][test]
                Ytrain[j] = Y[j][train]
                Ytest[j] = Y[j][test]
                
            if self._use_background_data:
                algo.hyperparams = hyperparams
                algo.fit(flatten_data(Xtrain), flatten_data(Ytrain))
                yfit = algo.predict(flatten_data(Xtest))
                y = flatten_data(Ytest)
                mse[i] = metric.mspe(y, yfit)
                print 'mse={0}'.format(mse[i])
                sse += metric.tspe(y, yfit)
            else:
                yfit = []
                for j in xrange(n):
                    algo.hyperparams = hyperparams
                    algo.fit(Xtrain[j], Ytrain[j])
                    yfit.extend(algo.predict(Xtest[j]))
                    
                yfit = np.array(yfit)
                y = flatten_data(Ytest)
                mse[i] = metric.mspe(y, yfit)
                sse += metric.tspe(y, yfit)
                print 'mse={0}'.format(mse[i])
    
        print 'task'    
        #compute the total error with its variance
        y = flatten_data(Y)
        ymean =  np.mean(y)
        sst = metric.tse(y, ymean)
        r2 = 1-sse/sst

        algo.hyperparams = hyperparams

        total_result = np.array([r2, np.mean(mse), np.std(mse)])
        self._total_result = total_result
        self._is_init = True
        
        #unseed
        np.random.seed(None)
        
        return total_result
    
class TransferLearningExperiment(MultitaskRegressionExperiment):
    
    __slots__ = ('_X',
                 '_Y',
                 '_Z',
                 '_nfolds',
                 '_seeds'
                 )
    
    def __init__(self, X, Y, Z, nfolds=10, seeds=None):
        MultitaskRegressionExperiment.__init__(self)
        
        self._X = asmtarray_object(X)
        self._Y = asmtarray_object(Y)
        self._Z = asmtarray_object(Z)
        

        if len(X) != len(Y) != len(Z):
            raise ValueError('Number of tasks must be equal in X,Y,Z.')
        
        self._nfolds = nfolds
        self._seeds = seeds

    def eval(self, algo):
        '''
        @todo: - check that the algo isnt a type of multitask learner
        '''
        X = self._X
        Y = self._Y
        Z = self._Z
        n = len(X)
        
        hyperparams = algo.hyperparams
    
        Ypred = np.empty(n, dtype='object')
        
        Xtrain = np.copy(X)
        Ytrain = np.copy(Y)
        
        weights = np.empty(n)
        task_result = np.empty((n,4))
        #evaluation is performed for each task individually 
        for i in xrange(n):
            print 'task={0}'.format(i+1)
            
            Xi = X[i]
            Yi = Y[i]
            
            m = len(Xi)
            
            if self._seeds != None:
                np.random.seed(self._seeds[i])
            loo = RandKFold(m, self._nfolds)
            Yhat = np.empty(m)
        
            train_err = []
            test_err = []
            train_w = []
            test_w = []
            #k-fold cv for eacht task
            for train, test in loo:
                Yi_hat = np.empty(m)
                
                Xtrain[i] = Xi[train]
                Ytrain[i] = Yi[train]
                
                #todo: filter the non-evaluted tasks by aquivalent bags
                
                
                algo.hyperparams = hyperparams
                algo.fit(Xtrain, Ytrain, Z)
                #todo: replace hack
                if len(Xi[test]) > 1:
                    result = algo.predict_by_task([Xi[test]], [i])
                else:
                    result = algo.predict_by_task(Xi[test], [i])
                Yi_hat[test] = result[0]
                
                if len(Xi[train]) > 1:
                    result = algo.predict_by_task([Xi[train]], [i])
                else:
                    result = algo.predict_by_task(Xi[train], [i])
                Yi_hat[train] = result[0]                
                Yhat[test] = Yi_hat[test]
                
                train_err.append(metric.mspe(Yi[train], Yi_hat[train]))
                test_err.append(metric.mspe(Yi[test], Yi_hat[test]))
                
                train_w.append(np.sum(train))
                test_w.append(np.sum(test))
            
            
            task_result[i] = self._calc_task_scores(test_err, test_w, train_err, train_w)
            print task_result[i]
            
            Xtrain[i] = X[i]
            Ytrain[i] = Y[i]
            
            Ypred[i] = Yhat
            weights[i] = m
            
        
        np.random.seed(None)
        
        #compute the total error with its variance (check if its true)
        mean_test_mse = stats.mean(task_result[:,0], weights)
        dev_test_mse = stats.stddev(task_result[:,0], weights)
        mean_train_mse = stats.mean(task_result[:,2], weights)
        dev_train_mse = stats.stddev(task_result[:,2], weights)
        
        total_result = np.array([mean_test_mse, dev_test_mse, mean_train_mse, 
                                 dev_train_mse])
        
        self._Ypred = Ypred
        self._task_result = task_result
        self._total_result = total_result
        self._is_init = True
        
        return total_result
    
class CVTransferLearningExperiment(MultitaskRegressionExperiment):
    
    __slots__ = ('_X',
                 '_Y',
                 '_Z',
                 '_nfolds',
                 '_seed',
                 '_use_meta_features'
                 )
    
    def __init__(self, X, Y, Z, nfolds=10, seed=None, use_meta_features=False):
        MultitaskRegressionExperiment.__init__(self)
        
        self._X = asmtarray_object(X)
        self._Y = asmtarray_object(Y)
        self._Z = asanyarray(Z)

        if len(X) != len(Y) != len(Z):
            raise ValueError('Number of tasks must be equal in X,Y,Z.')
        
        self._nfolds = nfolds
        self._seed = seed
        
        self._use_meta_features = use_meta_features

    def eval(self, algo):
        '''
        @todo: - check that the algo isnt a type of multitask learner
        '''
        
        #unseed
        np.random.seed(self._seed)
        
        X = self._X
        Y = self._Y
        Z = self._Z
        
        hyperparams = algo.hyperparams
    
        k = self._nfolds
        #preprocessing for transfer learning - remove all task with less than 10 samples
        rem = np.zeros(len(X),dtype=np.bool)
        for i in xrange(len(X)):
            if len(X[i]) < k:
                rem[i] = True
        X = X[np.nonzero(np.invert(rem))[0]]
        Y = Y[np.nonzero(np.invert(rem))[0]]
        Z = Z[np.nonzero(np.invert(rem))[0]]

        n = len(Z)
            
        if self._use_meta_features == True:
            X = compound_data(X, Z)
        
        mse = np.empty((k,2))
        sse = np.empty(2)
        
        task_folds = np.empty((n,k), dtype=np.object)
        for i in xrange(n):
            m = len(X[i])
            loo = RandKFold(m,k)
            j = 0
            for fold in loo:
                task_folds[i,j] = fold
                j += 1
            
            
        for i in xrange(k):
            Xtrain = np.empty(n, dtype='object')
            Xtest = np.empty(n, dtype='object')
            Ytrain = np.empty(n, dtype='object')
            Ytest = np.empty(n, dtype='object')
           
            for j in xrange(n):
                train, test = task_folds[j,i]
                
                Xtrain[j] = X[j][train]
                Xtest[j] = X[j][test]
                Ytrain[j] = Y[j][train]
                Ytest[j] = Y[j][test]
           
           
            y = flatten_data(Ytest)
            algo.hyperparams = hyperparams
            algo.fit(Xtrain, Ytrain, Z)
            
            
            result = algo.predict_by_task(Xtest, np.array(xrange(n)))
            yfit = flatten_data(result)
            mse[i,0] = metric.mspe(y, yfit)
            sse[0] += metric.tspe(y, yfit)
            
            result = algo.predict(Xtest, Z, 'complete')
            yfit = flatten_data(result)
            mse[i,1] = metric.mspe(y, yfit)
            sse[1] += metric.tspe(y, yfit)
            
            print 'fold_result({0})={1}'.format(i, mse[i])
        
        #compute the total error with its variance
        y = flatten_data(Y)
        ymean =  np.mean(y)
        sst = metric.tse(y, ymean)
        r2 = 1-sse/sst
            
        total_result = np.array([r2, np.mean(mse,0), np.std(mse,0)])
        self._total_result = total_result
        self._is_init = True
        
        #unseed
        np.random.seed(None)
        algo.hyperparams = hyperparams
        
        return total_result
    
class ZeroDataLearningExperiment(MultitaskRegressionExperiment):
    '''
    @todo: - parametrize evaluation criterions by functors
    '''
    
    __slots__ = ('_X',          #covariates of n tasks
                 '_Y',          #targets of n tasks
                 '_Z',          #meta description of each task
                 '_use_rbf',
                 '_use_meta_features',
                 '_ext_pred_methods'
                 )
    
    def __init__(self, X, Y, Z, use_rbf=False, use_meta_features=False, ext_pred_methods=False):
        '''
        '''
        MultitaskRegressionExperiment.__init__(self)
        
        self._X = asmtarray_object(X)
        self._Y = asmtarray_object(Y)
        self._Z = np.asanyarray(Z)
        
        if len(X) != len(Y) != len(Z):
            raise ValueError('Number of tasks must be equal in X,Y,Z.')
        
        self._use_rbf = use_rbf
        self._use_meta_features = use_meta_features
        self._ext_pred_methods = ext_pred_methods
        print 'ext'
        print ext_pred_methods
        

    def eval(self, algo):
        '''
        '''
        X = self._X
        Y = self._Y
        Z = self._Z
        n = len(X)
        
        hyperparams = algo.hyperparams
        
        if self._use_meta_features == True:
            X = compound_data(X, Z)
        
        weights = np.empty(n)
        
        if self._ext_pred_methods:
            task_result = np.empty((8,n))
        else:
            task_result = np.empty(n)
        
        i = 0
        loo = LeaveOneOut(n)
        for train, test in loo:
            print 'task={0}'.format(i+1) 
            
            if self._ext_pred_methods:
                algo.hyperparams = hyperparams
                algo.fit(X[train], Y[train], Z[train])
                y = Y[test][0]
                
                Ypred = algo.predict(X[test], Z[test], 'path', True)
                yfit = Ypred[0]
                task_result[0,i] = metric.mspe(y, yfit)
                
                Ypred = algo.predict(X[test], Z[test], 'path', False)
                yfit = Ypred[0]
                task_result[1,i] = metric.mspe(y, yfit)
                
                Ypred = algo.predict(X[test], Z[test], 'fullpath', True)
                yfit = Ypred[0]
                task_result[2,i] = metric.mspe(y, yfit)
                
                Ypred = algo.predict(X[test], Z[test], 'fullpath', False)
                yfit = Ypred[0]
                task_result[3,i] = metric.mspe(y, yfit)
                
                Ypred = algo.predict(X[test], Z[test], 'complete', True)
                yfit = Ypred[0]
                task_result[4,i] = metric.mspe(y, yfit)
                
                Ypred = algo.predict(X[test], Z[test], 'complete', False)
                yfit = Ypred[0]
                task_result[5,i] = metric.mspe(y, yfit)
                
                Ypred = algo.predict(X[test], Z[test], 'flat_maxcluster')
                yfit = Ypred[0]
                task_result[6,i] = metric.mspe(y, yfit)
                
                Ypred = algo.predict(X[test], Z[test], 'flat_complete')
                yfit = Ypred[0]
                task_result[7,i] = metric.mspe(y, yfit)
                
            else:
                algo.hyperparams = hyperparams
                if self._use_rbf:
                    rbf, Xrbf = make_rbf(X[train], kernel=GaussianBasis(0.1))
                    algo.fit(Xrbf, Y[train], Z[train])
                    
                    Ypred = algo.predict(rbf(flatten_data(X[test])), Z[test])
                    yfit = Ypred
                else:
                    algo.fit(X[train], Y[train], Z[train])
                    Ypred = algo.predict(X[test], Z[test])
                    yfit = Ypred[0]
                
                y = Y[test][0]
                
                
                #compute the task specific errors with variance
                task_result[i] = metric.mspe(y, yfit)
                
            k = len(y)
            weights[i] = k
            
            i += 1
            
            #print algo._model.alpha
            #print algo._model.beta
            #print algo._model.weights
            #print algo._model.intercept
        
        #compute the total error with its variance
        
        if self._ext_pred_methods:
            print task_result
            print weights
            mse = np.asarray([stats.mean(task_result[0], weights),stats.mean(task_result[1], weights),stats.mean(task_result[2], weights),stats.mean(task_result[3], weights),stats.mean(task_result[4], weights),stats.mean(task_result[5], weights),stats.mean(task_result[6], weights),stats.mean(task_result[7], weights)])
            stddev = np.asarray([stats.stddev(task_result[0], weights),stats.stddev(task_result[1], weights),stats.stddev(task_result[2], weights),stats.stddev(task_result[3], weights),stats.stddev(task_result[4], weights),stats.stddev(task_result[5], weights),stats.stddev(task_result[6], weights),stats.stddev(task_result[7], weights)])
        else:
            mse = stats.mean(task_result, weights)
            stddev = stats.stddev(task_result, weights)
            
        total_result = np.array([mse, stddev])
        
        
        algo.hyperparams = hyperparams
        
        self._task_result = task_result
        self._total_result = total_result
        self._is_init = True
        
        return total_result

class CVZeroDataLearningExperiment(MultitaskRegressionExperiment):
    '''
    @todo: - parametrize evaluation criterions by functors
    '''
    
    __slots__ = ('_X',          #covariates of n tasks
                 '_Y',          #targets of n tasks
                 '_Z',          #meta description of each task
                 '_k',          #number of folds
                 '_seed',        #random seed
                 '_use_rbf',
                 '_use_meta_features',
                 '_ext_pred_methods'
                 
                 )
    
    def __init__(self, X, Y, Z, k=10, seed=None, use_rbf=False, 
                 use_meta_features=False, ext_pred_methods=False):
        '''
        '''
        MultitaskRegressionExperiment.__init__(self)
        
        self._X = asmtarray_object(X)
        self._Y = asmtarray_object(Y)
        self._Z = np.asanyarray(Z)
        
        if len(X) != len(Y) != len(Z):
            raise ValueError('Number of tasks must be equal in X,Y,Z.')
        
        self._k = k
        self._seed = seed
        
        self._use_rbf = use_rbf
        self._use_meta_features = use_meta_features
        self._ext_pred_methods = ext_pred_methods
        

    def eval(self, algo):
        '''
        '''
        X = self._X
        Y = self._Y
        Z = self._Z
        k = self._k
        n = len(Z)
        
        hyperparams = algo.hyperparams
        
        if self._use_meta_features == True:
            X = compound_data(X, Z)
        
        weights = np.empty(k)
        
        if self._ext_pred_methods:
            task_result = np.empty((9,k))
            sse = np.zeros(9)
        else:
            sse = 0
            task_result = np.empty(k)
        
        
        np.random.seed(self._seed)
        i = 0
        loo = RandKFold(n, k)
        
        for train, test in loo:
            print 'task={0}'.format(i+1) 
        
            if i == 0:
                print 'folds'
                print train
            
            if self._ext_pred_methods:
                algo.hyperparams = hyperparams
                algo.fit(X[train], Y[train], Z[train])
                
                y = flatten_data(Y[test])
                
                Ypred = algo.predict(X[test], Z[test], 'path', True)
                yfit = flatten_data(Ypred)
                task_result[0,i] = metric.mspe(y, yfit)
                sse[0] += metric.tse(y, yfit)
                
                Ypred = algo.predict(X[test], Z[test], 'path', False)
                yfit = flatten_data(Ypred)
                task_result[1,i] = metric.mspe(y, yfit)
                sse[1] += metric.tse(y, yfit)
                
                Ypred = algo.predict(X[test], Z[test], 'fullpath', True)
                yfit = flatten_data(Ypred)
                task_result[2,i] = metric.mspe(y, yfit)
                sse[2] += metric.tse(y, yfit)
                
                Ypred = algo.predict(X[test], Z[test], 'fullpath', False)
                yfit = flatten_data(Ypred)
                task_result[3,i] = metric.mspe(y, yfit)
                sse[3] += metric.tse(y, yfit)
                
                Ypred = algo.predict(X[test], Z[test], 'complete', True)
                yfit = flatten_data(Ypred)
                task_result[4,i] = metric.mspe(y, yfit)
                sse[4] += metric.tse(y, yfit)
                
                Ypred = algo.predict(X[test], Z[test], 'complete', False)
                yfit = flatten_data(Ypred)
                task_result[5,i] = metric.mspe(y, yfit)
                sse[5] += metric.tse(y, yfit)
                
                Ypred = algo.predict(X[test], Z[test], 'flat_maxcluster')
                yfit = flatten_data(Ypred)
                task_result[6,i] = metric.mspe(y, yfit)
                sse[6] += metric.tse(y, yfit)
                
                Ypred = algo.predict(X[test], Z[test], 'flat_complete')
                yfit = flatten_data(Ypred)
                task_result[7,i] = metric.mspe(y, yfit)
                sse[7] += metric.tse(y, yfit)
                
                Ypred = algo.predict(X[test], Z[test], 'path1')
                yfit = flatten_data(Ypred)
                task_result[8,i] = metric.mspe(y, yfit)
                sse[8] += metric.tse(y, yfit)
                print 'task_result({0})={1}'.format(i, task_result[:,i])
            else:
                algo.hyperparams = hyperparams
                if self._use_rbf:
                    rbf, Xrbf = make_rbf(X[train], kernel=GaussianBasis(0.1))
                    algo.fit(Xrbf, Y[train], Z[train])
                    
                    Ypred = algo.predict(rbf(flatten_data(X[test])), Z[test], 'complete')
                    yfit = flatten_data(Ypred)
                else:
                    algo.fit(X[train], Y[train], Z[train])
                    Ypred = algo.predict(X[test], Z[test])
                    yfit = flatten_data(Ypred)
                
                
                y = flatten_data(Y[test])
                
                
                #compute the task specific errors with variance
                task_result[i] = metric.mspe(y, yfit)
                sse += metric.tse(y, yfit)
                print 'task_result({0})={1}'.format(i, task_result[i])
                
            m = len(y)
            weights[i] = m
            
            i += 1
            
            
            #print algo._model.alpha
            #print algo._model.beta
            #print algo._model.weights
            #print algo._model.intercept
        
        #compute the total error with its variance
        y = flatten_data(Y)
        ymean =  np.mean(y)
        sst = metric.tse(y, ymean)
        
        if self._ext_pred_methods:
            print task_result
            print weights
            mse = np.asarray([stats.mean(task_result[0], weights),stats.mean(task_result[1], weights),stats.mean(task_result[2], weights),stats.mean(task_result[3], weights),stats.mean(task_result[4], weights),stats.mean(task_result[5], weights),stats.mean(task_result[6], weights),stats.mean(task_result[7], weights),stats.mean(task_result[8], weights)])
            stddev = np.asarray([stats.stddev(task_result[0], weights),stats.stddev(task_result[1], weights),stats.stddev(task_result[2], weights),stats.stddev(task_result[3], weights),stats.stddev(task_result[4], weights),stats.stddev(task_result[5], weights),stats.stddev(task_result[6], weights),stats.stddev(task_result[7], weights),stats.stddev(task_result[8], weights)])
            r2 = 1-sse/sst
        else:
            mse = stats.mean(task_result, weights)
            stddev = stats.stddev(task_result, weights)
            r2 = 1-sse/sst
            
        total_result = np.array([r2, mse, stddev])
        
        self._task_result = task_result
        self._total_result = total_result
        self._is_init = True
        
        #unseed
        np.random.seed(None)
        
        algo.hyperparams = hyperparams
        
        return total_result


def _norm_metadata(Z, index):
    '''
    -remove this hack
    '''
    n = len(Z)
    data = np.empty(n)
    for i in xrange(n):
        data[i] = Z[i][index]
    
    mean = np.mean(data)
    dev  = np.std(data)
    for i in xrange(n):
        Z[i][index] = (Z[i][index]-mean)/dev 
        
def main():
    '''
    @todo: - fieldnames must be in lowercase
    '''
    outfile = '/home/marcel/datasets/multilevel/allen/eval/transfer_eqjp_norm_pga_reg.txt'
    #outfile = '/home/marcel/datasets/multilevel/ilea/eval/exam_zt_result_bhcgrad.txt'
    
    #filename = '/home/marcel/datasets/multilevel/ilea/exam_london.csv'
    #task_key = 'school'
    #task_fields = ['school_mixed', 'school_boys', 'school_girls', 
    #               'intake_score', 'school_vrb']
    #task_fields = ['intake_score', 'school_vrb']
    #target_field = 'exam_score'
    
    #filename = '/home/marcel/datasets/multilevel/ilea/school_norm_vrb.csv'
    #task_key = 'school'
    #task_fields = ['fsm', 'school_vr1', 'school_mixed', 'school_male', 'school_female', 
    #               'sdenom_maintain', 'sdenom_coe', 'sdenom_rc']
    #target_field = 'exam_score'

    filename = '/home/mhermkes/datasets/multilevel/allen/reg_norm_pga.csv'
    task_key = 'region_id'
    task_fields = ['region_id']
    #task_fields = ['pref_vs30', 'z1', 'z1_5', 'z2_5'] 
    target_field = 'pga'
    
    #filename = '/home/marcel/datasets/multilevel/allen/reg_mmi_norm_pga.csv'
    #task_key = 'region_id'
    #task_fields = ['region_id']
    #task_fields = ['pref_vs30', 'z1', 'z1_5', 'z2_5'] 
    #target_field = 'pga'
    
    #filename = '/home/marcel/datasets/multilevel/allen/stn20_jp_norm_pga.csv'
    #task_key = 'station_id'
    #task_fields = ['vs30']
    #task_key = 'eq_id'
    #task_fields = ['mag', 'depth', 'mechanism', 'ztor']
    #task_fields = ['mag', 'mechanims', 'ztor'] 
    #target_field = 'pga'
    
    
    X,Y,Z = load_data(filename, task_key, task_fields, target_field)
    #X,Z,Y = del_rare_tasks(10, X, Z, Y)
    #_norm_metadata(Z, 1)
    #X = X[50:70]
    #Y = Y[50:70]
    #Z = Z[50:70]
 
    print 'ntasks={0}'.format(len(X))
 
    XZ = compound_data(X,Z)
    rbf, Xrbf = make_rbf(X, kernel=GaussianBasis(0.1), ratio=0.02)
    print 'shape'
    print Xrbf[0].shape
    
    #np.arange(0.1, 5.1, 0.1)
    #reg_model = OptRBFRegression(EMBayesRegression(), GaussianBasis,  
    #                             np.arange(0.1, 5.1, 0.1), 0.05)
    #algo = MultiISpaceLearner(EBChenRegression(), use_meta_features=True)
    #algo = BHCRegression(200,0.56, 2.396)
    algo = ItGradBHCRegression(100, use_meta_features=True)
    #algo = MetaSpaceClusterLearner(RidgeRegression(), max_k=10, use_meta_features=True)
    
    
    #algo = EBChenRegression()
    
    #experiment = ZeroDataLearningExperiment(X, Y, Z, False)
    #experiment = TaskLearningExperiment(X, Y, Z, nfolds=10, use_meta_features=True, use_background_data=True)
    experiment = TransferLearningExperiment(X, Y, Z, nfolds=10)
    result = experiment.eval(algo)
    print array_str(result, precision=16)
    
    #savetxt(outfile, experiment.task_result)
    
if __name__ == '__main__':
    main()
