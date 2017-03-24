'''
Created on Jul 13, 2011

@author: marcel
'''

import numpy as np
import upgeo.util.metric as metric
import upgeo.util.stats as stats

from upgeo.ml.multitask.base import asmtarray_object, compound_data,\
    flatten_data
from upgeo.eval.cv import RandKFold, BagFold
from upgeo.data import dataset2array
from upgeo.eval.multitask.trial import MultitaskRegressionExperiment
from upgeo.eval.multitask.base import load_multidataset, make_rbf
from upgeo.ml.regression.function import GaussianBasis
from upgeo.ml.multitask.regression.dpm import ItGradBHCRegression
from numpy.core.numeric import array_str
from upgeo.ml.regression.bayes import EBChenRegression, EMBayesRegression
from numpy.lib.npyio import savetxt
from upgeo.ml.regression.linear import RidgeRegression

class BagTaskLearningExperiment(MultitaskRegressionExperiment):
    '''
    @todo: - parametrize the cross validation method
    '''
    __slots__ = ('_X',
                 '_Y',
                 '_Z',
                 '_bags',
                 '_use_background_data',
                 '_use_meta_features'
                 )
    
    def __init__(self, X, Y, Z, bags ,use_meta_features=False, use_background_data=False):
        
        MultitaskRegressionExperiment.__init__(self)
        
        self._X = asmtarray_object(X)
        self._Y = asmtarray_object(Y)
        self._Z = asmtarray_object(Z)
        self._bags = asmtarray_object(bags)
        
        if len(X) != len(Y) != len(Z):
            raise ValueError('Number of tasks must be equal in X,Y,Z.')
        
        self._use_meta_features = use_meta_features
        self._use_background_data = use_background_data
        
    def eval(self, algo):
        '''
        @todo: - check that the algo isnt a type of multitask learner
        '''
        X = self._X
        Y = self._Y
        Z = self._Z
        bags = self._bags
        
        hyperparams = algo.hyperparams
        
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
            
            bag_i = bags[i]
            
            m = len(Xi)
            if self._use_background_data == True:
                indices = np.ones(n, dtype=bool)
                indices[i] = False
                Xback = flatten_data(X[indices])
                Yback = flatten_data(Y[indices])
            
            
            loo = BagFold(bag_i)
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
            print 'task{0}={1}'.format(i,task_result[i])
            
            Ypred[i] = Yhat
            weights[i] = m
        
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

class BagTransferLearningExperiment(MultitaskRegressionExperiment):
    
    __slots__ = ('_X',
                 '_Y',
                 '_Z',
                 '_bags'
                )
    
    def __init__(self, X, Y, Z, bags):
        MultitaskRegressionExperiment.__init__(self)
        
        self._X = asmtarray_object(X)
        self._Y = asmtarray_object(Y)
        self._Z = asmtarray_object(Z)
        self._bags = asmtarray_object(bags)

        if len(X) != len(Y) != len(Z):
            raise ValueError('Number of tasks must be equal in X,Y,Z.')
        
        self._is_init = False

    def eval(self, algo):
        '''
        @todo: - check that the algo isnt a type of multitask learner
        '''
        X = self._X
        Y = self._Y
        Z = self._Z
        bags = self._bags
        
        hyperparams = algo.hyperparams
        
        n = len(X)
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
            
            bag_i = bags[i]
            m = len(Xi)
            
            loo = BagFold(bag_i)
            #loo = LeaveOneOut(m)
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
            print 'task{0}={1}'.format(i,task_result[i])
            
            Xtrain[i] = X[i]
            Ytrain[i] = Y[i]
            
            Ypred[i] = Yhat
            weights[i] = m
            
        
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

def _preprocess_dataset(dataset):
    '''
    '''
    X,Y,Z = dataset2array(dataset)
    n = dataset.ntasks
    bags = [0]*n
     
    for i in xrange(n):
        Xi = X[i]
        bags[i] = Xi[:,0]
        X[i] = Xi[:,1:Xi.shape[1]]
    return X,Y,Z,bags

if __name__ == '__main__':
    outfile = '/home/marcel/datasets/multilevel/nga/eval/test.txt'
    
    infile = '/home/marcel/datasets/multilevel/nga/reg_eq_norm_pga.csv'
    task_key = 'region_id'
    task_fields = ['region_id']
    #task_fields = ['pref_vs30', 'z1', 'z1_5', 'z2_5'] 
    target_field = 'pga'
    
    dataset = load_multidataset(infile, task_key, task_fields, target_field)
    print dataset.data_fields
    X, Y, Z, bags = _preprocess_dataset(dataset)
    
    print Z
    
    rbf, Xrbf = make_rbf(X, kernel=GaussianBasis(0.05), ratio=0.05)
    #print 'hallo'
    print Xrbf[0].shape
    
    #algo = ItGradBHCRegression(100, use_meta_features=False)
    #algo = EBChenRegression()
    #algo = RidgeRegression()
    algo = EMBayesRegression()
    
    experiment = BagTaskLearningExperiment(Xrbf, Y, Z, bags, use_meta_features=False, use_background_data=False)
    #experiment = BagTransferLearningExperiment(X, Y, Z, bags)
    result = experiment.eval(algo)
    print 'fuck'
    print array_str(result, precision=16)
    
    
    savetxt(outfile, experiment.task_result)