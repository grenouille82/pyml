'''
Created on Oct 6, 2011

@author: marcel
'''
'''
Created on May 13, 2011

@author: marcel
'''
import sys

import numpy as np

#from scikits.learn.svm import SVR
from upgeo.eval.multitask.base import load_data
from upgeo.ml.regression.bayes import EBChenRegression, EMBayesRegression,\
    RobustBayesRegression
from upgeo.ml.multitask.base import MultiISpaceLearner, flatten_data
from upgeo.eval.multitask.trial import ZeroDataLearningExperiment,\
    CVZeroDataLearningExperiment, CVTaskLearningExperiment,\
    TaskLearningExperiment, CVTransferLearningExperiment
from numpy.core.numeric import array_str
from upgeo.ml.regression.linear import RidgeRegression
from upgeo.ml.regression.np.gp import SparseGPRegression, GPRegression
from upgeo.ml.regression.np.kernel import SEKernel, NoiseKernel, ARDSEKernel,\
    MaternKernel, BiasedLinearKernel, SqConstantKernel, LinearKernel,\
    PolynomialKernel
from upgeo.ml.regression.np.selector import RandomSubsetSelector, FixedSelector,\
    KMeansSelector
from upgeo.ml.regression.np.infer import FITCExactInference, ExactInference
from numpy.linalg.linalg import LinAlgError
from upgeo.ml.multitask.regression.bhce import BHCLinearRegressionExpert,\
    SparseBHCGPRegregressionExpert
from upgeo.ml.multitask.regression.bhc import BHCLinearRegression,\
    SparseBHCGPRegression, BHCRobustRegression, RobustSparseBHCGPRegression

    
def eval_linreg_transfer(X, Y, Z, use_meta_features, use_background, runs=10, seeds=None):
    
    r2 = np.zeros(runs)
    mse = np.zeros(runs)
    stdmse = np.zeros(runs) 
    
    for i in xrange(runs):
        seed = None if seeds == None else seeds[i]
        algo = EMBayesRegression()
        experiment = CVTaskLearningExperiment(X, Y, Z, 10, seed, use_meta_features, use_background)
        result = experiment.eval(algo)
        r2[i] = result[0]
        mse[i] = result[1]
        stdmse[i] = result[2]
        
    total_result = np.asarray([np.mean(r2), np.std(r2), np.mean(mse), np.std(mse), np.mean(stdmse), np.std(stdmse)])
        
    print 'Run Results'
    print 'R2={0}'.format(array_str(r2, precision=16))
    print 'mse={0}'.format(array_str(mse, precision=16))
    print 'dev={0}'.format(array_str(stdmse, precision=16))

        
    print 'total_result:'
    print array_str(total_result, precision=16)
    
def eval_gp_transfer(X, Y, Z, kernel, use_meta_features, use_background, runs=10, seeds=None):
    init_params = np.copy(kernel.params)
    
    r2 = np.zeros(runs)
    mse = np.zeros(runs)
    stdmse = np.zeros(runs) 
    
    for i in xrange(runs):
        seed = None if seeds == None else seeds[i]
        
        kernel.params = init_params
        
        selector = KMeansSelector(100)
        algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector)
            
        experiment = CVTaskLearningExperiment(X, Y, Z, 10, seed, use_meta_features, use_background)
        result = experiment.eval(algo)
        r2[i] = result[0]
        mse[i] = result[1]
        stdmse[i] = result[2]
        
    total_result = np.asarray([np.mean(r2), np.std(r2), np.mean(mse), np.std(mse), np.mean(stdmse), np.std(stdmse)])
        
    print 'Run Results'
    print 'R2={0}'.format(array_str(r2, precision=16))
    print 'mse={0}'.format(array_str(mse, precision=16))
    print 'dev={0}'.format(array_str(stdmse, precision=16))

        
    print 'total_result:'
    print array_str(total_result, precision=16)

    
def eval_linreg_bhc_transfer(X, Y, Z, use_meta_features, runs=10, seeds=None):
    
    r2 = np.empty((runs,2))
    mse = np.empty((runs,2))
    stdmse = np.empty((runs,2)) 
    
    d = X[0].shape[1]
    
    for i in xrange(runs):
        seed = None if seeds == None else seeds[i]
    
        hyperparams = RobustBayesRegression.wrap(np.log(0.1), np.log(4), np.eye(d+1).ravel())
        algo = BHCRobustRegression(np.log(10.5), hyperparams) 

        #algo = BHCLinearRegression(np.log(10.5), np.log(np.array([1,1])))
        experiment = CVTransferLearningExperiment(X, Y, Z, 10, seed, use_meta_features)
        result = experiment.eval(algo)
        
        r2[i] = result[0]
        mse[i] = result[1]
        stdmse[i] = result[2]

        
    total_result = np.asarray([np.mean(r2,0), np.std(r2,0), np.mean(mse,0), np.std(mse,0), np.mean(stdmse,0), np.std(stdmse,0)])
        
    print 'Run Results'
    print 'R2={0}'.format(array_str(r2, precision=16))
    print 'mse={0}'.format(array_str(mse, precision=16))
    print 'dev={0}'.format(array_str(stdmse, precision=16))

        
    print 'total_result:'
    print array_str(total_result, precision=16)


def eval_gp_bhc_transfer(X, Y, Z, kernel, use_meta_features, runs=10, seeds=None):
    #copy kernel parameters, because gp working on a reference
    init_params = np.copy(kernel.params)
    
    r2 = np.empty((runs,2))
    mse = np.empty((runs,2))
    stdmse = np.empty((runs,2)) 
    
    for i in xrange(runs):
        seed = None if seeds == None else seeds[i]
        kernel.params = init_params
        algo = SparseBHCGPRegression(np.log(1.5), kernel, None, 100, True, True)
        experiment = CVTransferLearningExperiment(X, Y, Z, 10, seed, use_meta_features)
        result = experiment.eval(algo)
        
        r2[i] = result[0]
        mse[i] = result[1]
        stdmse[i] = result[2]

        
    total_result = np.asarray([np.mean(r2,0), np.std(r2,0), np.mean(mse,0), np.std(mse,0), np.mean(stdmse,0), np.std(stdmse,0)])
        
    print 'Run Results'
    print 'R2={0}'.format(array_str(r2, precision=16))
    print 'mse={0}'.format(array_str(mse, precision=16))
    print 'dev={0}'.format(array_str(stdmse, precision=16))

        
    print 'total_result:'
    print array_str(total_result, precision=16)
    



if __name__ == '__main__':
    #np.random.seed(304)
    #x = np.random.rand(10,1)
    #y = x[:,0]*4.3+2.4#+np.random.randn(100)*0.5
    #reg_model = EMBayesRegression()
    #reg_model.fit(x,y)
    #print reg_model.wgeights
    #print reg_model.intercept
    #print reg_model.log_evidence
    #print np.exp(reg_model.log_evidence)
    
    #exam london dataset
    filename = '/home/marcel/datasets/multilevel/nga/multitask_norm_pga_gp1.csv'
    task_key = 'region_id'
    task_fields = ['region_id']
    target_field = 'pga'
    
    X,Y,Z = load_data(filename, task_key, task_fields, target_field)
    
    seeds = None
    seeds = np.array([948234775757475])
    #seeds = np.array([2289758437583, 312872384375,
    #                  2878748748294, 958328375868,
    #                  9399857246245])


    #gp stuff
    kernel = ARDSEKernel(np.log(1)*np.ones(12), np.log(1)) + NoiseKernel(np.log(0.1))
    #kernel = MaternKernel(3, np.log(1), np.log(0.5)) + NoiseKernel(np.log(1))
    #kernel = MaternKernel(5, np.log(1), np.log(0.5)) + NoiseKernel(np.log(1))
    #kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.1))
    #kernel = SEKernel(np.log(1), np.log(0.5)) + BiasedLinearKernel(np.log(1)) + NoiseKernel(np.log(1)) 
    #kernel = ARDSEKernel(np.log(1)*np.ones(12), np.log(0.5)) + BiasedLinearKernel(np.log(1)) + NoiseKernel(np.log(1))
    #kernel = SEKernel(np.log(1), np.log(0.5)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
    #kernel = ARDSEKernel(np.log(1)*np.ones(18), np.log(0.5)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
    #kernel = PolynomialKernel(2, np.log(2), np.log(4)) + NoiseKernel(np.log(1)) 
    
    #np.random.seed(1245)
    
    #np.random.seed(293843475847)
    
    #exp = TaskLearningExperiment(X, Y, Z, nfolds=10, use_meta_features=False, use_background_data=False)
    #print exp.eval(EMBayesRegression())
    
    #eval_linreg_transfer(X, Y, Z, False, False, 1, seeds)
    eval_gp_transfer(X, Y, Z, kernel, False, True, 1, seeds)
    #eval_linreg_bhc_transfer(X, Y, Z, False, 1, seeds)
    #eval_gp_bhc_transfer(X, Y, Z, kernel, False, 1, seeds)
    
