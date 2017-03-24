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
    SparseBHCGPRegregressionExpert, BHCRobustRegressionExpert
from upgeo.ml.multitask.regression.bhc import BHCLinearRegression,\
    SparseBHCGPRegression, BHCRobustRegression

def eval_linreg_zerotask(X, Y, Z, use_meta_features, runs=5, seeds=None):
    
    r2 = np.zeros(runs)
    mse = np.zeros(runs)
    stdmse = np.zeros(runs) 
    
    for i in xrange(runs):
        seed = None if seeds == None else seeds[i]
        lin_reg = EMBayesRegression()
        algo = MultiISpaceLearner(lin_reg)
        experiment = CVZeroDataLearningExperiment(X, Y, Z, 10, seed, use_meta_features=use_meta_features)
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
        
        if use_background:
            selector = KMeansSelector(30)
            algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector)
        else:
            print 'fuck'
            algo = GPRegression(kernel,infer_method=ExactInference)
            
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

def eval_gp_zerotask(X, Y, Z, kernel, use_meta_features, runs=10, seeds=None):
    #copy kernel parameters, because gp working on a reference
    init_params = np.copy(kernel.params)
    
    r2 = np.empty(runs)
    mse = np.empty(runs)
    stdmse = np.empty(runs)

    i = 0
    while i < runs:
        seed = None if seeds == None else seeds[i]
        
        kernel.params = init_params
        selector = KMeansSelector(15)
        gp = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector)
        algo = MultiISpaceLearner(gp)
        
        try:
            experiment = CVZeroDataLearningExperiment(X, Y, Z, 10, seed, use_rbf=False, use_meta_features=use_meta_features)
            result = experiment.eval(algo)
            print 'result:'
            print array_str(result, precision=16)
            r2[i] = result[0]
            mse[i] = result[1]
            stdmse[i] = result[2]
        except LinAlgError:
            print "Unexpected error:", sys.exc_info()[0]
            i -= 1
            
        i += 1

    total_result = np.asarray([np.mean(r2), np.std(r2), np.mean(mse), np.std(mse), np.mean(stdmse), np.std(stdmse)])
        
    print 'Run Results'
    print 'R2={0}'.format(array_str(r2, precision=16))
    print 'mse={0}'.format(array_str(mse, precision=16))
    print 'dev={0}'.format(array_str(stdmse, precision=16))

        
    print 'total_result:'
    print array_str(total_result, precision=16)

def eval_linreg_bhce_zerotask(X, Y, Z, use_task_features, use_meta_features, runs=5, seeds=None):
    
    r2 = np.empty((runs,9))
    mse = np.empty((runs,9))
    stdmse = np.empty((runs,9)) 
    
    d = X[0].shape[1]+2
    
    
    for i in xrange(runs):
        seed = None if seeds == None else seeds[i]
    
        #algo = BHCLinearRegressionExpert(np.log(10.5), np.log(np.array([1,1])), use_task_features=use_task_features)
        hyperparams = RobustBayesRegression.wrap(np.log(0.001), np.log(0.001), np.eye(d+1).ravel())
        algo = BHCRobustRegressionExpert(np.log(10.5), hyperparams, use_task_features=use_task_features)
        experiment = CVZeroDataLearningExperiment(X, Y, Z, 10, seed, False, use_meta_features, True)
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
    
def eval_linreg_bhc_transfer(X, Y, Z, use_meta_features, runs=10, seeds=None):
    
    r2 = np.empty((runs,2))
    mse = np.empty((runs,2))
    stdmse = np.empty((runs,2)) 
    
    d = X[0].shape[1]
    
    for i in xrange(runs):
        seed = None if seeds == None else seeds[i]
    
        #algo = BHCLinearRegression(np.log(10.5), np.log(np.array([1,1])))
        hyperparams = RobustBayesRegression.wrap(np.log(0.001), np.log(0.001), np.eye(d+1).ravel())
        algo = BHCRobustRegression(np.log(10.5), hyperparams) 
   
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
    
def eval_linreg_bhc_zerotask(X, Y, Z, use_meta_features, runs=10, seeds=None):
    
    r2 = np.empty((runs,2))
    mse = np.empty((runs,2))
    stdmse = np.empty((runs,2)) 
    
    d = X[0].shape[1]
    
    for i in xrange(runs):
        seed = None if seeds == None else seeds[i]
    
        #algo = BHCLinearRegression(np.log(10.5), np.log(np.array([1,1])))
        hyperparams = RobustBayesRegression.wrap(np.log(0.001), np.log(0.001), np.eye(d+1).ravel())
        algo = BHCRobustRegression(np.log(10.5), hyperparams) 
   
        experiment = CVZeroDataLearningExperiment(X, Y, Z, 10, seed, False, use_meta_features)
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
        algo = SparseBHCGPRegression(np.log(10.5), kernel, None, 15, opt_kernel_global=True)
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
    

def eval_gp_bhce_zerotask(X, Y, Z, kernel, use_meta_features, max_runs=10, seeds=None):
    #copy kernel parameters, because gp working on a reference
    init_params = np.copy(kernel.params)
    
    r2 = np.empty((max_runs,9))
    mse = np.empty((max_runs,9))
    stdmse = np.empty((max_runs,9))

    i = 0
    while i < max_runs:
        seed = None if seeds == None else seeds[i]
        kernel.params = init_params
        #selector = RandomSubsetSelector(10)
        #Xt = flatten_data(X)
        #perm = np.random.permutation(len(Xt))
        #selector = FixedSelector(Xt[perm[0:10]])
        algo = SparseBHCGPRegregressionExpert(np.log(10.5), kernel, None, 15)
        
        try:
            experiment = CVZeroDataLearningExperiment(X, Y, Z, 10, seed, False, use_meta_features, True)
            result = experiment.eval(algo)
            print 'result:'
            print array_str(result, precision=16)
            
            r2[i] = result[0]
            mse[i] = result[1]
            stdmse[i] = result[2]
        except LinAlgError:
            print "Unexpected error:", sys.exc_info()[0]
            i -= 1
            
        i += 1
    
    total_result = np.asarray([np.mean(r2,0), np.std(r2,0), np.mean(mse,0), np.std(mse,0), np.mean(stdmse,0), np.std(stdmse,0)])
        
    print 'Run Results'
    print 'R2={0}'.format(array_str(r2, precision=16))
    print 'mse={0}'.format(array_str(mse, precision=16))
    print 'dev={0}'.format(array_str(stdmse, precision=16))

        
    print 'total_result:'
    print array_str(total_result, precision=16)

def eval_gp_bhc_zerotask(X, Y, Z, kernel, use_meta_features, max_runs=10, seeds=None):
    #copy kernel parameters, because gp working on a reference
    init_params = np.copy(kernel.params)
    
    r2 = np.empty((max_runs,2))
    mse = np.empty((max_runs,2))
    stdmse = np.empty((max_runs,2))

    i = 0
    while i < max_runs:
        seed = None if seeds == None else seeds[i]
        kernel.params = init_params
        #selector = RandomSubsetSelector(10)
        #Xt = flatten_data(X)
        #perm = np.random.permutation(len(Xt))
        #selector = FixedSelector(Xt[perm[0:10]])
        algo = SparseBHCGPRegression(np.log(10.5), kernel, None, 15, opt_kernel_global=True)
        
        try:
            experiment = CVZeroDataLearningExperiment(X, Y, Z, 10, seed, False, use_meta_features, True)
            result = experiment.eval(algo)
            print 'result:'
            print array_str(result, precision=16)
            
            r2[i] = result[0]
            mse[i] = result[1]
            stdmse[i] = result[2]
        except LinAlgError:
            print "Unexpected error:", sys.exc_info()[0]
            i -= 1
            
        i += 1
    
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
    filename = '/home/marcel/datasets/multilevel/ilea/zt_school_norm_split1.csv'
    task_key = 'school'
    #task_fields = ['fsm', 'school_vr1', 'school_mixed', 'school_male', 'school_female', 
    #               'sdenom_maintain', 'sdenom_coe', 'sdenom_rc']
   
    task_fields = ['fsm', 'school_vr1']
    #task_fields = ['school_mixed', 'school_boys', 'school_girls', 
    #              'intake_score', 'school_vrb']
    #task_fields = ['school_type','intake_score', 'school_vrb']
    #task_fields = ['intake_score']
    target_field = 'exam_score'
    
    
    #school dataset
    #filename = '/home/marcel/datasets/multilevel/ilea/school_norm_nvrb.csv'
    #task_key = 'school'
    #task_fields = ['fsm', 'school_vr1', 'school_mixed', 'school_male', 'school_female', 
    #               'sdenom_maintain', 'sdenom_coe', 'sdenom_rc']
    #task_fields = ['fsm', 'school_vr1', 'school_type', 
    #               'sdenom_maintain', 'sdenom_coe', 'sdenom_rc']
#    task_fields = ['fsm']
    #target_field = 'exam_score'

    seeds = None
    seeds = np.array([948234775757475, 382938984745774755,
                      923848274777222, 211324948828384842,
                      424384328855552])
    #seeds = np.array([2289758437583, 312872384375,
    #                  2878748748294, 958328375868,
    #                  9399857246245])


    
    X,Y,Z = load_data(filename, task_key, task_fields, target_field)
    #gp stuff
    kernel = ARDSEKernel(np.log(1)*np.ones(18), np.log(0.5)) + NoiseKernel(np.log(1))
    #kernel = MaternKernel(3, np.log(1), np.log(0.5)) + NoiseKernel(np.log(1))
    #kernel = MaternKernel(5, np.log(1), np.log(0.5)) + NoiseKernel(np.log(1))
    #kernel = SEKernel(np.log(1), np.log(0.5)) + NoiseKernel(np.log(1))
    #kernel = SEKernel(np.log(1), np.log(0.5)) + BiasedLinearKernel(np.log(1)) + NoiseKernel(np.log(1)) 
    #kernel = ARDSEKernel(np.log(1)*np.ones(3), np.log(0.5)) + BiasedLinearKernel(np.log(1)) + NoiseKernel(np.log(1))
    #kernel = SEKernel(np.log(1), np.log(0.5)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
    #kernel = ARDSEKernel(np.log(1)*np.ones(3), np.log(0.5)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
    #kernel = PolynomialKernel(2, np.log(2), np.log(4)) + NoiseKernel(np.log(1)) 
    
    #np.random.seed(1245)
    
    #np.random.seed(293843475847)
    
    #exp = TaskLearningExperiment(X, Y, Z, nfolds=10, use_meta_features=False, use_background_data=False)
    #print exp.eval(EMBayesRegression())
    
    #eval_gp_transfer(X, Y, Z, kernel, False, False, 5, seeds)
    #eval_linreg_bhc_transfer(X, Y, Z, False, 5, seeds)
    #eval_gp_bhc_transfer(X, Y, Z, kernel, False, 5, seeds)
    #eval_linreg_zerotask(X, Y, Z, True, 5, seeds)
    #eval_gp_zerotask(X, Y, Z, kernel, False, 5, seeds)
    eval_linreg_bhce_zerotask(X, Y, Z, False, True, 5, seeds)
    #eval_linreg_bhc_zerotask(X, Y, Z, False, 5, seeds)
    
    
    #np.random.seed(1245)
    #eval_gp_bhce_zerotask(X, Y, Z, kernel, False, 5)