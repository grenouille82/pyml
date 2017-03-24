'''
Created on Oct 6, 2011

@author: marcel
'''
import numpy as np

from upgeo.eval.multitask.base import load_multidataset
from upgeo.eval.multitask.eqtrail import BagTaskLearningExperiment,\
    BagTransferLearningExperiment
from upgeo.data import dataset2array
from upgeo.ml.regression.bayes import EMBayesRegression, RobustBayesRegression
from numpy.core.numeric import array_str
from upgeo.eval.multitask.trial import TaskLearningExperiment
from upgeo.ml.regression.np.selector import KMeansSelector
from upgeo.ml.regression.np.gp import SparseGPRegression
from upgeo.ml.regression.np.infer import FITCExactInference
from upgeo.ml.multitask.regression.bhc import BHCRobustRegression,\
    SparseBHCGPRegression
from upgeo.ml.regression.np.kernel import ARDSEKernel, NoiseKernel, SEKernel,\
    SqConstantKernel, LinearKernel

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

def eval_linreg_eq(X, Y, Z, bags, use_background, seeds=None):
    
    algo = EMBayesRegression()
    experiment = BagTaskLearningExperiment(X, Y, Z, bags, False, use_background)
    result = experiment.eval(algo)
    
    print 'Region Result'
    print array_str(experiment.task_result, precision=16)
    print 'Total Result'
    print array_str(result, precision=16)

def eval_linreg(X, Y, Z, use_background, seeds=None):
    
    algo = EMBayesRegression()
    experiment = TaskLearningExperiment(X, Y, Z, 10, False, use_background, seeds)
    result = experiment.eval(algo)
    
    print 'Region Result'
    print array_str(experiment.task_result, precision=16)
    print 'Total Result'
    print array_str(result, precision=16)
    
def eval_gp_eq(X, Y, Z, bags, kernel, use_background, seeds=None):
    
    selector = KMeansSelector(50)
    algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector)
    
    experiment = BagTaskLearningExperiment(X, Y, Z, bags, False, use_background)
    result = experiment.eval(algo)
    
    print 'Region Result'
    print array_str(experiment.task_result, precision=16)
    print 'Total Result'
    print array_str(result, precision=16)

def eval_gp(X, Y, Z, kernel, use_background, seeds=None):
 
    selector = KMeansSelector(15)
    algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector)
 
    experiment = TaskLearningExperiment(X, Y, Z, 10, False, use_background, seeds)
    result = experiment.eval(algo)
    
    print 'Region Result'
    print array_str(experiment.task_result, precision=16)
    print 'Total Result'
    print array_str(result, precision=16)

def eval_bhcreg_eq(X, Y, Z, bags):

    d = X[0].shape[1]
    hyperparams = RobustBayesRegression.wrap(np.log(0.1), np.log(4), np.eye(d+1).ravel())
    algo = BHCRobustRegression(np.log(10.5), hyperparams)
        
    experiment = BagTransferLearningExperiment(X, Y, Z, bags)
    result = experiment.eval(algo)
    
    print 'Region Result'
    print array_str(experiment.task_result, precision=16)
    print 'Total Result'
    print array_str(result, precision=16)

def eval_bhcreg(X, Y, Z, use_background, seeds=None):
    
    d = X[0].shape[1]
    hyperparams = RobustBayesRegression.wrap(np.log(0.1), np.log(4), np.eye(d+1).ravel())
    algo = BHCRobustRegression(np.log(10.5), hyperparams)
        
    experiment = BagTransferLearningExperiment(X, Y, Z, bags, seeds)
    result = experiment.eval(algo)
    
    print 'Region Result'
    print array_str(experiment.task_result, precision=16)
    print 'Total Result'
    print array_str(result, precision=16)
    
def eval_bhcsgp_eq(X, Y, Z, bags, kernel):

    algo = SparseBHCGPRegression(np.log(10.5), kernel, None, 50, True, True)      
    experiment = BagTransferLearningExperiment(X, Y, Z, bags)
    result = experiment.eval(algo)
    
    print 'Region Result'
    print array_str(experiment.task_result, precision=16)
    print 'Total Result'
    print array_str(result, precision=16)

def eval_bhcsgp(X, Y, Z, kernel, use_background, seeds=None):
    
    algo = SparseBHCGPRegression(np.log(10.5), kernel, None, 50, True, True)
    experiment = BagTransferLearningExperiment(X, Y, Z, bags, seeds)
    result = experiment.eval(algo)
    
    print 'Region Result'
    print array_str(experiment.task_result, precision=16)
    print 'Total Result'
    print array_str(result, precision=16)





if __name__ == '__main__':
    outfile = '/home/marcel/datasets/multilevel/nga/eval/test.txt'
    
    infile = '/home/marcel/datasets/multilevel/nga/eq_multitask_norm_pga_gp1.csv'
    task_key = 'region_id'
    task_fields = ['region_id'] 
    target_field = 'pga'
    
    seeds = [2323435346, 43285345345, 84572364744, 1294857345, 5663453455]
    
    np.random.seed(3498795868756)
    
    dataset = load_multidataset(infile, task_key, task_fields, target_field)
    print dataset.data_fields
    X, Y, Z, bags = _preprocess_dataset(dataset)
    
    #gp stuff
    #kernel = ARDSEKernel(np.log(1)*np.ones(12), np.log(1)) + NoiseKernel(np.log(0.1))
    #kernel = MaternKernel(3, np.log(1), np.log(0.5)) + NoiseKernel(np.log(1))
    #kernel = MaternKernel(5, np.log(1), np.log(0.5)) + NoiseKernel(np.log(1))
    kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.1))
    #kernel = SEKernel(np.log(1), np.log(0.5)) + BiasedLinearKernel(np.log(1)) + NoiseKernel(np.log(1)) 
    #kernel = ARDSEKernel(np.log(1)*np.ones(12), np.log(0.5)) + BiasedLinearKernel(np.log(1)) + NoiseKernel(np.log(1))
    #kernel = SEKernel(np.log(1), np.log(0.5)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
    #kernel = ARDSEKernel(np.log(1)*np.ones(12), np.log(0.5)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
    #kernel = PolynomialKernel(2, np.log(2), np.log(4)) + NoiseKernel(np.log(1)) 
 
    
    #eval_linreg_eq(X, Y, Z, bags, False)
    eval_gp_eq(X, Y, Z, bags, kernel, True)
    #eval_bhcreg_eq(X,Y,Z,bags)
    #eval_bhcsgp_eq(X,Y,Z,bags,kernel)
    
    