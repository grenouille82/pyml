'''
Created on Sep 26, 2011

@author: marcel
'''

import numpy as np
import scipy.io as sio  

from upgeo.eval.multitask.base import load_data
from upgeo.ml.multitask.base import compound_data, flatten_data
from upgeo.util import metric
from upgeo.ml.regression.bayes import EMBayesRegression, RobustBayesRegression
from upgeo.ml.regression.np.gp import SparseGPRegression, GPRegression
from upgeo.ml.regression.np.infer import FITCExactInference, ExactInference
from upgeo.ml.regression.np.selector import KMeansSelector
from numpy.core.numeric import array_str
from upgeo.ml.regression.np.kernel import SEKernel, NoiseKernel,\
    SqConstantKernel, LinearKernel, ARDSEKernel
from upgeo.ml.multitask.regression.bhc import BHCLinearRegression,\
    BHCRobustRegression, BHCGPRegression, SparseBHCGPRegression,\
    RobustSparseBHCGPRegression
from sklearn.decomposition.kernel_pca import KernelPCA

def eval_linreg(train, test, use_meta_features, use_background_data):
    algo = EMBayesRegression() 
    
    X_train = train[0]
    Y_train = train[1]
    Z_train = train[2]

    X_test = test[0]
    Y_test = test[1]
    Z_test = test[2]
    
    if use_meta_features:
        X_train = compound_data(X_train, Z_train)
        X_test = compound_data(X_test, Z_test)
        
    if use_background_data:
        x_train = flatten_data(X_train)
        y_train = flatten_data(Y_train)
        x_test = flatten_data(X_test)
        y_test = flatten_data(Y_test)
        
        algo.fit(x_train, y_train)
        yfit = algo.predict(x_test)
        
        mse = metric.mspe(y_test, yfit)
        sse = metric.tspe(y_test, yfit)
        ymean =  np.mean(y_test)
        sst = metric.tse(y_test, ymean)
        r2 = 1-sse/sst
    else:
        n = len(Z_test)
        yfit = []
        for i in xrange(n):
            if len(X_test[i]) > 0:
                algo.fit(X_train[i], Y_train[i])
                yfit.extend(algo.predict(X_test[i]))
                
        y = flatten_data(Y_test)
        
        mse = metric.mspe(y, yfit)
        sse = metric.tspe(y, yfit)
        ymean =  np.mean(y)
        sst = metric.tse(y, ymean)
        r2 = 1-sse/sst
    
    return (mse, r2)

def eval_bhclinreg(train, test, use_meta_features):
    
    algo = BHCLinearRegression(np.log(10.5), np.log(np.array([2,2]))) 
    
    X_train = train[0]
    Y_train = train[1]
    Z_train = train[2]

    X_test = test[0]
    Y_test = test[1]
    Z_test = test[2]
    
    n = len(Z_test)
    
    if use_meta_features:
        X_train = compound_data(X_train, Z_train)
        X_test = compound_data(X_test, Z_test)
        
    algo.fit(X_train, Y_train, Z_train)
    result = algo.predict_by_task(X_test, np.array(xrange(n)))
    
    y = flatten_data(Y_test)
    yfit = result = flatten_data(result)
    
    mse = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    ymean =  np.mean(y)
    sst = metric.tse(y, ymean)
    r2 = 1-sse/sst
    
    return (mse, r2)

def eval_bhcrobustreg(train, test, use_meta_features):    
    X_train = train[0]
    Y_train = train[1]
    Z_train = train[2]

    X_test = test[0]
    Y_test = test[1]
    Z_test = test[2]
    
    n = len(Z_test)
    d = X_train[0].shape[1]
    
    hyperparams = RobustBayesRegression.wrap(np.log(0.001), np.log(0.001), np.eye(d+1).ravel())
    algo = BHCRobustRegression(np.log(10.5), hyperparams) 
    
    if use_meta_features:
        X_train = compound_data(X_train, Z_train)
        X_test = compound_data(X_test, Z_test)
        
    algo.fit(X_train, Y_train, Z_train)
    result = algo.predict_by_task(X_test, np.array(xrange(n)))
    
    y = flatten_data(Y_test)
    yfit = result = flatten_data(result)
    
    mse = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    ymean =  np.mean(y)
    sst = metric.tse(y, ymean)
    r2 = 1-sse/sst
    
    return (mse, r2)

def eval_bhcsgp(train, test, kernel, use_meta_features):
    algo = SparseBHCGPRegression(np.log(10.5), kernel, k=15, opt_kernel_global=True)
    
    X_train = train[0]
    Y_train = train[1]
    Z_train = train[2]

    X_test = test[0]
    Y_test = test[1]
    Z_test = test[2]
    
    n = len(Z_test)
    
    if use_meta_features:
        X_train = compound_data(X_train, Z_train)
        X_test = compound_data(X_test, Z_test)
        
    algo.fit(X_train, Y_train, Z_train)
    result = algo.predict_by_task(X_test, np.array(xrange(n)))
    
    y = flatten_data(Y_test)
    yfit = result = flatten_data(result)
    
    mse = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    ymean =  np.mean(y)
    sst = metric.tse(y, ymean)
    r2 = 1-sse/sst
    
    return (mse, r2)

def eval_bhcrobustsgp(train, test, kernel, use_meta_features):
    algo = RobustSparseBHCGPRegression(np.log(10.5), kernel, k=15)
    
    X_train = train[0]
    Y_train = train[1]
    Z_train = train[2]

    X_test = test[0]
    Y_test = test[1]
    Z_test = test[2]
    
    n = len(Z_test)
    
    if use_meta_features:
        X_train = compound_data(X_train, Z_train)
        X_test = compound_data(X_test, Z_test)
        
    algo.fit(X_train, Y_train, Z_train)
    result = algo.predict_by_task(X_test, np.array(xrange(n)))
    
    y = flatten_data(Y_test)
    yfit = result = flatten_data(result)
    
    mse = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    ymean =  np.mean(y)
    sst = metric.tse(y, ymean)
    r2 = 1-sse/sst
    
    return (mse, r2)



def eval_sgp(train, test, kernel, k, use_meta_features, use_background_data):
    selector = KMeansSelector(k)
    algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector) 
    
    X_train = train[0]
    Y_train = train[1]
    Z_train = train[2]

    X_test = test[0]
    Y_test = test[1]
    Z_test = test[2]
    
    if use_meta_features:
        X_train = compound_data(X_train, Z_train)
        X_test = compound_data(X_test, Z_test)
        
    if use_background_data:
        x_train = flatten_data(X_train)
        y_train = flatten_data(Y_train)
        x_test = flatten_data(X_test)
        y_test = flatten_data(Y_test)
        
        algo.fit(x_train, y_train)
        yfit = algo.predict(x_test)
        
        mse = metric.mspe(y_test, yfit)
        sse = metric.tspe(y_test, yfit)
        ymean =  np.mean(y_test)
        sst = metric.tse(y_test, ymean)
        r2 = 1-sse/sst
    else:
        n = len(Z_test)
        yfit = []
        for i in xrange(n):
            if len(X_test[i]) > 0:
                algo.fit(X_train[i], Y_train[i])
                yfit.extend(algo.predict(X_test[i]))
                
        y = flatten_data(Y_test)
        
        mse = metric.mspe(y, yfit)
        sse = metric.tspe(y, yfit)
        ymean =  np.mean(y)
        sst = metric.tse(y, ymean)
        r2 = 1-sse/sst
    
    return (mse, r2)

def eval_gp(train, test, kernel, use_meta_features, use_background_data):
    algo = GPRegression(kernel, infer_method=ExactInference) 
    
    X_train = train[0]
    Y_train = train[1]
    Z_train = train[2]

    X_test = test[0]
    Y_test = test[1]
    Z_test = test[2]
    
    if use_meta_features:
        X_train = compound_data(X_train, Z_train)
        X_test = compound_data(X_test, Z_test)
        
    if use_background_data:
        x_train = flatten_data(X_train)
        y_train = flatten_data(Y_train)
        x_test = flatten_data(X_test)
        y_test = flatten_data(Y_test)
        
        algo.fit(x_train, y_train)
        yfit = algo.predict(x_test)
        
        mse = metric.mspe(y_test, yfit)
        sse = metric.tspe(y_test, yfit)
        ymean =  np.mean(y_test)
        sst = metric.tse(y_test, ymean)
        r2 = 1-sse/sst
    else:
        n = len(Z_test)
        yfit = []
        for i in xrange(n):
            if len(X_test[i]) > 0:
                algo.fit(X_train[i], Y_train[i])
                yfit.extend(algo.predict(X_test[i]))
                
        y = flatten_data(Y_test)
        
        mse = metric.mspe(y, yfit)
        sse = metric.tspe(y, yfit)
        ymean =  np.mean(y)
        sst = metric.tse(y, ymean)
        r2 = 1-sse/sst
    
    return (mse, r2)


def load_folds(filename):
    '''
    Load the training and test folds from data. The method
    returns the indices of the training and test samples.
    '''
    mat_dict = sio.loadmat(filename)
    train = np.squeeze(mat_dict['tr'])
    test = np.squeeze(mat_dict['tst'])
    
    return (train, test)

def retrieve_samples(data, indices):
    '''
    data is a tiplet (X,Y,Z) where each entry represents a task
    '''
    indices = indices-1
    X = data[0]
    Y = data[1]
    Z = data[2]
    
    n = len(Z) # number of tasks
    
    Xret = [[]]*n
    Yret = [[]]*n
    
    offset = 0
    for i in xrange(n):
        m = len(X[i])
        
        i_fold = indices[np.all([indices >= offset, indices < offset+m],0)]
        if len(i_fold) > 0:
            Xret[i] = X[i][i_fold-offset]
            Yret[i] = Y[i][i_fold-offset]
        
        offset += m
        
    return (Xret, Yret, Z)

def eval_pcagp(train, test, kernel, use_meta_features, use_background_data, dim):
    reg_model = GPRegression(kernel, infer_method=ExactInference) 
    algo = PCARegression(reg_model, dim)
    
    X_train = train[0]
    Y_train = train[1]
    Z_train = train[2]

    X_test = test[0]
    Y_test = test[1]
    Z_test = test[2]
    
    if use_meta_features:
        X_train = compound_data(X_train, Z_train)
        X_test = compound_data(X_test, Z_test)
        
    if use_background_data:
        x_train = flatten_data(X_train)
        y_train = flatten_data(Y_train)
        x_test = flatten_data(X_test)
        y_test = flatten_data(Y_test)
        
        algo.fit(x_train, y_train)
        yfit = algo.predict(x_test)
        
        mse = metric.mspe(y_test, yfit)
        sse = metric.tspe(y_test, yfit)
        ymean =  np.mean(y_test)
        sst = metric.tse(y_test, ymean)
        r2 = 1-sse/sst
    else:
        n = len(Z_test)
        yfit = []
        for i in xrange(n):
            if len(X_test[i]) > 0:
                algo.fit(X_train[i], Y_train[i])
                yfit.extend(algo.predict(X_test[i]))
                
        y = flatten_data(Y_test)
        
        mse = metric.mspe(y, yfit)
        sse = metric.tspe(y, yfit)
        ymean =  np.mean(y)
        sst = metric.tse(y, ymean)
        r2 = 1-sse/sst
    
    return (mse, r2)

def eval_kernelpcagp(train, test, kernel, use_meta_features, use_background_data, dim):
    reg_model = GPRegression(kernel, infer_method=ExactInference) 
    algo = KernelPCARegression(reg_model, dim,  kernel='rbf', degree=3)
    
    X_train = train[0]
    Y_train = train[1]
    Z_train = train[2]

    X_test = test[0]
    Y_test = test[1]
    Z_test = test[2]
    
    if use_meta_features:
        X_train = compound_data(X_train, Z_train)
        X_test = compound_data(X_test, Z_test)
        
    if use_background_data:
        x_train = flatten_data(X_train)
        y_train = flatten_data(Y_train)
        x_test = flatten_data(X_test)
        y_test = flatten_data(Y_test)
        
        algo.fit(x_train, y_train)
        yfit = algo.predict(x_test)
        
        mse = metric.mspe(y_test, yfit)
        sse = metric.tspe(y_test, yfit)
        ymean =  np.mean(y_test)
        sst = metric.tse(y_test, ymean)
        r2 = 1-sse/sst
    else:
        n = len(Z_test)
        yfit = []
        for i in xrange(n):
            if len(X_test[i]) > 0:
                algo.fit(X_train[i], Y_train[i])
                yfit.extend(algo.predict(X_test[i]))
                
        y = flatten_data(Y_test)
        
        mse = metric.mspe(y, yfit)
        sse = metric.tspe(y, yfit)
        ymean =  np.mean(y)
        sst = metric.tse(y, ymean)
        r2 = 1-sse/sst
    
    return (mse, r2)

    
if __name__ == '__main__':
    nfolds = 10
    fold_filename = '/home/marcel/datasets/multilevel/school_splitz/school_{0}_indexes.mat'
    
    filename = '/home/marcel/datasets/multilevel/ilea/school_norm.csv'
    task_key = 'school'
    task_fields = ['fsm', 'school_vr1', 'school_mixed', 'school_male', 'school_female', 
                   'sdenom_maintain', 'sdenom_coe', 'sdenom_rc']
    # task_fields = ['school']
    target_field = 'exam_score'
    
    
    
    X,Y,Z = load_data(filename, task_key, task_fields, target_field)
    
    r2 = np.empty(nfolds)
    mse = np.empty(nfolds)
    
    for i in xrange(nfolds):
        train, test = load_folds(fold_filename.format(i+1))
        data_train = retrieve_samples((X,Y,Z), train)
        data_test = retrieve_samples((X,Y,Z), test)
        
        #mse[i], r2[i] = eval_linreg(data_train, data_test, False, True)
        
        #kernel = SEKernel(np.log(1), np.log(0.5)) + NoiseKernel(np.log(1))
        #kernel = SEKernel(np.log(1), np.log(0.5)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
        #kernel = ARDSEKernel(np.log(1)*np.ones(18), np.log(0.5)) + NoiseKernel(np.log(1))
        #kernel = ARDSEKernel(np.log(1)*np.ones(18), np.log(0.5)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
        #mse[i], r2[i] = eval_bhcsgp(data_train, data_test, kernel, False)
        #mse[i], r2[i] = eval_gp(data_train, data_test, kernel, False, False)
        #mse[i], r2[i] = eval_sgp(data_train, data_test, kernel, 15, False, True)
        #mse[i], r2[i] = eval_bhcrobustsgp(data_train, data_test, kernel, False)
        #mse[i], r2[i] = eval_bhclinreg(data_train, data_test, True)
        mse[i], r2[i] = eval_bhcrobustreg(data_train, data_test, False)
        print 'task({0}): mse={1}, r2={2}'.format(i, mse[i], r2[i])
        
    results = np.vstack((mse, r2)) 
    
    print 'CV Results:'
    print 'mse,r2'
    print array_str(results.T, precision=16)
    print 'Total Results:'
    print 'mean={0}'.format(array_str(np.mean(results,1), precision=16))
    print 'err={0}'.format(array_str(np.std(results,1), precision=16))


    
    
