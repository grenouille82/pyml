'''
Created on Oct 5, 2011

@author: marcel
'''

import numpy as np
import scipy.io as sio  
import upgeo.util.stats as stats

from upgeo.eval.multitask.base import load_data
from upgeo.ml.multitask.base import compound_data, flatten_data
from upgeo.util import metric
from upgeo.ml.regression.bayes import EMBayesRegression, RobustBayesRegression,\
    EBChenRegression
from upgeo.ml.regression.np.gp import SparseGPRegression, GPRegression
from upgeo.ml.regression.np.infer import FITCExactInference, ExactInference
from upgeo.ml.regression.np.selector import KMeansSelector, RandomSubsetSelector,\
    FixedSelector
from numpy.core.numeric import array_str
from upgeo.ml.regression.np.kernel import SEKernel, NoiseKernel,\
    SqConstantKernel, LinearKernel, ARDSEKernel, MaternKernel
from upgeo.ml.multitask.regression.bhc import BHCLinearRegression,\
    BHCRobustRegression, BHCGPRegression, SparseBHCGPRegression,\
    RobustSparseBHCGPRegression
from upgeo.ml.multitask.regression.bhce import SparseBHCGPRegregressionExpert,\
    BHCRobustRegressionExpert

def eval_linreg(train, test, use_meta_features, use_background_data):
    algo = EMBayesRegression(alpha0=1, beta0=1, weight_bias=True)
    
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
        
        terr = 0
        for i in xrange(n):
            
            print 'shapes'
            print np.asarray(X_train[i]).shape
            print np.asarray(Y_train[i]).shape
            print np.asarray(X_test[i]).shape
            print np.asarray(Y_test[i]).shape

            if len(X_test[i]) > 0:
                algo.fit(X_train[i], Y_train[i])
                yfit.extend(algo.predict(X_test[i]))
                
                #debug
                yhat = algo.predict(X_test[i])
                err  = (yhat-Y_test[i])**2
                outlier = (err > 1.5)
                if np.sum(outlier > 0):
                    print 'outlier region={0}, number={1}:'.format(i, np.sum(outlier))
                    print np.c_[X_test[i][outlier],Y_test[i][outlier],yhat[outlier], err[outlier]]
                terr += np.sum(err)
            
        print 'fuck={0}'.format(terr/len(flatten_data(Y_test)))
        
                
        y = flatten_data(Y_test)
        
        mse = metric.mspe(y, yfit)
        sse = metric.tspe(y, yfit)
        ymean =  np.mean(y)
        sst = metric.tse(y, ymean)
        r2 = 1-sse/sst
    
    return (mse, r2)

def eval_bhclinreg(train, test, use_meta_features):
    
    algo = BHCLinearRegression(np.log(10.5), np.log(np.array([1,1]))) 
    
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
    
    #hyperparams = RobustBayesRegression.wrap(np.log(20), np.log(4), np.eye(d+1).ravel())
    #hyperparams = RobustBayesRegression.wrap(np.log(0.00000000001), np.log(0.00000000001), np.eye(d+1).ravel())
    #hyperparams = RobustBayesRegression.wrap(np.log(0.00001), np.log(0.00001), np.eye(d+1).ravel())
    #hyperparams = RobustBayesRegression.wrap(np.log(0.0001), np.log(0.0001), np.eye(d+1).ravel())
    #hyperparams = RobustBayesRegression.wrap(np.log(0.001), np.log(0.001), np.eye(d+1).ravel())
    hyperparams = RobustBayesRegression.wrap(np.log(0.1), np.log(0.1), np.eye(d+1).ravel())
    #hyperparams = RobustBayesRegression.wrap(np.log(1), np.log(1), np.eye(d+1).ravel())
    algo = BHCRobustRegression(np.log(10.5), hyperparams) 
    
    if use_meta_features:
        X_train = compound_data(X_train, Z_train)
        X_test = compound_data(X_test, Z_test)
        
    algo.fit(X_train, Y_train, Z_train)
    result = algo.predict_by_task(X_test, np.array(xrange(n)))
    
    #marshal
    algo.marshal_xml('nga_clustertree.xml')
    
    
    y = flatten_data(Y_test)
    yfit = result = flatten_data(result)
    
    mse = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    ymean =  np.mean(y)
    sst = metric.tse(y, ymean)
    r2 = 1-sse/sst
    
    return (mse, r2)

def eval_bhcsgp(train, test, kernel, use_meta_features):
    algo = SparseBHCGPRegression(np.log(10.5), kernel, k=50)
    
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

def eval_bhcgp(train, test, kernel, use_meta_features):
    algo = BHCGPRegression(np.log(10.5), kernel)
    
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


def eval_bhcrobustreg_eq(train, test, use_meta_features):    
    X_train = train[0]
    Y_train = train[1]
    Z_train = train[2]

    X_test = test[0]
    Y_test = test[1]
    Z_test = test[2]
    
    n = len(Z_test)
    d = X_train[0].shape[1]
    
    hyperparams = RobustBayesRegression.wrap(np.log(0.1), np.log(4), np.eye(d+1).ravel())
    algo = BHCRobustRegression(np.log(10.5), hyperparams) 
    
    if use_meta_features:
        X_train = compound_data(X_train, Z_train)
        X_test = compound_data(X_test, Z_test)
        
    algo.fit(X_train, Y_train, Z_train)
    result = algo.predict(X_test, Z_test, method='complete')
    
    y = flatten_data(Y_test)
    yfit = result = flatten_data(result)
    
    mse = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    ymean =  np.mean(y)
    sst = metric.tse(y, ymean)
    r2 = 1-sse/sst
    
    return (mse, r2)

def eval_bhcerobustreg_eq(train, test, use_meta_features):    
    X_train = train[0]
    Y_train = train[1]
    Z_train = train[2]

    X_test = test[0]
    Y_test = test[1]
    Z_test = test[2]
    
    if use_meta_features:
        X_train = compound_data(X_train, Z_train)
        X_test = compound_data(X_test, Z_test)
    
    n = len(Z_test)
    d = X_train[0].shape[1]
    
    hyperparams = RobustBayesRegression.wrap(np.log(0.1), np.log(4), np.eye(d+1).ravel())
    algo = BHCRobustRegressionExpert(np.log(10.5), hyperparams) 
    
     
    y = flatten_data(Y_test)
    ymean =  np.mean(y)
    
    mse = np.empty(5)
    r2 = np.empty(5)
        
    algo.fit(X_train, Y_train, Z_train)

    result = algo.predict(X_test, Z_test, method='path', incl_tree_weights=True)
    yfit = result = flatten_data(result)
    mse[0] = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    sst = metric.tse(y, ymean)
    r2[0] = 1-sse/sst
    
    result = algo.predict(X_test, Z_test, method='path', incl_tree_weights=False)
    yfit = result = flatten_data(result)
    mse[1] = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    sst = metric.tse(y, ymean)
    r2[1] = 1-sse/sst
    
    result = algo.predict(X_test, Z_test, method='path1', incl_tree_weights=False)
    yfit = result = flatten_data(result)
    mse[2] = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    sst = metric.tse(y, ymean)
    r2[2] = 1-sse/sst
    
    result = algo.predict(X_test, Z_test, method='complete', incl_tree_weights=True)
    yfit = result = flatten_data(result)
    mse[3] = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    sst = metric.tse(y, ymean)
    r2[3] = 1-sse/sst
    
    result = algo.predict(X_test, Z_test, method='complete', incl_tree_weights=False)
    yfit = result = flatten_data(result)
    mse[4] = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    sst = metric.tse(y, ymean)
    r2[4] = 1-sse/sst


    
    return (mse, r2)

def eval_bhcsgp_eq(train, test, kernel, use_meta_features):
    algo = SparseBHCGPRegression(np.log(10.5), kernel, k=30)
    
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
    result = algo.predict(X_test, Z_test, method='complete')
    
    y = flatten_data(Y_test)
    yfit = result = flatten_data(result)
    
    mse = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    ymean =  np.mean(y)
    sst = metric.tse(y, ymean)
    r2 = 1-sse/sst
    
    return (mse, r2)

def eval_bhcesgp_eq(train, test, kernel, use_meta_features):
    algo = SparseBHCGPRegregressionExpert(np.log(10.5), kernel, k=50)
    
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
        
    
    y = flatten_data(Y_test)
    ymean =  np.mean(y)
    
    mse = np.empty(5)
    r2 = np.empty(5)
        
    algo.fit(X_train, Y_train, Z_train)

    result = algo.predict(X_test, Z_test, method='path', incl_tree_weights=True)
    yfit = result = flatten_data(result)
    mse[0] = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    sst = metric.tse(y, ymean)
    r2[0] = 1-sse/sst
    
    result = algo.predict(X_test, Z_test, method='path', incl_tree_weights=False)
    yfit = result = flatten_data(result)
    mse[1] = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    sst = metric.tse(y, ymean)
    r2[1] = 1-sse/sst
    
    result = algo.predict(X_test, Z_test, method='path1', incl_tree_weights=False)
    yfit = result = flatten_data(result)
    mse[2] = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    sst = metric.tse(y, ymean)
    r2[2] = 1-sse/sst
    
    result = algo.predict(X_test, Z_test, method='complete', incl_tree_weights=True)
    yfit = result = flatten_data(result)
    mse[3] = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    sst = metric.tse(y, ymean)
    r2[3] = 1-sse/sst
    
    result = algo.predict(X_test, Z_test, method='complete', incl_tree_weights=False)
    yfit = result = flatten_data(result)
    mse[4] = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    sst = metric.tse(y, ymean)
    r2[4] = 1-sse/sst


    
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
    X_train = train[0]
    Y_train = train[1]
    Z_train = train[2]

    X_test = test[0]
    Y_test = test[1]
    Z_test = test[2]
    
    
#    if use_background_data:
#        #selecting pseudo inputs for each task
#        n = len(X_train)
#        d = X_train[0].shape[1]
#        m = np.int(np.floor(k/n))
#        Xu = np.empty((1,d))
#        for i in xrange(n):
#            selector = KMeansSelector(k)
#            Xu = np.r_[Xu, selector.apply(X_train[i], Y_train[i])]
#            
#        selector = FixedSelector(Xu)
#    else:
        #selector = KMeansSelector(k)
    selector = KMeansSelector(k, False) #include targets
    #selector = RandomSubsetSelector(k)
    

    
    if use_meta_features:
        X_train = compound_data(X_train, Z_train)
        X_test = compound_data(X_test, Z_test)
        
    if use_background_data:
        algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector)
        
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
        hyperparams = np.array(kernel.params).copy()
        n = len(Z_test)
        yfit = []
        for i in xrange(n):
            if len(X_test[i]) > 0:
                kernel.params = hyperparams
                algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector)
                #if len(X_test[i]) > 500:
                    #sparse gp approximation
                #    algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector)
                #else:
                    #full gp
                #    algo = GPRegression(kernel, infer_method=ExactInference)
                
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
    indices = indices - 1
    X = data[0]
    Y = data[1]
    Z = data[2]
    
    n = len(Z) # number of tasks
    
    Xret = [[]]*n
    Yret = [[]]*n
    
    offset = 0
    for i in xrange(n):
        Xi = np.atleast_2d(X[i])
        Yi = np.atleast_1d(Y[i])
        m = len(Xi)
        
        i_fold = indices[np.all([indices >= offset, indices < offset+m],0)]
        if len(i_fold) > 0:
            Xret[i] = Xi[i_fold-offset]
            Yret[i] = Yi[i_fold-offset]
        
        offset += m
        
    return (Xret, Yret, Z)

def delete_empty_tasks(data):
    X = data[0]
    Y = data[1]
    Z = data[2]
    
    Xret = []
    Yret = []
    Zret = []
    
    n = len(X)
    for i in xrange(n):
        if len(X[i]) > 0:
            Xret.append(X[i])
            Yret.append(Y[i])
            Zret.append(Z[i])
            
    return (Xret, Yret, Zret)
            
    
if __name__ == '__main__':
    nsplitz = 5
    nfolds = 10
    fold_filename = '/home/mhermkes/datasets/multilevel/nga_splitz{0}/nga_{1}_indexes.mat'
    
    filename = '/home/mhermkes/datasets/multilevel/nga/multitask_norm_pga7.csv'
    #filename = '/home/marcel/datasets/multilevel/nga/nga_norm1a.csv'
    task_key = 'region_id'
    task_fields = ['region_id']
    target_field = 'pga' 
    
    #filename = '/home/mhermkes/datasets/multilevel/nga/eq_multitask_norm_pga_test.csv'
    #task_key = 'eq_id'
    #task_fields = ['eq_mg', 'dip', 'rake_angle', 'hypctr_depth']
    #target_field = 'pga'
    
    
    X,Y,Z = load_data(filename, task_key, task_fields, target_field)
    split_results = np.empty((nsplitz,2))
    
    for j in xrange(nsplitz):

        r2 = np.empty(nfolds)
        mse = np.empty(nfolds)
        weights = np.empty(nfolds)
    
        for i in xrange(nfolds):
            train, test = load_folds(fold_filename.format(j+1,i+1))
        
            print 'intersect={0}'.format( np.intersect1d(train, test))
        
            data_train = retrieve_samples((X,Y,Z), train)
            data_test = retrieve_samples((X,Y,Z), test)
            
            #print 'length'
            #for j in xrange(len(data_train[0])):
            #    print np.asarray(data_train[0][j]).shape
            
            #mse[i], r2[i] = eval_linreg(data_train, data_test, False, False)
            weights[i] = len(test)
            
            #kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(1))
            #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
            #kernel = MaternKernel(5, np.log(1), np.log(1)) + NoiseKernel(np.log(1))
            #kernel = ARDSEKernel(np.log(1)*np.ones(16), np.log(1)) + NoiseKernel(np.log(1))
            kernel = ARDSEKernel(np.log(1)*np.ones(16), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
            mse[i], r2[i] = eval_sgp(data_train, data_test, kernel, 50, False, True)
            #mse[i], r2[i] = eval_gp(data_train, data_test, kernel, False, True)
            #mse[i], r2[i] = eval_bhcsgp(data_train, data_test, kernel, False)
            #mse[i], r2[i] = eval_bhcrobustsgp(data_train, data_test, kernel, False)
            #mse[i], r2[i] = eval_bhclinreg(data_train, data_test, False)
            #mse[i], r2[i] = eval_bhcrobustreg(data_train, data_test, False)
            print 'task({0}): mse={1}, r2={2}'.format(i, mse[i], r2[i])
            
        results = np.vstack((mse, r2)) 
        
        print weights
        print 'CV Results:'
        print 'mse,r2'
        print array_str(results.T, precision=16)
        print 'Total Results:'
        means = np.asarray([stats.mean(results[0], weights), stats.mean(results[1], weights)])
        std = np.asarray([stats.stddev(results[0], weights), stats.stddev(results[1], weights)])
        
        print 'mean={0}'.format(array_str(means, precision=16))
        print 'err={0}'.format(array_str(std, precision=16))
        split_results[j] = np.asarray([means[0], std[0]])
        
        
    print 'overall split results:'
    print array_str(split_results, precision=16)
    


    
    
