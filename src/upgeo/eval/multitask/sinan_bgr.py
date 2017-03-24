'''
Created on Mar 5, 2012

@author: marcel
'''
import numpy as np
import scipy.io as sio 

import upgeo.util.metric as metric
import upgeo.util.stats as stats
from numpy.core.numeric import array_str
from upgeo.ml.regression.bayes import EMBayesRegression
from upgeo.ml.regression.np.gp import GPRegression, SparseGPRegression
from upgeo.ml.regression.np.infer import ExactInference, FITCExactInference
from upgeo.ml.regression.np.kernel import NoiseKernel, SEKernel,\
    SqConstantKernel, LinearKernel, ARDSEKernel
from upgeo.ml.regression.np.selector import KMeansSelector, RandomSubsetSelector

def load_folds(filename):
    '''
    Load the training and test folds from data. The method
    returns the indices of the training and test samples.
    '''
    mat_dict = sio.loadmat(filename)
    train = np.squeeze(mat_dict['tr'])
    test = np.squeeze(mat_dict['tst'])
    
    train = train-1
    test = test-1
    
    return (train, test)

def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    d = data.shape[1]
    X = data[:, 0:d-1]
    y = data[:, d-1]
    print y.shape
    return X,y

def eval_only_background(algo, data, data_bgr, nfolds, fold_fname):
    
    X,y = data
    Xb,yb = data_bgr
    
    algo.fit(Xb, yb)
    
    r2_total = np.empty(nfolds)
    mse_total = np.empty(nfolds)
    weights = np.empty(nfolds)
    
    for i in xrange(nfolds):
        train, test = load_folds(fold_fname.format(i+1))
        yfit = algo.predict(X[test])
        
        mse = metric.mspe(y[test], yfit)
        sse = metric.tspe(y[test], yfit)
        ymean =  np.mean(y[test])
        sst = metric.tse(y[test], ymean)
        r2 = 1-sse/sst
        mse_total[i] = mse
        r2_total[i] = r2
        weights[i] = len(test)
                
        print 'task={0}, mse={1}'.format(i, mse)
        
    results = np.vstack((mse_total, r2_total)) 
    
    print weights
    print 'CV Results:'
    print 'mse,r2'
    print array_str(results, precision=16)
    print 'Total Results:'
    means = np.asarray([stats.mean(results[0], weights), stats.mean(results[1], weights)])
    std = np.asarray([stats.stddev(results[0], weights), stats.stddev(results[1], weights)])
    #means = np.asarray([stats.mean(results[:,0], weights), stats.mean(results[:,1], weights), stats.mean(results[:,2], weights), stats.mean(results[:,3], weights),
    #                    stats.mean(results[:,4], weights), stats.mean(results[:,5], weights), stats.mean(results[:,6], weights), stats.mean(results[:,7], weights), 
    #                    stats.mean(results[:,8], weights), stats.mean(results[:,9], weights)])
    #std = np.asarray([stats.stddev(results[:,0], weights), stats.stddev(results[:,1], weights), stats.stddev(results[:,2], weights), stats.stddev(results[:,3], weights),
    #                  stats.stddev(results[:,4], weights), stats.stddev(results[:,5], weights), stats.stddev(results[:,6], weights), stats.stddev(results[:,7], weights),
    #                  stats.stddev(results[:,8], weights), stats.stddev(results[:,9], weights)])
    
    print 'mean={0}'.format(array_str(means, precision=16))
    print 'err={0}'.format(array_str(std, precision=16))

def eval_background(algo, data, data_bgr, nfolds, fold_fname):
    
    params = algo.hyperparams
    
    X,y = data
    Xb,yb = data_bgr
    
    r2_total = np.empty(nfolds)
    mse_total = np.empty(nfolds)
    weights = np.empty(nfolds)
    
    for i in xrange(nfolds):
        algo.hyperparams = params
        
        
        print 'hyper={0}'.format(algo.hyperparams)
        
        train, test = load_folds(fold_fname.format(i+1))
        Xtrain = np.vstack((Xb, X[train]))
        ytrain = np.hstack((yb, y[train]))
        algo.fit(Xtrain, ytrain)
        
        yfit = algo.predict(X[test])
        
        mse = metric.mspe(y[test], yfit)
        sse = metric.tspe(y[test], yfit)
        ymean =  np.mean(y[test])
        sst = metric.tse(y[test], ymean)
        r2 = 1-sse/sst
        mse_total[i] = mse
        r2_total[i] = r2
        weights[i] = len(test)
        
        print 'task={0}, mse={1}'.format(i, mse)
        
    results = np.vstack((mse_total, r2_total)) 
    
    print weights
    print 'CV Results:'
    print 'mse,r2'
    print array_str(results, precision=16)
    print 'Total Results:'
    means = np.asarray([stats.mean(results[0], weights), stats.mean(results[1], weights)])
    std = np.asarray([stats.stddev(results[0], weights), stats.stddev(results[1], weights)])
    #means = np.asarray([stats.mean(results[:,0], weights), stats.mean(results[:,1], weights), stats.mean(results[:,2], weights), stats.mean(results[:,3], weights),
    #                    stats.mean(results[:,4], weights), stats.mean(results[:,5], weights), stats.mean(results[:,6], weights), stats.mean(results[:,7], weights), 
    #                    stats.mean(results[:,8], weights), stats.mean(results[:,9], weights)])
    #std = np.asarray([stats.stddev(results[:,0], weights), stats.stddev(results[:,1], weights), stats.stddev(results[:,2], weights), stats.stddev(results[:,3], weights),
    #                  stats.stddev(results[:,4], weights), stats.stddev(results[:,5], weights), stats.stddev(results[:,6], weights), stats.stddev(results[:,7], weights),
    #                  stats.stddev(results[:,8], weights), stats.stddev(results[:,9], weights)])
    
    print 'mean={0}'.format(array_str(means, precision=16))
    print 'err={0}'.format(array_str(std, precision=16))

def eval_notransfer(algo, data, nfolds, fold_fname):
    
    params = algo.hyperparams
    
    X,y = data
    
    r2_total = np.empty(nfolds)
    mse_total = np.empty(nfolds)
    weights = np.empty(nfolds)
    
    for i in xrange(nfolds):
        algo.hyperparams = params
        
        train, test = load_folds(fold_fname.format(i+1))
        
        
        print train, test
        algo.fit(X[train], y[train])
        
        yfit = algo.predict(X[test])
        
        mse = metric.mspe(y[test], yfit)
        sse = metric.tspe(y[test], yfit)
        ymean =  np.mean(y[test])
        sst = metric.tse(y[test], ymean)
        r2 = 1-sse/sst
        mse_total[i] = mse
        r2_total[i] = r2
        weights[i] = len(test)
        
        print 'task={0}, mse={1}'.format(i, mse)
        
    results = np.vstack((mse_total, r2_total)) 
    
    print weights
    print 'CV Results:'
    print 'mse,r2'
    print array_str(results, precision=16)
    print 'Total Results:'
    means = np.asarray([stats.mean(results[0], weights), stats.mean(results[1], weights)])
    std = np.asarray([stats.stddev(results[0], weights), stats.stddev(results[1], weights)])
    #means = np.asarray([stats.mean(results[:,0], weights), stats.mean(results[:,1], weights), stats.mean(results[:,2], weights), stats.mean(results[:,3], weights),
    #                    stats.mean(results[:,4], weights), stats.mean(results[:,5], weights), stats.mean(results[:,6], weights), stats.mean(results[:,7], weights), 
    #                    stats.mean(results[:,8], weights), stats.mean(results[:,9], weights)])
    #std = np.asarray([stats.stddev(results[:,0], weights), stats.stddev(results[:,1], weights), stats.stddev(results[:,2], weights), stats.stddev(results[:,3], weights),
    #                  stats.stddev(results[:,4], weights), stats.stddev(results[:,5], weights), stats.stddev(results[:,6], weights), stats.stddev(results[:,7], weights),
    #                  stats.stddev(results[:,8], weights), stats.stddev(results[:,9], weights)])
    
    print 'mean={0}'.format(array_str(means, precision=16))
    print 'err={0}'.format(array_str(std, precision=16))

if __name__ == '__main__':
    #fold_filename = '/home/marcel/datasets/multilevel/eueq_pga_nico_splitz/eueq_{0}_indexes.mat'
    
    #data = load_data('/home/marcel/datasets/multilevel/bgr_eqeu/nga/eueq_norm3.csv')
    #data_bgr = load_data('/home/marcel/datasets/multilevel/bgr_eqeu/nga/bgr_nga_norm3.csv')
    
    #data = load_data('/home/marcel/datasets/multilevel/bgr_eqeu/allen/eueq_norm3.csv')
    #data_bgr = load_data('/home/marcel/datasets/multilevel/bgr_eqeu/allen/bgr_allen_norm3.csv')

    fold_filename = '/home/marcel/datasets/multilevel/nga_splitz/nga_{0}_indexes.mat'
    
    #data_bgr = load_data('/home/marcel/datasets/multilevel/bgr_eqeu/nga/eueq_norm3.csv')
    #data = load_data('/home/marcel/datasets/multilevel/bgr_eqeu/nga/bgr_nga_norm3.csv')

    data_bgr = load_data('/home/marcel/datasets/multilevel/bgr_eqeu/awnga/bgr_allen_norm3.csv')
    data = load_data('/home/marcel/datasets/multilevel/bgr_eqeu/awnga/nga_norm3.csv')


    
    algo = EMBayesRegression()
    kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(1))
    #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
    #kernel = MaternKernel(5, np.log(1), np.log(1)) + NoiseKernel(np.log(1))
    #kernel = ARDSEKernel(np.log(1)*np.ones(13), np.log(1)) + NoiseKernel(np.log(1))
    #kernel = ARDSEKernel(np.log(1)*np.ones(13), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
    
    
    #algo = GPRegression(kernel, infer_method=ExactInference)
    selector = KMeansSelector(30)
    #selector = RandomSubsetSelector(50)
    #algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector)    
        
    #eval_notransfer(algo, data, 10, fold_filename)
    #eval_only_background(algo, data, data_bgr, 10, fold_filename)
    eval_background(algo, data, data_bgr, 10, fold_filename)
    
    
    
    
    
    