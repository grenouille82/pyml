'''
Created on Feb 3, 2012

@author: marcel
'''
import numpy as np
import scipy.io as sio  
import upgeo.util.stats as stats
import upgeo.util.metric as metric
import scipy.sparse as sparse

from upgeo.eval.multitask.base import load_data
from upgeo.ml.regression.np.kernel import SEKernel, NoiseKernel, HiddenKernel
from upgeo.ml.regression.np.gp import GPRegression
from upgeo.ml.regression.np.infer import ExactInference
from upgeo.ml.multitask.base import flatten_data
from scipy.sparse.construct import kron

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
        Xi = np.atleast_2d(X[i])
        Yi = np.atleast_1d(Y[i])
        m = len(Xi)
        
        i_fold = indices[np.all([indices >= offset, indices < offset+m],0)]
        if len(i_fold) >= 0:
            Xret[i] = Xi[i_fold-offset]
            Yret[i] = Yi[i_fold-offset]
        
        offset += m
        
    return (Xret, Yret, Z)


if __name__ == '__main__':
    fold_filename = '/home/marcel/datasets/multilevel/eueq_pga_nico_splitz/eueq_1_indexes.mat'
    filename = '/home/marcel/datasets/multilevel/eueq/mt_eueq_nico_norm_pga6.csv'
    
    task_key = 'region'
    task_fields = ['region']
    target_field = 'pga'
    
    #load data, and training and test folds
    X,Y,Z = load_data(filename, task_key, task_fields, target_field)
    train_folds, test_folds = load_folds(fold_filename)
    
    #retrieve test and train data from folds
    X_train, Y_train, _ = retrieve_samples((X,Y,Z), train_folds)
    X_test, Y_test, _ = retrieve_samples((X,Y,Z), test_folds)
    
    m = len(X) #number of tasks
    
    for i in xrange(m):
        #kernel = SEKernel(np.log(1), np.log(1))
        kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(1))
        
        if len(Y_test[i]) > 0:
            print X_train[i].shape
            gp = GPRegression(kernel, infer_method=ExactInference)
            gp.fit(X_train[i], Y_train[i])
        
            yhat = gp.predict(X_test[i])
            mse = metric.mspe(Y_test[i], yhat)
            print 'task{0}: mse={1}'.format(i, mse)
    
    #learning block diagonal covariance matrx by fitting each tasks individually
    #kernel = SEKernel(np.log(1), np.log(1))
    kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(1))
    
    gp = GPRegression(kernel, infer_method=ExactInference)
    gp.fit(flatten_data(X_train), flatten_data(Y_train))
    opt_params = kernel.params
    print 'global opt kernel params: {0}'.format(np.exp(opt_params))
    
    X = flatten_data(X_train)
    y = flatten_data(Y_train)
    n = len(X)
    
    Kx = kernel(X,X) # no noise at the diag eleents 
    iKx = np.linalg.inv(Kx + np.eye(n)*1e-6) 
    
    #build matrix D (noise for all specific tasks
    I = np.eye(n)
    D = np.eye(m)
    for i in xrange(m):
        masked_kernel = HiddenKernel(SEKernel(opt_params[0], opt_params[1])) + NoiseKernel(np.log(1))
        gp = GPRegression(masked_kernel, infer_method=ExactInference)
        gp.fit(X_train[i], Y_train[i])
        D[i,i] = masked_kernel.params[0]
    
    
    Kf1 = np.eye(m) #uncorrelated task matrix
    
    DI = kron(D,I).todense()
    Kfx = kron(Kf1,Kx).todense()
    S = Kfx + DI
    
    print S.shape
    print Kfx.shape
    
    #iS = np.linalg.inv(S)
    #iKfx = np.linalg.inv(Kfx)
    
    xs = X_test[0][0]
    kx = kernel(xs, X)
    kf = Kf1[0]
    
    kfx = kron(kf,kx).todense()
    print kfx.shape
    print y.shape
    Sy = np.dot(S.T,y)
    
    
    print Sy.shape
    
    #Kx = []
    #Y = []
    #for i in xrange(m):
    #    Kxi = kernel(X_train[i])
    #    Kx.append(Kxi)
    #    Y.append(Y_train[i])
    #Kx = block_diag(Kx.tolist()).todense()
    #Y = block_diag(Y).todense()
     
    #Kf2 = np.dot(np.dot(Y.T, iKx), Y)/n
    
      
    
    
    
    
    
    