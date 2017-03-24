'''
Created on Feb 20, 2012

@author: marcel
'''

import numpy as np
import scipy.io as sio

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from upgeo.filter import BinarizeFilter, StandardizeFilter, MeanShiftFilter
from upgeo.eval.regression.trial import CVRegressionExperiment
from upgeo.ml.regression.bayes import EMBayesRegression
from upgeo.ml.regression.np.gp import GPRegression
from upgeo.ml.regression.np.infer import ExactInference, OnePassInference
from upgeo.ml.regression.np.kernel import NoiseKernel, SEKernel, ARDSEKernel,\
    SqConstantKernel, LinearKernel, MaternKernel
from upgeo.ml.regression.plot import plot1d_reg, plot2d_reg, plot1d_cov_gp

def load_data(filename):
    mat_dict = sio.loadmat(filename)
    data = mat_dict['data']
    
    k = data.shape[1]
    X = data[:,0:(k-1)]
    y = data[:,k-1]
    return X,y


def cv_eval(X, y, algo, runs=1, seeds=None, filter=None, norm_mask=None):
    
    results = np.empty((runs,2))
    for i in xrange(runs):
        seed = None
        if seeds is not None:
            seed = seeds[i]
            
        experiment = CVRegressionExperiment(X, y, 10, filter, norm_mask, seed)
        mse, err = experiment.eval(algo)
        results[i] = mse, err 

    print 'Run Results:'
    print results
    print 'Total Result'
    print np.mean(results,0)
    
def mag_loo_eval(X, y, algo, mag_idx, min_mag=6.0, max_mag=8.0, step_size=0.25, norm_mask=None):
    mag_ranges = np.arange(min_mag, max_mag, step_size)[::-1]
    result = np.empty((len(mag_ranges),2))
    
    hyperparams = algo.hyperparams
    
    k = X.shape[1]+1
    i = 0
    for mag in mag_ranges:
        test = X[:,mag_idx] >= mag
        train = X[:,mag_idx] < mag
        print 'test'
        print np.sum(train)
        print np.sum(test)
        
        Z = np.c_[X,y]
        if norm_mask is None:
            Z = StandardizeFilter().process(Z)
        else:
            Z[:,norm_mask] = StandardizeFilter().process(Z[:,norm_mask])
            print Z[1:5,:]
            
        Xt = Z[:,0:(k-1)]
        yt = Z[:,k-1]
        
        algo.hyperparams = hyperparams
        print Xt[train].shape
        algo.fit(Xt[train], yt[train])
        yhat = algo.predict(Xt[test])
        
        se = (yt[test]-yhat)**2
        print 'mse={0}'.format((np.linalg.norm(yt[test]-yhat)**2)/len(yt[test]))
        result[i,0] = se.mean()
        result[i,1] = se.std()
        i = i+1
        
    algo.hyperparams = hyperparams
    
    print 'Results'
    for i in xrange(len(mag_ranges)):
        print 'Mag={0}: {1}'.format(mag_ranges[i], result[i])
        

def run_cv_simdata(algo, runs=1, seeds=None):
    
    X, y = load_data('/home/marcel/datasets/multilevel/synth_data/eval_smsim_pga_sample.mat')
    
    X = StandardizeFilter().process(X)
    y = StandardizeFilter().process(y)

    cv_eval(X, y, algo, runs, seeds)
    
def run_cv_gmmdata(algo, runs=1, seeds=None):
    
    X, y = load_data('/home/marcel/datasets/multilevel/synth_data/eval_synth_ab.mat')
    
    X = StandardizeFilter().process(X)
    y = MeanShiftFilter().process(y)
    
    k = X.shape[1]
    Z = BinarizeFilter().process(X[:,np.newaxis,k-1])
    X = np.c_[X[:,0:(k-1)], Z]
    
    cv_eval(X, y, algo, runs, seeds)

def run_mag_loo_simdata(algo):
    X, y = load_data('/home/marcel/datasets/multilevel/synth_data/eval_smsim_pga_sample2.mat')
    mag_loo_eval(X, y, algo, mag_idx=4, min_mag=6, max_mag=7.5, step_size=0.25)
    
def run_mag_loo_gmmdata(algo):
    X, y = load_data('/home/marcel/datasets/multilevel/synth_data/eval_synth_ab2.mat')
    
    k = X.shape[1]
    Z = BinarizeFilter().process(X[:,np.newaxis,k-1])
    X = np.c_[X[:,0:(k-1)], Z]
    mask = np.r_[np.arange(k-1), k+2]
    
    mag_loo_eval(X, y, algo, mag_idx=0, min_mag=5.5, max_mag=7.5, step_size=0.25, norm_mask=mask)

def plot_magdist(mag, dist, algo, ):
    pass

def plot_contour(x, y, z1, z2):
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    
    plt.subplots_adjust(hspace=0.5)
    
    plt.subplot(121)
    plt.contourf(x,y,z1,25,cmap=cm.jet)
    plt.colorbar()
    
    plt.subplot(122)
    plt.contourf(x,y,z2,25,cmap=cm.jet)
    plt.colorbar()
    
    plt.show()
    
    
    
    
    
if  __name__ == '__main__':
    
    #kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.1))
    #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.1))
    #kernel = MaternKernel(5, np.log(1), np.log(0.5)) + NoiseKernel(np.log(1))
    kernel = ARDSEKernel(np.log(1)*np.ones(6), np.log(1)) + NoiseKernel(np.log(0.1))
    #kernel = ARDSEKernel(np.log(1)*np.ones(2), np.log(0.5)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
    
    #algo = EMBayesRegression()
    algo = GPRegression(kernel, infer_method=ExactInference)
    #run_cv_simdata(algo, runs=5)
    #run_cv_gmmdata(algo, runs=5)
    run_mag_loo_simdata(algo)
    #run_mag_loo_gmmdata(algo)
    
    X, y = load_data('/home/marcel/datasets/multilevel/synth_data/eval_synth_ab1.mat')
    k = X.shape[1]
    Z = BinarizeFilter().process(X[:,np.newaxis,k-1])
    X = np.c_[X[:,0:(k-1)], Z]
    print X.shape
    #X = X[0:1000,0:3]
    #y = y[0:1000]
    
    X = StandardizeFilter().process(X)
    y = StandardizeFilter().process(y)

    
#    algo.fit(X, y)
#    
#    
#    xi = np.linspace(5, 7.5, 200)
#    yi = np.linspace(0, 200, 200)
#    
#    xx, yy = np.meshgrid(xi, yi)
#    
#    
#    #z,zi = algo.predict(np.c_[xx.ravel(),yy.ravel()], True)
#    #zi = np.sqrt(zi)
#    
#    print 'puff'
#    #print zi.shape
#    
#    #print z.shape
#    #print np.c_[xx.ravel(),yy.ravel()]
#    #print xx.shape
#    #print yy.shape
#    #print z
#    #z = z.reshape(200,200)
#    #zi = zi.reshape(200,200)
#    #print z.shape
#    
#    #print 'pop'
#    #plot_contour(xx, yy, z, zi)
#    #print 'pup'
#    
#    plot1d_reg(4, 7.5, 0, algo, np.array([100, 500, 1,0,0]), 200, (X,y))
#    plot1d_reg(0, 200, 1, algo, np.array([7, 500, 1,0,0]), 200, (X,y))
#    plot1d_reg(0, 200, 1, algo, np.array([6, 500, 1,0,0]), 200, (X,y))
#    #plot1d_reg(0, 400, 1, algo, np.array([6]), 200, (X,y))
#    #plot1d_reg(0, 400, 1, algo, np.array([7]), 200, (X,y))
#
#    plot2d_reg(4, 7.5, 0, 0, 200, 1, algo, np.array([500, 1,0,0]), n=50)
#    plot2d_reg(4, 7.5, 0, 0, 200, 1, algo, np.array([200, 1,0,0]), n=50)
#    plot2d_reg(4, 7.5, 0, 0, 200, 1, algo, np.array([800, 1,0,0]), n=50)
#    plot2d_reg(4, 7.5, 0, 0, 200, 1, algo, np.array([500, 0,1,0]), n=50)
#    plot2d_reg(4, 7.5, 0, 0, 200, 1, algo, np.array([200, 0,1,0]), n=50)
#    plot2d_reg(4, 7.5, 0, 0, 200, 1, algo, np.array([800, 0,1,0]), n=50)
#    #plot2d_reg(4, 7.5, 0, 0, 200, 1, algo, np.array([200]), n=200)
#    #plot2d_reg(4, 7.5, 0, 0, 200, 1, algo, np.array([200]), n=200)
#    #plot2d_reg(4, 7.5, 0, 0, 200, 1, algo, np.array([200]), n=200)
#    #plot2d_reg(4, 7.5, 0, 0, 200, 1, algo, np.array([400]), n=200)
#    #plot2d_reg(4, 7.5, 0, 0, 200, 1, algo, np.array([400]), n=200)
#    #plot2d_reg(4, 7.5, 0, 0, 200, 1, algo, np.array([400]), n=200)
#    #plot2d_reg(4, 7.5, 0, 0, 200, 1, algo, np.array([600]), n=200)
#    #plot2d_reg(4, 7.5, 0, 0, 200, 1, algo, np.array([600]), n=200)
#    #plot2d_reg(4, 7.5, 0, 0, 200, 1, algo, np.array([600]), n=200)
#
#
#    plot1d_cov_gp(4, 7.5, 0, algo, np.array([0]), 200)
#    plot1d_cov_gp(4, 7.5, 0, algo, np.array([50]), 200)
#    plot1d_cov_gp(4, 7.5, 0, algo, np.array([100]), 200)
#    plot1d_cov_gp(4, 7.5, 0, algo, np.array([200]), 200)
#    plot1d_cov_gp(0, 200, 1, algo, np.array([5]), 200)
#    plot1d_cov_gp(0, 200, 1, algo, np.array([6]), 200)
#    plot1d_cov_gp(0, 200, 1, algo, np.array([7.5]), 200)
#    
#    
    

    