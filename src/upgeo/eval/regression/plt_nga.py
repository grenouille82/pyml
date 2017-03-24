'''
Created on Jul 17, 2012

@author: marcel
'''
import numpy as np
import matplotlib.pyplot as plt

from upgeo.eval.multitask.base import load_data
from upgeo.ml.multitask.base import flatten_data
from upgeo.filter import BagFilter, StandardizeFilter, FunctionFilter,\
    MeanShiftFilter, CompositeFilter
from upgeo.ml.regression.np.kernel import ARDSEKernel, SqConstantKernel,\
    LinearKernel, NoiseKernel, SEKernel, ARDLinearKernel, PolynomialKernel,\
    MaternKernel, MaskedFeatureKernel, ARDPolynomialKernel, ARDRBFKernel,\
    RBFKernel, ARDSELinKernel
from upgeo.ml.regression.np.gp import GPRegression, SparseGPRegression
from upgeo.ml.regression.np.infer import ExactInference, OnePassInference,\
    FITCExactInference
from upgeo.ml.regression.np.selector import RandomSubsetSelector, KMeansSelector
from upgeo.ml.regression.bayes import EMBayesRegression
from upgeo.util.metric import mspe
from upgeo.ml.regression.plot import plot1d_reg, plot2d_reg, plot_resid, plot_qq
from upgeo.ml.regression.np.mean import BiasedLinearMean, MaskedFeatureMean,\
    HiddenMean
from upgeo.ml.regression.linear import LSRegresion

    
def gen_dist_data(min, max, z, mag_idxdist_map):
    '''
    '''
    
def gen_mag_data(min, max, dist_map):
    '''
    '''
    
def gen_param_data(min, max, dist_map):
    '''
    '''
    
def gen_data(xmin, xmax, step, z, xidx):
    x = np.arange(xmin, xmax, step)

if __name__ == '__main__':
    
    #load and preprocess data 
    #data_fname = '/home/marcel/datasets/multilevel/nga/pooled/mainshock/viz_nga_mainshock1.csv'
    #data_fname = '/home/marcel/datasets/multilevel/nga/pooled/mainshock/viz_nga_mainshock5.csv'
    #data_fname = '/home/marcel/datasets/multilevel/nga/pooled/50/viz_nga_pga50_1.csv'
    data_fname = '/home/marcel/datasets/multilevel/eusinan/viz_eu_data_200d_pga.csv'
    task_key = 'region'
    task_fields = ['region']
    target_field = 'pga' 
    #target_field = 'pgv'
    
    X,Y,Z = load_data(data_fname, task_key, task_fields, target_field)
    n = len(Z)    
    
    X = flatten_data(X)
    y = flatten_data(Y)

    jbd_trans_fun = lambda x: np.log(np.sqrt(x**2 + 12**2))
    jbd_inv_fun = lambda x: np.sqrt(np.exp(x)**2 - 12**2)
    
    
    event_idx = 1   #index of the event id row
    site_idx = 0    #index of the site id row
    
    #event_mask = [0,4]    #mask of the event features, which should be normalized
    #site_mask = [6]            #mask of the site features, which should be normalized
    #record_mask = [5]    #mask of the record features, which should be normalized
    #dist_mask =  [5]
    
    
    #event_mask = [0,3]    #mask of the event features, which should be normalized
    #site_mask = [5]            #mask of the site features, which should be normalized
    #record_mask = [4]    #mask of the record features, which should be normalized
    #dist_mask =  [4]
    
    #for eu datatset
    event_mask = [1,5]    #mask of the event features, which should be normalized
    site_mask = [6]            #mask of the site features, which should be normalized
    record_mask = [0]    #mask of the record features, which should be normalized
    dist_mask =  [0]

    
    #event_mask = [0]    #mask of the event features, which should be normalized
    #site_mask = [5]            #mask of the site features, which should be normalized
    #record_mask = [4]    #mask of the record features, which should be normalized
    #dist_mask =  [4]

    
    data_mask = np.ones(X.shape[1], 'bool')
    data_mask[event_idx] = data_mask[site_idx] = 0
    
    #periodic_mask = []  #mask of periodic features
    
    
    event_bag = X[event_idx]
    site_bag = X[site_idx]
    X = X[:,data_mask]
    event_filter = BagFilter(X[event_idx], StandardizeFilter(1,event_mask))
    site_filter = BagFilter(X[site_idx], StandardizeFilter(1,site_mask))
    record_filter = StandardizeFilter(1, record_mask)
    #dist_filter = FunctionFilter(np.log, np.exp, dist_mask)
    dist_filter = FunctionFilter(jbd_trans_fun, jbd_inv_fun, dist_mask)
    #periodic_filter = FunctionFilter(np.cos, periodic_mask)
    
    cov_filter = CompositeFilter([dist_filter, event_filter, site_filter, record_filter])
    #cov_filter = CompositeFilter([event_filter, site_filter, record_filter])
    target_filter = MeanShiftFilter()
    
    
    Xnorm = cov_filter.process(X)
    ynorm = np.squeeze(target_filter.process(y[:,np.newaxis]))
    
    
    #learn GP
    ll = (np.max(Xnorm,0)-np.min(Xnorm,0))
    ll[ll == 0] = 1e-4   
    
    #create kernel
    #kernel = SEKernel(np.log(np.mean(ll)), np.log(1)) + NoiseKernel(np.log(0.1))
    
    
    
    #kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = SEKernel(np.log(1), np.log(1))+SqConstantKernel(np.log(1))*LinearKernel()+ NoiseKernel(np.log(0.5))
    #kernel = RBFKernel(np.log(1), np.log(1)) +SqConstantKernel(np.log(1))*LinearKernel()+ NoiseKernel(np.log(0.5))
    kernel = ARDRBFKernel(np.log(1)*np.ones(len(ll)), np.log(1)) +SqConstantKernel(np.log(1))*LinearKernel()+ NoiseKernel(np.log(0.5))
    #kernel = ARDSEKernel(np.log(ll/2), np.log(1)) +SqConstantKernel(np.log(1))*LinearKernel()+ NoiseKernel(np.log(0.5))
    #kernel = ARDSEKernel(np.log(1)*np.ones(len(ll)), np.log(1)) +ARDLinearKernel(np.log(1)*np.ones(len(ll)), np.log(1))+ NoiseKernel(np.log(0.5))
    #kernel = ARDSELinKernel(np.log(ll/2), np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = ARDSEKernel(np.log(1)*np.ones(len(ll)), np.log(1)) +  PolynomialKernel(3, np.log(1), np.log(1)) + SqConstantKernel(np.log(1))*LinearKernel()+ NoiseKernel(np.log(0.5)) 
    #kernel = MaternKernel(1, np.log(1), np.log(1)) +  PolynomialKernel(3, np.log(1), np.log(1))+ NoiseKernel(np.log(0.5))
    #kernel = MaternKernel(1, np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = MaternKernel(1, np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = ARDSEKernel(np.log(1)*np.ones(len(ll)), np.log(1)) + ARDLinearKernel(np.log(1)*np.ones(len(ll)), np.log(1)) + NoiseKernel(np.log(0.5))
    rmodel = LSRegresion()
    rmodel.fit(Xnorm, ynorm)
    meanfct = BiasedLinearMean(rmodel.weights, rmodel.intercept)
    meanfct = HiddenMean(meanfct)
    #meanfct = BiasedLinearMean(np.zeros(len(ll)), 0)
    meanfct = None
    
    selector = KMeansSelector(15)
    #selector = RandomSubsetSelector(15)
    
    gp = GPRegression(kernel, meanfct, infer_method=ExactInference)
    #gp = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector, fix_inducing=False)
    gp.fit(Xnorm, ynorm)
    #dist_map =  np.loadtxt('/home/marcel/datasets/multilevel/nga/pooled/distances.csv', delimiter=',',skiprows=1)    
    
    linreg = EMBayesRegression(alpha0=1, beta0=1, weight_bias=True)
    linreg.fit(Xnorm, ynorm)
    
    
    data = (Xnorm, ynorm)
    
    #plot magnitude
    #z = np.array([0,1,0,5,20,600])
    #z = np.array([0.3,0,5,20,600])
    #z = np.array([0,1,0,20,270])
    z = np.array([20, 10, 1, 0, 0, 400])
    plot1d_reg(3, 8, 5, gp, z, 100, cov_filter, target_filter, data, 0.5, 'Mag', 'log PGA')
    plot1d_reg(3, 8, 5, linreg, z, 100, cov_filter, target_filter, data, 0.5, 'Mag', 'log PGA')
    z = np.array([20, 10, 0, 1, 0, 400])
    plot1d_reg(3, 8, 5, gp, z, 100, cov_filter, target_filter, data, 0.5, 'Mag', 'log PGA')
    plot1d_reg(3, 8, 5, linreg, z, 100, cov_filter, target_filter, data, 0.5, 'Mag', 'log PGA')
    z = np.array([20, 10, 0, 0, 1, 400])
    plot1d_reg(3, 8, 5, gp, z, 100, cov_filter, target_filter, data, 0.5, 'Mag', 'log PGA')
    plot1d_reg(3, 8, 5, linreg, z, 100, cov_filter, target_filter, data, 0.5, 'Mag', 'log PGA')
    #plot1d_reg(3, 8, 5, linreg, z, 100, cov_filter, None, data, 0.5,'Mag', 'log PGA')
    
    #plot distance
    #z = np.array([7.0,0,1,0,5,600])
    #z = np.array([7.0,0.3,0,5,600])
    #z = np.array([6.0,0,1,0,270])
    z = np.array([10, 0, 1, 0, 7, 400])
    plot1d_reg(0.0000, 200, 0, gp, z, 100, cov_filter, None, data,  0.5, 'RupDist', 'log PGA')
    plot1d_reg(0.0000, 200, 0, linreg, z, 100, cov_filter, None, data,  0.5, 'RupDist', 'log PGA')
    #plot1d_reg(0.0000, 200, 0, linreg, z, 100, cov_filter, None, data,  0.5, 'RupDist', 'log PGA')
    
    #plot vs30
    #z = np.array([6.0,0,1,0,5,20])
    z = np.array([6.0,0.3,0,5,20])
    #z = np.array([6.0,0,1,0,20])
    z = np.array([20, 10, 0, 1, 0, 7])
    plot1d_reg(0, 2500, 6, gp, z, 100, cov_filter, None, data,  0.5, 'Vs30', 'log PGA')
    #plot1d_reg(0, 2500, 6, linreg, z, 100, cov_filter, None, data,  0.5, 'Vs30', 'log PGA')
    
    #plot depth
    #z = np.array([6.0,0,1,0,20,270])
    #z = np.array([6.0,0.3,0,20,270])
    z = np.array([20, 0, 1, 0, 7, 400])
    plot1d_reg(0, 30, 1, gp, z, 100, cov_filter, None, data, 0.5,'Depth', 'log PGA')
    #plot1d_reg(0, 30, 2, linreg, z, 100, cov_filter, None, data, 0.5, 'Depth', 'log PGA')
    
    #z = np.array([6.0,0,5,20,270])
    #plot1d_reg(0, 1, 1, gp, z, 100, cov_filter, None, data, 0.5,'Dip', 'log PGA')
    #plot1d_reg(0, 1, 1, linreg, z, 100, cov_filter, None, data, 0.5, 'Dip', 'log PGA')
    
    #z = np.array([6.0,0.3,5,20,270])
    #plot1d_reg(-1, 1, 2, gp, z, 100, cov_filter, None, data, 0.5,'Dip', 'log PGA')
    #plot1d_reg(-1, 1, 2, linreg, z, 100, cov_filter, None, data, 0.5, 'Dip', 'log PGA')

    #plot mag vs dist
    #z = np.array([0,1,0,5,600])
    #z = np.array([0.3,0,5,600])
    z = np.array([10, 1, 0, 0, 400])
    #plot2d_reg(0.5, 200, 0, 3, 8, 5, gp, z, 50, cov_filter, target_filter, data, None, 'Rup Dist', 'Mag')
    #plot2d_reg(4.5, 8, 5, 0.5, 200, 0, gp, z, 50, cov_filter, target_filter, data, None, 'Mag', 'Rup Dist')
    z = np.array([10, 1, 0, 0, 400])
    #plot2d_reg(0.5, 200, 0, 3, 8, 5, gp, z, 50, cov_filter, target_filter, data, None, 'Rup Dist', 'Mag')
    plot2d_reg(4.5, 8, 5, 0.5, 200, 0, gp, z, 50, cov_filter, target_filter, data, None, 'Mag', 'Rup Dist')
    plot2d_reg(4.5, 8, 5, 0.5, 200, 0, linreg, z, 50, cov_filter, target_filter, data, None, 'Mag', 'Rup Dist')
    z = np.array([10, 0, 1, 0, 400])
    plot2d_reg(4.5, 8, 5, 0.5, 200, 0, gp, z, 50, cov_filter, target_filter, data, None, 'Mag', 'Rup Dist')
    plot2d_reg(4.5, 8, 5, 0.5, 200, 0, linreg, z, 50, cov_filter, target_filter, data, None, 'Mag', 'Rup Dist')
    z = np.array([10, 0, 0, 1, 400])
    plot2d_reg(4.5, 8, 5, 0.5, 200, 0, gp, z, 50, cov_filter, target_filter, data, None, 'Mag', 'Rup Dist')
    plot2d_reg(4.5, 8, 5, 0.5, 200, 0, linreg, z, 50, cov_filter, target_filter, data, None, 'Mag', 'Rup Dist')
    
    #z = np.array([0,1,0,5,600])
    #z = np.array([0.3,0,5,600])
    
    #plot2d_reg(4.5, 8, 5, 0.5, 200, 0, linreg, z, 50, cov_filter, target_filter, data, None, 'Mag', 'Rup Dist')
    
    plot_resid((X,y), gp, 0, cov_filter, target_filter, xlabel='JBDist')
    plot_resid((X,y), gp, 5, cov_filter, target_filter, xlabel='Mag')
    plot_resid((X,y), gp, 6, cov_filter, target_filter, xlabel='VS30')
    plot_resid((X,y), gp, 1, cov_filter, target_filter, xlabel='Depth')
    plot_resid((X,y), gp, -1, cov_filter, target_filter, xlabel='Target')
    plot_qq((X,y), gp, cov_filter, target_filter)
    
    plot_resid((X,y), linreg, 0, cov_filter, target_filter, xlabel='JBDist')
    plot_resid((X,y), linreg, 5, cov_filter, target_filter, xlabel='Mag')
    plot_resid((X,y), linreg, 6, cov_filter, target_filter, xlabel='VS30')
    plot_resid((X,y), linreg, 1, cov_filter, target_filter, xlabel='Depth')
    plot_resid((X,y), linreg, -1, cov_filter, target_filter, xlabel='Target')
    plot_qq((X,y), linreg, cov_filter, target_filter)
    
    
    