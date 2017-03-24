'''
Created on Aug 8, 2011

@author: marcel
'''

import numpy as np
import pylab as pl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def plot1d_gp(gp, xmin, xmax, n=100):
    '''
    validate 1 dimensionality
    '''
    if xmin >= xmax:
        raise ValueError('xmin must be smaller than xmax.')
    
    #test cases
    data = gp.training_set
    likel = gp.log_likel
    params = gp.hyperparams
    
    x = np.linspace(xmin, xmax, num=n)
    x = x[:,np.newaxis]
    mean, var = gp.predict(x, ret_var=True)
    mean = np.squeeze(mean)
    sd = np.sqrt(var)
    
    
    pl.figure()
    
    #plot mean and std error area
    pl.plot(x, mean, color='k', linestyle=':')
    f = np.r_[mean+2.0*sd, (mean-2.0*sd)[::-1]]
    pl.fill(np.r_[x, x[::-1]], f, edgecolor='w', facecolor='#d3d3d3')
    
    #plot training samples
    pl.plot(data[0], data[1], 'rs')
    
    #print plot title
    t = 'Log likelihood: {0}\n{1}'.format(likel, np.exp(params))
    pl.title(t)
    pl.show()
    
def plot1d_cov_gp(gp, xmin, xmax, n=100):
    
    likel = gp.log_likel
    params = gp.hyperparams
    kernel = gp.kernel
    
    x = np.linspace(xmin, xmax, num=n)
    x = x[:,np.newaxis]
    
    cov = kernel(x)    
    
    
    cmap = cm.get_cmap('jet') 
    plt.imshow(cov, interpolation='nearest', cmap=cmap)
    t = 'Log likelihood: {0}\n{1}'.format(likel, np.exp(params))
    plt.title(t)
    plt.colorbar()
    plt.show()

    
    
def gendata_1d(fun, min, max, n=100, sigma=1, noise_fun=None):
    '''
    '''
    if min > max:
        raise ValueError('min must be less than max')
    if sigma < 0:
        raise ValueError('sigma must be positive')
    
    x = np.random.uniform(min, max, n)
    noise = np.random.normal(0, sigma, n)
    if noise_fun:
        noise *= noise_fun(x)
    y = fun(x) + noise
    return (x,y)
    
    
def f1(x):
    x = np.ravel(np.asarray(x))
    return np.sin(x)/x

def f2(x):
    x = np.ravel(np.asarray(x))
    return 3.0*np.sin(x**2) + 2.0*np.sin(1.5*x+1)