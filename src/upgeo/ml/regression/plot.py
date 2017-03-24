'''
Created on Feb 24, 2012

@author: marcel
'''

import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt


import upgeo.util.metric as metric

def plot_resid(data, algo, xidx, data_filter=None, target_filter=None, xlabel=None):
    X = data[0]
    y = data[1]
    
    Xn =  data_filter.process(X, True) if data_filter else X 
    
    yfit, var = algo.predict(Xn, ret_var=True)
    if target_filter:
        yfit = np.squeeze(target_filter.invprocess(yfit[:,np.newaxis]))
    
    resid = (y-yfit)#/np.sqrt(var)
    print 'var={0}'.format(var)
    
    plt.figure()
    if xidx > -1:
        plt.plot(X[:,xidx], resid, 'bs') 
        plt.plot([np.floor(np.min(X[:,xidx])),np.ceil(np.max(X[:,xidx]))],[0,0], color='k', linestyle='-')
    else:
        plt.plot(y, resid, 'bs')
        plt.plot([np.floor(np.min(y)),np.ceil(np.max(y))],[0,0], color='k', linestyle='-')
        print 'minmax'
        print [np.floor(np.min(y)),np.ceil(np.max(y))]
    plt.xlabel(xlabel)
    plt.show()
    
def plot_qq(data, algo, data_filter, target_filter):
    X = data[0]
    y = data[1]
    
    if data_filter:
        Xn = data_filter.process(X, True) 
    yfit = algo.predict(Xn)
    if target_filter:
        yfit = np.squeeze(target_filter.invprocess(yfit[:,np.newaxis]))
    
    xlim = np.floor(min(np.min(y), np.min(yfit)))
    ylim = np.ceil(max(np.max(y), np.max(yfit)))+1
    
    plt.figure()
    plt.plot(np.arange(xlim,ylim),np.arange(xlim,ylim), color='k', linestyle='-')
    plt.plot(yfit, y, 'bs')
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.show()    

def plot1d_reg(xmin, xmax, xidx, algo, z=None, n=100, 
               data_filter=None, target_filter=None, 
               data=None, data_sigma=None, 
               xlabel=None, ylabel=None):
    '''
        xmin - min value on the x-axis
        xmax - max value on the x-axis
        xidx - index of the x value correspon
        algo - regression model
        z    - constant data vector
        n    - number of points
        
    '''
    if xmin >= xmax:
        raise ValueError('xmin must be smaller than xmax.')
    
    likel = algo.log_likel
    params = algo.hyperparams
    #params = np.array([])

    x = np.linspace(xmin, xmax, num=n)
    x = x[:,np.newaxis]
    if z is None:
        X = x
    else:
        m = len(z)
        Z = np.tile(z, (n,1))
        X = np.hstack((Z[:,0:xidx],x,(Z[:,xidx:m])))
        
    if data_filter:
        X = data_filter.process(X, True)
        
    yfit, var = algo.predict(X, ret_var=True)
    #yfit = np.squeeze(yfit)
    #BUG: if we are not using a linear filter of the predicted value yfit 
    #then the corresponding becomes meaningless. Thus we have to  
    #specify explictly a filter for the predictive variance.
    if target_filter:
        yfit = np.squeeze(target_filter.invprocess(yfit[:,np.newaxis]))
    sd = np.sqrt(var)
    plt.figure()
        
    #plot mean and std error area
    plt.plot(x, yfit, color='k', linestyle='-')
    f = np.r_[yfit+2.0*sd, (yfit-2.0*sd)[::-1]]
    plt.fill(np.r_[x, x[::-1]], f, edgecolor='w', facecolor='#d3d3d3')
    
    #plot training samples       
    if data is not None:
        Xtrain = data[0]
        ytrain = data[1]
        yfit = algo.predict(Xtrain)
        mse = metric.mspe(ytrain, yfit)
        
        if data_sigma and z is not None:
            #extract the datapoints for plotting in the pre-specified range
            mask = np.ones(X.shape[1], 'bool')
            mask[xidx] = 0
            if data_filter:
                #normalize z vector, by definiton training data should be normalized
                #this is done by just replacing z from first normalized vector of X
                z = X[0,mask]
            
            idx = ~np.any(np.abs(Xtrain[:,mask]-z) > data_sigma,1)
            if data_filter:
                Xtrain = data_filter.invprocess(Xtrain)
            if target_filter:
                ytrain = np.squeeze(target_filter.invprocess(ytrain[:,np.newaxis]))
            Xtrain = Xtrain[idx]
            ytrain = ytrain[idx]
        else:
            if data_filter:
                Xtrain = data_filter.invprocess(Xtrain)
            if target_filter:
                ytrain = np.squeeze(target_filter.invprocess(ytrain[:,np.newaxis]))
            
            
        plt.plot(Xtrain[:,xidx], ytrain, 'rs')
        t = 'Log likelihood: {0}\n params: {1}\n mse: {2}'.format(likel, np.exp(params), mse)
    else:
        t = 'Log likelihood: {0}\n params: {1}'.format(likel, np.exp(params))
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #print plot title
    plt.suptitle(t)
    plt.show()
    
def plot2d_reg(xmin, xmax, xidx, ymin, ymax, yidx, algo, z=None, n=100, 
               data_filter=None, target_filter=None, data=None, data_sigma=None,
               xlabel=None, ylabel=None):
    
    if xmin >= xmax:
        raise ValueError('xmin must be smaller than xmax.')
    if ymin >= ymax:
        raise ValueError('ymin must be smaller than ymax.')
    if xidx == yidx:
        raise ValueError('xidx must be different to yidx.')
    likel = algo.log_likel
    params = algo.hyperparams
    
    x = np.linspace(xmin, xmax, num=n)
    y = np.linspace(ymin, ymax, num=n)
    
    xi, yi = np.meshgrid(x, y)
    xxi = xi.ravel()
    yyi = yi.ravel()
    xxi = xxi[:,np.newaxis]
    yyi = yyi[:,np.newaxis]
    
    if z is None:
        if xidx < yidx:
            Z = np.c_[xi.ravel(), yi.ravel()]
        else:
            Z = np.c_[yi.ravel(), xi.ravel()]
    else:
        m = len(z)
        Z = np.tile(z, (xi.size,1))
        if xidx < yidx:
            Z = np.hstack((Z[:,0:xidx],xxi, Z[:,xidx:yidx-1], yyi, Z[:,yidx-1:m]))
        else:
            Z = np.hstack((Z[:,0:yidx],yyi, Z[:,yidx:xidx-1], xxi, Z[:,xidx-1:m]))
            
    if data_filter != None:
        Z = data_filter.process(Z, True)
    yfit, var = algo.predict(Z, ret_var=True)
    se = 2.0*np.sqrt(var)
    
    if target_filter:
        yfit = np.squeeze(target_filter.invprocess(yfit[:,np.newaxis]))
    yfit = yfit.reshape(n,n)
    se = se.reshape(n,n)
    
    cmap = cm.get_cmap('jet')
    
    plt.subplots_adjust(hspace=0.5)
    
    plt.subplot(121)
    plt.contourf(xi,yi,yfit,25,cmap=cmap)
    plt.colorbar()
    plt.subplot(122)
    plt.contourf(xi,yi,se,25,cmap=cmap)
    plt.colorbar()

    #plot training samples
    if data is not None:
        Xtrain = data[0]
        ytrain = data[1]
        yfit = algo.predict(Xtrain)
        mse = metric.mspe(ytrain, yfit) 

        Xtrain = data[0]
        ytrain = data[1]
        yfit = algo.predict(Xtrain)
        mse = metric.mspe(ytrain, yfit)
        
        if data_sigma and z is not None:
            #extract the datapoints for plotting in the pre-specified range
            mask = np.ones(Z.shape[1], 'bool')
            mask[xidx] = 0
            mask[yidx] = 0
            if data_filter:
                #normalize z vector, by definiton training data should be normalized
                #this is done by just replacing z from first normalized vector of X
                z = Z[0,mask]
            
            idx = ~np.any(np.abs(Xtrain[:,mask]-z) > data_sigma,1)
            if data_filter:
                Xtrain = data_filter.invprocess(Xtrain)
            if target_filter:
                ytrain = np.squeeze(target_filter.invprocess(ytrain[:,np.newaxis]))
            Xtrain = Xtrain[idx]
            ytrain = ytrain[idx]
        else:
            if data_filter:
                Xtrain = data_filter.invprocess(Xtrain)
            if target_filter:
                ytrain = np.squeeze(target_filter.invprocess(ytrain[:,np.newaxis]))


        #plt.plot(Xtrain[:,xidx], ytrain, 'rs')
        t = 'Log likelihood: {0}\n params: {1}\n mse: {2}'.format(likel, np.exp(params), mse)
    else:
        t = 'Log likelihood: {0}\n params: {1}'.format(likel, np.exp(params))
        
    #print plot title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(t)
    
    plt.show()


def plot1d_cov_gp(xmin, xmax, xidx, gp, z=None, n=100):
    
    likel = gp.log_likel
    params = gp.hyperparams
    kernel = gp.kernel
    
    x = np.linspace(xmin, xmax, num=n)
    x = x[:,np.newaxis]
    if z is None:
        X = x
    else:
        m = len(z)
        Z = np.tile(z, (n,1))
        X = np.hstack((Z[:,0:xidx],x,(Z[:,xidx:m])))

    
    cov = kernel(X)    
    
    
    cmap = cm.get_cmap('jet') 
    plt.imshow(cov, interpolation='nearest', cmap=cmap, extent=[xmin, xmax, xmax, xmin])
    t = 'Log likelihood: {0}\n{1}'.format(likel, np.exp(params))
    plt.title(t)
    plt.colorbar()
    plt.show()    