'''
Created on Mar 28, 2012

@author: marcel
'''
import numpy as np

import matplotlib.cm as cm
import matplotlib.mpl as mpl
import matplotlib.pyplot as plt

import os
import shutil

from upgeo.eval.multitask.base import load_data
from upgeo.ml.multitask.regression.bhc import BHCGPRegression,\
    BHCRobustRegression, RobustBHCGPRegression
from upgeo.ml.regression.np.kernel import SEKernel, SqConstantKernel,\
    LinearKernel, NoiseKernel, ARDSEKernel
from upgeo.ml.multitask.base import flatten_data
from upgeo.util import metric
from upgeo.ml.regression.bayes import EMBayesRegression, RobustBayesRegression
from upgeo.ml.regression.plot import plot2d_reg
from upgeo.ml.regression.np.gp import GPRegression
from upgeo.ml.regression.np.infer import ExactInference

def generate_bhc_plots(xmin, xmax, xidx, ymin, ymax, yidx, bhc_algo, z=None, n=100, dir=None):    
    
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
            Z = np.hstack((Z[:,0:yidx],yi.ravel(), Z[:,yidx:xidx], yi.ravel(), Z[:,xidx:m]))

    
    data = []
    min_mean = np.Inf
    max_mean = -np.Inf
    min_se = np.Inf
    max_se = -np.Inf
    cluster_tree = bhc_algo._task_tree
    for cl in cluster_tree:
        descr = cl.descr
        reg_model = cl.model
        
        mean, var = reg_model.predict(Z, ret_var=True)
        se = 2.0*np.sqrt(var)
        
        data.append((descr, mean, se))
        min_mean = np.min(np.array([min_mean, np.min(mean)]))
        max_mean = np.max(np.array([max_mean, np.max(mean)]))
        min_se = np.min(np.array([min_se, np.min(se)]))
        max_se = np.max(np.array([max_se, np.max(se)]))
        
    cmap = cm.get_cmap('jet')
    
    fname = None
    if not dir is None:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            del_dir_content(dir)
    
        fname_fmt = os.path.join(dir, 'plt_predict_pga_cluster{0}.png')
    for (descr, mean, se) in data:
        print 'descr={0}'.format(descr)
        mean = mean.reshape(n,n)
        se = se.reshape(n,n)
        
        if not dir is None:
            fname = fname_fmt.format(descr)
        plot2d_contour(xi, yi, mean, se, cmap, min_mean, max_mean, min_se, max_se, fname)
    
    
    if not dir is None:
        fname = os.path.join(dir, 'cb_pga.png')
    plot_colorbar(cmap, min_mean, max_mean, 'log PGA', fname)
    if not dir is None:
        fname = os.path.join(dir, 'cb_se.png')
    plot_colorbar(cmap, min_se, max_se, 'standard error', fname)
    
def plot2d_contour(X, Y, Z1, Z2, cmap, z1min, z1max, z2min, z2max, fname=None):

    plt.figure(None, figsize=(4,2))
    plt.subplots_adjust(wspace=0.5)
    plt.subplots_adjust(hspace=0.05)
    
    ax = plt.subplot(121)
    plt.contourf(X,Y,Z1,25,cmap=cmap)
    plt.clim(z1min,z1max)
    #plt.colorbar()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    

    
    ax = plt.subplot(122)
    plt.contourf(X,Y,Z2,25,cmap=cmap)
    #plt.colorbar()
    plt.clim(z2min, z2max)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
        
    #print plot title
    #plt.suptitle(t)

    if fname == None:
        plt.show()
    else:
        plt.savefig(fname, transparent=True)
        
    plt.clf()
    

def plot_colorbar(cmap, vmin, vmax, label=None, fname=None):
    fig = plt.figure(None, figsize=(8,1))
    ax = fig.add_axes([0.05, 0.65, 0.9, 0.15])
    
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb1.set_label(label)

    if fname == None:
        plt.show()
    else:
        plt.savefig(fname, transparent=True)
    plt.clf()


def del_dir_content(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


if __name__ == '__main__':
    
    homedir = os.path.expanduser('~')
    basedir = os.path.join(homedir, 'datasets/multilevel/eueq/viz_egu2')
    
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    else:
        del_dir_content(basedir)

    
    filename = '/home/marcel/datasets/multilevel/eueq/viz_eueq_nico_norm_pga.csv'
    task_key = 'region'
    task_fields = ['region']
    target_field = 'pga' 
    
    X,Y,Z = load_data(filename, task_key, task_fields, target_field)
    n = len(Z)

    kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
    #kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(1))
    #kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(1))
    #kernel = ARDSEKernel(np.log(1)*np.ones(13), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
    #kernel = ARDSEKernel(np.log(1)*np.ones(6), np.log(1)) + NoiseKernel(np.log(1))
    #algo = BHCGPRegression(np.log(10.5), kernel)
    algo = RobustBHCGPRegression(np.log(10.5), kernel)
    d = X[0].shape[1]
    #hyperparams = RobustBayesRegression.wrap(np.log(0.1), np.log(0.1), np.eye(d+1).ravel())
    #hyperparams = RobustBayesRegression.wrap(np.log(1), np.log(1), np.eye(d+1).ravel())
    #hyperparams = RobustBayesRegression.wrap(np.log(0.000000001), np.log(0.000000001), np.eye(d+1).ravel())
    #algo = BHCRobustRegression(np.log(10.5), hyperparams)         
    algo.fit(X,Y,Z)
    result = algo.predict_by_task(X, np.array(xrange(n)))
    #algo = EMBayesRegression()
    #algo = GPRegression(kernel, infer_method=ExactInference)
    #algo = RobustBayesRegression(np.log(0.1), np.log(0.1), np.eye(d+1))
    #algo.fit(flatten_data(X), flatten_data(Y))
    #result = algo.predict(flatten_data(X))
    
    y = flatten_data(Y)
    yfit = result = flatten_data(result)
    #yfit = result
    
    mse = metric.mspe(y, yfit)
    sse = metric.tspe(y, yfit)
    ymean =  np.mean(y)
    sst = metric.tse(y, ymean)
    r2 = 1-sse/sst
    
    print 'mse={0}'.format(mse)
    #print 'likel={0}'.format(algo.log_likel)
    

    generate_bhc_plots(-2, 4, 0, -1.5, 2, 3, algo, np.array([0,0]), n=50, dir=basedir)
    #generate_bhc_plots(-2, 4, 0, -1.5, 2, 6, algo, np.array([0,1,0,0,0]), n=50, dir=os.path.join(basedir, 'mnorm'))
    #generate_bhc_plots(-2, 4, 0, -1.5, 2, 6, algo, np.array([0,0,1,0,0]), n=50, dir=os.path.join(basedir, 'mrev'))
    #generate_bhc_plots(-2, 4, 0, -1.5, 2, 6, algo, np.array([0,0,0,1,0]), n=50, dir=os.path.join(basedir, 'mss'))
    #generate_bhc_plots(-2, 4, 0, -1.5, 2, 3, algo, np.array([]), n=50, dir=basedir)
    algo.marshal_xml(os.path.join(basedir, 'cluster_tree.xml'))

    
    