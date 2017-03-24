'''
Created on Jun 19, 2011

@author: marcel
'''

import numpy as np

from upgeo.ml.multitask.regression.dpm import _norm, _normln, _pi, _piln, _marg,\
    _margln, _post, _postln
from upgeo.util.array import invweight, weight

if __name__ == '__main__':
    alpha = 21.25
    n = 2
    dl = np.exp(3.05635689537)
    dr = np.exp(3.05635689537)
    
    pi = np.exp(-10.5452316004)
    model_likel = np.exp(-402.116996831)
    margl = np.exp(-39.5715169253)
    margr = np.exp(-374.15387782)
    
    marg = np.exp(-412.365573138)
    
    w = np.array([np.exp(-77.9664965235), np.exp(-7.29790103606), np.exp(-2.37172466398), np.exp(0)])
    
    print 'norm={0}, ln norm={1} '.format(_norm(alpha, n, dl, dr), np.log(_norm(alpha, n, dl, dr)))
    print 'exp normln={0}, normln={1} '.format(np.exp(_normln(alpha, n, np.log(dl), np.log(dr))), _normln(alpha, n, np.log(dl), np.log(dr)))
    
    print '-------------------------------------------------------'
    
    print 'pi={0}, ln pi={1} '.format(_pi(alpha, n, dl, dr), np.log(_pi(alpha, n, dl, dr)))
    print 'exp piln={0}, piln={1} '.format(np.exp(_piln(alpha, n, np.log(dl), np.log(dr))), _piln(alpha, n, np.log(dl), np.log(dr)))
    
    print '-------------------------------------------------------'
    
    print 'marg={0}, ln marg={1}'.format(_marg(pi, model_likel, margl, margr), np.log(_marg(pi, model_likel, margl, margr)))
    print 'exp margln={0}, margln={1}'.format(np.exp(_margln(np.log(pi), np.log(model_likel), np.log(margl), np.log(margr))), _margln(np.log(pi), np.log(model_likel), np.log(margl), np.log(margr)))
    
    print '-------------------------------------------------------'

    print 'post={0}, ln post={1}'.format(_post(pi, model_likel, marg), np.log(_post(pi, model_likel, marg)))
    print 'exp postln={0}, postln={1}'.format(np.exp(_postln(np.log(pi), np.log(model_likel), np.log(marg))), _postln(np.log(pi), np.log(model_likel), np.log(marg)))

    print '-------------------------------------------------------'

    print 'invweights={0}'.format(invweight(w))
    print 'ln invweights={0}'.format(np.log(invweight(w)))
    
    print '-------------------------------------------------------'
    
    print 'weights={0}'.format(weight(w))
    print 'ln weights={0}'.format(np.log(weight(w)))
    
    print 'sum w={0}'.format(np.sum(w))
    print 'sum weights={0}'.format(np.sum(weight(w)))
    
    