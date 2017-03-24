'''
Created on Aug 17, 2011

@author: marcel
'''

import numpy as np
import scipy as sp
import scipy.optimize as spopt
import scipy.special as sps

from upgeo.ml.regression.bayes import BayesEvidenceLikelihood,\
    EvidenceLikelihood
from scipy.linalg.decomp_cholesky import cho_solve
from numpy.core.numeric import array_str
from numpy.lib.twodim_base import tril

np.random.seed(23942874)

n = 100
d = 3

#X = np.array([[ 0.22283113, -2.68437278]])#,[ 0.67450033, 0.6392598 ], [-1.66874772, -2.63491929],[ 0.18726887, -0.06809102], [ 0.05295961,  1.17990371], [ 1.00059796, -2.41591204], [-0.16933247, -0.77339388], [ 0.42445738, -1.8778914 ], [-0.7516727,  -0.32769361], [ 0.09046034, -1.0223605 ]])
X = np.random.randn(n,d)
y = 1.5+2.0*X[:,0]-0.4*X[:,1] + np.random.randn(n)*0.5

#a = 2.5
#b = 1.5
#
#v = 20
#r = 0.5
##m = np.array([2,2])
##S = np.array([[0.5, 0.0], [0.0, 0.5]])
#m = np.mean(X,0)
#S = np.linalg.inv(np.cov(X.T))
#
#def f_alpha(alpha):
#    alpha = np.exp(alpha)
#    
#    D = np.diag(np.ones(d))
#    A = alpha*D + b*np.dot(X.T,X)
#    invA = np.linalg.inv(A)
#    
#    w = np.dot(np.dot(b*invA,X.T), y)
#    yhat = np.dot(X, w)
#    err = (alpha*np.dot(w,w) + b*np.dot(y-yhat, y-yhat))/2 
#    
#    _, ln_detA = np.linalg.slogdet(A)
#    return d/2.0*np.log(alpha) - ln_detA/2.0 - n/2.0 * np.log(2.0*np.pi) - err + n/2.0*np.log(b)
#
#def grad_alpha(alpha):
#    alpha = np.exp(alpha)
#    
#    D = np.diag(np.ones(d))
#    A = alpha*D + b*np.dot(X.T,X)
#    invA = np.linalg.inv(A)
#    invAA = np.dot(-invA, invA)
#    
#    w = np.dot(np.dot(b*invA,X.T), y)
#    yhat = np.dot(X, w)
#    w_prime = np.dot(np.dot(b*invAA,X.T), y)
#    err = (alpha*2.0*np.dot(w,w_prime)+np.dot(w,w) - b*2.0*np.dot(np.dot(X.T,(y-yhat)), w_prime))/2.0
#    
#    
#    return (d/(2.0*alpha) - np.trace(np.linalg.inv(A))/2.0 - err) * alpha
#
#def f_beta(beta):
#    beta = np.exp(beta)
#    
#    D = np.diag(np.ones(d))
#    A = a*D + beta*np.dot(X.T,X)
#    invA = np.linalg.inv(A)
#    
#    w = np.dot(np.dot(beta*invA,X.T), y)
#    yhat = np.dot(X, w)
#    err = (a*np.dot(w,w) + beta*np.dot(y-yhat, y-yhat))/2 
#    
#    _, ln_detA = np.linalg.slogdet(A)
#    return  n/2.0*np.log(beta) - ln_detA - n/2.0 * np.log(2.0*np.pi) - err #+ m/2.0*np.log(a) 
#
#def grad_beta(beta):
#    beta = np.exp(beta)
#    
#    D = np.diag(np.ones(d))
#    XX = np.dot(X.T,X)
#    A = a*D + beta*XX
#    invA = np.linalg.inv(A)
#    
#    w = np.dot(np.dot(beta*invA,X.T), y)
#    yhat = np.dot(X, w)
#    
#    w_prime = np.dot(np.dot(invA, X.T),y) - beta* np.dot(np.dot(np.dot(np.dot(invA, XX), invA), X.T), y)
#    err_prime = -2.0*np.dot(np.dot(X.T,(y-yhat)), w_prime)
#    err = (a*2.0*np.dot(w,w_prime) + np.dot((y-yhat),(y-yhat)) + beta*err_prime)/2.0
#       
#    return (n/(2.0*beta) - np.trace(np.dot(np.linalg.inv(A),XX)) - err)* beta
#
#def f_niw(m):
#    XX = np.dot(X.T,X)
#    mm = np.outer(m,m.T)
#    
#    r_prime = np.float64(r+n)
#    v_prime = np.float64(v+n)
#    
#    m_prime = (r*m + X)/np.float(r_prime) 
#    mm_prime = np.dot(m_prime.T, m_prime)
#    S_prime = S + XX + r*mm - r_prime*mm_prime
#    
#    detS = float(np.linalg.det(S))
#    detS_prime = float(np.linalg.det(S_prime))
#    
#    A = np.pi**(-n*d/2.0)
#    B = (r**(d/2.0) * detS**(v/2.0)) / (r_prime**(d/2.0) * detS_prime**(v_prime/2.0))
#    
#    C = 1.0
#    for i in xrange(d):
#        C *= sps.gamma((v_prime+1.0-i)/2.0) / sps.gamma((v+1.0-i)/2.0)
#    
#    l = A*B*C
#    print 'A={0},B={1},C={2}'.format(A,B,C)
#    return l
#
#def f_niw_heller(m):
#    mu = np.mean(X,0)
#    
#    x = np.sum(X,0)
#    xx = np.outer(x,x)
#    XX = np.dot(X.T,X)
#    mm = np.outer(m,m.T)
#    
#    Q = np.dot((X-mu).T, X-mu)
#    print 'Q={0}'.format(Q)
#    print mu
#    r_prime = np.float64(r+n)
#    v_prime = np.float64(v+n)
#    
#    m_prime = (r*m + X)/np.float(r_prime) 
#    mm_prime = np.dot(m_prime.T, m_prime)
#    
#    
#    S_prime = S + Q + r*n/(r+n)*np.outer(mu-m,mu-m)
#    
#    detS = float(np.linalg.det(S))
#    detS_prime = float(np.linalg.det(S_prime))
#    
#    A = np.pi**(-n*d/2.0) #* (r/(n+r))**(d/2.0)
#    B = (r**(d/2.0) * detS**(v/2.0)) / (r_prime**(d/2.0) * detS_prime**(v_prime/2.0))
#    C = 1.0
#    for i in xrange(d):
#        C *= sps.gamma((v_prime+1.0-i)/2.0) / sps.gamma((v+1.0-i)/2.0)
#    
#    l = A*B*C
#    print 'A={0},B={1},C={2}'.format(A,np.log(B),C)
#    return l
#    
#def f_log_niw(m):
#    XX = np.dot(X.T,X)
#    mm = np.outer(m,m.T)
#    
#    r_prime = np.float64(r+n)
#    v_prime = np.float64(v+n)
#    
#    m_prime = (r*m + X)/np.float(r_prime) 
#    mm_prime = np.dot(m_prime.T, m_prime)
#    S_prime = S + XX + r*mm - r_prime*mm_prime
#    
#    
#    _, detS = np.linalg.slogdet(S)
#    _, detS_prime = np.linalg.slogdet(S_prime) 
#    
#    A = -n*d/2.0 * np.log(np.pi)
#    B = d/2.0*np.log(r) - d/2.0*np.log(r_prime) + v/2.0*detS - v_prime/2.0*detS_prime
#    
#    C = 0
#    for i in xrange(d):
#        C += sps.gammaln((v_prime+1.0-i)/2.0) - sps.gammaln((v+1.0-i)/2.0)
#    
#    l = A+B+C
#    print 'A={0},B={1},C={2}'.format(A,B,C)
#    return l
#
#def f_niw_gelman(m):
#    mu = np.mean(X,0)
#    
#    XX = np.dot(X.T,X)
#    mm = np.outer(m, m)
#    
#    Q = np.dot((X-mu).T, X-mu) #opt
#   
#    
#    r_prime = r+n
#    v_prime = v+n
#    m_prime = (r*m + n*mu) / (r_prime)
#    S_prime = S + Q + r*n*np.outer(mu-m, mu-m)/r_prime
#
#    L = np.linalg.cholesky(S)
#    L_prime = np.linalg.cholesky(S_prime)
#
#    detS = np.prod(np.diag(L)**2.0)
#    detS_prime = np.prod(np.diag(L_prime)**2.0)
#    
#    A = 1.0/np.pi**(n*d/2.0)
#    B = (r**(d/2.0) * detS**(v/2.0)) / (r_prime**(d/2.0) * detS_prime**(v_prime/2.0))
#    
#    C = 1.0
#    for i in xrange(d):
#        C *= sps.gamma((v_prime+1.0-i)/2.0) / sps.gamma((v+1.0-i)/2.0) 
#    
#    l = A*B*C
#    print 'A={0},B={1},C={2}'.format(A,B,C)
#    return l
#
#def f_log_niw_gelman(m):
#    mu = np.mean(X,0)
#    
#    XX = np.dot(X.T,X)
#    mm = np.outer(m, m)
#    
#    Q = np.dot((X-mu).T, X-mu) #opt
#    
#    r_prime = r+n
#    v_prime = v+n
#    m_prime = (r*m + n*mu) / (r_prime)
#    S_prime = S + Q + r*n*np.outer(mu-m, mu-m)/r_prime
#
#    L = np.linalg.cholesky(S)
#    L_prime = np.linalg.cholesky(S_prime)
#
#    detS = 2.0*np.sum(np.log(np.diag(L)))
#    detS_prime = 2.0*np.sum(np.log(np.diag(L_prime)))
#    
#    A = -n*d*np.log(np.pi)/2.0
#    B = (d*np.log(r) + v*detS - d*np.log(r_prime) - v_prime*detS_prime) / 2.0
#    #B = d/2.0*np.log(r) - d/2.0*np.log(r_prime) + v/2.0*detS - v_prime/2.0*detS_prime
#    
#    C = 0.0
#    for i in xrange(d):
#        C += sps.gammaln((v_prime+1.0-i)/2.0) - sps.gammaln((v+1.0-i)/2.0) 
#    
#    
#    l = A+B+C
#    return l
#
#def gradm_log_niw(m):
#    mu = np.mean(X,0)
#    
#    XX = np.dot(X.T,X)
#    mm = np.outer(m, m)
#    
#    Q = np.dot((X-mu).T, X-mu) #opt
#    
#    r_prime = r+n
#    v_prime = v+n
#    m_prime = (r*m + n*mu) / (r_prime)
#    S_prime = S + Q + r*n*np.outer(mu-m, mu-m)/r_prime
#    
#    L = np.linalg.cholesky(S)
#    L_prime = np.linalg.cholesky(S_prime)
#    
#    grad = cho_solve((L_prime,1), (r*n)*(mu-m)/r_prime)*v_prime #inverse of cholesky decomposition
#  
#    return grad
#
#def fS_log_niw_gelman(S):
#    S = np.reshape(S, (d,d))
#    
#    mu = np.mean(X,0)
#    
#    XX = np.dot(X.T,X)
#    mm = np.outer(m, m)
#    
#    Q = np.dot((X-mu).T, X-mu) #opt
#    
#    r_prime = r+n
#    v_prime = v+n
#    m_prime = (r*m + n*mu) / (r_prime)
#    S_prime = S + Q + r*n*np.outer(mu-m, mu-m)/r_prime
#
#    L = np.linalg.cholesky(S)
#    L_prime = np.linalg.cholesky(S_prime)
#
#    detS = 2.0*np.sum(np.log(np.diag(L)))
#    #detS_prime = 2.0*np.sum(np.log(np.diag(L_prime)))
#    detS_prime = np.log(np.linalg.det(S_prime))
#    print 'detS'
#    print detS_prime
#    print detS_prime
#    
#    
#    A = -n*d*np.log(np.pi)/2.0
#    B = (d*np.log(r) + v*detS - d*np.log(r_prime)- v_prime*detS_prime) / 2.0
#    B = (d*np.log(r) - d*np.log(r_prime)- v_prime*detS_prime) / 2.0
#    B =  -(v_prime*detS_prime) / 2.0
#    #B = d/2.0*np.log(r) - d/2.0*np.log(r_prime) + v/2.0*detS - v_prime/2.0*detS_prime
#    
#    C = 0.0
#    for i in xrange(d):
#        C += sps.gammaln((v_prime+1.0-i)/2.0) - sps.gammaln((v+1.0-i)/2.0) 
#    
#    
#    l = A+B+C
#    return l
#
#
#def gradS_log_niw(S):
#    S = np.reshape(S, (d,d))
#    mu = np.mean(X,0)
#    
#    XX = np.dot(X.T,X)
#    mm = np.outer(m, m)
#    
#    Q = np.dot((X-mu).T, X-mu) #opt
#    
#    r_prime = r+n
#    v_prime = v+n
#    m_prime = (r*m + n*mu) / (r_prime)
#    S_prime = S + Q + r*n*np.outer(mu-m, mu-m)/r_prime
#
#    
#
#    invS = np.linalg.inv(S)
#    invS_prime = np.linalg.inv(S_prime)
#    
#    grad = v/2.0*invS - v_prime/2.0*invS_prime.T
#    grad =  - v_prime/2.0*invS_prime
#    return grad.ravel()
#
#
#def f_old(S):
#    S = np.reshape(S, (2,2))
#    mu = np.mean(X,0)
#    v_prime = v+n
#    Q = np.dot((X-mu).T, X-mu)
#    D = S +Q+np.outer(m,m)
#    L = np.linalg.cholesky(D)
#    detL = 2.0*np.sum(np.log(np.diag(L)))
#    #return -detL*v_prime/2.0
#    return -2.0*np.sum(np.log(np.diag(L)))
#
#def g_old(S):
#    v_prime = v+n
#    mu = np.mean(X,0)
#    S = np.reshape(S, (2,2))
#    Q = np.dot((X-mu).T, X-mu)
#    D = S +Q+np.outer(m,m)
#    L = np.linalg.cholesky(D)
#    detL = 2.0*np.sum(np.log(np.diag(L)))
#    #return -cho_solve((L, 1), np.eye(d))*v_prime/2.0
#    return -cho_solve((L, 1), np.eye(d)) 
#
#def f(L):
#    L = np.tril(np.reshape(L, (2,2)))
#    
#    v_prime = v+n
#    mu = np.mean(X,0)
#    S = np.dot(L,L.T)
#    
#    Q = S+np.dot((X-mu).T, X-mu)+np.outer(m,m)
#    LQ = np.linalg.cholesky(Q)
#    
#    return (2.0*np.sum(np.log(np.diag(L)))*v - 2*np.sum(np.log(np.diag(LQ)))*v_prime) / 2.0
#
#def g(L):
#    L = np.tril(np.reshape(L, (2,2)))
#    
#    v_prime = v+n
#    mu = np.mean(X,0)
#    S = np.dot(L,L.T)
#    
#    Q = S+np.dot((X-mu).T, X-mu)+np.outer(m,m)
#    LQ = np.linalg.cholesky(Q)
#    
#    #return -2*cho_solve((L, 1), L)
#    return np.tril(cho_solve((L,1),L)*v - cho_solve((LQ,1), L)*v_prime)
#    #return -2*np.dot(np.linalg.inv(Q),L)
#    
#
#
#def fr_log_niw_gelman(r):
#    mu = np.mean(X,0)
#    
#    XX = np.dot(X.T,X)
#    mm = np.outer(m, m)
#    
#    Q = np.dot((X-mu).T, X-mu) #opt
#    
#    r_prime = r+n
#    v_prime = v+n
#    m_prime = (r*m + n*mu) / (r_prime)
#    S_prime = S + Q + r*n*np.outer(mu-m, mu-m)/r_prime
#
#    L = np.linalg.cholesky(S)
#    L_prime = np.linalg.cholesky(S_prime)
#
#    detS = 2.0*np.sum(np.log(np.diag(L)))
#    detS_prime = 2.0*np.sum(np.log(np.diag(L_prime)))
#    
#    A = -n*d*np.log(np.pi)/2.0
#    B = (d*np.log(r) + v*detS - d*np.log(r_prime) - v_prime*detS_prime) / 2.0
#    #B = d/2.0*np.log(r) - d/2.0*np.log(r_prime) + v/2.0*detS - v_prime/2.0*detS_prime
#    
#    C = 0.0
#    for i in xrange(d):
#        C += sps.gammaln((v_prime+1.0-i)/2.0) - sps.gammaln((v+1.0-i)/2.0) 
#    
#    
#    l = A+B+C
#    return l
#
#
#def gradr_log_niw(r):
#    mu = np.mean(X,0)
#    
#    XX = np.dot(X.T,X)
#    mm = np.outer(m, m)
#    
#    Q = np.dot((X-mu).T, X-mu) #opt
#    
#    r_prime = r+n
#    v_prime = v+n
#    m_prime = (r*m + n*mu) / (r_prime)
#    S_prime = S + Q + r*n*np.outer(mu-m, mu-m)/r_prime
#    D = n*np.outer(mu-m, mu-m)/(n+r) - n*r*np.outer(mu-m, mu-m)/(n+r)**2.0
#    
#    L = np.linalg.cholesky(S)
#    L_prime = np.linalg.cholesky(S_prime)
#    
#    
#    
#    #grad = d/(2*r) - d/(2*(n+r))- v_prime/2* np.trace(np.dot(np.linalg.inv(S_prime), D))
#    grad = d/(2*r) - d/(2*(n+r))- v_prime/2* np.trace(cho_solve((L_prime,1), D))
#    return grad
#
#
#def fv_log_niw_gelman(v):
#    mu = np.mean(X,0)
#    
#    XX = np.dot(X.T,X)
#    mm = np.outer(m, m)
#    
#    Q = np.dot((X-mu).T, X-mu) #opt
#    
#    r_prime = r+n
#    v_prime = v+n
#    m_prime = (r*m + n*mu) / (r_prime)
#    S_prime = S + Q + r*n*np.outer(mu-m, mu-m)/r_prime
#
#    L = np.linalg.cholesky(S)
#    L_prime = np.linalg.cholesky(S_prime)
#
#    detS = 2.0*np.sum(np.log(np.diag(L)))
#    detS_prime = 2.0*np.sum(np.log(np.diag(L_prime)))
#    
#    A = -n*d*np.log(np.pi)/2.0
#    B = (d*np.log(r) + v*detS - d*np.log(r_prime) - v_prime*detS_prime) / 2.0
#    #B = d/2.0*np.log(r) - d/2.0*np.log(r_prime) + v/2.0*detS - v_prime/2.0*detS_prime
#    
#    C = 0.0
#    for i in xrange(d):
#        C += sps.gammaln((v_prime+1.0-i)/2.0) - sps.gammaln((v+1.0-i)/2.0) 
#    
#    
#    l = A+B+C
#    return l
#
#
#def gradv_log_niw(v):
#    mu = np.mean(X,0)
#    
#    XX = np.dot(X.T,X)
#    mm = np.outer(m, m)
#    
#    Q = np.dot((X-mu).T, X-mu) #opt
#    
#    r_prime = r+n
#    v_prime = v+n
#    m_prime = (r*m + n*mu) / (r_prime)
#    S_prime = S + Q + r*n*np.outer(mu-m, mu-m)/r_prime
#    D = n*np.outer(mu-m, mu-m)/(n+r) - n*r*np.outer(mu-m, mu-m)/(n+r)**2.0
#    
#    L = np.linalg.cholesky(S)
#    L_prime = np.linalg.cholesky(S_prime)
#    
#    detS = 2.0*np.sum(np.log(np.diag(L)))
#    detS_prime = 2.0*np.sum(np.log(np.diag(L_prime)))
#    
#    C = 0.0
#    for i in xrange(d):
#        C += (sps.digamma((v_prime+1.0-i)/2.0) - sps.digamma((v+1.0-i)/2.0))/2.0 
#    
#    grad = (detS-detS_prime)/2.0 + C
#    return grad
#
#def h1(m,S):
#    return np.dot(np.dot(m,np.linalg.inv(S)),m)
#
#def h2(m,S):
#    L = np.linalg.cholesky(S)
#    return np.sum(np.dot(m,np.linalg.inv(L.T))**2)
#
#def h3(m,S):
#    d = len(S)
#    L = np.linalg.cholesky(S)
#    return np.dot(np.dot(m,cho_solve((L, 1), np.eye(d))),m)
#
#def h4(m,S):
#    d = len(S)
#    L = np.linalg.cholesky(S)
#    return np.sum(np.dot(m, np.linalg.solve(L.T, np.eye(d)))**2)
#
#def h5(m,S):
#    G = np.outer(m,m)
#    return np.trace(np.dot(np.linalg.inv(S), G))
#
#def h6(m,S):
#    G = np.outer(m,m)
#    L = np.linalg.cholesky(S)
#    return np.trace(cho_solve((L, 1), G))



a0 = 1e-16
b0 = 3.0
V0 = np.eye(d)#*0.5

def nig_likel_a(a):
    a = np.exp(a)
    
    n = len(X)
    
    XX = np.dot(X.T,X)
    yy = np.dot(y,y)
    Xy = np.dot(X.T,y)
    
    #posteriors
    LV = np.linalg.cholesky(V0)
    Vn = V0 + XX
    LVn = np.linalg.cholesky(Vn)
    mn = cho_solve((LVn, 1), Xy)
    an = a + n/2.0
    bn = b0 + (yy - np.sum(np.dot(mn, LVn)**2))/2.0
    
    
    A = np.sum(np.log(np.diag(LV))) - np.sum(np.log(np.diag(LVn)))
    B = a*np.log(b0) - an*np.log(bn)
    C = sps.gammaln(an) - sps.gammaln(a)
    likel = A + B + C - 2.0/n*np.log(2.0*np.pi)
    return likel

def nig_grad_a(a):
    a = np.exp(a)
    
    n = len(X)
    
    XX = np.dot(X.T,X)
    yy = np.dot(y,y)
    Xy = np.dot(X.T,y)
    
    #posteriors
    LV = np.linalg.cholesky(V0)
    Vn = V0 + XX
    LVn = np.linalg.cholesky(Vn)
    mn = cho_solve((LVn, 1), Xy)
    an = a + n/2.0
    bn = b0 + (yy - np.sum(np.dot(mn, LVn)**2))/2.0
    
    #gradients
    B = np.log(b0) - np.log(bn)
    C = sps.digamma(an) - sps.digamma(a)
    
    grad = (B+C) * a
    return grad
    
def nig_likel_b(b):
    b = np.exp(b)
    
    n = len(X)
    
    XX = np.dot(X.T,X)
    yy = np.dot(y,y)
    Xy = np.dot(X.T,y)
    
    #posteriors
    LV = np.linalg.cholesky(V0)
    Vn = V0 + XX
    LVn = np.linalg.cholesky(Vn)
    mn = cho_solve((LVn, 1), Xy)
    an = a0 + n/2.0
    bn = b + (yy - np.sum(np.dot(mn, LVn)**2))/2.0
    
    
    A =  np.sum(np.log(np.diag(LV))) - np.sum(np.log(np.diag(LVn)))
    B = a0*np.log(b) - an*np.log(bn)
    C = sps.gammaln(an) - sps.gammaln(a0)
    likel = A + B + C - 2.0/n*np.log(2.0*np.pi)
    return likel

def nig_grad_b(b):
    b = np.exp(b)
    
    n = len(X)
    
    XX = np.dot(X.T,X)
    yy = np.dot(y,y)
    Xy = np.dot(X.T,y)
    
    #posteriors
    LV = np.linalg.cholesky(V0)
    Vn = V0 + XX
    LVn = np.linalg.cholesky(Vn)
    mn = cho_solve((LVn, 1), Xy)
    an = a0 + n/2.0
    bn = b + (yy - np.sum(np.dot(mn, LVn)**2))/2.0
    
    #gradients
    B = a0/b - an/bn
    grad = B * b
    return grad

def nig_likel_V(V):
    V = unwrap(V)
    
    n = len(X)
    
    XX = np.dot(X.T,X)
    yy = np.dot(y,y)
    Xy = np.dot(X.T,y)
    
    #posteriors
    #print V
    LV = np.linalg.cholesky(V)
    Vn = V + XX
    LVn = np.linalg.cholesky(Vn)
    mn = cho_solve((LVn, 1), Xy)
    an = a0 + n/2.0
    bn = b0 + (yy - np.sum(np.dot(mn, LVn)**2))/2.0
    #bn = b0 + (yy - np.dot(np.dot(mn, np.eye(d)), mn))/2.0
    #bn = np.dot(mn, mn)
    
    A =  np.sum(np.log(np.diag(LV))) - np.sum(np.log(np.diag(LVn)))
    B = a0*np.log(b0) - an*np.log(bn)
    C = sps.gammaln(an) - sps.gammaln(a0)
    likel = A + B + C - 2.0/n*np.log(2.0*np.pi)
    #likel = A + B
    return likel

def nig_likel_L(L):
    L = unwrap(L)
    
    n = len(X)
    
    XX = np.dot(X.T,X)
    yy = np.dot(y,y)
    Xy = np.dot(X.T,y)
    
    V = np.dot(L,L.T)
    
    #posteriors
    #print V
    Vn = V + XX
    Ln = np.linalg.cholesky(Vn)
    mn = cho_solve((Ln, 1), Xy)
    an = a0 + n/2.0
    bn = b0 + (yy - np.sum(np.dot(mn, Ln)**2))/2.0
    
    A =  np.sum(np.log(np.diag(L))) - np.sum(np.log(np.diag(Ln)))
    B = a0*np.log(b0) - an*np.log(bn)
    C = sps.gammaln(an) - sps.gammaln(a0)
    likel = A + B + C - 2.0/n*np.log(2.0*np.pi)
    #likel = B + C - 2.0/n*np.log(2.0*np.pi)
    return likel

def nig_grad_L(L):
    L = unwrap(L)
    
    n = len(X)
    
    XX = np.dot(X.T,X)
    yy = np.dot(y,y)
    Xy = np.dot(X.T,y)
    XyyX = np.outer(Xy,Xy)
    
    V = np.dot(L,L.T)
    
    #posteriors
    #print V
    Vn = V + XX
    Ln = np.linalg.cholesky(Vn)
    mn = cho_solve((Ln, 1), Xy)
    an = a0 + n/2.0
    bn = b0 + (yy - np.sum(np.dot(mn, Ln)**2))/2.0
    
    A =  np.sum(np.log(np.diag(L))) - np.sum(np.log(np.diag(Ln)))
    B = a0*np.log(b0) - an*np.log(bn)
    C = sps.gammaln(an) - sps.gammaln(a0)

    #Inverses
    H1 = cho_solve((Ln, 1), XyyX)
    H2 = cho_solve((Ln, 1), L)
    H3 = cho_solve((L, 1), L)

    A_prime = H3 - H2
    B_prime = (a0/b0 - an/bn) * np.dot(H1,H2)  
    
    grad = np.tril(A_prime+B_prime)
    
    return grad    
    


def nig_grad_V(V):
    V = unwrap(V)
    
    n = len(X)
    
    XX = np.dot(X.T,X)
    yy = np.dot(y,y)
    Xy = np.dot(X.T,y)
    
    #posteriors
    #print V
    LV = np.linalg.cholesky(V)
    Vn = V + XX
    LVn = np.linalg.cholesky(Vn)
    mn = cho_solve((LVn, 1), Xy)
    an = a0 + n/2.0
    bn = b0 + (yy - np.sum(np.dot(mn, LVn)**2))/2.0
    
    iV = cho_solve((LV, 1), np.eye(d))
    iVn =  cho_solve((LVn, 1), np.eye(d))
    
    #gradients
    gmn = - np.dot(np.dot(iVn, iVn), Xy)
    #gbn = -(np.outer(np.dot(gmn, Vn), mn) + np.outer(np.dot(mn, Vn), gmn) + np.outer(mn, mn))/2.0
    #gbn =   -(np.outer(np.dot(gmn, np.eye(d)), mn) + np.outer(np.dot(mn, np.eye(d)), gmn))/2.0
    #gbn = -np.outer(np.dot(2*np.eye(d),mn), gmn)/2.0
    gbn = -(np.outer(np.dot(V+V.T, mn), gmn) + np.outer(mn,mn))/2.0 
    #gbn = -(np.outer(np.dot(gmn, Vn), mn) + np.outer(np.dot(mn, Vn), gmn))/2.0
    #gbn = -(np.outer(np.ones(d), np.ones(d)))/2.0
    print 'gmn={0}'.format(gmn)
    print 'gbn={0}'.format(gbn)
    A = iV.T - iVn.T
    B = -an/bn * gbn
    
    grad = (A/2.0 +B) * V
    
    #return 2*grad - grad*np.eye(d)
    return grad

def wrap(V):
    return V.ravel()

def unwrap(a):
    V = np.reshape(a, (d,d))
    return V


    

if __name__ == '__main__':
    import time
    
    G =  np.ones((d,d))*0.2
    G[np.diag_indices(d, 2)] = 3
    #G[2,2] = 0.5
    #G[0,0] = 3
    #G[1,2] = 1.5
    
    
    print nig_likel_a(np.log(2.0))
    print nig_grad_a(np.log(2.0))
    print spopt.approx_fprime(np.array([np.log(2.0)]), nig_likel_a, np.sqrt(np.finfo(float).eps))
    print nig_likel_b(np.log(3))
    print nig_grad_b(np.log(1.5))
    print spopt.approx_fprime(np.array([np.log(1.5)]), nig_likel_b, np.sqrt(np.finfo(float).eps))
    print nig_likel_V(wrap(G))
    L = np.linalg.cholesky(G)
    print nig_likel_L(wrap(L))
    print unwrap(nig_grad_L(wrap(L)))
    
    print unwrap(spopt.approx_fprime(wrap(L), nig_likel_L, np.sqrt(np.finfo(float).eps)))
#    
#    G = 2.0*np.eye(d)
#    L = np.linalg.cholesky(G)
#    print g1(wrap(L))
#    print unwrap(spopt.approx_fprime(wrap(L), l1, np.sqrt(np.finfo(float).eps)))
#    print g2(wrap(L))
#    print unwrap(spopt.approx_fprime(wrap(L), l2, np.sqrt(np.finfo(float).eps)))
#    print g3(wrap(G))
#    print unwrap(spopt.approx_fprime(wrap(G), l3, np.sqrt(np.finfo(float).eps)))
#    print 'g4'
#    print g4(wrap(G))
#    print unwrap(spopt.approx_fprime(wrap(G), l4, np.sqrt(np.finfo(float).eps)))
#
#    print l3(G)
#    print l4(G)
#    print 'likel'
#    print l6(wrap(L))
#    print g6(wrap(L))
#    print g7(wrap(L))
#    print unwrap(spopt.approx_fprime(wrap(L), l6, np.sqrt(np.finfo(float).eps)))

#    print '3'
#    print g3(wrap(G))
#    print unwrap(spopt.approx_fprime(wrap(G), l3, np.sqrt(np.finfo(float).eps)))
#    print g4(wrap(G))
#    print unwrap(spopt.approx_fprime(wrap(G), l4, np.sqrt(np.finfo(float).eps)))
#    print g5(wrap(G))
#    print unwrap(spopt.approx_fprime(wrap(G), l5, np.sqrt(np.finfo(float).eps)))
#    
#    print g6(wrap(G))
#    print unwrap(spopt.approx_fprime(wrap(G), l6, np.sqrt(np.finfo(float).eps)))

    
    
#    
##    m = np.array([2,2])
#    print f_log_niw_gelman(m)
##    #print gradm_log_niw(m)
##    #print spopt.approx_fprime(m, f_log_niw_gelman, np.sqrt(np.finfo(float).eps))
##    #print f_log_niw_gelman(m)
##    #print fS_log_niw_gelman(S.ravel())
##    #print gradS_log_niw(S.ravel())
##    #print spopt.approx_fprime(S.ravel(), fS_log_niw_gelman, np.sqrt(np.finfo(float).eps))
##    #print spopt.check_grad(fS_log_niw_gelman, gradm_log_niw, S.ravel())
##    
##    D = np.array([[3,0.5], [0.5,2]])
##    L = np.linalg.cholesky(D)
##    print f_old(D.ravel())
##    print g_old(D.ravel())
##    print f(L.ravel())
##    print g(L.ravel())
##    print spopt.approx_fprime(L.ravel(), f, np.sqrt(np.finfo(float).eps))
##    
##    print '----------'
##    print f_log_niw_gelman(m)
##    print fr_log_niw_gelman(0.5)
##    print gradr_log_niw(0.5)
##    print spopt.approx_fprime(np.atleast_1d(np.array([0.5])), fr_log_niw_gelman, np.sqrt(np.finfo(float).eps))
##    print fv_log_niw_gelman(2.0)
##    print gradv_log_niw(2.0)
##    print spopt.approx_fprime(np.atleast_1d(np.array([2.0])), fv_log_niw_gelman, np.sqrt(np.finfo(float).eps))
##    
#    
#    
#    X = np.random.randn(1000,100)
#    m = np.mean(X,0)
#    S = np.cov(X.T)
#    x = np.random.randn(10,100)
#        
#    
#    print h1(m,S)
#    print h2(m,S)
#    print h3(m,S)
#    print h4(m,S)
#    print h5(m,S)
#    print h6(m,S)
#    
#    print 'result'
#    from timeit import Timer
#    
#    t = Timer("h1(m,S)", "from __main__ import h1,m,S")
#    print np.mean(t.repeat(50,10))
#    
#    t = Timer("h2(m,S)", "from __main__ import h2,m,S")
#    print np.mean(t.repeat(50,10))
#    
#    t = Timer("h3(m,S)", "from __main__ import h3,m,S")
#    print np.mean(t.repeat(50,10))
#    
#    t = Timer("h4(m,S)", "from __main__ import h4,m,S")
#    print np.mean(t.repeat(50,10))
#    
#    t = Timer("h5(m,S)", "from __main__ import h5,m,S")
#    print np.mean(t.repeat(50,10))
#    
#    t = Timer("h6(m,S)", "from __main__ import h6,m,S")
#    print np.mean(t.repeat(50,10))
#    
#        
#    #print np.linalg.inv(L.T)
#    #print cho_solve((L, 1), np.eye(2))
#    #print np.linalg.inv(D)
#    #print np.linalg.solve(L.T, np.eye(2))
#    
#    
#    