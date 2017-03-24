'''
Created on Feb 28, 2011

@author: marcel
'''

import numpy as np

from numpy.linalg import inv, eigvalsh
from numpy.random import (rand, multivariate_normal, permutation)
from scipy.sparse import lil_matrix
from scikits.learn.base import BaseEstimator 

from .prototype import KMeans

from upgeo.exception import NotFittedError
from upgeo.util.array import sqix_
from upgeo.util.metric import distance_matrix
from upgeo.util.stats import nanmvnpdf



class MoG(BaseEstimator):
    '''
    classdocs
    
    TODO:
    -----
    - Make the covariance types globals like enums.
    - use decorators to check whether
    '''
    
    __slots__ = ("__nstates", "__nfeatures", "__cov_type", "__weights",  
                 "__means", "__covars", "__is_init")
    
    def __init__(self, nstates=1, cov_type='full'):
        '''
        Constructor
        '''
        if not cov_type in ['spherical', 'tied', 'diag', 'full']:
            raise TypeError("bad cov_type: '{0}'".format(cov_type))
        
        self.__nstates = nstates
        self.__is_init = False
        self.__cov_type = cov_type
    
    def activation(self, x):
        '''
        Computes the activation, i.e. the probability P(X|J) of the data 
        conditioned on each component density, for the underlying mixture model. 
        '''
        self.__init_check()
        
        x = np.asarray(x)
        m = x.shape[0]
        k = self.__nstates
        
        act = np.zeros((m, k))
        for j in xrange(k):
            _, mu, sigma = self.get_component_parameters(j)
            act[:,j] = nanmvnpdf(x, mu, sigma)
        
        return act
         
    def posterior(self, x, act=None):
        '''
        Computes the class (component) probabilities, i.e. each component 
        conditioned on the data P(J|X), for the underlying mixture model.
        '''
        self.__init_check()
        x = np.asarray(x)
        
        if act is None:
            act = self.activation(x)
            
        comp_likel = act * self.__weights
        post = comp_likel.T / np.sum(comp_likel, 1).T
        post = post.T
        
        return post #, act
    
    def log_likelihood(self, x, act=None):
        '''
        Computes the complete log-likelihood of the gaussian mixture model for the 
        specified data.
        
        TODO:
        -----
        - handling zero probabilities 
        '''
        self.__init_check()
        x = np.asarray(x)
        
        if act is None:
            act = self.activation(x)
        
        prob = self.pdf(x, act)
        log_prob = np.log(prob)
        likel = np.sum(log_prob)
        
        return likel #, act
        
    def likelihood(self, x, act=None):
        '''
        Computes the complete likelihood of the gaussian mixture model for the 
        specified data.
        TODO:
        -----
        - handling zero probabilities 

        '''
        self.__init_check()
        x = np.asarray(x)
        
        if act is None:
            act = self.activation(x)
        
        prob = self.pdf(x, act)
        likel = np.prod(prob)
        
        return likel #, act
        
    def pdf(self, x, act=None):
        '''
        Computes the probability density P(X) of the data for the gaussian 
        mixture distribution.
        '''
        self.__init_check()
        x = np.asarray(x)
        
        if act is None:
            act = self.activation(x)
    
        p = np.dot(act, self.__weights)
        return p #, act
        
    def cdf(self, x):
        '''
        '''
        raise NotImplementedError()
    
    def marginalize(self, features):
        '''
        Returns a marginalized mixture model of the specified features.
        
        TODO:
        ----- 
        - handling logical features
        - exception handling    
        '''
        self.__init_check()
        features = self.__asarray_indices(features)
        n = features.size
        k = self.__nstates
        
        model = self.copy()
        model.__nfeatures = self.__nfeatures-n
        model.__means = self.__means[:,features]
        
        model.__covars = np.squeeze(np.zeros((k,n,n)))
        for j in xrange(k):
            model.__covars[j] = self.__covars[sqix_(j, features, features)]
        
        return model

    def rvs(self, n=1):
        '''
        Generate a fixed number of random variates by using the underlying model.
        '''
        assert n > 0, "sample size must be greater than 0"
        
        self.__init_check()
        
        r = rand(n)
        d = self.__nfeatures
        k = self.__nstates
        
        samples = np.ones((n,d))*np.nan
        components = np.ones(n, dtype='int')*-1
        
        total_samples = 0
        cum_weight = 0
        for j in xrange(k):
            weight = self.__weights[j]
            mu = self.__means[j]
            sigma = self.__covars[j]
            
            n_samples = np.sum(np.all([r >= cum_weight , r < cum_weight+weight], 0))
            i = np.arange(total_samples, total_samples+n_samples)
            
            samples[i,:]  = multivariate_normal(mu, sigma, n_samples)
            components[i] = j
            
            cum_weight += weight
            total_samples += n_samples
            
             
        perm = permutation(n)
        return samples[perm], components[perm]
        
    def predict(self, x, act=None, bias=True):
        '''
        Predicts the labels for data, i.e. estimate for each case the most likely 
        mixture component, and estimates their probability for each component. The
        component probability can be biased, i.e. the component priors are included
        by the computation. 
        '''
        self.__init_check()
        x = np.asarray(x)
        
        if act is None:
            act = self.activation(x)
    
        if bias:
            prob = self.posterior(x, act)
        else:
            prob = act / np.sum(act, 1)
        
        label = np.argmax(prob, 1)
        return label, prob
    
    def expectation(self, x, covariates, targets):
        '''
        TODO:
        -----
        - optimize implementation
        '''
        self.__init_check()
        
        x = np.asarray(x)
        covariates = self.__asarray_indices(covariates)
        targets = self.__asarray_indices(targets)
 
        #x = x[:,covariates]
        marg_gmm = self.marginalize(covariates)
        act = self.__weights * marg_gmm.activation(x[:,covariates]) #is the multiplication with the priors true??
        weights = act.T / np.sum(act, 1).T
        weights = weights.T
        
        n = x.shape[0]
        k = self.__nstates
        d = targets.size
        
        y = np.squeeze(np.zeros((n,d)))       
        for j in xrange(k):
            expectations = self.component_expectation(j, x, covariates, targets)
            y += weights[:,j]*expectations
            
        return y
        
    def component_expectation(self, j, x, covariates, targets):
        '''
        TODO:
        -----
        - check whether covariates and targets contain same elements
        - check the dimensionality of x
        - speed up computation by caching the inverse covariance matrix for data
          with missing values
        '''
        self.__init_check()
        self.__state_range_check(j)
        
        x = np.asarray(x)
        covariates = self.__asarray_indices(covariates)
        targets = self.__asarray_indices(targets)
        
        #x = x[:,covariates]
        nan_mask = np.isnan(x)
        nan_rows = np.any(nan_mask, 1)
        
        n = x.shape[0]
        d = targets.size
        y = np.squeeze(np.ones((n,d))*np.nan)
        
        mu = self.__means[j]
        sigma = self.__covars[j]
        
        #compute the expectation for complete cases
        x_center = x[sqix_(~nan_rows,covariates)] - mu[covariates]
        a = np.dot(inv(sigma[sqix_(covariates,covariates)]), 
                   sigma[sqix_(targets,covariates)].T)
        a = np.dot(a, x_center.T)
        y[~nan_rows] = mu[targets] - a

        #compute the expectation for incomplete cases
        if np.any(nan_rows):
            cases = np.flatnonzero(nan_rows)
            for i in cases:
                ov = np.intersect1d(np.flatnonzero(~nan_mask[i]), covariates, \
                                    assume_unique=True)
                x_center = x[i,ov]-mu[ov]
                a = np.dot(inv(sigma[sqix_(ov,ov)]), sigma[sqix_(targets,ov)].T)
                a = np.dot(a, x_center.T)
                y[i] = mu[targets] + a
                
        return y
           
    def fit(self, x, init='kmeans', max_iter=100, min_covar=1e-3, thresh=1e-5):
        '''
        Estimates the parameters of the gaussian mixture model given the 
        parameters with the expectation maximation
        
        Parameters
        ---------- 
        '''
        x = np.asarray(x)
        nan_mask = np.isnan(x)
        
        #model initialization
        if init == 'kmeans':
            self.__init_kmeans(x, nan_mask)
        elif init == 'random':
            self.__init_random(x, nan_mask)
        else:
            raise TypeError('bad init value: {0}'.format(init))
        
        act = self.activation(x)
        
        likels = [self.log_likelihood(x, act)] 
        for it in xrange(max_iter):
            
            post, moment, moment_square = self.__e_step(x, act, nan_mask)
            weights, means, covars = self.__m_step(x, post, moment, moment_square)
            
            self.__weights = weights
            self.__means = means
            self.__covars = covars
            
            #check for convergence
            act = self.activation(x)
            likels.append(self.log_likelihood(x, act))
            if (likels[-1] - likels[-2]) < thresh:
                break
            
        return likels
    
    def refit(self, x, max_iter=100, min_covar=1e-3, thresh=1e-2):
        '''
        '''
        raise NotImplementedError
    
    def _get_nstates(self):
        '''
        '''
        return self.__nstates

    nstates = property(fget=_get_nstates)
    
    def _get_cov_type(self):
        '''
        '''
        return self.__cov_type
    
    cov_type = property(fget=_get_cov_type)
    
    def _get_nfeatures(self):
        '''
        '''
        self.__init_check()
        return self.__nfeatures
    
    ndim = property(fget=_get_nfeatures)
        
    def _get_weights(self):
        '''
        '''
        self.__init_check()
        return self.__weights
    
    weights = property(fget=_get_weights)
    
    def _get_means(self):
        '''
        '''
        self.__init_check()
        return self.__means.copy()
    
    means = property(fget=_get_means)
    
    def _get_covars(self):
        '''
        '''
        return self.__covars.copy()
    
    covars = property(fget=_get_covars)
    
    def get_weight(self, j):
        '''
        Returns the weight of the j-th component in the mixture model.
        '''
        self.__init_check()
        self.__state_range_check(j)
        return self.__weights[j]
    
    def _set_weight(self, j, value):
        '''
        Sets the weight of the j-th component in the mixture model to the given
        value. 
        '''
        assert (0 <= value <= 1), "weight must be in the range [0,1]"
        self.__state_range_check(j)
        self.__weights[j] = value
        
    def get_mean(self, j):
        '''
        Returns the mean vector of the j-th component in the mixture model.
        '''
        self.__init_check()
        self.__state_range_check(j)
        return self.__means[j].copy()
    
    def _set_means(self, j, mu):
        '''
        Sets the mean vector of the j-th component in the mixture to the 
        specified vector.
        '''
        self.__state_range_check(j)
        mu = np.asarray(mu)
        assert mu.ndims == 1 and self.__nfeatures == mu.shape[0], \
            'mu must have shape(nfeatures)'
        self.__means[j] = mu.copy()
    
    def get_covariance(self, j):
        '''
        Returns the covariance matrix of the j-th component in the mixture model.
        '''
        self.__init_check()
        self.__state_range_check(j)
        return self.__covars[j].copy()
    
    def _set_covariance(self, j, sigma):
        '''
        Sets the covariance matrix of the j-th component in the mixture model to 
        the specified matrix.
        '''
        self.__state_range_check(j)
        sigma = self._chk_ascovar(sigma)
        
        if self.__cov_type == 'tied':
            #make a copy for all components
            for i in xrange(self.__nst):
                self.__covars[i] = sigma.copy
        else:
            self.__covars[j] = sigma.copy()
        
        
    def get_component_parameters(self, j):
        '''
        Returns all parameters of the j-th specified component in the mixture
        model.
        '''
        self.__init_check()
        
        weight = self.get_weight(j)
        mu = self.get_mean(j)
        sigma = self.get_covariance(j)
        return weight, mu, sigma
    
    def copy(self):
        '''
        '''
        cp = MoG(self.__nstates, self.__cov_type)
        cp.__is_init = self._is_init
        cp.__nfeatures = self.__nfeatures
        cp.__weights = self.__weights
        cp.__means = self.__means
        cp.__covars = self.__covars
        return cp
        
    def __init_random(self, x, nan_mask=None):
        '''
        Initializes the means for the different component means randomly. The 
        weihts are equal distributed over the components and the covariances is
        set to the data variance and/or covariance, based on the choosen covariance
        type. Initialization is done only on complete cases.
        '''
        if nan_mask != None:
            x =  x[~np.any(nan_mask, 1),:]
            
        k = self.__nstates
        [m,d] = x.shape
        
        self.__nfeatures = d
        
        #init equal weights
        self.__weights = np.ones(k) / k
      
        #choose randomly k mean vectors from x
        seeds = np.random.permutation(m)[:k]
        self.__means = x[seeds]
        
        #compute covariance of the whole dataset
        if self.__cov_type == 'spherical':
            sigma = np.identity(d) * np.mean(np.var(x, 0))
        elif self.__cov_type == 'diag':
            sigma = np.diag(np.var(x, 0))
        elif self.__cov_type == 'full':
            sigma = np.cov(x.T)
        elif self.__cov_type == 'tied':
            sigma = np.cov(x.T)
        else:
            raise TypeError('unknown covariance type')
        
        #distribute the covariance to all components
        self.__covars = np.tile(sigma, (k,1,1))
        
        self.__is_init = True
    
    def __init_kmeans(self, x, nan_mask=None):
        '''
        Initialize the means of the underlying mixture model by the k-means 
        algorithm. The component weights estimated by the cluster size and 
        the covariances by the elements in the cluster. It should be mentioned
        that only complete cases are considered by the initialization procedure.
         
        '''
        if nan_mask != None:
            x = x[~np.any(nan_mask, 1),:]


        k = self.__nstates
        [m,d] = x.shape
        
        self.__nfeatures = d
        
        kmeans_cluster = KMeans(k)
        labels, _ = kmeans_cluster.fit(x) #@todo: refactor return values
        cluster_size = np.array(np.bincount(labels), dtype='float')
        
        #init the weights by the relative size of each cluster
        self.__weights = cluster_size / m
       
        #init the components to the cluster centers of kmeans
        self.__means = kmeans_cluster.centers
        
        #init the covariances by each cluster covariance
        self.__covars = np.zeros((k,d,d))
        if self.__cov_type == 'spherical':
            
            for j in xrange(k):
                sigma = np.identity(d) * np.mean(np.var(x[labels == j], 0))
                self.__covars[j] = sigma
                
        elif self.__cov_type == 'diag':
            
            for j in xrange(k):
                sigma = np.diag(np.var(x[labels == j], 0))
                self.__covars[j] = sigma
                
        elif self.__cov_type == 'full':
            
            for j in xrange(k):
                sigma = np.cov(x[labels == j].T) 
                self.__covars[j] = sigma
                
        elif self.__cov_type == 'tied':
            sigma = np.cov(x.T)
        else:
            raise TypeError('unknown covariance type')
        
        self.__is_init = True    
        
    def __e_step(self, x, act=None, nan_mask=None):
        '''
        The Expectation-Step of the EM-Algorithm, which computes the component
        posterior P(J|X) and if requested the first P(X_mis|X_obs,J) and second 
        moment P(X_mis X_mis.T|X_obs,J) of missing values in the data.
        ''' 
        post = self.posterior(x, act)
        
        [m,d] = x.shape
        k = self.__nstates
        
        moment = None
        moment_square = None
        
        if nan_mask is not None:
            moment = tuple(lil_matrix((m,d)) for _ in xrange(k))
            if self.__cov_type == 'full':
                moment_square = tuple({} for _ in xrange(k));
            
            #TODO: - code optimization
            nan_rows = np.any(nan_mask, 1)
            for i in np.flatnonzero(nan_rows):
                mv = nan_mask[i,:]
                ov = ~mv
                
                #calculate the moments of the missing attributes for each 
                #component
                for j in xrange(k):
                    mu = self.__means[j]
                    sigma = self.__covars[j]
                    
                    sigma_inv_ov = inv(sigma[sqix_(ov,ov)])
                    
                    #calculate the first moment of the missing values in case i
                    k_moment = moment[j]
                    
                    x_center = x[i,ov]-mu[ov]
                    a = np.dot(sigma_inv_ov, sigma[sqix_(mv,ov)].T)
                    a = np.dot(a, x_center.T)
                    x_moment = mu[mv] + a
                    #indexing scheme np.where(ov)[0] is necessary for sparse matrices
                    k_moment[i, np.flatnonzero(ov)] = x_moment
                    
                    #calculate the second moment of the missing values in case i
                    if self.__cov_type == 'full':
                        k_moment_square = moment_square[j]
                        
                        a = np.dot(sigma_inv_ov, sigma[sqix_(mv,ov)].T)
                        a = np.outer(a, sigma[sqix_(mv,ov)])
                        a = a + np.outer(x_moment, x_moment)
                        x_moment_square = sigma[sqix_(mv,mv)] - a
                        k_moment_square[i] = x_moment_square
                
        return post, moment, moment_square
    
    def __m_step(self, x, post, moment=None, moment_square=None, nan_mask=None):        
        '''
        TODO:
        ----
        - work on a copy of x???
        
        '''
        complete_x = x.copy()
        
        k = self.__nstates
        [m,d] = x.shape
        
        if nan_mask is not None:
            nan_rows, nan_cols = np.where(nan_mask)
                
        #maximize the weights of each component
        sum_post = sum(post, 0)
        weights = sum_post / m
        
        means = np.zeros((k,d))
        covars = np.zeros((k,d,d))  
        for j in xrange(k):
            #work on the reference of the component mean and covariance
            #mu = means[j] 
            #sigma = covars[j]
            
            #complete the dataset for the component j with the corresponding moment
            if nan_mask is not None:
                k_moment = moment[j]
                k_moment = k_moment[nan_rows, nan_cols].toarray().flatten()
                complete_x[nan_rows, nan_cols] = k_moment
     
            
            #maximize the mean vector of component j
            mu = np.dot(post[:,j], complete_x) / sum_post[j]
            
            #maximize the covariance matrix of component j
            if self.__cov_type == 'spherical':
                sigma = self.__maximize_spherical_cov(complete_x, mu, post[:,j], sum_post[j])
            elif self.__cov_type == 'diag':
                sigma = self.__maximize_diag_cov(complete_x, mu, post[:,j], sum_post[j])
            elif self.__cov_type == 'full':
                sigma = self.__maximize_full_cov(complete_x, mu, post[:,j], sum_post[j], 
                                                 moment_square[j], nan_mask)
            elif self.__cov_type == 'tied':
                raise NotImplementedError('covariance optimization for tied ' \
                                          'covars is not implemented yet');
            else:
                raise TypeError('unknown covariance type')
            
            means[j] = mu
            covars[j] = sigma
        
        return weights, means, covars
    
    def __maximize_spherical_cov(self, complete_x, mu, post, sum_post):
        '''
        Maximizes the covariance of a spherical form for a component
        
        TODO:
        -----
        - Return a covariance matrix in compressed for or not?
        - Check if the estimator is biased
        '''
        d = complete_x.shape[1]
        dist  = distance_matrix(complete_x, mu, 'sqeuclidean')
        sigma = np.dot(post, dist) / (sum_post*d)
        sigma = np.identity(d)*sigma
        return sigma
    
    def __maximize_diag_cov(self, complete_x, mu, post, sum_post):
        '''
        Maximizes the covariance of a diagonal form for a component
        
        TODO:
        -----
        - Return a covariance matrix in compressed for or not?
        - Check if the estimator is biased
        '''
        x_center = complete_x-mu
        sigma = np.dot((x_center**2).T, post) / sum_post
        sigma = np.diag(sigma)
        return sigma
    
    def __maximize_full_cov(self, complete_x, mu, post, sum_post, 
                            moment_square=None, nan_mask=None):
        '''
        Maximizes the covariance of a full form for a component
        
        TODO:
        -----
        - Return a covariance matrix in compressed for or not?
        - Check if the estimator is biased
        '''
        m = complete_x.shape[0]
        
        if nan_mask is None:
            nan_rows = np.zeros(m, dtype='bool') 
        else:
            nan_rows = np.any(nan_mask, 1)
        
        #optimization including complete cases
        x_center = complete_x[~nan_rows,:] - mu
        sigma = np.dot(x_center.T*post[~nan_rows], x_center)
        
        for i in np.flatnonzero(nan_rows):
            mv = nan_mask[i,:]
            ov = ~mv;
            
            x = complete_x[i,:]
            s = np.outer(-x, mu) - np.outer(x, mu) + np.outer(mu, mu)
            s[sqix_(ov,ov)] += np.outer(x[ov], x[ov])
            s[sqix_(ov,mv)] += np.outer(x[ov], x[mv])
            s[sqix_(mv,ov)] += np.outer(x[mv], x[ov])
            s[sqix_(mv,mv)] += moment_square[i]
            
            sigma += post[i] * s
        
        sigma /= sum_post
        return sigma
    
    def __init_check(self):
        '''
        '''
        if not self.__is_init:
            raise NotFittedError('fit was not invoked')
        

    def __state_range_check(self, index):
        '''
        Check whether the specified index is in range [0, nstates-1].
        '''
        if index < 0 or index >= self.__nstates:
            raise IndexError("Index '{0}' out of bounds".format(index))
    
    def __feature_range_check(self, index):
        '''
        Check whether the specified index is in range [0, nfeatures-1].
        '''
        if index < 0 or index >= self.__nfeatures:
            raise IndexError("Index '{0}' out of bounds".format(index))
        
    def __asarray_indices(self, indices):
        '''
        '''
        indices = np.ravel(indices)
        
        if indices.dtype == 'bool':
            indices = np.flatnonzero(indices)
        else:
            indices = np.unique(indices);
            indices = np.asarray(indices, dtype='int')

        if indices.size > 0:
            self.__feature_range_check(indices[0])
            self.__feature_range_check(indices[-1])
            
        return indices

    def __chk_ascovar(self, sigma):
        '''
        '''
        sigma = np.asarray(sigma)
        d = self.__nfeatures
        if self.__cov_type == 'spherical':    
            
            if sigma.size != 1:
                raise TypeError('sigma must be a scalar')
            sigma = np.diag(d) * sigma
            
        elif self.__cov_type == 'diag':
            
            if sigma.ndim != 1 or sigma.size != d:
                raise TypeError('sigma must be an array of size d')
            elif np.any(sigma < 0):
                raise TypeError('sigma must be non-negative') 
            sigma = np.diag(sigma)
            
        else:
            if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
                raise TypeError('sigma must be a square matrix')
            elif not np.allclose(sigma, sigma.T) or np.any(eigvalsh(sigma) <= 0):
                raise TypeError('sigma must be symmetric and positive-definite')

        return sigma 
