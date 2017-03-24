'''
Created on Aug 22, 2011

@author: marcel
'''
import os
import sys

import numpy as np
import scipy.special as sps
import scipy.optimize as spopt

from collections import deque
from xml.etree import ElementTree


from scikits.learn.base import RegressorMixin, BaseEstimator

from upgeo.ml.multitask.base import asmtarray_object, flatten_data,\
    compound_data
from upgeo.ml.regression.bayes import EBChenRegression, FastBayesRegression,\
    RobustBayesRegression
from upgeo.prior.normal import NormalInvWishart
from upgeo.util.math import sumln
from upgeo.adt.tree import MutableBTree, path_to_root
from upgeo.exception import NotFittedError
from upgeo.ml.regression.np.selector import RandomSubsetSelector, FixedSelector
from upgeo.ml.regression.linear import LinearRegressionModel
from upgeo.ml.regression.np.gp import SparseGPRegression
from upgeo.ml.cluster.prototype import KMeans
from upgeo.ml.regression.np.infer import FITCExactInference

class BHCLinearRegressionExpert(BaseEstimator, RegressorMixin):
    '''     
    @todo: - 
    '''
    
    __slots__ = ('_alpha',          #cluster concentration hyperparameter in log space
                 '_beta',           #hyperparameters of regression model
                 '_gamma',          #hyperparameters of the gating function (NiW distribution)
                 '_task_tree',      #the cluster tree of regression models
                 '_use_task_features',
                 '_is_init'
                 )        
    
    def __init__(self, alpha, beta, use_task_features=False):
        '''
        @todo: - check the number of hyperparams and priors
        '''
        self._alpha = alpha
        self._beta = beta
        self._use_task_features = use_task_features
        
        self._is_init = False
        
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        return np.r_[self._alpha, self._beta]
    
    def _set_hyperparams(self, params):
        '''
        '''
        self._alpha = params[0]
        self._beta = params[1:]
        
    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)

    def fit(self, X, Y, Z, max_it=50, tol=1.e-5):
        '''
        '''
        
        X = asmtarray_object(X)
        Y = asmtarray_object(Y)
        Z = np.asanyarray(Z)
        
        if len(X) != len(Y) != len(Z):
            raise ValueError('number of task data X, Y and task ' + 
                             'description Z must be the same')
        
        alpha = self._alpha
        beta = self._beta
        
        #todo: should be an optional preprocessing step
        if True:
            #Estimate hyperparameters of the linear model by expectation propagation algorithm
            covars = flatten_data(X)
            targets = flatten_data(Y)
            reg_model = EBChenRegression()
            reg_model.fit(covars, targets)
            
            beta = np.log(np.array([reg_model.alpha, reg_model.beta]))
            print 'alpha={0}, beta={1}'.format(alpha, beta)
    
        #preprocess
        if self._use_task_features:
            n = len(X)
            d = X[0].shape[1]
            Xmean = np.zeros((n,d))
            for i in xrange(len(X)):
                Xmean[i] = np.mean(X[i], 0)
            
            Z = np.c_[Xmean, Z]
            
        #estimate hyperparameters of the gating model (NiW distribution)
        #todo: add a small amount to non diagonal elements, because the matrix is optimized in log space
        #wenn die werte null sind, wird nur die diag optimiert
        m = np.mean(Z, 0)
        S = np.diag(1/np.var(Z, 0)) #+ np.ones((Z.shape[1],Z.shape[1]))*1e-05
        #S = np.eye(Z.shape[1]) #+ np.ones((Z.shape[1],Z.shape[1]))*1e-05
        r = np.log(0.5)
        v = Z.shape[0]+1
        gamma = (m, S, r, v) 
        
        #optimize the concentration parameter alpha and the model hyperparameters
        # by an iterative procedure where in the first step the best tree structure 
        #is find for fixed parameters and in the second step the parameters is 
        #optimized by an alternating line search and gradient descent mehtod
        i = 0
        bhc_model = None
        likel = likel_old = np.Inf
        while i < max_it:
            
            if i > 0 and likel_old-likel < tol:
                break
            
            likel_old = likel
            
            #E-Step: estimate the new tree of the optimized parameters
            model = self._build_bhc_model(alpha, beta, gamma, X, Y, Z)
            
            #M-Step: optimize the parameters on a fixed tree
            #beta are the model hyperparameters
            likel, alpha, beta, gamma = self._optimize_bhc_parameters(model, alpha, beta, gamma)
            bhc_model = model
            i += 1
            
        self._update_model(alpha, beta, gamma, bhc_model)
        
        self._task_tree = bhc_model
        self._alpha = alpha
        
        self._is_init = True  
        
    def _optimize_bhc_parameters(self, model, alpha, beta, gamma):
        '''
        '''        
        print model.cargo.marg
        alpha_likel_fun = BHCAlphaLikelihood(model)
        print 'alpha={0}'.format(alpha)
        
        try:
            opt_alpha = spopt.brent(alpha_likel_fun, brack=(alpha-10, alpha, alpha+10))
        except AssertionError:
            opt_alpha = spopt.brent(alpha_likel_fun, brack=(alpha-2, alpha+2))
        
        
        print 'opt_alpha={0}'.format(opt_alpha)
        print model.cargo.marg
        self.__update_alpha(opt_alpha, model)
        
        
        #todo: hack
        x0 = self._wrap(beta, gamma)
        n = len(beta)
        d = len(gamma[0])
        
        #constraints for the precision matrix - elements must be posititive
        constraints = [(None, None)]*(n+d) 
        constraints.extend([(0,None)]*(d*d)) #L, r
        constraints.extend([(None,None), (d+1,None)])#r,v
            
            
        print 'likel_before={0}'.format(model.cargo.marg)
            
        model_likel_fun = _BHCModelLikelihood(model)
        model_likel_grad = model_likel_fun.gradient
        print 'foo'
        #opt_result = spopt.fmin_bfgs(model_likel_fun, x0, model_likel_grad, full_output=True)
        opt_result = spopt.fmin_tnc(model_likel_fun, x0, model_likel_grad, bounds=constraints)
        opt_beta, opt_gamma = self._unwrap(opt_result[0], n, d)
        opt_gamma = list(opt_gamma)
        likel = opt_result[1]
        print 'opt_beta={0}'.format(np.exp(opt_beta))
        print 'opt_gamma={0}'.format(opt_gamma)
        print likel
        
        return (likel, opt_alpha, opt_beta, opt_gamma) 
    
    def _wrap(self, beta, gamma):
        '''
        @todo: make static
        '''
        m, R, r, v = gamma
        L = np.linalg.cholesky(R)
        
        params = np.r_[beta, m, L.ravel(), r, v]
        return params
    
    def _unwrap(self, params, n, d):
        '''
        @todo: make static
        '''
        beta = params[0:n]
        m = params[n:n+d]
        L = params[n+d:n+d+d*d]
        L = np.reshape(L, (d,d))
        L = np.tril(L)
        R = np.dot(L, L.T)
        r = params[-2]
        v = params[-1]
        gamma = (m,R,r,v)
        return beta, gamma
         
    
    def predict(self, X, Z, method='path', incl_tree_weights=True):
        '''  
        @todo: - more generic parametrization for invoking the different prediction methods
        '''
        self._init_check()
        X = asmtarray_object(X)
        Z = asmtarray_object(Z)
        
        if len(X) != len(Z):
            raise ValueError('number of task data X and task' + 
                             'description Z must be the same')
            
        if method == 'path':
            pred_method = self.__predict_path
        elif method == 'path1':
            pred_method = self.__predict_path1
        elif method == 'fullpath':
            pred_method = self.__predict_fullpath
        elif method == 'maxcluster':
            pred_method = self.__predict_maxcluster
        elif method == 'complete':
            pred_method = self.__predict_complete
        elif method == 'flat_complete':
            pred_method = self.__predict_flat_complete
        elif method == 'flat_maxcluster':
            pred_method = self.__predict_flat_maxcluster
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        n = len(X)
        Y = np.empty(n, dtype='object')
        if self._use_task_features:
            #todo: its a hack... optimize
            for i in xrange(n):
                Xi = X[i]
                Zi = Z[i]
                m = len(Xi)
                Yi = np.empty(m)
                for j in xrange(m):
                    Xij = Xi[j]
                    Zij = np.r_[Xij, Zi]
                    Xij = np.atleast_2d(Xij)
                    Yi[j] = np.float(np.asarray(pred_method(Xij, Zij, incl_tree_weights)))
                    
                Y[i] = Yi
        else:
            for i in xrange(n):
                yhat = pred_method(X[i], Z[i], incl_tree_weights)
                Y[i] = yhat
            
        return Y
    
    def _get_likelihood(self):
        '''
        '''
        self._init_check()
        cluster = self._task_tree.cargo
        return cluster.marg
    
    likelihood = property(fget=_get_likelihood)
    
    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')
    
    
    #Methods for Learning
    
    def _build_bhc_model(self, alpha, beta, gamma, X, Y, Z):
        '''
        @todo: - implement the access of the block matrix R
        '''
        
        n = len(X)
        
        task_cluster = [None]*n
        nodes = [None]*n
         
        #initialize the leafs of the cluster hierarchy by each task of its own
        for i in xrange(n):
            Xi = X[i]
            Yi = Y[i]
            Zi = Z[i]
            
            model = self.__make_expert_model(Xi, Yi, Zi, beta, gamma)
            task_cluster[i] = _ClusterNode.make_leaf(i, Xi, Yi, Zi, model, alpha)
            nodes[i] = MutableBTree(task_cluster[i])
        
        #initialize the model matrix and the probability matrix of merged 
        #hypothesis
        model_matrix = np.empty([n,n], dtype=object)
        prob_matrix = np.ones([n,n], dtype=float)*-np.inf
        for i in xrange(n):
            cluster_i = task_cluster[i]
            
            for j in xrange(i+1, n):
                    
                cluster_j = task_cluster[j]
                Xij = np.vstack((cluster_i.X, cluster_j.X))
                yij = np.hstack((cluster_i.y, cluster_j.y))
                Zij = np.vstack((cluster_i.Z, cluster_j.Z))
                    
                model = self.__make_expert_model(Xij, yij, Zij, beta, gamma)
                cluster = _ClusterNode.make_internal(cluster_i, cluster_j, 
                                                     model, alpha)
                    
                model_matrix[i,j] = model_matrix[j,i] = model
                prob_matrix[i,j] = prob_matrix[j,i] = cluster.post
                
        #print 'pobmatr'
        #print prob_matrix
        #main-loop of the hierarchical task clustering process
        n_cluster = n
        while n_cluster > 1:
            
            
            #print 'pobmatr'
            #print prob_matrix
            
            #find the two tasks with highest merge hypothesis posterior
            max_prob = -np.inf 
            p = q = -1
            for i in xrange(n):
                if task_cluster[i] != None:
                    for j in xrange(i+1,n):
                        if task_cluster[j] != None and prob_matrix[i,j] > max_prob:
                            max_prob = prob_matrix[i,j]
                            p = i
                            q = j
            
            #print 'max_prob={0}'.format(max_prob)
            #print 'p={0},q={1}'.format(p,q)     
            #merge the two tasks p,q with highest posterior
            cluster_p = task_cluster[p]
            cluster_q = task_cluster[q]            
            cluster = _ClusterNode.make_internal(cluster_p, cluster_q, 
                                                 model_matrix[p,q], alpha)
            
            task_cluster[p] = cluster
            task_cluster[q] = None
            nodes[p] = MutableBTree(cluster, None, nodes[p], nodes[q])
            nodes[q] = None 
            
            #update model and probability matrix
            model_matrix[p,q] = model_matrix[q,p] = None
            prob_matrix[p,q] = prob_matrix[q,p] = -1.0
            for i in xrange(n):
                if task_cluster[i] != None and i != p:
                    cluster_i = task_cluster[i]
                    
                    Xi = np.vstack((cluster.X, cluster_i.X)) #same as np.r_[X[p],X[q],X[i]]
                    yi = np.hstack((cluster.y, cluster_i.y)) #same as np.r_[Y[p],Y[q],Y[i]]
                    Zi = np.vstack((cluster.Z, cluster_i.Z))
                    
                    model = self.__make_expert_model(Xi, yi, Zi, beta, gamma)
                    tmp_cluster = _ClusterNode.make_internal(cluster, cluster_i, 
                                                             model, alpha)
                    
                    
                    model_matrix[p,i] = model_matrix[i,p] = model
                    model_matrix[q,i] = model_matrix[i,q] = None
                    prob_matrix[p,i] = prob_matrix[i,p] = tmp_cluster.post
                    prob_matrix[q,i] = prob_matrix[i,q] = -1
                    
                    #todo: prepare block matrix
                    
            n_cluster -= 1
            
        cluster_tree = nodes[0]
        self.__update_weights(cluster_tree)
        return cluster_tree
    
    def __make_expert_model(self, X, y, Z, beta, gamma):
        reg_model = FastBayesRegression(np.exp(beta[0]), np.exp(beta[1]), weight_bias=True)
        reg_model.fit(X,y)
        
        m, R, r, v = gamma
        density_model = NormalInvWishart(Z, m, R, np.exp(r), v)
        
        model = _ExpertModel(X, y, Z, reg_model, density_model) 
        return model
    
    def _update_model(self, alpha, beta, gamma, treenode):
        self._beta = beta
        self._alpha = alpha
        
        self.__update_alpha(alpha, treenode)
       
        self.__update_model_parameters(beta, gamma, treenode)
        self.__update_weights(treenode)

    def __update_weights(self, treenode):
        '''
        @todo: - weight each instance by the number of elements in parent and sibling
        '''
        for node in treenode.subtree():
            cluster = node.cargo
            cluster.weight = np.exp(cluster.post) 
            cluster.inv_weight = 1.0-cluster.weight
            #if cluster.inv_weight == 0.0:
            #    cluster.inv_weight += 1e-16
            if not node.isroot():
                parent = node.parent
                super_cluster = parent.cargo
                
                #new stuff for normalizing (see phd thesis of heller)
                ns = np.float(super_cluster.ntasks)
                n = np.float(cluster.ntasks)
                norm = n/ns
                #norm = 1
                
                inv_weight = super_cluster.inv_weight*norm
                cluster.weight *= inv_weight
                cluster.inv_weight *= inv_weight
            
                
            #print 'cluster_weight={0}'.format(cluster.weight)

    def __update_alpha(self, alpha, treenode):
        '''
        '''
        if treenode.isatom():
            cluster = treenode.cargo
            cluster.norm_const = alpha+sps.gammaln(cluster.ntasks)
        else:
            self.__update_alpha(alpha, treenode.left)
            self.__update_alpha(alpha, treenode.right)
            
            cluster = treenode.cargo
            left = treenode.left.cargo
            right = treenode.right.cargo
            
            cluster = treenode.cargo
            
            model = cluster.model
            n = cluster.ntasks
        
            #compute the cluster prior 'pi'. Because numerical issues, the norm_const
            #is compute by the sum of individual logs.
            nom = alpha+sps.gammaln(n) 
            cluster.norm_const = sumln([nom, left.norm_const+right.norm_const]) 
            cluster.pi = nom - cluster.norm_const
        
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = left.norm_const + right.norm_const - cluster.norm_const
            a = cluster.pi + model.log_likel
            b = npi+left.marg+right.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + model.log_likel - cluster.marg
            

    def __update_model_parameters(self, beta, gamma, treenode):
        '''
        @todo: kernel is not used
        '''
        if treenode.isatom():
            cluster = treenode.cargo
            
            #rebuild the gp model
            model = self.__make_expert_model(cluster.X, cluster.y, cluster.Z, beta, gamma)
            cluster.model = model
            cluster.marg = model.log_likel
        else:
            self.__update_model_parameters(beta, gamma, treenode.left)
            self.__update_model_parameters(beta, gamma, treenode.right)
            
            cluster = treenode.cargo
            left = treenode.left.cargo
            right = treenode.right.cargo
            
            cluster = treenode.cargo

            #rebuild the gp model
            model = self.__make_expert_model(cluster.X, cluster.y, cluster.Z, beta, gamma)
            cluster.model = model            
            
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = left.norm_const + right.norm_const - cluster.norm_const
            a = cluster.pi + model.log_likel
            b = npi+left.marg+right.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + model.log_likel - cluster.marg
            
    def __predict_complete(self, X, z, incl_tree_weights=True):
        '''
        '''
        tree = self._task_tree
        n = X.shape[0]
        yhat = np.zeros(n)
        
        total_weight = 0
        for cluster in tree:
            model = cluster.model
            weight = model.responsibility(z)
            if incl_tree_weights:
                weight *= cluster.weight
            total_weight += weight
            
        for cluster in tree:
            model = cluster.model
            weight = model.responsibility(z)
            if incl_tree_weights:
                weight *= cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat
    
    def __predict_maxcluster(self, X, z, incl_tree_weights=True):
        tree = self._task_tree
        
        max_cluster = None
        max_response = -1
        for cluster in tree:
            model = cluster.model
            response = model.responsibility(z)
            if response > max_response:
                max_response = response
                max_cluster = cluster
                
        yhat = max_cluster.model.predict(X)
        return yhat
        
    def __predict_path(self, X, z, incl_tree_weights=True):
        '''
        @todo: - check whether the treenodes putting into the cluster_set are correctly 
                 inserted, that means no collosion occurs. Is the id() function sufficient
                 to compute the hash value of the object? What is better, using treenode or
                 cluster nodes?
        '''
        n = X.shape[0]
        yhat = np.zeros(n)
        
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing

        max_node = None
        max_response = -1
        for node in leaves:
            cluster = node.cargo
            model = cluster.model
            response = model.responsibility(z)
            if response > max_response:
                max_response = response
                max_node = node
        
     
        path = path_to_root(max_node)
        
        print '-----------'
        total_weight = 0
        for node in path:
            cluster = node.cargo
            model = cluster.model
            weight = model.responsibility(z)
            print 'responsibility={0}'.format(weight)
            if incl_tree_weights:
                print 'cluster weight={0}'.format(cluster.weight)
                weight *= cluster.weight
            total_weight += weight
            
        for node in path:
            cluster = node.cargo
            model = cluster.model
            weight = model.responsibility(z)
            if incl_tree_weights:
                weight *= cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat
    
    def __predict_path1(self, X, z, incl_tree_weights=True):
        '''
        @todo: - check whether the treenodes putting into the cluster_set are correctly 
                 inserted, that means no collosion occurs. Is the id() function sufficient
                 to compute the hash value of the object? What is better, using treenode or
                 cluster nodes?
        '''
        n = X.shape[0]
        yhat = np.zeros(n)
        
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing

        max_node = None
        max_response = -1
        for node in leaves:
            cluster = node.cargo
            model = cluster.model
            response = model.responsibility(z)
            if response > max_response:
                max_response = response
                max_node = node
        
     
        path = path_to_root(max_node)
        
        total_weight = 0
        for node in path:
            cluster = node.cargo
            model = cluster.model
            total_weight += cluster.weight
            
        for node in path:
            cluster = node.cargo
            model = cluster.model
            weight = cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat

    

    def __predict_fullpath(self, X, z, incl_tree_weights=True):
        '''
        @todo: - check whether the treenodes putting into the cluster_set are correctly 
                 inserted, that means no collosion occurs. Is the id() function sufficient
                 to compute the hash value of the object? What is better, using treenode or
                 cluster nodes?
        '''
        n = X.shape[0]
        yhat = np.zeros(n)
        
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing

        max_node = None
        max_response = -1
        for node in leaves:
            cluster = node.cargo
            model = cluster.model
            response = model.responsibility(z)
            if response > max_response:
                max_response = response
                max_node = node
        
     
        path = path_to_root(max_node)
        
        total_weight = 0
        for node in path:
            cluster = node.cargo
            model = cluster.model
            weight = model.responsibility(z)
            if incl_tree_weights:
                weight *= cluster.weight
            total_weight += weight
            
        for node in path:
            cluster = node.cargo
            model = cluster.model
            weight = model.responsibility(z)
            if incl_tree_weights:
                weight *= cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat

    def __predict_flat_maxcluster(self,  X, z, incl_tree_weights=True, r=0.5):
        tree = self._task_tree
        clusters = self.__flatten_hierarchy(tree, r)
        
        max_cluster = None
        max_response = -1
        for c in clusters:
            model = c.model
            response = model.responsibility(z)
            if response > max_response:
                max_response = response
                max_cluster = c
                
        yhat = max_cluster.model.predict(X)
        return yhat
    
    def __predict_flat_complete(self,  X, z, incl_tree_weights=True, r=0.5):
        tree = self._task_tree
        clusters = self.__flatten_hierarchy(tree, r)
        
        n = X.shape[0]
        yhat = np.zeros(n)
        
        total_weight = 0
        for c in clusters:
            model = c.model
            total_weight += model.responsibility(z)
            
        for c in clusters:
            model = c.model
            weight = model.responsibility(z)
            yhat += weight*model.predict(X)/total_weight
        
        return yhat
        
    def __flatten_hierarchy(self, treenode, r=0.5):
        '''
        '''
        clusters = list()
        queue = deque([treenode])
        
        #iterate aslong the queue is empty
        while queue:
            node = queue.pop()
            c = node.cargo
            if c.post >= np.log(r):
                clusters.append(c)
            else:
                queue.append(node.left)
                queue.append(node.right)
        
        return clusters
        
        

    #Misc methods
    def xml_element(self):
        '''
        '''
        root_element = ElementTree.Element('bhc_model')
        
        #build parameter element 
        param_element = ElementTree.SubElement(root_element, 'parameters')
        param_element.attrib['rho'] = str(self._rho)
        param_element.attrib['alpha'] = str(self._alpha)
        param_element.attrib['beta'] = str(self._beta)
        
        #build task tree element
        task_tree_element = ElementTree.SubElement(root_element, 'task_tree')
        task_tree_element.append(self._tree2xml(self._task_tree)) 
        
        return root_element

    def _tree2xml(self, treenode):
        '''
        '''
        cluster = treenode.cargo
        reg_model = cluster.model
        
        if treenode.isatom():
            cluster_element = ElementTree.Element('leaf')
            cluster_element.attrib['id'] = str(cluster.task_id+1)
        else:
            cluster_element = ElementTree.Element('task_cluster')
            
        cluster_element.attrib['descr'] = str(cluster.z)
        
        likel_element = ElementTree.SubElement(cluster_element, 'likel')
        likel_element.attrib['post'] = str(cluster.post)
        likel_element.attrib['marg'] = str(cluster.marg)
        likel_element.attrib['mlikel'] = str(reg_model.log_evidence)
        likel_element.attrib['pi'] = str(cluster.pi)
        likel_element.attrib['norm'] = str(cluster.norm_const)
        
        pred_element = ElementTree.SubElement(cluster_element, 'predict')
        pred_element.attrib['weight'] = str(cluster.weight)
        pred_element.attrib['invweight'] = str(cluster.inv_weight)
        
        prop_element = ElementTree.SubElement(cluster_element, 'props')
        prop_element.attrib['ntask'] = str(cluster.ntasks)
        prop_element.attrib['size'] = str(cluster.size)
        prop_element.attrib['shape'] = str(cluster.covariates.shape)
        
        model_element = ElementTree.SubElement(cluster_element, 'reg_model')
        model_element.attrib['intercept'] = str(reg_model.intercept)
        model_element.attrib['weights'] = str(reg_model.weights)
        
        if not treenode.isatom():
            cluster_element.append(self._tree2xml(treenode.left))
            cluster_element.append(self._tree2xml(treenode.right))
            
        return cluster_element
        
    def marshal_xml(self, filename):
        '''
        '''
        element = self.xml_element()
        tree = ElementTree.ElementTree(element)
        try:
            tree.write(filename, "UTF-8")
        except EnvironmentError as err:
            print("{0}: import error {1}".format(os.path.basename(sys.argv[0]), err))

class BHCRobustRegressionExpert(BaseEstimator, RegressorMixin):
    '''     
    @todo: - 
    '''
    
    __slots__ = ('_alpha',          #cluster concentration hyperparameter in log space
                 '_beta',           #hyperparameters of regression model
                 '_gamma',          #hyperparameters of the gating function (NiW distribution)
                 '_task_tree',      #the cluster tree of regression models
                 '_use_task_features',
                 '_is_init'
                 )        
    
    def __init__(self, alpha, beta, use_task_features=False):
        '''
        @todo: - check the number of hyperparams and priors
        '''
        self._alpha = alpha
        self._beta = beta
        self._use_task_features = use_task_features
        
        self._is_init = False
        
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        return np.r_[self._alpha, self._beta]
    
    def _set_hyperparams(self, params):
        '''
        '''
        self._alpha = params[0]
        self._beta = params[1:]
        
    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)

    def fit(self, X, Y, Z, max_it=50, tol=1.e-5):
        '''
        '''
        
        X = asmtarray_object(X)
        Y = asmtarray_object(Y)
        Z = np.asanyarray(Z)
        
        if len(X) != len(Y) != len(Z):
            raise ValueError('number of task data X, Y and task ' + 
                             'description Z must be the same')
        
        alpha = self._alpha
        beta = self._beta
    
        #preprocess
        if self._use_task_features:
            n = len(X)
            d = X[0].shape[1]
            Xmean = np.zeros((n,d))
            for i in xrange(len(X)):
                Xmean[i] = np.mean(X[i], 0)
            
            Z = np.c_[Xmean, Z]
            
        #estimate hyperparameters of the gating model (NiW distribution)
        #todo: add a small amount to non diagonal elements, because the matrix is optimized in log space
        #wenn die werte null sind, wird nur die diag optimiert
        m = np.mean(Z, 0)
        S = np.diag(1/np.var(Z, 0)) #+ np.ones((Z.shape[1],Z.shape[1]))*1e-05
        #S = np.eye(Z.shape[1]) #+ np.ones((Z.shape[1],Z.shape[1]))*1e-05
        r = np.log(0.5)
        v = Z.shape[0]+1
        gamma = (m, S, r, v) 
        
        #optimize the concentration parameter alpha and the model hyperparameters
        # by an iterative procedure where in the first step the best tree structure 
        #is find for fixed parameters and in the second step the parameters is 
        #optimized by an alternating line search and gradient descent mehtod
        i = 0
        bhc_model = None
        likel = likel_old = np.Inf
        while i < max_it:
            
            if i > 0 and likel_old-likel < tol:
                break
            
            likel_old = likel
            
            #E-Step: estimate the new tree of the optimized parameters
            model = self._build_bhc_model(alpha, beta, gamma, X, Y, Z)
            
            #M-Step: optimize the parameters on a fixed tree
            #beta are the model hyperparameters
            likel, alpha, beta, gamma = self._optimize_bhc_parameters(model, alpha, beta, gamma)
            bhc_model = model
            i += 1
            
        self._update_model(alpha, beta, gamma, bhc_model)
        
        self._task_tree = bhc_model
        self._alpha = alpha
        
        self._is_init = True  
        
    def _optimize_bhc_parameters(self, model, alpha, beta, gamma):
        '''
        '''        
        #print model.cargo.marg
        alpha_likel_fun = BHCAlphaLikelihood(model)
        print 'alpha={0}'.format(alpha)
        
        try:
            opt_alpha = spopt.brent(alpha_likel_fun, brack=(alpha-10.0, alpha, alpha+10.0))
        except AssertionError:
            opt_alpha = spopt.brent(alpha_likel_fun, brack=(alpha-2.0, alpha+2.0))
        
        
        print 'opt_alpha={0}'.format(opt_alpha)
        print model.cargo.marg
        self.__update_alpha(opt_alpha, model)
        
        x0 = self._wrap(beta, gamma)
        n = len(beta)
        d = len(gamma[0])
        
        print 'beta={0}'.format(beta)
        print 'gamma={0}'.format(gamma)
        print 'X0={0}'.format(x0)
            
        
        #constraints for the precision matrix - elements must be posititive
        constraints = [(None, None)]*2
        constraints.extend([(0,None)]*(n-2))
        constraints.extend([(None, None)]*d)
        constraints.extend([(0,None)]*(d*d)) #L
        constraints.extend([(None,None), (d+1,None)])#r,v
            
            
        print 'likel_before={0}'.format(model.cargo.marg)
            
        model_likel_fun = _BHCModelLikelihood(model)
        model_likel_grad = model_likel_fun.gradient
        print 'foo'
           
        #opt_result = spopt.fmin_bfgs(model_likel_fun, x0, model_likel_grad, full_output=True)
        opt_result = spopt.fmin_tnc(model_likel_fun, x0, model_likel_grad, bounds=constraints)
        opt_beta, opt_gamma = self._unwrap(opt_result[0], n, d)
        opt_gamma = list(opt_gamma)
        likel = opt_result[1]
        print 'opt_beta={0}'.format(opt_beta)
        print 'opt_gamma={0}'.format(opt_gamma)
        print likel
            
        return (likel, opt_alpha, opt_beta, opt_gamma) 
    
    def _wrap(self, beta, gamma):
        '''
        @todo: make static
        '''
        m, R, r, v = gamma
        L = np.linalg.cholesky(R)
        
        params = np.r_[beta, m, L.ravel(), r, v]
        return params
    
    def _unwrap(self, params, n, d):
        '''
        @todo: make static
        '''
        beta = params[0:n]
        m = params[n:n+d]
        L = params[n+d:n+d+d*d]
        L = np.reshape(L, (d,d))
        L = np.tril(L)
        R = np.dot(L, L.T)
        r = params[-2]
        v = params[-1]
        gamma = (m,R,r,v)
        return beta, gamma
         
    
    def predict(self, X, Z, method='path', incl_tree_weights=True):
        '''  
        @todo: - more generic parametrization for invoking the different prediction methods
        '''
        self._init_check()
        X = asmtarray_object(X)
        Z = asmtarray_object(Z)
        
        if len(X) != len(Z):
            raise ValueError('number of task data X and task' + 
                             'description Z must be the same')
            
        if method == 'path':
            pred_method = self.__predict_path
        elif method == 'path1':
            pred_method = self.__predict_path1
        elif method == 'fullpath':
            pred_method = self.__predict_fullpath
        elif method == 'maxcluster':
            pred_method = self.__predict_maxcluster
        elif method == 'complete':
            pred_method = self.__predict_complete
        elif method == 'flat_complete':
            pred_method = self.__predict_flat_complete
        elif method == 'flat_maxcluster':
            pred_method = self.__predict_flat_maxcluster
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        n = len(X)
        Y = np.empty(n, dtype='object')
        if self._use_task_features:
            #todo: its a hack... optimize
            for i in xrange(n):
                Xi = X[i]
                Zi = Z[i]
                m = len(Xi)
                Yi = np.empty(m)
                for j in xrange(m):
                    Xij = Xi[j]
                    Zij = np.r_[Xij, Zi]
                    Xij = np.atleast_2d(Xij)
                    Yi[j] = np.float(np.asarray(pred_method(Xij, Zij, incl_tree_weights)))
                    
                Y[i] = Yi
        else:
            for i in xrange(n):
                yhat = pred_method(X[i], Z[i], incl_tree_weights)
                Y[i] = yhat
            
        return Y
    
    def _get_likelihood(self):
        '''
        '''
        self._init_check()
        cluster = self._task_tree.cargo
        return cluster.marg
    
    likelihood = property(fget=_get_likelihood)
    
    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')
    
    
    #Methods for Learning
    
    def _build_bhc_model(self, alpha, beta, gamma, X, Y, Z):
        '''
        @todo: - implement the access of the block matrix R
        '''
        
        n = len(X)
        
        task_cluster = [None]*n
        nodes = [None]*n
         
        #initialize the leafs of the cluster hierarchy by each task of its own
        for i in xrange(n):
            Xi = X[i]
            Yi = Y[i]
            Zi = Z[i]
            
            model = self.__make_expert_model(Xi, Yi, Zi, beta, gamma)
            task_cluster[i] = _ClusterNode.make_leaf(i, Xi, Yi, Zi, model, alpha)
            nodes[i] = MutableBTree(task_cluster[i])
        
        #initialize the model matrix and the probability matrix of merged 
        #hypothesis
        model_matrix = np.empty([n,n], dtype=object)
        prob_matrix = np.ones([n,n], dtype=float)*-np.inf
        for i in xrange(n):
            cluster_i = task_cluster[i]
            
            for j in xrange(i+1, n):
                    
                cluster_j = task_cluster[j]
                Xij = np.vstack((cluster_i.X, cluster_j.X))
                yij = np.hstack((cluster_i.y, cluster_j.y))
                Zij = np.vstack((cluster_i.Z, cluster_j.Z))
                    
                model = self.__make_expert_model(Xij, yij, Zij, beta, gamma)
                cluster = _ClusterNode.make_internal(cluster_i, cluster_j, 
                                                     model, alpha)
                    
                model_matrix[i,j] = model_matrix[j,i] = model
                prob_matrix[i,j] = prob_matrix[j,i] = cluster.post
                
        #print 'pobmatr'
        #print prob_matrix
        #main-loop of the hierarchical task clustering process
        n_cluster = n
        while n_cluster > 1:
            
            
            #print 'pobmatr'
            #print prob_matrix
            
            #find the two tasks with highest merge hypothesis posterior
            max_prob = -np.inf 
            p = q = -1
            for i in xrange(n):
                if task_cluster[i] != None:
                    for j in xrange(i+1,n):
                        if task_cluster[j] != None and prob_matrix[i,j] > max_prob:
                            max_prob = prob_matrix[i,j]
                            p = i
                            q = j
            
            #print 'max_prob={0}'.format(max_prob)
            #print 'p={0},q={1}'.format(p,q)     
            #merge the two tasks p,q with highest posterior
            cluster_p = task_cluster[p]
            cluster_q = task_cluster[q]            
            cluster = _ClusterNode.make_internal(cluster_p, cluster_q, 
                                                 model_matrix[p,q], alpha)
            
            task_cluster[p] = cluster
            task_cluster[q] = None
            nodes[p] = MutableBTree(cluster, None, nodes[p], nodes[q])
            nodes[q] = None 
            
            #update model and probability matrix
            model_matrix[p,q] = model_matrix[q,p] = None
            prob_matrix[p,q] = prob_matrix[q,p] = -1.0
            for i in xrange(n):
                if task_cluster[i] != None and i != p:
                    cluster_i = task_cluster[i]
                    
                    Xi = np.vstack((cluster.X, cluster_i.X)) #same as np.r_[X[p],X[q],X[i]]
                    yi = np.hstack((cluster.y, cluster_i.y)) #same as np.r_[Y[p],Y[q],Y[i]]
                    Zi = np.vstack((cluster.Z, cluster_i.Z))
                    
                    model = self.__make_expert_model(Xi, yi, Zi, beta, gamma)
                    tmp_cluster = _ClusterNode.make_internal(cluster, cluster_i, 
                                                             model, alpha)
                    
                    
                    model_matrix[p,i] = model_matrix[i,p] = model
                    model_matrix[q,i] = model_matrix[i,q] = None
                    prob_matrix[p,i] = prob_matrix[i,p] = tmp_cluster.post
                    prob_matrix[q,i] = prob_matrix[i,q] = -1
                    
                    #todo: prepare block matrix
                    
            n_cluster -= 1
            
        cluster_tree = nodes[0]
        self.__update_weights(cluster_tree)
        return cluster_tree
    
    def __make_expert_model(self, X, y, Z, beta, gamma):
        a,b,L = RobustBayesRegression.unwrap(beta)
        reg_model = RobustBayesRegression(a,b,L)
        reg_model.fit(X,y)
        
        m, R, r, v = gamma
        density_model = NormalInvWishart(Z, m, R, np.exp(r), v)
        
        model = _ExpertModel(X, y, Z, reg_model, density_model) 
        return model
    
    def _update_model(self, alpha, beta, gamma, treenode):
        self._beta = beta
        self._alpha = alpha
        
        self.__update_alpha(alpha, treenode)
        self.__update_model_parameters(beta, gamma, treenode)
        self.__update_weights(treenode)

    def __update_weights(self, treenode):
        '''
        @todo: - weight each instance by the number of elements in parent and sibling
        '''
        for node in treenode.subtree():
            cluster = node.cargo
            cluster.weight = np.exp(cluster.post) 
            cluster.inv_weight = 1.0-cluster.weight
            #if cluster.inv_weight == 0.0:
            #    cluster.inv_weight += 1e-16
            if not node.isroot():
                parent = node.parent
                super_cluster = parent.cargo
                
                #new stuff for normalizing (see phd thesis of heller)
                ns = np.float(super_cluster.ntasks)
                n = np.float(cluster.ntasks)
                norm = n/ns
                #norm = 1
                
                inv_weight = super_cluster.inv_weight*norm
                cluster.weight *= inv_weight
                cluster.inv_weight *= inv_weight
            
                
            #print 'cluster_weight={0}'.format(cluster.weight)

    def __update_alpha(self, alpha, treenode):
        '''
        '''
        if treenode.isatom():
            cluster = treenode.cargo
            cluster.norm_const = alpha+sps.gammaln(cluster.ntasks)
        else:
            self.__update_alpha(alpha, treenode.left)
            self.__update_alpha(alpha, treenode.right)
            
            cluster = treenode.cargo
            left = treenode.left.cargo
            right = treenode.right.cargo
            
            cluster = treenode.cargo
            
            model = cluster.model
            n = cluster.ntasks
        
            #compute the cluster prior 'pi'. Because numerical issues, the norm_const
            #is compute by the sum of individual logs.
            nom = alpha+sps.gammaln(n) 
            cluster.norm_const = sumln([nom, left.norm_const+right.norm_const]) 
            cluster.pi = nom - cluster.norm_const
        
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = left.norm_const + right.norm_const - cluster.norm_const
            a = cluster.pi + model.log_likel
            b = npi+left.marg+right.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + model.log_likel - cluster.marg
            

    def __update_model_parameters(self, beta, gamma, treenode):
        '''
        @todo: kernel is not used
        '''
        if treenode.isatom():
            cluster = treenode.cargo
            
            #rebuild the gp model
            model = self.__make_expert_model(cluster.X, cluster.y, cluster.Z, beta, gamma)
            cluster.model = model
            cluster.marg = model.log_likel
        else:
            self.__update_model_parameters(beta, gamma, treenode.left)
            self.__update_model_parameters(beta, gamma, treenode.right)
            
            cluster = treenode.cargo
            left = treenode.left.cargo
            right = treenode.right.cargo
            
            cluster = treenode.cargo

            #rebuild the gp model
            model = self.__make_expert_model(cluster.X, cluster.y, cluster.Z, beta, gamma)
            cluster.model = model            
            
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = left.norm_const + right.norm_const - cluster.norm_const
            a = cluster.pi + model.log_likel
            b = npi+left.marg+right.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + model.log_likel - cluster.marg
            
    def __predict_complete(self, X, z, incl_tree_weights=True):
        '''
        '''
        tree = self._task_tree
        n = X.shape[0]
        yhat = np.zeros(n)
        
        total_weight = 0
        for cluster in tree:
            model = cluster.model
            weight = model.responsibility(z)
            if incl_tree_weights:
                weight *= cluster.weight
            total_weight += weight
            
        for cluster in tree:
            model = cluster.model
            weight = model.responsibility(z)
            if incl_tree_weights:
                weight *= cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat
    
    def __predict_maxcluster(self, X, z, incl_tree_weights=True):
        tree = self._task_tree
        
        max_cluster = None
        max_response = -1
        for cluster in tree:
            model = cluster.model
            response = model.responsibility(z)
            if response > max_response:
                max_response = response
                max_cluster = cluster
                
        yhat = max_cluster.model.predict(X)
        return yhat
        
    def __predict_path(self, X, z, incl_tree_weights=True):
        '''
        @todo: - check whether the treenodes putting into the cluster_set are correctly 
                 inserted, that means no collosion occurs. Is the id() function sufficient
                 to compute the hash value of the object? What is better, using treenode or
                 cluster nodes?
        '''
        n = X.shape[0]
        yhat = np.zeros(n)
        
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing

        max_node = None
        max_response = -1
        for node in leaves:
            cluster = node.cargo
            model = cluster.model
            response = model.responsibility(z)
            if response > max_response:
                max_response = response
                max_node = node
        
     
        path = path_to_root(max_node)
        
        print '-----------'
        total_weight = 0
        for node in path:
            cluster = node.cargo
            model = cluster.model
            weight = model.responsibility(z)
            print 'responsibility={0}'.format(weight)
            if incl_tree_weights:
                print 'cluster weight={0}'.format(cluster.weight)
                weight *= cluster.weight
            total_weight += weight
            
        for node in path:
            cluster = node.cargo
            model = cluster.model
            weight = model.responsibility(z)
            if incl_tree_weights:
                weight *= cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat
    
    def __predict_path1(self, X, z, incl_tree_weights=True):
        '''
        @todo: - check whether the treenodes putting into the cluster_set are correctly 
                 inserted, that means no collosion occurs. Is the id() function sufficient
                 to compute the hash value of the object? What is better, using treenode or
                 cluster nodes?
        '''
        n = X.shape[0]
        yhat = np.zeros(n)
        
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing

        max_node = None
        max_response = -1
        for node in leaves:
            cluster = node.cargo
            model = cluster.model
            response = model.responsibility(z)
            if response > max_response:
                max_response = response
                max_node = node
        
     
        path = path_to_root(max_node)
        
        total_weight = 0
        for node in path:
            cluster = node.cargo
            model = cluster.model
            total_weight += cluster.weight
            
        for node in path:
            cluster = node.cargo
            model = cluster.model
            weight = cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat


    def __predict_fullpath(self, X, z, incl_tree_weights=True):
        '''
        @todo: - check whether the treenodes putting into the cluster_set are correctly 
                 inserted, that means no collosion occurs. Is the id() function sufficient
                 to compute the hash value of the object? What is better, using treenode or
                 cluster nodes?
        '''
        n = X.shape[0]
        yhat = np.zeros(n)
        
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing

        max_node = None
        max_response = -1
        for node in leaves:
            cluster = node.cargo
            model = cluster.model
            response = model.responsibility(z)
            if response > max_response:
                max_response = response
                max_node = node
        
     
        path = path_to_root(max_node)
        
        total_weight = 0
        for node in path:
            cluster = node.cargo
            model = cluster.model
            weight = model.responsibility(z)
            if incl_tree_weights:
                weight *= cluster.weight
            total_weight += weight
            
        for node in path:
            cluster = node.cargo
            model = cluster.model
            weight = model.responsibility(z)
            if incl_tree_weights:
                weight *= cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat

    def __predict_flat_maxcluster(self,  X, z, incl_tree_weights=True, r=0.5):
        tree = self._task_tree
        clusters = self.__flatten_hierarchy(tree, r)
        
        max_cluster = None
        max_response = -1
        for c in clusters:
            model = c.model
            response = model.responsibility(z)
            if response > max_response:
                max_response = response
                max_cluster = c
                
        yhat = max_cluster.model.predict(X)
        return yhat
    
    def __predict_flat_complete(self,  X, z, incl_tree_weights=True, r=0.5):
        tree = self._task_tree
        clusters = self.__flatten_hierarchy(tree, r)
        
        n = X.shape[0]
        yhat = np.zeros(n)
        
        total_weight = 0
        for c in clusters:
            model = c.model
            total_weight += model.responsibility(z)
            
        for c in clusters:
            model = c.model
            weight = model.responsibility(z)
            yhat += weight*model.predict(X)/total_weight
        
        return yhat
        
    def __flatten_hierarchy(self, treenode, r=0.5):
        '''
        '''
        clusters = list()
        queue = deque([treenode])
        
        #iterate aslong the queue is empty
        while queue:
            node = queue.pop()
            c = node.cargo
            if c.post >= np.log(r):
                clusters.append(c)
            else:
                queue.append(node.left)
                queue.append(node.right)
        
        return clusters
        
        

    #Misc methods
    def xml_element(self):
        '''
        '''
        root_element = ElementTree.Element('bhc_model')
        
        #build parameter element 
        param_element = ElementTree.SubElement(root_element, 'parameters')
        param_element.attrib['rho'] = str(self._rho)
        param_element.attrib['alpha'] = str(self._alpha)
        param_element.attrib['beta'] = str(self._beta)
        
        #build task tree element
        task_tree_element = ElementTree.SubElement(root_element, 'task_tree')
        task_tree_element.append(self._tree2xml(self._task_tree)) 
        
        return root_element

    def _tree2xml(self, treenode):
        '''
        '''
        cluster = treenode.cargo
        reg_model = cluster.model
        
        if treenode.isatom():
            cluster_element = ElementTree.Element('leaf')
            cluster_element.attrib['id'] = str(cluster.task_id+1)
        else:
            cluster_element = ElementTree.Element('task_cluster')
            
        cluster_element.attrib['descr'] = str(cluster.z)
        
        likel_element = ElementTree.SubElement(cluster_element, 'likel')
        likel_element.attrib['post'] = str(cluster.post)
        likel_element.attrib['marg'] = str(cluster.marg)
        likel_element.attrib['mlikel'] = str(reg_model.log_evidence)
        likel_element.attrib['pi'] = str(cluster.pi)
        likel_element.attrib['norm'] = str(cluster.norm_const)
        
        pred_element = ElementTree.SubElement(cluster_element, 'predict')
        pred_element.attrib['weight'] = str(cluster.weight)
        pred_element.attrib['invweight'] = str(cluster.inv_weight)
        
        prop_element = ElementTree.SubElement(cluster_element, 'props')
        prop_element.attrib['ntask'] = str(cluster.ntasks)
        prop_element.attrib['size'] = str(cluster.size)
        prop_element.attrib['shape'] = str(cluster.covariates.shape)
        
        model_element = ElementTree.SubElement(cluster_element, 'reg_model')
        model_element.attrib['intercept'] = str(reg_model.intercept)
        model_element.attrib['weights'] = str(reg_model.weights)
        
        if not treenode.isatom():
            cluster_element.append(self._tree2xml(treenode.left))
            cluster_element.append(self._tree2xml(treenode.right))
            
        return cluster_element
        
    def marshal_xml(self, filename):
        '''
        '''
        element = self.xml_element()
        tree = ElementTree.ElementTree(element)
        try:
            tree.write(filename, "UTF-8")
        except EnvironmentError as err:
            print("{0}: import error {1}".format(os.path.basename(sys.argv[0]), err))


class SparseBHCGPRegregressionExpert(BaseEstimator, RegressorMixin):
    '''     
    @todo: - 
    '''
    
    __slots__ = ('_alpha',          #cluster concentration hyperparameter in log space
                 '_gamma',          #hyperparameters of the gating function (NiW distribution)
                 '_kernel',         #initial kernel
                 '_priors',         #priors of kernel hyperparameters
                 '_selector',       #selection method for inducing points
                 '_k',              #number of inducing points
                 '_task_tree',      #the cluster tree of regression models
                 '_use_task_features',
                 '_is_init'
                 )        
    
    def __init__(self, alpha, kernel, priors=None, k=10, 
                 use_task_features=False):
        '''
        @todo: - check the number of hyperparams and priors
        '''
        self._alpha = alpha
        self._kernel = kernel
        self._priors = priors
        #self._selector = selector
        self._use_task_features = use_task_features
        
        self._k = k
        
        self._is_init = False
        
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        return np.r_[self._alpha, self._kernel.params]
    
    def _set_hyperparams(self, params):
        '''
        '''
        self._alpha = params[0]
        self._kernel.params = params[1:]
        
    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)

    def fit(self, X, Y, Z, max_it=50, tol=1.e-3):
        '''
        '''
        
        X = asmtarray_object(X)
        Y = asmtarray_object(Y)
        Z = np.asanyarray(Z)
        
        if len(X) != len(Y) != len(Z):
            raise ValueError('number of task data X, Y and task ' + 
                             'description Z must be the same')
        
        #todo: make the code more  clear
        #determine a set of inducing points using k-means
        
        kmeans = KMeans(self._k)
        kmeans.fit(flatten_data(X))
        Xu = kmeans.centers
        self._selector = FixedSelector(Xu)
        
        kernel = self._kernel
        priors = self._priors
        alpha = self._alpha
        beta = np.copy(kernel.params)
        
        #preprocess
        if self._use_task_features:
            n = len(X)
            d = X[0].shape[1]
            Xmean = np.zeros((n,d))
            for i in xrange(len(X)):
                Xmean[i] = np.mean(X[i], 0)
            
            Z = np.c_[Xmean, Z]
            
        #estimate hyperparameters of the gating model (NiW distribution)
        #todo: add a small amount to non diagonal elements, because the matrix is optimized in log space
        #wenn die werte null sind, wird nur die diag optimiert
        m = np.mean(Z, 0)
        S = np.diag(1.0/np.var(Z, 0)) #+ np.ones((Z.shape[1],Z.shape[1]))*1e-05
        #S = np.eye(Z.shape[1]) + np.ones((Z.shape[1],Z.shape[1]))*1e-05
        r = np.log(1)
        v = Z.shape[1]+1
        gamma = (m, S, r, v) 
        
        print 'mean={0}'.format(m)
        print 'prec={0}'.format(S)
        
        #optimize the concentration parameter alpha and the model hyperparameters
        # by an iterative procedure where in the first step the best tree structure 
        #is find for fixed parameters and in the second step the parameters is 
        #optimized by an alternating line search and gradient descent mehtod
        i = 0
        bhc_model = None
        likel = likel_old = np.Inf
        while i < max_it:
            
            if i > 0 and likel_old-likel < tol:
                break
            
            likel_old = likel
            
            #E-Step: estimate the new tree of the optimized parameters
            model = self._build_bhc_model(alpha, gamma, X, Y, Z)
            
            #M-Step: optimize the parameters on a fixed tree
            #beta are the model hyperparameters
            likel, alpha, beta, gamma = self._optimize_bhc_parameters(model, alpha, beta, gamma)
            bhc_model = model
            i += 1
            
        self._update_model(alpha, beta, gamma, bhc_model)
        self.__refit_local_models(bhc_model)

        
        self._task_tree = bhc_model
        self._alpha = alpha
        
        self._is_init = True  
        
    def _optimize_bhc_parameters(self, model, alpha, beta, gamma):
        '''
        '''        
        print model.cargo.marg
        alpha_likel_fun = BHCAlphaLikelihood(model)
        print 'alpha={0}'.format(alpha)
        
        try:
            opt_alpha = spopt.brent(alpha_likel_fun, brack=(alpha-10, alpha, alpha+10))
        except AssertionError:
            opt_alpha = spopt.brent(alpha_likel_fun, brack=(alpha-2, alpha+2))
        
        
        print 'opt_alpha={0}'.format(opt_alpha)
        print model.cargo.marg
        self.__update_alpha(opt_alpha, model)
        
        #todo: hack
        x0 = self._wrap(beta, gamma)
        n = len(beta)
        d = len(gamma[0])
        
        #constraints for the precision matrix - elements must be posititive
        
        constraints = [(None, None)]*(n+d) 
        constraints.extend([(0,None)]*(d*d)) #L
        constraints.extend([(None,None), (d+1,None)])#r,v
            
            
        print 'likel_before={0}'.format(model.cargo.marg)
            
        model_likel_fun = _BHCModelLikelihood(model)
        model_likel_grad = model_likel_fun.gradient
        print 'foo'
        #opt_result = spopt.fmin_bfgs(model_likel_fun, x0, model_likel_grad, full_output=True)
        opt_result = spopt.fmin_tnc(model_likel_fun, x0, model_likel_grad, bounds=constraints)
        opt_beta, opt_gamma = self._unwrap(opt_result[0], n, d)
        opt_gamma = list(opt_gamma)
        likel = opt_result[1]
        print 'opt_beta={0}'.format(np.exp(opt_beta))
        print 'opt_gamma={0}'.format(opt_gamma)
        print likel
        
        return (likel, opt_alpha, opt_beta, opt_gamma) 
        
    def _wrap(self, beta, gamma):
        '''
        @todo: make static
        '''
        m, R, r, v = gamma
        L = np.linalg.cholesky(R)
        
        params = np.r_[beta, m, L.ravel(), r, v]
        return params
    
    def _unwrap(self, params, n, d):
        '''
        @todo: make static
        '''
        beta = params[0:n]
        m = params[n:n+d]
        L = params[n+d:n+d+d*d]
        L = np.reshape(L, (d,d))
        L = np.tril(L)
        R = np.dot(L, L.T)
        r = params[-2]
        v = params[-1]
        gamma = (m,R,r,v)
        return beta, gamma         
    
    def predict(self, X, Z, method='path', incl_tree_weights=True):
        '''  
        @todo: - more generic parametrization for invoking the different prediction methods
        '''
        self._init_check()
        X = asmtarray_object(X)
        Z = asmtarray_object(Z)
        
        if len(X) != len(Z):
            raise ValueError('number of task data X and task' + 
                             'description Z must be the same')
            
        if method == 'path':
            pred_method = self.__predict_path
        elif method == 'path1':
            pred_method = self.__predict_path1
        elif method == 'fullpath':
            pred_method = self.__predict_fullpath
        elif method == 'maxcluster':
            pred_method = self.__predict_maxcluster
        elif method == 'complete':
            pred_method = self.__predict_complete
        elif method == 'flat_complete':
            pred_method = self.__predict_flat_complete
        elif method == 'flat_maxcluster':
            pred_method = self.__predict_flat_maxcluster
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        n = len(X)
        Y = np.empty(n, dtype='object')
        if self._use_task_features:
            #todo: its a hack... optimize
            for i in xrange(n):
                Xi = X[i]
                Zi = Z[i]
                m = len(Xi)
                Yi = np.empty(m)
                for j in xrange(m):
                    Xij = Xi[j]
                    Zij = np.r_[Xij, Zi]
                    Xij = np.atleast_2d(Xij)
                    Yi[j] = np.float(np.asarray(pred_method(Xij, Zij, incl_tree_weights)))
                    
                Y[i] = Yi
        else:
            for i in xrange(n):
                yhat = pred_method(X[i], Z[i], incl_tree_weights)
                Y[i] = yhat
            
        return Y
    
    def _get_likelihood(self):
        '''
        '''
        self._init_check()
        cluster = self._task_tree.cargo
        return cluster.marg
    
    likelihood = property(fget=_get_likelihood)
    
    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')
    
    
    #Methods for Learning
    
    def _build_bhc_model(self, alpha, gamma, X, Y, Z):
        '''
        @todo: - implement the access of the block matrix R
        '''
        
        n = len(X)
        
        task_cluster = [None]*n
        nodes = [None]*n
         
        #initialize the leafs of the cluster hierarchy by each task of its own
        for i in xrange(n):
            Xi = X[i]
            Yi = Y[i]
            Zi = Z[i]
            
            model = self.__make_expert_model(Xi, Yi, Zi, gamma)
            task_cluster[i] = _ClusterNode.make_leaf(i, Xi, Yi, Zi, model, alpha)
            nodes[i] = MutableBTree(task_cluster[i])
        
        #initialize the model matrix and the probability matrix of merged 
        #hypothesis
        model_matrix = np.empty([n,n], dtype=object)
        prob_matrix = np.ones([n,n], dtype=float)*-np.inf
        for i in xrange(n):
            cluster_i = task_cluster[i]
            
            for j in xrange(i+1, n):
                    
                cluster_j = task_cluster[j]
                Xij = np.vstack((cluster_i.X, cluster_j.X))
                yij = np.hstack((cluster_i.y, cluster_j.y))
                Zij = np.vstack((cluster_i.Z, cluster_j.Z))
                    
                model = self.__make_expert_model(Xij, yij, Zij, gamma)
                cluster = _ClusterNode.make_internal(cluster_i, cluster_j, 
                                                     model, alpha)
                    
                model_matrix[i,j] = model_matrix[j,i] = model
                prob_matrix[i,j] = prob_matrix[j,i] = cluster.post
                
        #print prob_matrix
        #main-loop of the hierarchical task clustering process
        n_cluster = n
        while n_cluster > 1:
            
            #find the two tasks with highest merge hypothesis posterior
            max_prob = -np.inf 
            p = q = -1
            for i in xrange(n):
                if task_cluster[i] != None:
                    for j in xrange(i+1,n):
                        if task_cluster[j] != None and prob_matrix[i,j] > max_prob:
                            max_prob = prob_matrix[i,j]
                            p = i
                            q = j
                            
            #merge the two tasks p,q with highest posterior
            cluster_p = task_cluster[p]
            cluster_q = task_cluster[q]            
            cluster = _ClusterNode.make_internal(cluster_p, cluster_q, 
                                                 model_matrix[p,q], alpha)
            
            task_cluster[p] = cluster
            task_cluster[q] = None
            nodes[p] = MutableBTree(cluster, None, nodes[p], nodes[q])
            nodes[q] = None 
            
            #update model and probability matrix
            model_matrix[p,q] = model_matrix[q,p] = None
            prob_matrix[p,q] = prob_matrix[q,p] = -1.0
            for i in xrange(n):
                if task_cluster[i] != None and i != p:
                    cluster_i = task_cluster[i]
                    
                    Xi = np.vstack((cluster.X, cluster_i.X)) #same as np.r_[X[p],X[q],X[i]]
                    yi = np.hstack((cluster.y, cluster_i.y)) #same as np.r_[Y[p],Y[q],Y[i]]
                    Zi = np.vstack((cluster.Z, cluster_i.Z))
                    
                    model = self.__make_expert_model(Xi, yi, Zi, gamma)
                    tmp_cluster = _ClusterNode.make_internal(cluster, cluster_i, 
                                                             model, alpha)
                    
                    
                    model_matrix[p,i] = model_matrix[i,p] = model
                    model_matrix[q,i] = model_matrix[i,q] = None
                    prob_matrix[p,i] = prob_matrix[i,p] = tmp_cluster.post
                    prob_matrix[q,i] = prob_matrix[i,q] = -1
                    
                    #todo: prepare block matrix
                    
            n_cluster -= 1
            
        cluster_tree = nodes[0]
        self.__update_weights(cluster_tree)
        return cluster_tree
    
    def __make_expert_model(self, X, y, Z, gamma):
        kernel = self._kernel
        priors = self._priors
        selector = self._selector
        
        reg_model = SparseGPRegression(kernel, priors=priors, selector=selector)
        reg_model.fit(X,y)
        
        
        m, R, r, v = gamma
        density_model = NormalInvWishart(Z, m, R, np.exp(r), v)
        
        model = _ExpertModel(X, y, Z, reg_model, density_model) 
        return model
    
    def _update_model(self, alpha, beta, gamma, treenode):
        self._beta = beta
        self._alpha = alpha
        
        self.__update_alpha(alpha, treenode)
        self.__update_model_parameters(beta, gamma, treenode)
        self.__update_weights(treenode)

    def __update_weights(self, treenode):
        '''
        @todo: - weight each instance by the number of elements in parent and sibling
        '''
        for node in treenode.subtree():
            cluster = node.cargo
            cluster.weight = np.exp(cluster.post) 
            cluster.inv_weight = 1.0-cluster.weight
            #if cluster.inv_weight == 0.0:
            #    cluster.inv_weight += 1e-16
            if not node.isroot():
                parent = node.parent
                super_cluster = parent.cargo
                
                #new stuff for normalizing (see phd thesis of heller)
                ns = np.float(super_cluster.ntasks)
                n = np.float(cluster.ntasks)
                norm = n/ns
                #norm = 1
                
                inv_weight = super_cluster.inv_weight*norm
                cluster.weight *= inv_weight
                cluster.inv_weight *= inv_weight
            
                
            #print 'cluster_weight={0}'.format(cluster.weight)

    def __update_alpha(self, alpha, treenode):
        '''
        '''
        if treenode.isatom():
            cluster = treenode.cargo
            cluster.norm_const = alpha+sps.gammaln(cluster.ntasks)
        else:
            self.__update_alpha(alpha, treenode.left)
            self.__update_alpha(alpha, treenode.right)
            
            cluster = treenode.cargo
            left = treenode.left.cargo
            right = treenode.right.cargo
            
            cluster = treenode.cargo
            
            model = cluster.model
            n = cluster.ntasks
        
            #compute the cluster prior 'pi'. Because numerical issues, the norm_const
            #is compute by the sum of individual logs.
            nom = alpha+sps.gammaln(n) 
            cluster.norm_const = sumln([nom, left.norm_const+right.norm_const]) 
            cluster.pi = nom - cluster.norm_const
        
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = left.norm_const + right.norm_const - cluster.norm_const
            a = cluster.pi + model.log_likel
            b = npi+left.marg+right.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + model.log_likel - cluster.marg
            
    def __refit_local_models(self, treenode):
        if treenode.isatom():
            cluster = treenode.cargo
            
            #rebuild the gp model
            kernel = self._kernel.copy()
            priors = self._priors
            selector = self._selector
        
            reg_model = SparseGPRegression(kernel, infer_method=FITCExactInference, priors=priors, selector=selector)
            reg_model.fit(cluster.X, cluster.y)
            
            cluster.model._reg_model = reg_model
            cluster.marg = cluster.model.log_likel
        else:
            self.__refit_local_models(treenode.left)
            self.__refit_local_models(treenode.right)
            
            cluster = treenode.cargo
            left = treenode.left.cargo
            right = treenode.right.cargo
            
            cluster = treenode.cargo

            #rebuild the gp model
            kernel = self._kernel.copy()
            priors = self._priors
            selector = self._selector
        
            reg_model = SparseGPRegression(kernel, infer_method=FITCExactInference, priors=priors, selector=selector)
            reg_model.fit(cluster.X, cluster.y)
            
            cluster.model._reg_model = reg_model            
            
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = left.norm_const + right.norm_const - cluster.norm_const
            a = cluster.pi + cluster.model.log_likel
            b = npi+left.marg+right.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + cluster.model.log_likel - cluster.marg

    def __update_model_parameters(self, beta, gamma, treenode):
        '''
        @todo: kernel is not used
        '''
        if treenode.isatom():
            cluster = treenode.cargo
            
            #rebuild the gp model
            model = self.__make_expert_model(cluster.X, cluster.y, cluster.Z, gamma)
            cluster.model = model
            cluster.marg = model.log_likel
        else:
            self.__update_model_parameters(beta, gamma, treenode.left)
            self.__update_model_parameters(beta, gamma, treenode.right)
            
            cluster = treenode.cargo
            left = treenode.left.cargo
            right = treenode.right.cargo
            
            cluster = treenode.cargo

            #rebuild the gp model
            model = self.__make_expert_model(cluster.X, cluster.y, cluster.Z, gamma)
            cluster.model = model            
            
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = left.norm_const + right.norm_const - cluster.norm_const
            a = cluster.pi + model.log_likel
            b = npi+left.marg+right.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + model.log_likel - cluster.marg
            
    def __predict_complete(self, X, z, incl_tree_weights=True):
        '''
        '''
        tree = self._task_tree
        n = X.shape[0]
        yhat = np.zeros(n)
        
        print 'complete'
        total_weight = 0
        for cluster in tree:
            model = cluster.model
            weight = model.responsibility(z)
            print 'response={0}'.format(weight)
            if incl_tree_weights:
                weight *= cluster.weight
            total_weight += weight
            
        for cluster in tree:
            model = cluster.model
            weight = model.responsibility(z)
            if incl_tree_weights:
                weight *= cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat
    
    def __predict_maxcluster(self, X, z, incl_tree_weights=True):
        tree = self._task_tree
        
        max_cluster = None
        max_response = -1
        for cluster in tree:
            model = cluster.model
            response = model.responsibility(z)
            if response > max_response:
                max_response = response
                max_cluster = cluster
                
        yhat = max_cluster.model.predict(X)
        return yhat
        
    def __predict_path(self, X, z, incl_tree_weights=True):
        '''
        @todo: - check whether the treenodes putting into the cluster_set are correctly 
                 inserted, that means no collosion occurs. Is the id() function sufficient
                 to compute the hash value of the object? What is better, using treenode or
                 cluster nodes?
        '''
        n = X.shape[0]
        yhat = np.zeros(n)
        
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing

        max_node = None
        max_response = -1
        for node in leaves:
            cluster = node.cargo
            model = cluster.model
            response = model.responsibility(z)
            if response > max_response:
                max_response = response
                max_node = node
        
     
        path = path_to_root(max_node)
        
        print '-----------------'
        total_weight = 0
        for node in path:
            cluster = node.cargo
            model = cluster.model
            weight = model.responsibility(z)
            print 'cluster response={0}'.format(weight)
            if incl_tree_weights:
                print 'cluster weight={0}'.format(cluster.weight)
                weight *= cluster.weight
            total_weight += weight
            
        for node in path:
            cluster = node.cargo
            model = cluster.model
            weight = model.responsibility(z)
            if incl_tree_weights:
                weight *= cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat
    
    def __predict_path1(self, X, z, incl_tree_weights=True):
        '''
        @todo: - check whether the treenodes putting into the cluster_set are correctly 
                 inserted, that means no collosion occurs. Is the id() function sufficient
                 to compute the hash value of the object? What is better, using treenode or
                 cluster nodes?
        '''
        n = X.shape[0]
        yhat = np.zeros(n)
        
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing

        max_node = None
        max_response = -1
        for node in leaves:
            cluster = node.cargo
            model = cluster.model
            response = model.responsibility(z)
            if response > max_response:
                max_response = response
                max_node = node
        
     
        path = path_to_root(max_node)
        
        total_weight = 0
        for node in path:
            cluster = node.cargo
            model = cluster.model
            total_weight += cluster.weight
            
        for node in path:
            cluster = node.cargo
            model = cluster.model
            weight = cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat


    def __predict_fullpath(self, X, z, incl_tree_weights=True):
        '''
        @todo: - check whether the treenodes putting into the cluster_set are correctly 
                 inserted, that means no collosion occurs. Is the id() function sufficient
                 to compute the hash value of the object? What is better, using treenode or
                 cluster nodes?
        '''
        n = X.shape[0]
        yhat = np.zeros(n)
        
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing

        max_node = None
        max_response = -1
        for node in leaves:
            cluster = node.cargo
            model = cluster.model
            response = model.responsibility(z)
            if response > max_response:
                max_response = response
                max_node = node
        
     
        path = path_to_root(max_node)
        
        total_weight = 0
        for node in path:
            cluster = node.cargo
            model = cluster.model
            weight = model.responsibility(z)
            if incl_tree_weights:
                weight *= cluster.weight
            total_weight += weight
            
        for node in path:
            cluster = node.cargo
            model = cluster.model
            weight = model.responsibility(z)
            if incl_tree_weights:
                weight *= cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat

    def __predict_flat_maxcluster(self,  X, z, incl_tree_weights=True, r=0.5):
        tree = self._task_tree
        clusters = self.__flatten_hierarchy(tree, r)
        
        max_cluster = None
        max_response = -1
        for c in clusters:
            model = c.model
            response = model.responsibility(z)
            if response > max_response:
                max_response = response
                max_cluster = c
                
        yhat = max_cluster.model.predict(X)
        return yhat
    
    def __predict_flat_complete(self,  X, z, incl_tree_weights=True, r=0.5):
        tree = self._task_tree
        clusters = self.__flatten_hierarchy(tree, r)
        
        n = X.shape[0]
        yhat = np.zeros(n)
        
        total_weight = 0
        for c in clusters:
            model = c.model
            total_weight += model.responsibility(z)
            
        for c in clusters:
            model = c.model
            weight = model.responsibility(z)
            yhat += weight*model.predict(X)/total_weight
        
        return yhat
        
    def __flatten_hierarchy(self, treenode, r=0.5):
        '''
        '''
        clusters = list()
        queue = deque([treenode])
        
        #iterate aslong the queue is empty
        while queue:
            node = queue.pop()
            c = node.cargo
            if c.post >= np.log(r):
                clusters.append(c)
            else:
                queue.append(node.left)
                queue.append(node.right)
        
        return clusters
        
        

    #Misc methods
    def xml_element(self):
        '''
        '''
        root_element = ElementTree.Element('bhc_model')
        
        #build parameter element 
        param_element = ElementTree.SubElement(root_element, 'parameters')
        param_element.attrib['rho'] = str(self._rho)
        param_element.attrib['alpha'] = str(self._alpha)
        param_element.attrib['beta'] = str(self._beta)
        
        #build task tree element
        task_tree_element = ElementTree.SubElement(root_element, 'task_tree')
        task_tree_element.append(self._tree2xml(self._task_tree)) 
        
        return root_element

    def _tree2xml(self, treenode):
        '''
        '''
        cluster = treenode.cargo
        reg_model = cluster.model
        
        if treenode.isatom():
            cluster_element = ElementTree.Element('leaf')
            cluster_element.attrib['id'] = str(cluster.task_id+1)
        else:
            cluster_element = ElementTree.Element('task_cluster')
            
        cluster_element.attrib['descr'] = str(cluster.z)
        
        likel_element = ElementTree.SubElement(cluster_element, 'likel')
        likel_element.attrib['post'] = str(cluster.post)
        likel_element.attrib['marg'] = str(cluster.marg)
        likel_element.attrib['mlikel'] = str(reg_model.log_evidence)
        likel_element.attrib['pi'] = str(cluster.pi)
        likel_element.attrib['norm'] = str(cluster.norm_const)
        
        pred_element = ElementTree.SubElement(cluster_element, 'predict')
        pred_element.attrib['weight'] = str(cluster.weight)
        pred_element.attrib['invweight'] = str(cluster.inv_weight)
        
        prop_element = ElementTree.SubElement(cluster_element, 'props')
        prop_element.attrib['ntask'] = str(cluster.ntasks)
        prop_element.attrib['size'] = str(cluster.size)
        prop_element.attrib['shape'] = str(cluster.covariates.shape)
        
        model_element = ElementTree.SubElement(cluster_element, 'reg_model')
        model_element.attrib['intercept'] = str(reg_model.intercept)
        model_element.attrib['weights'] = str(reg_model.weights)
        
        if not treenode.isatom():
            cluster_element.append(self._tree2xml(treenode.left))
            cluster_element.append(self._tree2xml(treenode.right))
            
        return cluster_element
        
    def marshal_xml(self, filename):
        '''
        '''
        element = self.xml_element()
        tree = ElementTree.ElementTree(element)
        try:
            tree.write(filename, "UTF-8")
        except EnvironmentError as err:
            print("{0}: import error {1}".format(os.path.basename(sys.argv[0]), err))


class _ClusterNode:
    '''
    - alpha is handled in log space
    - model use log_likel
    - renamed covariates, targets, descriptor to X,y,z
    - rho renamed to alpha
    - parameter list changed (z <-> model)
    '''
    __slots__ = ('task_id',     #task identifier 
                 'X',  
                 'y',
                 'z', 
                 'post',        
                 'marg', 
                 'weight',
                 'inv_weight' 
                 'pi',  
                 'norm_const', 
                 'model',
                 'ntasks',
                 'size')

    @staticmethod
    def make_leaf(id, X, y, Z, model, alpha):
        '''
        This factory method creates a leaf _ClusterNode at which the node 
        parameters instantiated from scratch.
        '''
        n = 1
        norm_const = alpha+sps.gammaln(n)
        marg = model.log_likel
        node = _ClusterNode(X, y, Z, model, n, marg, 
                            norm_const=norm_const, id=id)
        return node
    
    @staticmethod
    def make_internal(left, right, model, alpha):
        '''
        This factory method creates an internal _ClusterNode, which is the 
        parent of the given left and right _ClusterNode. The instantiated 
        parameters of the created node are the merged one of both children.
        The concrete initialization depends on the parameters. The weight
        of the new created internal node is given as parameter.
        
        '''
        X = np.vstack((left.X, right.X))
        y = np.hstack((left.y, right.y))
        Z = np.vstack((left.Z, right.Z))
        n = left.ntasks+right.ntasks
        #n = len(covariates)
        
        #print 'left_model_likel={0}'.format(left.model.log_likel)
        #print 'right_model_likel={0}'.format(right.model.log_likel)
        #print 'model_likel={0}'.format(model.log_likel)
        
        #compute the cluster prior 'pi'. Because numerical issues, the norm_const
        #is compute by the sum of individual logs.
        nom = alpha+sps.gammaln(n) 
        norm_const = sumln([nom, left.norm_const+right.norm_const]) 
        pi = nom - norm_const
        
        #compute the log probability weight of the merged hypothesis p(H|D)
        npi = left.norm_const + right.norm_const - norm_const
        marg = sumln([pi + model.log_likel, npi+left.marg+right.marg])
        post = pi + model.log_likel - marg
        
        
        #print 'pi={0}'.format(pi)
        #print 'marg={0}'.format(marg)
        #print 'post={0}'.format(post)
        
        #print 'pi={0}, marg={1}, post={2}'.format(pi, marg, post)
        node = _ClusterNode(X, y, Z, model, n, 
                            marg, post, pi, norm_const)
        return node
    
    #@change: post and pi are initialized by default by the log values
    def __init__(self, X, y, Z, model, ntasks=1, 
                 marg=0.0, post=0.0, pi=0.0, norm_const=0.0, id=None):
        self.task_id = id
        self.X = X
        self.y = y
        self.Z = Z
        self.model = model
        self.post = post
        self.pi = pi
        self.norm_const = norm_const
        self.marg = marg
        self.ntasks = ntasks
        self.weight = np.nan
        self.inv_weight = np.nan
        
        self.size = len(X)
        
    def __hash__(self):
        return id(self)
    
    def __str__(self):
        str = 'post={0}, pi={1}, weight={2}, marg={3}'.format(self.post, self.pi, 
                                                    self.weight,self.marg)
        str += ',ncluster={0} ,evidence={1} ,norm_const={2}, invweight={3}'.format(self.ntasks, self.model.log_evidence, self.norm_const, self.inv_weight)
        str += ',size={0}, shape={1}'.format(self.size, self.model.X.shape)
        #str += 'descr={0}'.format(self.descriptor)
        #if self.task_id != None:
        #    str += ', task_id={0}'.format(self.task_id)
        return str
    
    def copy(self):
        '''
        '''
        copy_node = _ClusterNode(self.X, self.y, self.Z, self.model, 
                                 self.ntasks, self.marg, 
                                 self.post, self.pi, self.norm_const)
        return copy_node


class _ExpertModel(object):
    
    '''
    @todo: - dynamic type check of reg_model should be remove, this can be done as soon
             as gp models and lin models inherit from the same interface, hence they have
             the same behaviour with respect to their methods
    '''
    
    __slots__ = ('_reg_model',
                 '_density_model',
                 '_X',
                 '_y',
                 '_Z',
                 '_n',
                 '_d')
    
    
    def __init__(self, X, y, Z, reg_model, density_model):
        self._X = X
        self._y = y
        self._Z = Z
        self._reg_model = reg_model
        self._density_model = density_model
        
        if isinstance(self._reg_model, RobustBayesRegression):
            self._n = 2+(X.shape[1]+1)*(X.shape[1]+1)
        else:
            self._n = len(reg_model.hyperparams)
        self._d = density_model.ndim
        
    def predict(self, X):
        return self._reg_model.predict(X)
    
    def responsibility(self, Z):
        return np.exp(self._density_model.predictive(Z))
    
    def likelihood(self):
        #print 'reg_like={0}, dens_likel={1}'.format(self._reg_model.log_likel, self._density_model.log_likel)
        return self._reg_model.log_likel + self._density_model.log_likel
    
    log_likel = property(fget=likelihood)
    
    def refit(self, params):
        
        beta, m, L, r, v = self._unwrap(params)
        L = np.tril(L)
        R = np.dot(L, L.T)
        
        
        #print 'regparams_before={0}'.format(np.log(self._reg_model.hyperparams))
        #print 'densparams_before={0}'.format(self._density_model.hyperparams)
        
        #print 'params_after={0}'.format(params)
        
        #todo: remove hack - hyperparameters of regression model must be updated explicitly
        if isinstance(self._reg_model, LinearRegressionModel):
            self._reg_model.refit(self._X, self._y, beta)
        else:
            self._reg_model._kernel.params = beta
            self._reg_model.fit(self._X, self._y)
            
        self._density_model = NormalInvWishart(self._Z, m, R, np.exp(r), v) #implement a refit for niw
    
    @property
    def likel_fun(self):
        return _ExpertModel.LikelihoodFunction(self)
    
    def _unwrap(self, params):
        '''
        @todo: make static (there is a need in other places)
        '''
        beta = params[0:self._n]
            
        m = params[self._n:self._n+self._d]
        L = params[self._n+self._d:self._n+self._d+self._d*self._d]
        L = np.reshape(L, (self._d, self._d))
        r = params[-2]
        v = params[-1]
            
        return beta, m, L, r, v
        
    def _wrap(self, beta, m, L, r, v):
        '''
        @todo: make static (there is a need in other places)
        '''
        return np.r_[beta, m, L.ravel(), r, v]
    
    class LikelihoodFunction(object):
        '''
        - r is in log space
        '''
        __slots__ = ('_model',
                     '_reg_likel_fun',
                     '_dens_likel_fun',
                     '_n',              #number of hyperparameters of regression model
                     '_d'               #dimensionality of density model
                    )
        
        def __init__(self, model):
            
            self._model = model
            self._reg_likel_fun = model._reg_model.likel_fun
            self._dens_likel_fun = model._density_model.likel_fun
            
            self._n = len(model._reg_model.hyperparams)
            self._d = model._density_model.ndim
            
        def __call__(self, params):
            '''
            @todo check if beta is in log space
            '''
            beta, m, L, r, v = self._model._unwrap(params)
            #print 'beta={0}'.format(beta)
            #print 'm={0}'.format(m)
            #print 'L={0}'.format(L)
            #print 'r={0}'.format(r)
            #print 'v={0}'.format(v)
            #optimize just the diagonal elements of the preciosion matrix of the gaussian component
            L = np.eye(L.shape[0]) * L
            
            L = np.tril(L)
            #print 'likel_chol={0}'.format(L)
            r = np.exp(r)
            
            #todo: remove hack - 
            if isinstance(self._model._reg_model, LinearRegressionModel):
                #print 'reg_likel={0}'.format(self._reg_likel_fun(beta))
                #print 'dens_likel={0}'.format(self._dens_likel_fun(m, L, r, v))
                likel = self._reg_likel_fun(beta) + self._dens_likel_fun(m, L, r, v)
            else:
                X = self._model._X
                y = self._model._y
                
                reg_model = self._model._reg_model
                
                
                if np.any(reg_model._kernel.params != beta):
                    reg_model._kernel.params = beta
                    reg_model.fit(X,y)
                    self._reg_likel_fun = reg_model.likel_fun
                    
                likel = self._reg_likel_fun() + self._dens_likel_fun(m, L, r, v)
                #print 'reg_likel={0}'.format(self._reg_likel_fun())
                #print 'dens_likel={0}'.format(self._dens_likel_fun(m, L, r, v))

                
            #print 'log_likel={0}'.format(likel)
            return likel
        
        def gradient(self, params):
            beta, m, L, r, v = self._model._unwrap(params)
            #optimize just the diagonal elements of the preciosion matrix of the gaussian component
            L = np.eye(L.shape[0]) * L
            
            r = np.exp(r)
            #L[L <= 1.e-8] = 0.0
            L = np.tril(L)
            #print 'grad_chol={0}'.format(L)
            m_grad, L_grad, r_grad, v_grad = self._dens_likel_fun.gradient(m, L, r, v)
            r_grad *= r
            
            #optimize just the diagonal elements of the preciosion matrix of the gaussian component
            L_grad = np.eye(L_grad.shape[0]) * L_grad
            
            
            #todo: remove hack - 
            if isinstance(self._model._reg_model, LinearRegressionModel):
                beta_grad = self._reg_likel_fun.gradient(beta)
            else:
                #gp stuff
                X = self._model._X
                y = self._model._y
                
                reg_model = self._model._reg_model
                
                if np.any(reg_model._kernel.params != beta):
                    reg_model._kernel.params = beta
                    reg_model.fit(X,y)
                    self._reg_likel_fun = reg_model.likel_fun
                    
                beta_grad = self._reg_likel_fun.gradient()
            
            grad = self._model._wrap(beta_grad, m_grad, L_grad, r_grad, v_grad)
            return grad
        
class BHCAlphaLikelihood(object):
    '''
    @todo: - implement gradient function
    '''
    __slots__ = ('_bhc_model',
                 '_last_alpha',
                 '_likel_tree'
                 )
    
    def __init__(self, bhc_model):
        self._bhc_model = bhc_model 
        self._last_alpha = np.inf
    
    def __call__(self, alpha):
        '''
        '''
        #if alpha > 0:
        if alpha != self._last_alpha:
            self._likel_tree = self._estimate_likelihood(alpha, self._bhc_model)
            self._last_alpha = alpha
            
        likel = -self._likel_tree.cargo.marg
        #else:
        #    likel = 1e+100
       
        print 'alpha={0}'.format(alpha)
        print 'alpha_likel={0}'.format(likel)
        return likel#*-1
    
    def _estimate_likelihood(self, alpha, treenode):
        '''
        '''
        alpha = np.squeeze(alpha)
        if treenode.isatom():
            cluster = treenode.cargo.copy()
            cluster.norm_const = alpha+sps.gammaln(cluster.ntasks)
            
            likel_tree = MutableBTree(cluster)
        else:
            left = self._estimate_likelihood(alpha, treenode.left)
            right = self._estimate_likelihood(alpha, treenode.right)
            
            cluster = treenode.cargo.copy()
            cluster_l = left.cargo
            cluster_r = right.cargo
            
            model = cluster.model
            n = cluster.ntasks
        
            #compute the cluster prior 'pi'. Because numerical issues, the norm_const
            #is compute by the sum of individual logs.
            nom = alpha+sps.gammaln(n) 
            cluster.norm_const = sumln([nom, cluster_l.norm_const+cluster_r.norm_const]) 
            cluster.pi = nom - cluster.norm_const
        
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = cluster_l.norm_const + cluster_r.norm_const - cluster.norm_const
            a = cluster.pi + model.log_likel
            b = npi+cluster_l.marg+cluster_r.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + model.log_likel - cluster.marg
            
            likel_tree = MutableBTree(cluster, None, left, right)
            
        return likel_tree
    
class _BHCModelLikelihood(object):
    '''
    @todo: - build log gradient to permit negative values of rho
    '''
    __slots__ = ('_bhc_model',
                 '_last_params'
                 )
    
    def __init__(self, bhc_model):
        self._bhc_model = bhc_model
        self._last_params = np.inf
    
    def __call__(self, params):
        '''
        '''
        likel = -self._bhc_model.cargo.marg
        if np.any(params != self._last_params):
            #exception handling for no semi-definit kernel matrix
            try:
                #print 'likel params={0}'.format(params)
                self._estimate_likelihood(self._bhc_model, params)
                self._last_params = params
                likel = -self._bhc_model.cargo.marg
            except np.linalg.LinAlgError:
                params = self._last_params
                self._estimate_likelihood(self._bhc_model, params)
                likel = 1e+300
            
        #print 'likel={0}'.format(likel)
        return likel
    
    def gradient(self, params):
        '''
        '''
        #print 'grad_params={0}'.format(params)
        grad = self._compute_gradient(self._bhc_model, params)
        #print 'grad={0}'.format(grad)
        return -grad
    
    def _estimate_likelihood(self, treenode, params):
        '''
        '''
        if treenode.isatom():
            cluster = treenode.cargo
            
            #rebuild the gp model
            model = cluster.model
            #print 'mlikel_before={0}'.format(model.log_likel)
            model.refit(params)
            #print 'mlikel_after={0}'.format(model.log_likel)
            #print 'reg_likel={0}'.format(model._reg_model.log_likel)
            #print 'dens_likel={0}'.format(model._density_model.log_likel)
            cluster.marg = model.log_likel
        else:
            self._estimate_likelihood(treenode.left, params)
            self._estimate_likelihood(treenode.right, params)
            
            cluster = treenode.cargo
            left = treenode.left.cargo
            right = treenode.right.cargo
            
            cluster = treenode.cargo

            #rebuild the gp model
            model = cluster.model
            #print 'mlikel_before={0}'.format(model.log_likel)
            model.refit(params)
            #print 'mlikel_after={0}'.format(model.log_likel)
            
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = left.norm_const + right.norm_const - cluster.norm_const
            a = cluster.pi + model.log_likel
            b = npi+left.marg+right.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + model.log_likel - cluster.marg
                
    def _compute_gradient(self, treenode, params):
       
        cluster = treenode.cargo    #corresponding cluster of the treenode
        model = cluster.model       #regression model of the cluster
        likel_fun = model.likel_fun #likel function of the regression model
        
        r = np.exp(cluster.pi+likel_fun(params)-cluster.marg)
        
        #print 'marg={0}'.format(cluster.marg)
        #print 'likel={0}'.format(likel_fun(params))
        #print 'pi={0}'.format(cluster.pi)
        #print 'r={0}'.format(r)
        #print 'func_grad={0}'.format(likel_fun.gradient(params))
        if r > 1.0:
            r = 1.0
        if r < 0:
            r = 0.0
        if np.isinf(r) or np.isnan(r):
            r = 0.0   
            
        
        grad = r*likel_fun.gradient(params)
        grad[np.any([np.isinf(grad), np.isnan(grad)],0)] = 0.0
        if not treenode.isatom():
            grad_l = self._compute_gradient(treenode.left, params)
            grad_r = self._compute_gradient(treenode.right, params)
            grad += (1.0-r) * (grad_l+grad_r)
        
        return grad
    

    
if __name__ == '__main__':
    
    from upgeo.ml.regression.np.kernel import SEKernel, NoiseKernel
    from upgeo.ml.regression.np.infer import ExactInference
    
    from upgeo.util.metric import mspe
    from upgeo.eval.multitask.base import load_data
    from upgeo.ml.multitask.base import flatten_data

    #filename = '/home/marcel/datasets/multilevel/ilea/school_norm_vrb.csv'
    #task_key = 'school'
    #task_fields = ['fsm', 'school_vr1', 'school_mixed', 'school_male', 'school_female', 
    #               'sdenom_maintain', 'sdenom_coe', 'sdenom_rc']
    #target_field = 'exam_score'
    
    filename = '/home/marcel/datasets/multilevel/ilea/exam_london_linreg_normed.csv'
    task_key = 'school'
    task_fields = ['school_mixed', 'school_boys', 'school_girls', 
                   'intake_score', 'school_vrb']
    #task_fields = ['intake_score']
    target_field = 'exam_score'
    
    X,Y,Z = load_data(filename, task_key, task_fields, target_field)
    #X = compound_data(X, Z)
    x_test = X[64]
    y_test = Y[64]
    z_test = Z[64] 
    
    X = X[0:63]
    Y = Y[0:63]
    Z = Z[0:63]
 
    print Z
 
    print 'bhc_linreg'
 
    bhc_linreg = BHCLinearRegressionExpert(np.log(10), np.log(np.array([2,2])))
    bhc_linreg.fit(X, Y, Z)
    yhat = bhc_linreg.predict([x_test], [z_test], method='path', incl_tree_weights=True)
    print mspe(y_test, yhat[0])
    yhat = bhc_linreg.predict([x_test], [z_test], method='path', incl_tree_weights=False)
    print mspe(y_test, yhat[0])
    yhat = bhc_linreg.predict([x_test], [z_test], method='fullpath', incl_tree_weights=True)
    print mspe(y_test, yhat[0])
    yhat = bhc_linreg.predict([x_test], [z_test], method='fullpath', incl_tree_weights=False)
    print mspe(y_test, yhat[0])
    yhat = bhc_linreg.predict([x_test], [z_test], method='complete', incl_tree_weights=True)
    print mspe(y_test, yhat[0])
    yhat = bhc_linreg.predict([x_test], [z_test], method='complete', incl_tree_weights=False)
    print mspe(y_test, yhat[0])
    yhat = bhc_linreg.predict([x_test], [z_test], method='maxcluster', incl_tree_weights=True)
    print mspe(y_test, yhat[0])
    yhat = bhc_linreg.predict([x_test], [z_test], method='flat_maxcluster')
    print mspe(y_test, yhat[0])
    yhat = bhc_linreg.predict([x_test], [z_test], method='flat_complete')
    print mspe(y_test, yhat[0])
    
    print 'bhc_gp'
      

    selector = RandomSubsetSelector(50)
    kernel = SEKernel(np.log(10.5), np.log(1)) + NoiseKernel(np.log(0.5))

    bhc_gp = SparseBHCGPRegregressionExpert(np.log(1000), kernel, None, selector, opt_model_params=True)
    bhc_gp.fit(X, Y, Z)
    yhat = bhc_gp.predict([x_test], [z_test], method='path', incl_tree_weights=True)
    print mspe(y_test, yhat[0])
    yhat = bhc_gp.predict([x_test], [z_test], method='path', incl_tree_weights=False)
    print mspe(y_test, yhat[0])
    yhat = bhc_gp.predict([x_test], [z_test], method='fullpath', incl_tree_weights=True)
    print mspe(y_test, yhat[0])
    yhat = bhc_gp.predict([x_test], [z_test], method='fullpath', incl_tree_weights=False)
    print mspe(y_test, yhat[0])
    yhat = bhc_gp.predict([x_test], [z_test], method='complete', incl_tree_weights=True)
    print mspe(y_test, yhat[0])
    yhat = bhc_gp.predict([x_test], [z_test], method='complete', incl_tree_weights=False)
    print mspe(y_test, yhat[0])
    yhat = bhc_gp.predict([x_test], [z_test], method='maxcluster', incl_tree_weights=True)
    print mspe(y_test, yhat[0])
    yhat = bhc_gp.predict([x_test], [z_test], method='flat_maxcluster')
    print mspe(y_test, yhat[0])
    yhat = bhc_gp.predict([x_test], [z_test], method='flat_complete')
    print mspe(y_test, yhat[0])
