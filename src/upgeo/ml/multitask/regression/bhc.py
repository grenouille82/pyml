'''
Created on Aug 15, 2011

@author: marcel
'''
import os
import sys

import time

import numpy as np
import scipy.special as sps
import scipy.optimize as spopt

from xml.etree import ElementTree

from scikits.learn.base import RegressorMixin, BaseEstimator

from upgeo.adt.tree import MutableBTree, path_to_root
from upgeo.exception import NotFittedError
from upgeo.ml.regression.np.gp import GPRegression, SparseGPRegression
from upgeo.ml.multitask.base import asmtarray_object, compound_data
from upgeo.util.math import sumln
from upgeo.util.metric import NaiveMatrixSearch, mspe
from upgeo.ml.regression.bayes import EBChenRegression, FastBayesRegression,\
    RobustBayesRegression
from upgeo.ml.regression.np.kernel import MaternKernel, ARDSEKernel
from upgeo.ml.regression.np.selector import RandomSubsetSelector, FixedSelector,\
    KMeansSelector
from upgeo.ml.regression.linear import LinearRegressionModel
from upgeo.ml.multitask.base import flatten_data
from upgeo.ml.cluster.prototype import KMeans
from upgeo.ml.regression.np.infer import ExactInference, FITCExactInference

class BHCGPRegression(BaseEstimator, RegressorMixin):
    '''     
    @todo: - 
    '''
    
    __slots__ = ('_alpha',          #cluster concentration hyperparameter in log space
                 '_kernel',         #initial kernel
                 '_priors',         #priors of kernel hyperparameters
                 '_task_tree',      #the cluster tree of regression models
                 '_is_init'
                 )        
    
    def __init__(self, alpha, kernel, priors=None):
        '''
        @todo: - check the number of hyperparams and priors
        '''
        self._alpha = alpha
        self._kernel = kernel
        self._priors = priors
        self._is_init = False
        
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        return np.r_[self._alpha, np.copy(self._kernel.params)]
    
    def _set_hyperparams(self, params):
        '''
        '''
        self._alpha = params[0]
        self._kernel.params = np.copy(params[1:])
    
    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)

    def fit(self, X, Y, Z, max_it=50, tol=1.e-5):
        '''
        '''
        X = asmtarray_object(X)
        Y = asmtarray_object(Y)
        Z = asmtarray_object(Z)
        
        if len(X) != len(Y) != len(Z):
            raise ValueError('number of task data X, Y and task ' + 
                             'description Z must be the same')
        
        
        kernel = self._kernel
        gp = GPRegression(kernel, infer_method=ExactInference)
        X_p = flatten_data(X)
        Y_p = flatten_data(Y)
        gp.fit(X_p,Y_p)
   
        
        priors = self._priors
        alpha = self._alpha
        beta = np.copy(kernel.params)
        
        
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
            model = self._build_bhc_model(alpha, kernel, priors, X, Y, Z)
            
            #M-Step: optimize the parameters on a fixed tree
            #beta are the model hyperparameters
            likel, alpha, beta = self._optimize_bhc_parameters(model, alpha, beta, priors)
            bhc_model = model
            i += 1
            
        self._update_model(alpha, beta, bhc_model)
        self.__refit_local_models(bhc_model)
        self.__update_weights(bhc_model)
        
        self._task_tree = bhc_model
        self._alpha = alpha
        
        self._is_init = True  
        
    def _optimize_bhc_parameters(self, model, alpha, beta, priors):
        '''
        '''        
        print model.cargo.marg
        alpha_likel_fun = BHCAlphaLikelihood(model)
        opt_alpha = spopt.brent(alpha_likel_fun, brack=(alpha-2, alpha+2))
        
        #try:
        #    opt_alpha = spopt.brent(alpha_likel_fun, brack=(alpha-5, alpha, alpha+10))
        #except AssertionError:
        #    print 'bash'
        #    opt_alpha = spopt.brent(alpha_likel_fun, brack=(alpha-2, alpha+2))
        
        print 'opt_alpha={0}'.format(opt_alpha)
        self.__update_alpha(opt_alpha, model)
        print model.cargo.marg
        
        model_likel_fun = _GPModelLikelihood(model, self._kernel)
        model_likel_grad = model_likel_fun.gradient
        opt_result = spopt.fmin_bfgs(model_likel_fun, beta, model_likel_grad, full_output=True)
        opt_beta = opt_result[0]
        likel = opt_result[1]
        print 'opt_beta={0}'.format(np.exp(opt_beta))
        print likel
        
        return (likel, opt_alpha, opt_beta) 
        
    def predict_by_task(self, X, tasks, method='path'):
        '''
        @todo: - more generic parametrization for invoking the different prediction methods
        '''
        self._init_check()
        X = asmtarray_object(X)
        
        if len(X) != len(tasks):
            raise ValueError('number of task data X and tasks must be equal.')
        
        if method == 'path':
            pred_method = self.__predict_task_path
        elif method == 'maxcluster':
            pred_method = self.__predict_task_maxcluster
        elif method == 'cutpath':
            pred_method = self.__predict_task_cutpath
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        
        n = len(X)
        Y = np.empty(n, dtype='object')
        for i in xrange(n):
            if len(X[i]) > 0:
                yhat = pred_method(X[i], tasks[i])
                Y[i] = yhat
            else: 
                Y[i] = []
            
        return Y
        
    def predict(self, X, Z, method='path', k=1):
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
        elif method == 'maxpath':
            pred_method = self.__predict_maxpath
        elif method == 'cutpath':
            pred_method = self.__predict_cutpath
        elif method == 'complete':
            pred_method = self.__predict_complete
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        n = len(X)
        Y = np.empty(n, dtype='object')
        for i in xrange(n):
            yhat = pred_method(X[i], Z[i], k)
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
    
    def _build_bhc_model(self, alpha, kernel, priors, X, Y, Z):
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
            
            model = self.__make_gp_model(Xi, Yi)
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
                X = np.r_[cluster_i.X, cluster_j.X]
                y = np.r_[cluster_i.y, cluster_j.y]
                    
                model = self.__make_gp_model(X, y)
                cluster = _ClusterNode.make_internal(cluster_i, cluster_j, 
                                                     model, alpha)
                    
                model_matrix[i,j] = model_matrix[j,i] = model
                prob_matrix[i,j] = prob_matrix[j,i] = cluster.post
                
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
                    
                    X = np.r_[cluster.X, cluster_i.X] #same as np.r_[X[p],X[q],X[i]]
                    y = np.r_[cluster.y, cluster_i.y] #same as np.r_[Y[p],Y[q],Y[i]]
                    
                    model = self.__make_gp_model(X, y)
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
    
    def __make_gp_model(self, X, y):
        kernel = self._kernel
        priors = self._priors
        
        model = GPRegression(kernel, priors=priors)
        model.fit(X,y)
        return model
    
    def _update_model(self, alpha, beta, treenode):
        self._kernel.params = beta
        self._alpha = alpha
        
        self.__update_alpha(alpha, treenode)
        self.__update_model_parameters(beta, treenode)
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
                #ns = np.float(super_cluster.ntasks)
                #n = np.float(cluster.ntasks)
                #norm = n/ns
                norm = 1
                
                inv_weight = super_cluster.inv_weight*norm
                cluster.weight *= inv_weight
                cluster.inv_weight *= inv_weight
                

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
            

    def __update_model_parameters(self, beta, treenode):
        '''
        @todo: kernel is not used
        '''
        if treenode.isatom():
            cluster = treenode.cargo
            
            #rebuild the gp model
            model = self.__make_gp_model(cluster.X, cluster.y)
            cluster.model = model
            cluster.marg = model.log_likel
        else:
            self.__update_model_parameters(beta, treenode.left)
            self.__update_model_parameters(beta, treenode.right)
            
            cluster = treenode.cargo
            left = treenode.left.cargo
            right = treenode.right.cargo
            
            cluster = treenode.cargo

            #rebuild the gp model
            model = self.__make_gp_model(cluster.X, cluster.y)
            cluster.model = model            
            
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
            
            model = GPRegression(kernel, infer_method=ExactInference, priors=priors)
            model.fit(cluster.X, cluster.y)
            print 'refit_params={0}'.format(np.exp(model.hyperparams))
            
            cluster.model = model
            cluster.marg = model.log_likel
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
            
            model = GPRegression(kernel, infer_method=ExactInference, priors=priors)
            model.fit(cluster.X, cluster.y)
            print 'refit_params={0}'.format(np.exp(model.hyperparams))
            cluster.model = model            
            
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = left.norm_const + right.norm_const - cluster.norm_const
            a = cluster.pi + model.log_likel
            b = npi+left.marg+right.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + model.log_likel - cluster.marg

    
    #Prediction Methods
    def __predict_task_path(self, X, task):
        '''
        @todo: - meta features does not work, remove this shit
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)
        
        n = len(X)
        yhat = np.zeros(n)
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        print '------------------------'
        total_weight = 0
        for node in path:
            print 'weight={0}'.format(node.cargo.weight)
            total_weight += node.cargo.weight
                 
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        print '---------------------'
        for node in path:
            model = node.cargo.model
            yhat += node.cargo.weight*model.predict(X)/total_weight
            
            print 'model_params={0}'.format(np.exp(model.hyperparams))
        
        return yhat
        

    def __predict_task_maxcluster(self, X, task):
        '''
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)

        n = len(X)
        yhat = np.zeros(n)
        
        #determine the cluster with the highest weight
        max_weight = 0
        max_cluster = None
        for node in path:
            if node.cargo.weight > max_weight:
                max_weight = node.cargo.weight
                max_cluster = node.cargo
         
        model = max_cluster.model
        yhat = model.predict(X)
                
        return yhat
        
    def __predict_task_cutpath(self, X, task, r=0.5):
        '''
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)

        n = len(X)
        yhat = np.zeros(n)
        
        total_weight = 0
        for node in path:
            if node.cargo.post >= np.log(r):
                total_weight += node.cargo.weight
        
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for node in path:
            if node.cargo.post >= np.log(r):
                model = node.cargo.model
                yhat += node.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    def __lookup_tasknode(self, task_id):
        '''
        @todo: - use hashing for O(1) access
               - exception handling
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves())
        
        for node in leaves:
            if node.cargo.task_id == task_id:
                return node
            
        raise ValueError('task not found')
            
    def __predict_complete(self, X, z, k):
        '''
        @todo: - should the weights normalized
        '''
        tree = self._task_tree
        n = X.shape[0]
        yhat = np.zeros(n)
        
        total_weight = 0
        print '---------------'
        for cluster in tree:
            print 'cluster_weight={0}'.format(cluster.weight)
            total_weight += cluster.weight
            
        for cluster in tree:
            model = cluster.model
            weight = cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat
    
    def __predict_path(self, X, z, k):
        '''
        @todo: - check whether the treenodes putting into the cluster_set are correctly 
                 inserted, that means no collosion occurs. Is the id() function sufficient
                 to compute the hash value of the object? What is better, using treenode or
                 cluster nodes?
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks, dist = neighbor_struct.query_knn(z, k)
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
            
        #remove clusters with low probability by creating a new list
        #tmp_cluster_set = list()
        #for cluster in cluster_set:
        #    if cluster.cargo.weight > 1e-2:
        #        tmp_cluster_set.append(cluster)
        #cluster_set = tmp_cluster_set
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        for cluster in cluster_set:
            total_weight += cluster.cargo.weight
            
        path = path_to_root(neighbors[0])
        #print 'total_w={0}'.format(total_weight)
        for node in path:
            cluster = node.cargo
            print 'weight={0}, norm_weight={1}'.format(cluster.weight, cluster.weight/total_weight)
            
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for cluster in cluster_set:
            model = cluster.cargo.model
            yhat += cluster.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    def __predict_cutpath(self, X, z, k, r=0.5):
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks, dist = neighbor_struct.query_knn(z, k)
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #print 'meta task={0}'.format(z)
        #for i in knn_tasks:
        #    print 'matched task={0}, dist={1}'.format(leaves[i].cargo.z, dist)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
            
        #remove clusters with low probability by creating a new list
        #tmp_cluster_set = list()
        #for cluster in cluster_set:
        #    if cluster.cargo.weight > 1e-2:
        #        tmp_cluster_set.append(cluster)
        #cluster_set = tmp_cluster_set
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        for cluster in cluster_set:
            if cluster.cargo.post >= np.log(r):
                total_weight += cluster.cargo.weight
        
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        
        #print self._task_tree
        for cluster in cluster_set:
            if cluster.cargo.post >= np.log(r):
                model = cluster.cargo.model
                yhat += cluster.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    
    def __predict_maxpath(self, X, z, k):
        '''
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks,_ = neighbor_struct.query_knn(z, k)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
        
        #determine the cluster with the highest weight
        max_weight = 0
        max_cluster = None
        for cluster in cluster_set:
            if cluster.cargo.weight > max_weight:
                max_weight = cluster.cargo.weight
                max_cluster = cluster.cargo
         
        model = max_cluster.model
        yhat = model.predict(X)
                
        return yhat
            
    #Misc methods
    def xml_element(self):
        '''
        '''
        root_element = ElementTree.Element('bhc_model')
        
        #build parameter element 
        param_element = ElementTree.SubElement(root_element, 'parameters')
        #param_element.attrib['rho'] = str(self._rho)
        param_element.attrib['alpha'] = str(self._alpha)
        #param_element.attrib['beta'] = str(self._beta)
        
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
            
        cluster_element.attrib['descr'] = str(cluster.descr)
        
        likel_element = ElementTree.SubElement(cluster_element, 'likel')
        likel_element.attrib['post'] = str(np.exp(cluster.post))
        likel_element.attrib['marg'] = str(cluster.marg)
        likel_element.attrib['mlikel'] = str(reg_model.log_likel)
        likel_element.attrib['pi'] = str(cluster.pi)
        likel_element.attrib['norm'] = str(cluster.norm_const)
        
        pred_element = ElementTree.SubElement(cluster_element, 'predict')
        pred_element.attrib['weight'] = str(cluster.weight)
        pred_element.attrib['invweight'] = str(cluster.inv_weight)
        
        prop_element = ElementTree.SubElement(cluster_element, 'props')
        prop_element.attrib['ntask'] = str(cluster.ntasks)
        prop_element.attrib['size'] = str(cluster.size)
        #prop_element.attrib['shape'] = str(cluster.covariates.shape)
        
        model_element = ElementTree.SubElement(cluster_element, 'reg_model')
        model_element.attrib['params'] = str(np.exp(reg_model.hyperparams))
        #model_element.attrib['intercept'] = str(reg_model.intercept)
        #model_element.attrib['weights'] = str(reg_model.weights)
        
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



class SparseBHCGPRegression(BaseEstimator, RegressorMixin):
    '''     
    @todo: - 
    '''
    
    __slots__ = ('_alpha',          #cluster concentration hyperparameter in log space
                 '_kernel',         #initial kernel
                 '_priors',         #priors of kernel hyperparameters
                 '_selector',       #selection method for inducing points
                 '_k',
                 '_task_tree',      #the cluster tree of regression models
                 '_opt_kernel_global',
                 '_is_init'
                 )        
    
    def __init__(self, alpha, kernel, priors=None, k=10, opt_kernel_global=False):
        '''
        @todo: - check the number of hyperparams and priors
        '''
        self._alpha = alpha
        self._kernel = kernel
        self._priors = priors
        self._k = k
        
        self._opt_kernel_global = opt_kernel_global
        
        self._is_init = False
        
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        return np.r_[self._alpha, np.copy(self._kernel.params)]
    
    def _set_hyperparams(self, params):
        '''
        '''
        self._alpha = params[0]
        self._kernel.params = np.copy(params[1:])
        
    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)

    def fit(self, X, Y, Z, max_it=50, tol=1.e-5):
        '''
        '''
        X = asmtarray_object(X)
        Y = asmtarray_object(Y)
        Z = asmtarray_object(Z)
        
        if len(X) != len(Y) != len(Z):
            raise ValueError('number of task data X, Y and task ' + 
                             'description Z must be the same')
        
        
        #kernel = self._kernel
        priors = self._priors
        alpha = self._alpha
        
        
        
                
        #todo: make the code more  clear
        #determine a set of inducing points using k-means
        X_p = flatten_data(X)
        Y_p = flatten_data(Y)
        #Xu = KMeansSelector(self._k).apply(X_p, Y_p)
        Xu = RandomSubsetSelector(self._k).apply(X_p, Y_p)
        self._selector = FixedSelector(Xu)
        
        kernel = self._kernel
        gp = SparseGPRegression(kernel, infer_method=FITCExactInference, priors=priors, selector=self._selector)
        
        gp.fit(X_p,Y_p)
        beta = np.copy(kernel.params)
        
        #if self._opt_kernel_global:
        #    model = SparseGPRegression(kernel, infer_method=FITCExactInference, priors=priors, selector=self._selector)
        #    model.fit(flatten_data(X),flatten_data(Y))
            
        #    beta = np.copy(kernel.params)
             
        
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
            
            print 'kernel_params={0}'.format(kernel.params)
            
            likel_old = likel
            
            #E-Step: estimate the new tree of the optimized parameters
            t = time.time()
            print 'E-Step'
            model = self._build_bhc_model(alpha, kernel, priors, X, Y, Z)
            print 'time={0}'.format(time.time()-t)
            
            #M-Step: optimize the parameters on a fixed tree
            #beta are the model hyperparameters
            t = time.time()
            print 'M-Step'
            likel, alpha, beta = self._optimize_bhc_parameters(model, alpha, beta, priors)
            print 'time={0}'.format(time.time()-t)
            print 'beta={0}'.format(beta)
            bhc_model = model
            
            i += 1
            
        self._update_model(alpha, beta, bhc_model)
        self.__refit_local_models(bhc_model)
        self.__update_weights(bhc_model)
        
        self._task_tree = bhc_model
        self._alpha = alpha
        
        self._is_init = True  
        
    def _optimize_bhc_parameters(self, model, alpha, beta, priors):
        '''
        '''        
        print model.cargo.marg
        alpha_likel_fun = BHCAlphaLikelihood(model)
        print 'alpha={0}'.format(alpha)
        #opt_alpha = spopt.brent(alpha_likel_fun, brack=(np.log(1e-100), alpha, np.log(1e+100)))
        try:
            opt_alpha = spopt.brent(alpha_likel_fun, brack=(alpha-10, alpha, alpha+10))
        except AssertionError:
            opt_alpha = spopt.brent(alpha_likel_fun, brack=(alpha-2, alpha+2))
        
        #opt_alpha = spopt.brent(alpha_likel_fun, brack=(np.log(1e-100), np.log(1e+100)))
        
        
        print 'opt_alpha={0}'.format(opt_alpha)
        self.__update_alpha(opt_alpha, model)
        print model.cargo.marg
        
        if not self._opt_kernel_global:
            model_likel_fun = _GPModelLikelihood(model, self._kernel)
            model_likel_grad = model_likel_fun.gradient
            opt_result = spopt.fmin_bfgs(model_likel_fun, beta, model_likel_grad, full_output=True)
            opt_beta = opt_result[0]
            likel = opt_result[1]
            print 'opt_beta={0}'.format(np.exp(opt_beta))
            print likel
        else:
            likel = model.cargo.marg
            opt_beta = beta
        
        return (likel, opt_alpha, opt_beta) 
        
    def predict_by_task(self, X, tasks, method='path'):
        '''
        @todo: - more generic parametrization for invoking the different prediction methods
        '''
        self._init_check()
        X = asmtarray_object(X)
        
        if len(X) != len(tasks):
            raise ValueError('number of task data X and tasks must be equal.')
        
        if method == 'path':
            pred_method = self.__predict_task_path
        elif method == 'maxcluster':
            pred_method = self.__predict_task_maxcluster
        elif method == 'cutpath':
            pred_method = self.__predict_task_cutpath
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        
        n = len(X)
        Y = np.empty(n, dtype='object')
        for i in xrange(n):
            if len(X[i]) > 0:
                yhat = pred_method(X[i], tasks[i])
                Y[i] = yhat
            else: 
                Y[i] = []            
        return Y
        
    def predict(self, X, Z, method='path', k=1):
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
        elif method == 'maxpath':
            pred_method = self.__predict_maxpath
        elif method == 'cutpath':
            pred_method = self.__predict_cutpath
        elif method == 'complete':
            pred_method = self.__predict_complete
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        n = len(X)
        Y = np.empty(n, dtype='object')
        for i in xrange(n):
            yhat = pred_method(X[i], Z[i], k)
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
    
    def _build_bhc_model(self, alpha, kernel, priors, X, Y, Z):
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
            
            model = self.__make_gp_model(Xi, Yi)
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
                X = np.r_[cluster_i.X, cluster_j.X]
                y = np.r_[cluster_i.y, cluster_j.y]
                    
                model = self.__make_gp_model(X, y)
                cluster = _ClusterNode.make_internal(cluster_i, cluster_j, 
                                                     model, alpha)
                    
                model_matrix[i,j] = model_matrix[j,i] = model
                prob_matrix[i,j] = prob_matrix[j,i] = cluster.post
                
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
                    
                    X = np.r_[cluster.X, cluster_i.X] #same as np.r_[X[p],X[q],X[i]]
                    y = np.r_[cluster.y, cluster_i.y] #same as np.r_[Y[p],Y[q],Y[i]]
                    
                    model = self.__make_gp_model(X, y)
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
    
    def __make_gp_model(self, X, y):
        kernel = self._kernel
        priors = self._priors
        selector = self._selector
        
        model = SparseGPRegression(kernel, priors=priors, selector=selector)
        model.fit(X,y)
        return model
    
    def _update_model(self, alpha, beta, treenode):
        self._kernel.params = beta
        self._alpha = alpha
        
        self.__update_alpha(alpha, treenode)
        self.__update_model_parameters(beta, treenode)
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
            

    def __update_model_parameters(self, beta, treenode):
        '''
        @todo: kernel is not used
        '''
        if treenode.isatom():
            cluster = treenode.cargo
            
            #rebuild the gp model
            model = self.__make_gp_model(cluster.X, cluster.y)
            cluster.model = model
            cluster.marg = model.log_likel
        else:
            self.__update_model_parameters(beta, treenode.left)
            self.__update_model_parameters(beta, treenode.right)
            
            cluster = treenode.cargo
            left = treenode.left.cargo
            right = treenode.right.cargo
            
            cluster = treenode.cargo

            #rebuild the gp model
            model = self.__make_gp_model(cluster.X, cluster.y)
            cluster.model = model            
            
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
        
            model = SparseGPRegression(kernel, infer_method=FITCExactInference, priors=priors, selector=selector)
            model.fit(cluster.X, cluster.y)
            
            cluster.model = model
            cluster.marg = model.log_likel
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
        
            model = SparseGPRegression(kernel, infer_method=FITCExactInference, priors=priors, selector=selector)
            model.fit(cluster.X, cluster.y)
            
            cluster.model = model            
            
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = left.norm_const + right.norm_const - cluster.norm_const
            a = cluster.pi + model.log_likel
            b = npi+left.marg+right.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + model.log_likel - cluster.marg
        
    
    #Prediction Methods
    def __predict_task_path(self, X, task):
        '''
        @todo: - meta features does not work, remove this shit
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)
        
        n = len(X)
        yhat = np.zeros(n)
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        print '--------------'
        for node in path:
            print 'weight={0}'.format(node.cargo.weight)
            total_weight += node.cargo.weight
                 
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for node in path:
            model = node.cargo.model
            yhat += node.cargo.weight*model.predict(X)/total_weight
        
        return yhat
        

    def __predict_task_maxcluster(self, X, task):
        '''
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)

        n = len(X)
        yhat = np.zeros(n)
        
        #determine the cluster with the highest weight
        max_weight = 0
        max_cluster = None
        for node in path:
            if node.cargo.weight > max_weight:
                max_weight = node.cargo.weight
                max_cluster = node.cargo
         
        model = max_cluster.model
        yhat = model.predict(X)
                
        return yhat
        
    def __predict_task_cutpath(self, X, task, r=0.5):
        '''
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)

        n = len(X)
        yhat = np.zeros(n)
        
        total_weight = 0
        for node in path:
            if node.cargo.post >= np.log(r):
                total_weight += node.cargo.weight
        
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for node in path:
            if node.cargo.post >= np.log(r):
                model = node.cargo.model
                yhat += node.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    def __lookup_tasknode(self, task_id):
        '''
        @todo: - use hashing for O(1) access
               - exception handling
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves())
        
        for node in leaves:
            if node.cargo.task_id == task_id:
                return node
            
        raise ValueError('task not found')
            
    def __predict_complete(self, X, z, k):
        '''
        @todo: - should the weights normalized
        '''
        tree = self._task_tree
        n = X.shape[0]
        yhat = np.zeros(n)
        
        total_weight = 0
        print '---------------'
        for cluster in tree:
            print 'weight={0}'.format(cluster.weight)
            total_weight += cluster.weight
            
        for cluster in tree:
            model = cluster.model
            weight = cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat
    
    def __predict_path(self, X, z, k):
        '''
        @todo: - check whether the treenodes putting into the cluster_set are correctly 
                 inserted, that means no collosion occurs. Is the id() function sufficient
                 to compute the hash value of the object? What is better, using treenode or
                 cluster nodes?
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks, dist = neighbor_struct.query_knn(z, k)
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
            
        #remove clusters with low probability by creating a new list
        #tmp_cluster_set = list()
        #for cluster in cluster_set:
        #    if cluster.cargo.weight > 1e-2:
        #        tmp_cluster_set.append(cluster)
        #cluster_set = tmp_cluster_set
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        for cluster in cluster_set:
            total_weight += cluster.cargo.weight
            
        path = path_to_root(neighbors[0])
        #print 'total_w={0}'.format(total_weight)
        for node in path:
            cluster = node.cargo
            print 'weight={0}, norm_weight={1}'.format(cluster.weight, cluster.weight/total_weight)
            
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for cluster in cluster_set:
            model = cluster.cargo.model
            yhat += cluster.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    def __predict_cutpath(self, X, z, k, r=0.5):
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks, dist = neighbor_struct.query_knn(z, k)
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #print 'meta task={0}'.format(z)
        #for i in knn_tasks:
        #    print 'matched task={0}, dist={1}'.format(leaves[i].cargo.z, dist)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
            
        #remove clusters with low probability by creating a new list
        #tmp_cluster_set = list()
        #for cluster in cluster_set:
        #    if cluster.cargo.weight > 1e-2:
        #        tmp_cluster_set.append(cluster)
        #cluster_set = tmp_cluster_set
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        for cluster in cluster_set:
            if cluster.cargo.post >= np.log(r):
                total_weight += cluster.cargo.weight
        
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        
        #print self._task_tree
        for cluster in cluster_set:
            if cluster.cargo.post >= np.log(r):
                model = cluster.cargo.model
                yhat += cluster.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    
    def __predict_maxpath(self, X, z, k):
        '''
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks,_ = neighbor_struct.query_knn(z, k)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
        
        #determine the cluster with the highest weight
        max_weight = 0
        max_cluster = None
        for cluster in cluster_set:
            if cluster.cargo.weight > max_weight:
                max_weight = cluster.cargo.weight
                max_cluster = cluster.cargo
         
        model = max_cluster.model
        yhat = model.predict(X)
                
        return yhat
            
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
            
        cluster_element.attrib['descr'] = str(cluster.descr)
        
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

class RobustBHCGPRegression(BaseEstimator, RegressorMixin):
    '''     
    @todo: - 
    '''
    
    __slots__ = ('_alpha',          #cluster concentration hyperparameter in log space
                 '_kernel',         #initial kernel
                 '_priors',         #priors of kernel hyperparameters
                 '_selector',       #selection method for inducing points
                 '_k',
                 '_task_tree',      #the cluster tree of regression models
                 '_cached_data',
                 '_is_init'
                 )        
    
    def __init__(self, alpha, kernel, priors=None, k=15):
        '''
        @todo: - check the number of hyperparams and priors
        '''
        self._alpha = alpha
        self._kernel = kernel
        self._priors = priors
        self._k = k
        
        self._cached_data = None
        self._is_init = False
        
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        return np.r_[self._alpha, np.copy(self._kernel.params)]
    
    def _set_hyperparams(self, params):
        '''
        '''
        self._alpha = params[0]
        self._kernel.params = np.copy(params[1:])

    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)
    
    def fit(self, X, Y, Z, max_it=50, tol=1.e-5):
        '''
        '''
        X = asmtarray_object(X)
        Y = asmtarray_object(Y)
        Z = asmtarray_object(Z)
        
        if len(X) != len(Y) != len(Z):
            raise ValueError('number of task data X, Y and task ' + 
                             'description Z must be the same')
        
        kernel = self._kernel
        priors = self._priors
        alpha = self._alpha
        #beta = np.copy(kernel.params)
        
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
            t = time.time()
            print 'E-Step'
            model = self._build_bhc_model(alpha, kernel, priors, X, Y, Z)
            print 'time={0}'.format(time.time()-t)
            
            #M-Step: optimize the parameters on a fixed tree
            #beta are the model hyperparameters
            t = time.time()
            print 'M-Step'
            likel, alpha = self._optimize_bhc_parameters(model, alpha, priors)
            print 'time={0}'.format(time.time()-t)
            bhc_model = model
            i += 1
            
        self._update_model(alpha, bhc_model)
        
        self._task_tree = bhc_model
        self._alpha = alpha
        
        self._cached_data = None
        self._is_init = True  
        
    def _optimize_bhc_parameters(self, model, alpha, priors):
        '''
        '''        
        print model.cargo.marg
        alpha_likel_fun = BHCAlphaLikelihood(model)
        
        opt_alpha = spopt.brent(alpha_likel_fun, brack=(np.log(1e-100), alpha, np.log(1e+100)))
        
        print 'opt_alpha={0}'.format(opt_alpha)
        self.__update_alpha(opt_alpha, model)
        print model.cargo.marg
        likel = model.cargo.marg
        
        return (likel, opt_alpha) 
        
    def predict_by_task(self, X, tasks, method='path'):
        '''
        @todo: - more generic parametrization for invoking the different prediction methods
        '''
        self._init_check()
        X = asmtarray_object(X)
        
        if len(X) != len(tasks):
            raise ValueError('number of task data X and tasks must be equal.')
        
        if method == 'path':
            pred_method = self.__predict_task_path
        elif method == 'maxcluster':
            pred_method = self.__predict_task_maxcluster
        elif method == 'cutpath':
            pred_method = self.__predict_task_cutpath
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        
        n = len(X)
        Y = np.empty(n, dtype='object')
        for i in xrange(n):
            if len(X[i]) > 0:
                yhat = pred_method(X[i], tasks[i])
                Y[i] = yhat
            else: 
                Y[i] = []            
        return Y
        
    def predict(self, X, Z, method='path', k=1):
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
        elif method == 'maxpath':
            pred_method = self.__predict_maxpath
        elif method == 'cutpath':
            pred_method = self.__predict_cutpath
        elif method == 'complete':
            pred_method = self.__predict_complete
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        n = len(X)
        Y = np.empty(n, dtype='object')
        for i in xrange(n):
            yhat = pred_method(X[i], Z[i], k)
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
    
    def _build_bhc_model(self, alpha, kernel, priors, X, Y, Z):
        '''
        @todo: - implement the access of the block matrix R
        '''
        
        n = len(X)
        
        
        if self._cached_data == None:
            task_cluster = [None]*n
            nodes = [None]*n
        
            #initialize the leafs of the cluster hierarchy by each task of its own
            for i in xrange(n):
                Xi = X[i]
                Yi = Y[i]
                Zi = Z[i]
                
                model = self.__make_gp_model(Xi, Yi)
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
                    X = np.r_[cluster_i.X, cluster_j.X]
                    y = np.r_[cluster_i.y, cluster_j.y]
                        
                    model = self.__make_gp_model(X, y)
                    cluster = _ClusterNode.make_internal(cluster_i, cluster_j, 
                                                         model, alpha)
                        
                    model_matrix[i,j] = model_matrix[j,i] = model
                    prob_matrix[i,j] = prob_matrix[j,i] = cluster.post
                    
            self._cached_data = (np.copy(task_cluster), np.copy(nodes), 
                                 np.copy(model_matrix), np.copy(prob_matrix))
        else:
            task_cluster = np.copy(self._cached_data[0])
            nodes = np.copy(self._cached_data[1])
            model_matrix = np.copy(self._cached_data[2])
            prob_matrix = np.copy(self._cached_data[3])
            
            #todo: hack
            for i in xrange(n):
                nodes[i] = MutableBTree(task_cluster[i])
                
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
                    
                    X = np.r_[cluster.X, cluster_i.X] #same as np.r_[X[p],X[q],X[i]]
                    y = np.r_[cluster.y, cluster_i.y] #same as np.r_[Y[p],Y[q],Y[i]]
                    
                    model = self.__make_gp_model(X, y)
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
    
    def __make_gp_model(self, X, y):
        kernel = self._kernel.copy()
        priors = self._priors
        #selector = self._selector
        
        model = GPRegression(kernel, infer_method=ExactInference, priors=priors)
        model.fit(X,y)
        return model
    
    def _update_model(self, alpha, treenode):
        self._alpha = alpha
        
        self.__update_alpha(alpha, treenode)
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
    
    #Prediction Methods
    def __predict_task_path(self, X, task):
        '''
        @todo: - meta features does not work, remove this shit
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)
        
        n = len(X)
        yhat = np.zeros(n)
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        print '--------------'
        for node in path:
            print 'weight={0}'.format(node.cargo.weight)
            total_weight += node.cargo.weight
                 
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for node in path:
            model = node.cargo.model
            yhat += node.cargo.weight*model.predict(X)/total_weight
        
        return yhat
        
    def __predict_task_maxcluster(self, X, task):
        '''
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)

        n = len(X)
        yhat = np.zeros(n)
        
        #determine the cluster with the highest weight
        max_weight = 0
        max_cluster = None
        for node in path:
            if node.cargo.weight > max_weight:
                max_weight = node.cargo.weight
                max_cluster = node.cargo
         
        model = max_cluster.model
        yhat = model.predict(X)
                
        return yhat
        
    def __predict_task_cutpath(self, X, task, r=0.5):
        '''
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)

        n = len(X)
        yhat = np.zeros(n)
        
        total_weight = 0
        for node in path:
            if node.cargo.post >= np.log(r):
                total_weight += node.cargo.weight
        
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for node in path:
            if node.cargo.post >= np.log(r):
                model = node.cargo.model
                yhat += node.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    def __lookup_tasknode(self, task_id):
        '''
        @todo: - use hashing for O(1) access
               - exception handling
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves())
        
        for node in leaves:
            if node.cargo.task_id == task_id:
                return node
            
        raise ValueError('task not found')
            
    def __predict_complete(self, X, z, k):
        '''
        @todo: - should the weights normalized
        '''
        tree = self._task_tree
        n = X.shape[0]
        yhat = np.zeros(n)
        
        total_weight = 0
        for cluster in tree:
            total_weight += cluster.weight
            
        for cluster in tree:
            model = cluster.model
            weight = cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat
    
    def __predict_path(self, X, z, k):
        '''
        @todo: - check whether the treenodes putting into the cluster_set are correctly 
                 inserted, that means no collosion occurs. Is the id() function sufficient
                 to compute the hash value of the object? What is better, using treenode or
                 cluster nodes?
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks, dist = neighbor_struct.query_knn(z, k)
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
            
        #remove clusters with low probability by creating a new list
        #tmp_cluster_set = list()
        #for cluster in cluster_set:
        #    if cluster.cargo.weight > 1e-2:
        #        tmp_cluster_set.append(cluster)
        #cluster_set = tmp_cluster_set
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        for cluster in cluster_set:
            total_weight += cluster.cargo.weight
            
        path = path_to_root(neighbors[0])
        #print 'total_w={0}'.format(total_weight)
        for node in path:
            cluster = node.cargo
            print 'weight={0}, norm_weight={1}'.format(cluster.weight, cluster.weight/total_weight)
            
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for cluster in cluster_set:
            model = cluster.cargo.model
            yhat += cluster.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    def __predict_cutpath(self, X, z, k, r=0.5):
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks, dist = neighbor_struct.query_knn(z, k)
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #print 'meta task={0}'.format(z)
        #for i in knn_tasks:
        #    print 'matched task={0}, dist={1}'.format(leaves[i].cargo.z, dist)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
            
        #remove clusters with low probability by creating a new list
        #tmp_cluster_set = list()
        #for cluster in cluster_set:
        #    if cluster.cargo.weight > 1e-2:
        #        tmp_cluster_set.append(cluster)
        #cluster_set = tmp_cluster_set
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        for cluster in cluster_set:
            if cluster.cargo.post >= np.log(r):
                total_weight += cluster.cargo.weight
        
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        
        #print self._task_tree
        for cluster in cluster_set:
            if cluster.cargo.post >= np.log(r):
                model = cluster.cargo.model
                yhat += cluster.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    
    def __predict_maxpath(self, X, z, k):
        '''
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks,_ = neighbor_struct.query_knn(z, k)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
        
        #determine the cluster with the highest weight
        max_weight = 0
        max_cluster = None
        for cluster in cluster_set:
            if cluster.cargo.weight > max_weight:
                max_weight = cluster.cargo.weight
                max_cluster = cluster.cargo
         
        model = max_cluster.model
        yhat = model.predict(X)
                
        return yhat
            
    #Misc methods
    def xml_element(self):
        '''
        '''
        root_element = ElementTree.Element('bhc_model')
        
        #build parameter element 
        param_element = ElementTree.SubElement(root_element, 'parameters')
        #param_element.attrib['rho'] = str(self._rho)
        param_element.attrib['alpha'] = str(self._alpha)
        #param_element.attrib['beta'] = str(self._beta)
        
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
            
        cluster_element.attrib['descr'] = str(cluster.descr)
        
        likel_element = ElementTree.SubElement(cluster_element, 'likel')
        likel_element.attrib['post'] = str(np.exp(cluster.post))
        likel_element.attrib['marg'] = str(cluster.marg)
        likel_element.attrib['mlikel'] = str(reg_model.log_likel)
        likel_element.attrib['pi'] = str(cluster.pi)
        likel_element.attrib['norm'] = str(cluster.norm_const)
        
        pred_element = ElementTree.SubElement(cluster_element, 'predict')
        pred_element.attrib['weight'] = str(cluster.weight)
        pred_element.attrib['invweight'] = str(cluster.inv_weight)
        
        prop_element = ElementTree.SubElement(cluster_element, 'props')
        prop_element.attrib['ntask'] = str(cluster.ntasks)
        prop_element.attrib['size'] = str(cluster.size)
        #prop_element.attrib['shape'] = str(cluster.covariates.shape)
        
        model_element = ElementTree.SubElement(cluster_element, 'reg_model')
        model_element.attrib['params'] = str(np.exp(reg_model.hyperparams))
        #model_element.attrib['intercept'] = str(reg_model.intercept)
        #model_element.attrib['weights'] = str(reg_model.weights)
        
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


class RobustSparseBHCGPRegression(BaseEstimator, RegressorMixin):
    '''     
    @todo: - 
    '''
    
    __slots__ = ('_alpha',          #cluster concentration hyperparameter in log space
                 '_kernel',         #initial kernel
                 '_priors',         #priors of kernel hyperparameters
                 '_selector',       #selection method for inducing points
                 '_k',
                 '_task_tree',      #the cluster tree of regression models
                 '_cached_data',
                 '_is_init'
                 )        
    
    def __init__(self, alpha, kernel, priors=None, k=15):
        '''
        @todo: - check the number of hyperparams and priors
        '''
        self._alpha = alpha
        self._kernel = kernel
        self._priors = priors
        self._k = k
        
        self._cached_data = None
        self._is_init = False
        
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        return np.r_[self._alpha, np.copy(self._kernel.params)]
    
    def _set_hyperparams(self, params):
        '''
        '''
        self._alpha = params[0]
        self._kernel.params = np.copy(params[1:])

    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)
    
    def fit(self, X, Y, Z, max_it=50, tol=1.e-5):
        '''
        '''
        X = asmtarray_object(X)
        Y = asmtarray_object(Y)
        Z = asmtarray_object(Z)
        
        if len(X) != len(Y) != len(Z):
            raise ValueError('number of task data X, Y and task ' + 
                             'description Z must be the same')
        
        kernel = self._kernel
        priors = self._priors
        alpha = self._alpha
        #beta = np.copy(kernel.params)
                
        #todo: make the code more  clear
        #determine a set of inducing points using k-means
        X_p = flatten_data(X)
        Y_p = flatten_data(Y)
        #Xu = KMeansSelector(self._k).apply(X_p, Y_p)
        Xu = RandomSubsetSelector(self._k).apply(X_p, Y_p)
        self._selector = FixedSelector(Xu)
        
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
            t = time.time()
            print 'E-Step'
            model = self._build_bhc_model(alpha, kernel, priors, X, Y, Z)
            print 'time={0}'.format(time.time()-t)
            
            #M-Step: optimize the parameters on a fixed tree
            #beta are the model hyperparameters
            t = time.time()
            print 'M-Step'
            likel, alpha = self._optimize_bhc_parameters(model, alpha, priors)
            print 'time={0}'.format(time.time()-t)
            bhc_model = model
            i += 1
            
        self._update_model(alpha, bhc_model)
        
        self._task_tree = bhc_model
        self._alpha = alpha
        
        self._cached_data = None
        self._is_init = True  
        
    def _optimize_bhc_parameters(self, model, alpha, priors):
        '''
        '''        
        print model.cargo.marg
        alpha_likel_fun = BHCAlphaLikelihood(model)
        
        opt_alpha = spopt.brent(alpha_likel_fun, brack=(np.log(1e-100), alpha, np.log(1e+100)))
        
        print 'opt_alpha={0}'.format(opt_alpha)
        self.__update_alpha(opt_alpha, model)
        print model.cargo.marg
        likel = model.cargo.marg
        
        return (likel, opt_alpha) 
        
    def predict_by_task(self, X, tasks, method='path'):
        '''
        @todo: - more generic parametrization for invoking the different prediction methods
        '''
        self._init_check()
        X = asmtarray_object(X)
        
        if len(X) != len(tasks):
            raise ValueError('number of task data X and tasks must be equal.')
        
        if method == 'path':
            pred_method = self.__predict_task_path
        elif method == 'maxcluster':
            pred_method = self.__predict_task_maxcluster
        elif method == 'cutpath':
            pred_method = self.__predict_task_cutpath
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        
        n = len(X)
        Y = np.empty(n, dtype='object')
        for i in xrange(n):
            if len(X[i]) > 0:
                yhat = pred_method(X[i], tasks[i])
                Y[i] = yhat
            else: 
                Y[i] = []            
        return Y
        
    def predict(self, X, Z, method='path', k=1):
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
        elif method == 'maxpath':
            pred_method = self.__predict_maxpath
        elif method == 'cutpath':
            pred_method = self.__predict_cutpath
        elif method == 'complete':
            pred_method = self.__predict_complete
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        n = len(X)
        Y = np.empty(n, dtype='object')
        for i in xrange(n):
            yhat = pred_method(X[i], Z[i], k)
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
    
    def _build_bhc_model(self, alpha, kernel, priors, X, Y, Z):
        '''
        @todo: - implement the access of the block matrix R
        '''
        
        n = len(X)
        
        
        if self._cached_data == None:
            task_cluster = [None]*n
            nodes = [None]*n
        
            #initialize the leafs of the cluster hierarchy by each task of its own
            for i in xrange(n):
                Xi = X[i]
                Yi = Y[i]
                Zi = Z[i]
                
                model = self.__make_gp_model(Xi, Yi)
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
                    X = np.r_[cluster_i.X, cluster_j.X]
                    y = np.r_[cluster_i.y, cluster_j.y]
                        
                    model = self.__make_gp_model(X, y)
                    cluster = _ClusterNode.make_internal(cluster_i, cluster_j, 
                                                         model, alpha)
                        
                    model_matrix[i,j] = model_matrix[j,i] = model
                    prob_matrix[i,j] = prob_matrix[j,i] = cluster.post
                    
            self._cached_data = (np.copy(task_cluster), np.copy(nodes), 
                                 np.copy(model_matrix), np.copy(prob_matrix))
        else:
            task_cluster = np.copy(self._cached_data[0])
            nodes = np.copy(self._cached_data[1])
            model_matrix = np.copy(self._cached_data[2])
            prob_matrix = np.copy(self._cached_data[3])
            
            #todo: hack
            for i in xrange(n):
                nodes[i] = MutableBTree(task_cluster[i])
                
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
                    
                    X = np.r_[cluster.X, cluster_i.X] #same as np.r_[X[p],X[q],X[i]]
                    y = np.r_[cluster.y, cluster_i.y] #same as np.r_[Y[p],Y[q],Y[i]]
                    
                    model = self.__make_gp_model(X, y)
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
    
    def __make_gp_model(self, X, y):
        kernel = self._kernel.copy()
        priors = self._priors
        selector = self._selector
        
        model = SparseGPRegression(kernel, infer_method=FITCExactInference, priors=priors, selector=selector)
        model.fit(X,y)
        return model
    
    def _update_model(self, alpha, treenode):
        self._alpha = alpha
        
        self.__update_alpha(alpha, treenode)
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
    
    #Prediction Methods
    def __predict_task_path(self, X, task):
        '''
        @todo: - meta features does not work, remove this shit
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)
        
        n = len(X)
        yhat = np.zeros(n)
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        print '--------------'
        for node in path:
            print 'weight={0}'.format(node.cargo.weight)
            total_weight += node.cargo.weight
                 
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for node in path:
            model = node.cargo.model
            yhat += node.cargo.weight*model.predict(X)/total_weight
        
        return yhat
        
    def __predict_task_maxcluster(self, X, task):
        '''
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)

        n = len(X)
        yhat = np.zeros(n)
        
        #determine the cluster with the highest weight
        max_weight = 0
        max_cluster = None
        for node in path:
            if node.cargo.weight > max_weight:
                max_weight = node.cargo.weight
                max_cluster = node.cargo
         
        model = max_cluster.model
        yhat = model.predict(X)
                
        return yhat
        
    def __predict_task_cutpath(self, X, task, r=0.5):
        '''
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)

        n = len(X)
        yhat = np.zeros(n)
        
        total_weight = 0
        for node in path:
            if node.cargo.post >= np.log(r):
                total_weight += node.cargo.weight
        
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for node in path:
            if node.cargo.post >= np.log(r):
                model = node.cargo.model
                yhat += node.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    def __lookup_tasknode(self, task_id):
        '''
        @todo: - use hashing for O(1) access
               - exception handling
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves())
        
        for node in leaves:
            if node.cargo.task_id == task_id:
                return node
            
        raise ValueError('task not found')
            
    def __predict_complete(self, X, z, k):
        '''
        @todo: - should the weights normalized
        '''
        tree = self._task_tree
        n = X.shape[0]
        yhat = np.zeros(n)
        
        total_weight = 0
        for cluster in tree:
            total_weight += cluster.weight
            
        for cluster in tree:
            model = cluster.model
            weight = cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat
    
    def __predict_path(self, X, z, k):
        '''
        @todo: - check whether the treenodes putting into the cluster_set are correctly 
                 inserted, that means no collosion occurs. Is the id() function sufficient
                 to compute the hash value of the object? What is better, using treenode or
                 cluster nodes?
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks, dist = neighbor_struct.query_knn(z, k)
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
            
        #remove clusters with low probability by creating a new list
        #tmp_cluster_set = list()
        #for cluster in cluster_set:
        #    if cluster.cargo.weight > 1e-2:
        #        tmp_cluster_set.append(cluster)
        #cluster_set = tmp_cluster_set
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        for cluster in cluster_set:
            total_weight += cluster.cargo.weight
            
        path = path_to_root(neighbors[0])
        #print 'total_w={0}'.format(total_weight)
        for node in path:
            cluster = node.cargo
            print 'weight={0}, norm_weight={1}'.format(cluster.weight, cluster.weight/total_weight)
            
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for cluster in cluster_set:
            model = cluster.cargo.model
            yhat += cluster.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    def __predict_cutpath(self, X, z, k, r=0.5):
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks, dist = neighbor_struct.query_knn(z, k)
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #print 'meta task={0}'.format(z)
        #for i in knn_tasks:
        #    print 'matched task={0}, dist={1}'.format(leaves[i].cargo.z, dist)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
            
        #remove clusters with low probability by creating a new list
        #tmp_cluster_set = list()
        #for cluster in cluster_set:
        #    if cluster.cargo.weight > 1e-2:
        #        tmp_cluster_set.append(cluster)
        #cluster_set = tmp_cluster_set
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        for cluster in cluster_set:
            if cluster.cargo.post >= np.log(r):
                total_weight += cluster.cargo.weight
        
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        
        #print self._task_tree
        for cluster in cluster_set:
            if cluster.cargo.post >= np.log(r):
                model = cluster.cargo.model
                yhat += cluster.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    
    def __predict_maxpath(self, X, z, k):
        '''
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks,_ = neighbor_struct.query_knn(z, k)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
        
        #determine the cluster with the highest weight
        max_weight = 0
        max_cluster = None
        for cluster in cluster_set:
            if cluster.cargo.weight > max_weight:
                max_weight = cluster.cargo.weight
                max_cluster = cluster.cargo
         
        model = max_cluster.model
        yhat = model.predict(X)
                
        return yhat
            
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
            
        cluster_element.attrib['descr'] = str(cluster.descr)
        
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


class BHCLinearRegression(BaseEstimator, RegressorMixin):
    '''     
    @todo: - 
    '''
    
    __slots__ = ('_alpha',          #cluster concentration hyperparameter in log space
                 '_beta',           #hyperparameters of regression model
                 '_task_tree',      #the cluster tree of regression models
                 '_is_init'
                 )        
    
    def __init__(self, alpha, beta):
        '''
        @todo: - check the number of hyperparams and priors
        '''
        self._alpha = alpha
        self._beta = beta
        
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
        Z = asmtarray_object(Z)
        
        if len(X) != len(Y) != len(Z):
            raise ValueError('number of task data X, Y and task ' + 
                             'description Z must be the same')
        
        alpha = self._alpha
        beta = self._beta
        
        #todo: should be an optional preprocessing step
        if True:
            #Estimate hyperparameters alpha and beta by fitting the linear model
            #to the whole data
            covars = flatten_data(X)
            targets = flatten_data(Y)
            reg_model = EBChenRegression()
            reg_model.fit(covars, targets)
            
            beta = np.log(np.array([reg_model.alpha, reg_model.beta]))
            print 'alpha={0}, beta={1}'.format(alpha, beta)
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
            model = self._build_bhc_model(alpha, beta, X, Y, Z)
            
            #M-Step: optimize the parameters on a fixed tree
            #beta are the model hyperparameters
            likel, alpha, beta = self._optimize_bhc_parameters(model, alpha, beta)
            bhc_model = model
            i += 1
            
        self._update_model(alpha, beta, bhc_model)
        
        self._task_tree = bhc_model
        self._alpha = alpha
        
        self._is_init = True  
        
    def _optimize_bhc_parameters(self, model, alpha, beta):
        '''
        '''        
        print model.cargo.marg
        alpha_likel_fun = BHCAlphaLikelihood(model)
        opt_alpha = spopt.brent(alpha_likel_fun, brack=(np.log(1e-100), alpha, np.log(1e+100)))
        
        print 'opt_alpha={0}'.format(opt_alpha)
        self.__update_alpha(opt_alpha, model)
        print model.cargo.marg
        
        model_likel_fun = _LinearRegressionModelLikelihood(model)
        model_likel_grad = model_likel_fun.gradient
        opt_result = spopt.fmin_bfgs(model_likel_fun, beta, model_likel_grad, full_output=True)
            
        opt_beta = opt_result[0]
        likel = opt_result[1]
        print 'opt_beta={0}'.format(np.exp(opt_beta))
        print likel
        
        return (likel, opt_alpha, opt_beta) 
        
    def predict_by_task(self, X, tasks, method='path'):
        '''
        @todo: - more generic parametrization for invoking the different prediction methods
        '''
        self._init_check()
        X = asmtarray_object(X)
        
        if len(X) != len(tasks):
            raise ValueError('number of task data X and tasks must be equal.')
        
        if method == 'path':
            pred_method = self.__predict_task_path
        elif method == 'maxcluster':
            pred_method = self.__predict_task_maxcluster
        elif method == 'cutpath':
            pred_method = self.__predict_task_cutpath
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        
        n = len(X)
        Y = np.empty(n, dtype='object')
        for i in xrange(n):
            if len(X[i]) > 0:
                yhat = pred_method(X[i], tasks[i])
                Y[i] = yhat
            else: 
                Y[i] = []    
            
        return Y
        
    def predict(self, X, Z, method='path', k=1):
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
        elif method == 'maxpath':
            pred_method = self.__predict_maxpath
        elif method == 'cutpath':
            pred_method = self.__predict_cutpath
        elif method == 'complete':
            pred_method = self.__predict_complete
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        n = len(X)
        Y = np.empty(n, dtype='object')
        for i in xrange(n):
            yhat = pred_method(X[i], Z[i], k)
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
    
    def _build_bhc_model(self, alpha, beta, X, Y, Z):
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
            
            model = self.__make_regression_model(Xi, Yi, beta)
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
                X = np.r_[cluster_i.X, cluster_j.X]
                y = np.r_[cluster_i.y, cluster_j.y]
                    
                model = self.__make_regression_model(X, y, beta)
                cluster = _ClusterNode.make_internal(cluster_i, cluster_j, 
                                                     model, alpha)
                    
                model_matrix[i,j] = model_matrix[j,i] = model
                prob_matrix[i,j] = prob_matrix[j,i] = cluster.post
                
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
        
            #print 'max_prob={0}'.format(max_prob)
                                
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
                    
                    X = np.r_[cluster.X, cluster_i.X] #same as np.r_[X[p],X[q],X[i]]
                    y = np.r_[cluster.y, cluster_i.y] #same as np.r_[Y[p],Y[q],Y[i]]
                    
                    model = self.__make_regression_model(X, y, beta)
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
    
    def __make_regression_model(self, X, y, beta):
        model = FastBayesRegression(np.exp(beta[0]), np.exp(beta[1]), weight_bias=True)
        model.fit(X,y)
        return model
    
    def _update_model(self, alpha, beta, treenode):
        self._beta = beta
        self._alpha = alpha
        
        self.__update_alpha(alpha, treenode)
        self.__update_model_parameters(beta, treenode)
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
            

    def __update_model_parameters(self, beta, treenode):
        '''
        @todo: kernel is not used
        '''
        if treenode.isatom():
            cluster = treenode.cargo
            
            #rebuild the gp model
            model = self.__make_regression_model(cluster.X, cluster.y, beta)
            cluster.model = model
            cluster.marg = model.log_likel
        else:
            self.__update_model_parameters(beta, treenode.left)
            self.__update_model_parameters(beta, treenode.right)
            
            cluster = treenode.cargo
            left = treenode.left.cargo
            right = treenode.right.cargo
            
            cluster = treenode.cargo

            #rebuild the gp model
            model = self.__make_regression_model(cluster.X, cluster.y, beta)
            cluster.model = model            
            
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = left.norm_const + right.norm_const - cluster.norm_const
            a = cluster.pi + model.log_likel
            b = npi+left.marg+right.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + model.log_likel - cluster.marg
    
    #Prediction Methods
    def __predict_task_path(self, X, task):
        '''
        @todo: - meta features does not work, remove this shit
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)
        
        n = len(X)
        yhat = np.zeros(n)
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        print '-------------------'
        for node in path:
            
            print 'weight={0}'.format(node.cargo.weight)
            total_weight += node.cargo.weight
                 
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for node in path:
            model = node.cargo.model
            yhat += node.cargo.weight*model.predict(X)/total_weight
        
        return yhat
        

    def __predict_task_maxcluster(self, X, task):
        '''
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)

        n = len(X)
        yhat = np.zeros(n)
        
        #determine the cluster with the highest weight
        max_weight = 0
        max_cluster = None
        for node in path:
            if node.cargo.weight > max_weight:
                max_weight = node.cargo.weight
                max_cluster = node.cargo
         
        model = max_cluster.model
        yhat = model.predict(X)
                
        return yhat
        
    def __predict_task_cutpath(self, X, task, r=0.5):
        '''
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)

        n = len(X)
        yhat = np.zeros(n)
        
        total_weight = 0
        for node in path:
            if node.cargo.post >= np.log(r):
                total_weight += node.cargo.weight
        
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for node in path:
            if node.cargo.post >= np.log(r):
                model = node.cargo.model
                yhat += node.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    def __lookup_tasknode(self, task_id):
        '''
        @todo: - use hashing for O(1) access
               - exception handling
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves())
        
        for node in leaves:
            if node.cargo.task_id == task_id:
                return node
            
        raise ValueError('task not found')
            
    def __predict_complete(self, X, z, k):
        '''
        @todo: - should the weights normalized
        '''
        tree = self._task_tree
        n = X.shape[0]
        yhat = np.zeros(n)
        
        total_weight = 0
        for cluster in tree:
            total_weight += cluster.weight
            
        for cluster in tree:
            model = cluster.model
            weight = cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat
    
    def __predict_path(self, X, z, k):
        '''
        @todo: - check whether the treenodes putting into the cluster_set are correctly 
                 inserted, that means no collosion occurs. Is the id() function sufficient
                 to compute the hash value of the object? What is better, using treenode or
                 cluster nodes?
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks, dist = neighbor_struct.query_knn(z, k)
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
            
        #remove clusters with low probability by creating a new list
        #tmp_cluster_set = list()
        #for cluster in cluster_set:
        #    if cluster.cargo.weight > 1e-2:
        #        tmp_cluster_set.append(cluster)
        #cluster_set = tmp_cluster_set
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        for cluster in cluster_set:
            total_weight += cluster.cargo.weight
            
        path = path_to_root(neighbors[0])
        #print 'total_w={0}'.format(total_weight)
        for node in path:
            cluster = node.cargo
            print 'weight={0}, norm_weight={1}'.format(cluster.weight, cluster.weight/total_weight)
            
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for cluster in cluster_set:
            model = cluster.cargo.model
            yhat += cluster.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    def __predict_cutpath(self, X, z, k, r=0.5):
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks, dist = neighbor_struct.query_knn(z, k)
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #print 'meta task={0}'.format(z)
        #for i in knn_tasks:
        #    print 'matched task={0}, dist={1}'.format(leaves[i].cargo.z, dist)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
            
        #remove clusters with low probability by creating a new list
        #tmp_cluster_set = list()
        #for cluster in cluster_set:
        #    if cluster.cargo.weight > 1e-2:
        #        tmp_cluster_set.append(cluster)
        #cluster_set = tmp_cluster_set
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        for cluster in cluster_set:
            if cluster.cargo.post >= np.log(r):
                total_weight += cluster.cargo.weight
        
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        
        #print self._task_tree
        for cluster in cluster_set:
            if cluster.cargo.post >= np.log(r):
                model = cluster.cargo.model
                yhat += cluster.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    
    def __predict_maxpath(self, X, z, k):
        '''
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks,_ = neighbor_struct.query_knn(z, k)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
        
        #determine the cluster with the highest weight
        max_weight = 0
        max_cluster = None
        for cluster in cluster_set:
            if cluster.cargo.weight > max_weight:
                max_weight = cluster.cargo.weight
                max_cluster = cluster.cargo
         
        model = max_cluster.model
        yhat = model.predict(X)
                
        return yhat
            
    #Misc methods
    def xml_element(self):
        '''
        '''
        root_element = ElementTree.Element('bhc_model')
        
        #build parameter element 
        param_element = ElementTree.SubElement(root_element, 'parameters')
        #param_element.attrib['rho'] = str(self._rho)
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
            
        cluster_element.attrib['descr'] = str(cluster.descr)
        
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
        #prop_element.attrib['shape'] = str(cluster.covariates.shape)
        
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

class BHCRobustRegression(BaseEstimator, RegressorMixin):
    '''     
    @todo: - 
    '''
    
    __slots__ = ('_alpha',          #cluster concentration hyperparameter in log space
                 '_beta',           #hyperparameters of regression model
                 '_task_tree',      #the cluster tree of regression models
                 '_is_init'
                 )        
    
    def __init__(self, alpha, beta):
        '''
        @todo: - check the number of hyperparams and priors
        '''
        self._alpha = alpha
        self._beta = beta
        
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
        Z = asmtarray_object(Z)
        
        if len(X) != len(Y) != len(Z):
            raise ValueError('number of task data X, Y and task ' + 
                             'description Z must be the same')
        
        alpha = self._alpha
        beta = self._beta
        
        
        
        
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
            model = self._build_bhc_model(alpha, beta, X, Y, Z)
            
            #M-Step: optimize the parameters on a fixed tree
            #beta are the model hyperparameters
            likel, alpha, beta = self._optimize_bhc_parameters(model, alpha, beta)
            bhc_model = model
            i += 1
            print 'fuck_likel={0}'.format(likel)
            
        self._update_model(alpha, beta, bhc_model)
        
        self._task_tree = bhc_model
        self._alpha = alpha
        
        self._is_init = True  
        
    def _optimize_bhc_parameters(self, model, alpha, beta):
        '''
        '''        
        print model.cargo.marg
        alpha_likel_fun = BHCAlphaLikelihood(model)
        opt_alpha = spopt.brent(alpha_likel_fun, brack=(np.log(1e-50), alpha, np.log(1e+50)))
        
        #print 'opt_alpha={0}'.format(opt_alpha)
        #print 'opt_f={0}'.format(opt_f)
        self.__update_alpha(alpha, model)
        #self.__update_alpha(alpha, model)
        #print 'back'
        print model.cargo.marg
        
            
        #constraints for the precision matrix - elements must be posititive
        n = len(beta)-2
        constraints = [(None, 200)]*2 # upper bound for overflow
        constraints.extend([(0,None)]*n)
            
        model_likel_fun = _LinearRegressionModelLikelihood(model)
        model_likel_grad = model_likel_fun.gradient
        #todo: which method is the best for constraining the precision matrix
        #opt_result = spopt.fmin_bfgs(model_likel_fun, beta, model_likel_grad, full_output=True)
        #opt_result = spopt.fmin_l_bfgs_b(model_likel_fun, beta, model_likel_grad, bounds=constraints)
        
        #tnc optimization stuff. opt likelihood is not returned
        opt_result = spopt.fmin_tnc(model_likel_fun, beta, model_likel_grad, bounds=constraints)
        opt_beta = opt_result[0]
        #opt_beta = beta
        likel = model_likel_fun(opt_beta)
        
        #likel = opt_result[1]
        #print opt_result[2]
        print 'opt_beta={0}'.format(opt_beta)
        print 'opt_likel={0}'.format(likel)
        
        return (likel, opt_alpha, opt_beta) 
        
    def predict_by_task(self, X, tasks, method='path'):
        '''
        @todo: - more generic parametrization for invoking the different prediction methods
        '''
        
        self._init_check()
        X = asmtarray_object(X)
        
        if len(X) != len(tasks):
            raise ValueError('number of task data X and tasks must be equal.')
        
        if method == 'path':
            pred_method = self.__predict_task_path
        elif method == 'maxcluster':
            pred_method = self.__predict_task_maxcluster
        elif method == 'cutpath':
            pred_method = self.__predict_task_cutpath
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        
        n = len(X)
        Y = np.empty(n, dtype='object')
        for i in xrange(n):
            if len(X[i]) > 0:
                yhat = pred_method(X[i], tasks[i])
                Y[i] = yhat
            else: 
                Y[i] = []
            
        return Y
        
    def predict(self, X, Z, method='path', k=1):
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
        elif method == 'maxpath':
            pred_method = self.__predict_maxpath
        elif method == 'cutpath':
            pred_method = self.__predict_cutpath
        elif method == 'complete':
            pred_method = self.__predict_complete
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        n = len(X)
        Y = np.empty(n, dtype='object')
        for i in xrange(n):
            yhat = pred_method(X[i], Z[i], k)
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
    
    def _build_bhc_model(self, alpha, beta, X, Y, Z):
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
            
            model = self.__make_regression_model(Xi, Yi, beta)
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
                X = np.r_[cluster_i.X, cluster_j.X]
                y = np.r_[cluster_i.y, cluster_j.y]
                    
                model = self.__make_regression_model(X, y, beta)
                cluster = _ClusterNode.make_internal(cluster_i, cluster_j, 
                                                     model, alpha)
                    
                model_matrix[i,j] = model_matrix[j,i] = model
                prob_matrix[i,j] = prob_matrix[j,i] = cluster.post
                
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
        
            #print 'max_prob={0}'.format(max_prob)
                                
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
                    
                    X = np.r_[cluster.X, cluster_i.X] #same as np.r_[X[p],X[q],X[i]]
                    y = np.r_[cluster.y, cluster_i.y] #same as np.r_[Y[p],Y[q],Y[i]]
                    
                    model = self.__make_regression_model(X, y, beta)
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
    
    def __make_regression_model(self, X, y, beta):
        a,b,L = RobustBayesRegression.unwrap(beta)
        model = RobustBayesRegression(a,b,L)
        model.fit(X,y)
        return model
    
    def _update_model(self, alpha, beta, treenode):
        self._beta = beta
        self._alpha = alpha
        
        self.__update_alpha(alpha, treenode)
        self.__update_model_parameters(beta, treenode)
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
            

    def __update_model_parameters(self, beta, treenode):
        '''
        @todo: kernel is not used
        '''
        if treenode.isatom():
            cluster = treenode.cargo
            
            #rebuild the gp model
            model = self.__make_regression_model(cluster.X, cluster.y, beta)
            cluster.model = model
            cluster.marg = model.log_likel
        else:
            self.__update_model_parameters(beta, treenode.left)
            self.__update_model_parameters(beta, treenode.right)
            
            cluster = treenode.cargo
            left = treenode.left.cargo
            right = treenode.right.cargo
            
            cluster = treenode.cargo

            #rebuild the gp model
            model = self.__make_regression_model(cluster.X, cluster.y, beta)
            cluster.model = model            
            
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = left.norm_const + right.norm_const - cluster.norm_const
            a = cluster.pi + model.log_likel
            b = npi+left.marg+right.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + model.log_likel - cluster.marg
    
    #Prediction Methods
    def __predict_task_path(self, X, task):
        '''
        @todo: - meta features does not work, remove this shit
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)
        
        n = len(X)
        yhat = np.zeros(n)
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        print '-------------------'
        for node in path:
            
            print 'weight={0}'.format(node.cargo.weight)
            total_weight += node.cargo.weight
                 
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for node in path:
            model = node.cargo.model
            yhat += node.cargo.weight*model.predict(X)/total_weight
        
        return yhat
        

    def __predict_task_maxcluster(self, X, task):
        '''
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)

        n = len(X)
        yhat = np.zeros(n)
        
        #determine the cluster with the highest weight
        max_weight = 0
        max_cluster = None
        for node in path:
            if node.cargo.weight > max_weight:
                max_weight = node.cargo.weight
                max_cluster = node.cargo
         
        model = max_cluster.model
        yhat = model.predict(X)
                
        return yhat
        
    def __predict_task_cutpath(self, X, task, r=0.5):
        '''
        '''
        tasknode = self.__lookup_tasknode(task)
        path = path_to_root(tasknode)

        n = len(X)
        yhat = np.zeros(n)
        
        total_weight = 0
        for node in path:
            if node.cargo.post >= np.log(r):
                total_weight += node.cargo.weight
        
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for node in path:
            if node.cargo.post >= np.log(r):
                model = node.cargo.model
                yhat += node.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    def __lookup_tasknode(self, task_id):
        '''
        @todo: - use hashing for O(1) access
               - exception handling
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves())
        
        for node in leaves:
            if node.cargo.task_id == task_id:
                return node
            
        raise ValueError('task not found')
            
    def __predict_complete(self, X, z, k):
        '''
        @todo: - should the weights normalized
        '''
        tree = self._task_tree
        n = X.shape[0]
        yhat = np.zeros(n)
        
        total_weight = 0
        for cluster in tree:
            total_weight += cluster.weight
            
        for cluster in tree:
            model = cluster.model
            weight = cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat
    
    def __predict_path(self, X, z, k):
        '''
        @todo: - check whether the treenodes putting into the cluster_set are correctly 
                 inserted, that means no collosion occurs. Is the id() function sufficient
                 to compute the hash value of the object? What is better, using treenode or
                 cluster nodes?
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks, dist = neighbor_struct.query_knn(z, k)
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
            
        #remove clusters with low probability by creating a new list
        #tmp_cluster_set = list()
        #for cluster in cluster_set:
        #    if cluster.cargo.weight > 1e-2:
        #        tmp_cluster_set.append(cluster)
        #cluster_set = tmp_cluster_set
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        for cluster in cluster_set:
            total_weight += cluster.cargo.weight
            
        path = path_to_root(neighbors[0])
        #print 'total_w={0}'.format(total_weight)
        for node in path:
            cluster = node.cargo
            print 'weight={0}, norm_weight={1}'.format(cluster.weight, cluster.weight/total_weight)
            
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for cluster in cluster_set:
            model = cluster.cargo.model
            yhat += cluster.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    def __predict_cutpath(self, X, z, k, r=0.5):
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks, dist = neighbor_struct.query_knn(z, k)
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #print 'meta task={0}'.format(z)
        #for i in knn_tasks:
        #    print 'matched task={0}, dist={1}'.format(leaves[i].cargo.z, dist)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
            
        #remove clusters with low probability by creating a new list
        #tmp_cluster_set = list()
        #for cluster in cluster_set:
        #    if cluster.cargo.weight > 1e-2:
        #        tmp_cluster_set.append(cluster)
        #cluster_set = tmp_cluster_set
        
        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        for cluster in cluster_set:
            if cluster.cargo.post >= np.log(r):
                total_weight += cluster.cargo.weight
        
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        
        #print self._task_tree
        for cluster in cluster_set:
            if cluster.cargo.post >= np.log(r):
                model = cluster.cargo.model
                yhat += cluster.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    
    def __predict_maxpath(self, X, z, k):
        '''
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.z for node in leaves])
        knn_tasks,_ = neighbor_struct.query_knn(z, k)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = leaves[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node)) 
        
        #determine the cluster with the highest weight
        max_weight = 0
        max_cluster = None
        for cluster in cluster_set:
            if cluster.cargo.weight > max_weight:
                max_weight = cluster.cargo.weight
                max_cluster = cluster.cargo
         
        model = max_cluster.model
        yhat = model.predict(X)
                
        return yhat
            
    #Misc methods
    def xml_element(self):
        '''
        '''
        root_element = ElementTree.Element('bhc_model')
        
        #build parameter element 
        param_element = ElementTree.SubElement(root_element, 'parameters')
        #param_element.attrib['rho'] = str(self._rho)
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
            
        cluster_element.attrib['descr'] = str(cluster.descr)
        
        likel_element = ElementTree.SubElement(cluster_element, 'likel')
        likel_element.attrib['post'] = str(np.exp(cluster.post))
        likel_element.attrib['marg'] = str(cluster.marg)
        #likel_element.attrib['mlikel'] = str(reg_model.log_evidence)
        likel_element.attrib['pi'] = str(cluster.pi)
        likel_element.attrib['norm'] = str(cluster.norm_const)
        
        pred_element = ElementTree.SubElement(cluster_element, 'predict')
        pred_element.attrib['weight'] = str(cluster.weight)
        pred_element.attrib['invweight'] = str(cluster.inv_weight)
        
        prop_element = ElementTree.SubElement(cluster_element, 'props')
        prop_element.attrib['ntask'] = str(cluster.ntasks)
        prop_element.attrib['size'] = str(cluster.size)
        #prop_element.attrib['shape'] = str(cluster.covariates.shape)
        
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
       
        if alpha != self._last_alpha:
            self._likel_tree = self._estimate_likelihood(alpha, self._bhc_model)
            self._last_alpha = alpha
            
        likel = -self._likel_tree.cargo.marg
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
        
        
class _GPModelLikelihood(object):
    '''
    @todo: - build log gradient to permit negative values of rho
    '''
    __slots__ = ('_bhc_model',
                 '_kernel',
                 '_last_beta'
                 )
    
    def __init__(self, bhc_model, kernel):
        self._bhc_model = bhc_model
        self._kernel = kernel
        self._last_beta = None
    
    def __call__(self, beta):
        '''
        '''
        likel = -self._bhc_model.cargo.marg
        #if np.any(beta != self._last_beta):
        if self._last_beta == None or np.any((np.not_equal(self._last_beta, beta))):
            self._kernel.params = beta
            #exception handling for no semi-definit kernel matrix
            try:
                self._estimate_likelihood(self._bhc_model)
                self._last_beta = beta
                likel = -self._bhc_model.cargo.marg
            except np.linalg.LinAlgError:
                self._kernel.params = self._last_beta
                self._estimate_likelihood(self._bhc_model)
                likel = 1e+300
            
        return likel
    
    def gradient(self, beta):
        '''
        '''
        if self._last_beta == None or np.any((np.not_equal(self._last_beta, beta))):
            #reestimation should not occur, because the likel function is invoked before normally  
            self._estimate_likelihood(self._bhc_model)
        grad = self._compute_gradient(self._bhc_model)
        return -grad
    
    def _estimate_likelihood(self, treenode):
        '''
        '''
        if treenode.isatom():
            cluster = treenode.cargo
            
            #rebuild the gp model
            model = cluster.model
            model.fit(cluster.X, cluster.y)
            cluster.marg = model.log_likel
        else:
            self._estimate_likelihood(treenode.left)
            self._estimate_likelihood(treenode.right)
            
            cluster = treenode.cargo
            left = treenode.left.cargo
            right = treenode.right.cargo
            
            cluster = treenode.cargo

            #rebuild the gp model
            model = cluster.model
            model.fit(cluster.X, cluster.y)
            
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = left.norm_const + right.norm_const - cluster.norm_const
            a = cluster.pi + model.log_likel
            b = npi+left.marg+right.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + model.log_likel - cluster.marg
                
    def _compute_gradient(self, treenode):
        cluster = treenode.cargo    #corresponding cluster of the treenode
        model = cluster.model       #regression model of the cluster
        likel_fun = model.likel_fun #likel function of the regression model
        
        r = np.exp(cluster.pi+likel_fun()-cluster.marg)
        #print 'r={0}, likel={1}, marg={2}, pi={3}, post={4}'.format(r, likel_fun(), cluster.marg, cluster.pi, np.exp(cluster.post))
        if r > 1.0:
            r = 1.0
        elif r < 0.0:
            r = 0.0
        elif np.isnan(r) or np.isinf(r):
            r = 0.0
            
        
        grad = r*likel_fun.gradient()
        if not treenode.isatom():
            grad_l = self._compute_gradient(treenode.left)
            grad_r = self._compute_gradient(treenode.right)
            grad += (1.0-r) * (grad_l+grad_r)
        
        return grad
    
class _LinearRegressionModelLikelihood(object):
    '''
    @todo: - build log gradient to permit negative values of rho
    '''
    __slots__ = ('_bhc_model',
                 '_last_beta'
                 )
    
    def __init__(self, bhc_model):
        self._bhc_model = bhc_model
        self._last_beta = None
    
    def __call__(self, beta):
        '''
        '''
        #print 'fop'
        #print 'params={0}'.format(beta)
        likel = -self._bhc_model.cargo.marg
       
        #if np.any(beta != self._last_beta):
            #exception handling for no semi-definit kernel matrix
        if self._last_beta == None or np.any((np.not_equal(self._last_beta, beta))):
            try:
                self._estimate_likelihood(self._bhc_model, beta)
                self._last_beta = beta
                likel = -self._bhc_model.cargo.marg
            except np.linalg.LinAlgError:
                beta = self._last_beta
                self._estimate_likelihood(self._bhc_model, beta)
                likel = 1e+300
            
        

        #print 'likel beta'
        #print beta
        #print 'likel'
        #print likel    
        return likel
    
    def gradient(self, beta):
        '''
        '''
        #print 'fup'
        #print 'params={0}'.format(beta)
        if self._last_beta == None or np.any((np.not_equal(self._last_beta, beta))):
            #reestimation should not occur, because the likel function is invoked before normally  
            self._estimate_likelihood(self._bhc_model, beta)
        grad = self._compute_gradient(self._bhc_model, beta)
        return -grad
    
    def _estimate_likelihood(self, treenode, beta):
        '''
        '''
        if treenode.isatom():
            cluster = treenode.cargo
            
            #rebuild the gp model
            model = cluster.model
            #todo: remove this hack
            if isinstance(model, RobustBayesRegression):
                model.refit(cluster.X, cluster.y, beta)
               
            else:
                model.refit(cluster.X, cluster.y, np.exp(beta[0]), np.exp(beta[1]))

            cluster.marg = model.log_likel
        else:
            self._estimate_likelihood(treenode.left, beta)
            self._estimate_likelihood(treenode.right, beta)
            
            cluster = treenode.cargo
            left = treenode.left.cargo
            right = treenode.right.cargo
            
            cluster = treenode.cargo

            #rebuild the gp model
            model = cluster.model
            #todo: remove this hack
            if isinstance(model, RobustBayesRegression):
                model.refit(cluster.X, cluster.y, beta)
            else:
                model.refit(cluster.X, cluster.y, np.exp(beta[0]), np.exp(beta[1]))
            
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = left.norm_const + right.norm_const - cluster.norm_const
            a = cluster.pi + model.log_likel
            b = npi+left.marg+right.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + model.log_likel - cluster.marg
                
    def _compute_gradient(self, treenode, beta):
        cluster = treenode.cargo    #corresponding cluster of the treenode
        model = cluster.model       #regression model of the cluster
        likel_fun = model.likel_fun #likel function of the regression model

        #todo: likel_fun returns inf if we cannot compute the cholesky decomposition
        #by fitting the regression model. how we can handle this.
        r = np.exp(cluster.pi+likel_fun(beta)-cluster.marg)
        #print 'r={0}, likel={1}, marg={2}, pi={3}, post={4}'.format(r, likel_fun(beta), cluster.marg, cluster.pi, np.exp(cluster.post))
        if r > 1.0:
            r = 1.0
        elif r < 0.0:
            r = 0.0
        elif np.isnan(r) or np.isinf(r):
            r = 0.0
        
        
        
        grad = r*likel_fun.gradient(beta)
        if not treenode.isatom():
            grad_l = self._compute_gradient(treenode.left, beta)
            grad_r = self._compute_gradient(treenode.right, beta)
            grad += (1.0-r) * (grad_l+grad_r)
        return grad
        
         
        
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
                 'size',
                 'descr') #hack

    @staticmethod
    def make_leaf(id, X, y, z, model, alpha):
        '''
        This factory method creates a leaf _ClusterNode at which the node 
        parameters instantiated from scratch.
        '''
        n = 1
        norm_const = alpha+sps.gammaln(n)
        marg = model.log_likel
        descr = str(id+1)
        node = _ClusterNode(X, y, z, model, n, marg, 
                            norm_const=norm_const, id=id, descr=descr)
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
        X = np.r_[left.X, right.X]
        y = np.r_[left.y, right.y]
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
        
        #the descriptor is the mean of the both child nodes.
        z = (left.z+right.z) / 2.0
        
        #print 'pi={0}, marg={1}, post={2}'.format(pi, marg, post)
        #print 'pi={0}, alpha={1}, n={2}, dk={3}, dl={4}, dr={5}, post={6}'.format(pi, alpha, n, norm_const, left.norm_const, right.norm_const, post)
        
        
        descr = left.descr + right.descr
        node = _ClusterNode(X, y, z, model, n, 
                            marg, post, pi, norm_const, descr=descr)
        return node
    
    #@change: post and pi are initialized by default by the log values
    def __init__(self, X, y, z, model, ntasks=1, 
                 marg=0.0, post=0.0, pi=0.0, norm_const=0.0, id=None, descr=None):
        self.task_id = id
        self.X = X
        self.y = y
        self.z = z
        self.model = model
        self.post = post
        self.pi = pi
        self.norm_const = norm_const
        self.marg = marg
        self.ntasks = ntasks
        self.weight = np.nan
        self.inv_weight = np.nan
        self.descr = descr
        
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
        copy_node = _ClusterNode(self.X, self.y, self.z, self.model, 
                                 self.ntasks, self.marg, 
                                 self.post, self.pi, self.norm_const)
        return copy_node
    
def evaluate_gp_model(X, Y, X_test, y_test):
    n = len(X)
    d = np.size(X[0],1)
    
    #learn model for each task
    for i in xrange(n):
        Xi = X[i]
        yi = Y[i]
        kernel = SEKernel(np.log(0.5), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(4.5)*np.ones(d), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = MaternKernel(5, np.log(4.5), np.log(1)) + NoiseKernel(np.log(0.5))
        gp_model = GPRegression(kernel, infer_method=ExactInference)
        gp_model.fit(Xi, yi)
        
        yhat = gp_model.predict(x_test)
        
        print 'task {0}: likel={1}, mse={2}'.format(i, gp_model.log_likel, mspe(yhat, y_test))
        print 'hyperparams: {0}'.format(np.exp(gp_model.hyperparams))
        
    covars = flatten_data(X)
    targets = flatten_data(Y)

    kernel = SEKernel(np.log(0.5), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = ARDSEKernel(np.log(4.5)*np.ones(d), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = MaternKernel(5, np.log(4.5), np.log(1)) + NoiseKernel(np.log(0.5))
    gp_model = GPRegression(kernel, infer_method=ExactInference)
    gp_model.fit(covars, targets)
    yhat = gp_model.predict(x_test)
    print 'complete: likel={0}, mse={1}'.format(gp_model.log_likel, mspe(yhat, y_test))
    print 'hyperparams: {0}'.format(np.exp(gp_model.hyperparams))
    
    total_mse = 0
    for i in xrange(n):
        Xi = X[i]
        yi = Y[i]
        
        yhat = gp_model.predict(Xi)
        mse = mspe(yhat, yi)
        print 'task {0}: mse={1}'.format(i, mse)
        total_mse += mse
    print 'reg total_mse={0}'.format(total_mse/n)
    
def evaluate_lin_model(X, Y, X_test, y_test):
    n = len(X)
    
    #learn model for each task
    for i in xrange(n):
        Xi = X[i]
        yi = Y[i]
        reg_model = EBChenRegression()
        reg_model.fit(Xi, yi)
        
        yhat = reg_model.predict(x_test)
        
        print 'task {0}: likel={1}, mse={2}'.format(i, reg_model.log_evidence, mspe(yhat, y_test))
        print 'hyperparams: {0}'.format([reg_model.alpha, reg_model.beta])
        
    covars = flatten_data(X)
    targets = flatten_data(Y)

    reg_model = EBChenRegression()
    reg_model.fit(covars, targets)
    yhat = reg_model.predict(x_test)
    print 'complete: likel={0}, mse={1}'.format(reg_model.log_evidence, mspe(yhat, y_test))
    print 'hyperparams: {0}'.format([reg_model.alpha, reg_model.beta])
    
    total_mse = 0
    for i in xrange(n):
        Xi = X[i]
        yi = Y[i]
        
        yhat = reg_model.predict(Xi)
        mse = mspe(yhat, yi)
        print 'task {0}: mse={1}'.format(i, mse)
        total_mse += mse
    print 'reg total_mse={0}'.format(total_mse/n)

    
if __name__ == '__main__':
    
    from upgeo.ml.regression.np.kernel import SEKernel, NoiseKernel
    from upgeo.ml.regression.np.infer import ExactInference
    
    from upgeo.util.metric import mspe
    from upgeo.eval.multitask.base import load_data

    constraints = [(None,None)]*2
    constraints.extend([(0,None)]*4)
    print constraints


#    filename = '/home/marcel/datasets/multilevel/ilea/school_norm_vrb.csv'
#    task_key = 'school'
#    task_fields = ['fsm', 'school_vr1', 'school_mixed', 'school_male', 'school_female', 
#                   'sdenom_maintain', 'sdenom_coe', 'sdenom_rc']
#    target_field = 'exam_score'
    
    filename = '/home/marcel/datasets/multilevel/ilea/exam_london_bhce_normed.csv'
    task_key = 'school'
    #task_fields = ['school_mixed', 'school_boys', 'school_girls', 
    #               'intake_score', 'school_vrb']
    task_fields = ['school_type', 'intake_score', 'school_vrb']
    target_field = 'exam_score'
    
    X,Y,Z = load_data(filename, task_key, task_fields, target_field)
    #X = compound_data(X, Z)
    
    x_test = X[58]
    y_test = Y[58]
    z_test = Z[58] 
    
    X = X[0:60]
    Y = Y[0:60]
    Z = Z[0:60]
 
#    evaluate_gp_model(X, Y, x_test, y_test)
    #evaluate_lin_model(X, Y, x_test, y_test)
    
    print 'bhc_linreg'
    
    bhc_linreg = BHCLinearRegression(np.log(10.5), np.log(np.array([2,2])))
    bhc_linreg.fit(X, Y, Z)
    yhat = bhc_linreg.predict([x_test], [z_test])
    print 'hier'
    print mspe(y_test, yhat[0])
    yhat = bhc_linreg.predict([x_test], [z_test], method='complete')
    print mspe(y_test, yhat[0])
    
    print 'robust'
    
    hyperparams = RobustBayesRegression.wrap(np.log(0.1), np.log(4), np.eye(4).ravel())
    print hyperparams
    bhc_linreg = BHCRobustRegression(np.log(10.5), hyperparams)
    bhc_linreg.fit(X, Y, Z)
    yhat = bhc_linreg.predict([x_test], [z_test])
    print 'hier'
    print mspe(y_test, yhat[0])
    yhat = bhc_linreg.predict([x_test], [z_test], method='complete')
    print mspe(y_test, yhat[0])
    
    
    
#    print 'bhc_gp'
#        
#    #learn bhc model
#    #4.02926712  1.15771128  0.73712249
#    #3.74822022  1.76607879  0.93516676
##    kernel = SEKernel(np.log(10.5), np.log(1)) + NoiseKernel(np.log(0.5))
##    bhc_gp = BHCGPRegression(np.log(1000), kernel, None, opt_model_params=True)
##    bhc_gp.fit(X, Y, Z)
##    yhat = bhc_gp.predict([x_test], [z_test])
##    print mspe(y_test, yhat[0])
##    yhat = bhc_gp.predict([x_test], [z_test], method='complete')
##    print mspe(y_test, yhat[0])
#    
#    print 'bhc_sgp'
#    
#    selector = RandomSubsetSelector(50)
#    kernel = SEKernel(np.log(10.5), np.log(1)) + NoiseKernel(np.log(0.5))
#    bhc_sgp = SparseBHCGPRegression(np.log(1000), kernel, None, selector, opt_model_params=True)
#    bhc_sgp.fit(X, Y, Z)
#    yhat = bhc_sgp.predict([x_test], [z_test])
#    print mspe(y_test, yhat[0])
#    yhat = bhc_sgp.predict([x_test], [z_test], method='complete')
#    print mspe(y_test, yhat[0])
#    
#    
#    bhc_reg_mse = 0.0
##    bhc_gp_mse = 0.0
#    bhc_sgp_mse = 0.0
#    for i in xrange(len(X)):
#        Xi = X[i]
#        yi = Y[i]
#        zi = Z[i]
#        
# #       yhat = bhc_gp.predict_by_task([Xi], [i])
# #       mse = mspe(yhat[0], yi)
# #       print 'task {0}, gp-path-mspe={1}'.format(i, mse)
# #       bhc_gp_mse += mse
#        
#        yhat = bhc_sgp.predict_by_task([Xi], [i])
#        mse = mspe(yhat[0], yi)
#        print 'task {0}, sgp-path-mspe={1}'.format(i, mse)
#        bhc_sgp_mse += mse
#        
#        yhat = bhc_linreg.predict_by_task([Xi], [i])
#        mse = mspe(yhat[0], yi)
#        print 'task {0}, lin-path-mspe={1}'.format(i, mse)
#        bhc_reg_mse += mse
#        
##    print 'bhc_gp total_mse={0}'.format(bhc_gp_mse/len(X))
#    print 'bhc_sgp total_mse={0}'.format(bhc_sgp_mse/len(X))
#    print 'bhc_reg total_mse={0}'.format(bhc_reg_mse/len(X))