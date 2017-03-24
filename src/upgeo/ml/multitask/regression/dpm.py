'''
Created on Apr 19, 2011

@author: marcel
'''
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import scipy.special as sps
import time

from scikits.learn.base import BaseEstimator, RegressorMixin

from upgeo.util.math import sumln, diffln
from upgeo.util.metric import NaiveMatrixSearch, mspe, distance_matrix
from upgeo.adt.tree import common_ancestor, path_to_root, MutableBTree,\
    shortest_path
from upgeo.ml.regression.bayes import MapBayesRegression, EMBayesRegression,\
    EBChenRegression, FastBayesRegression
from upgeo.exception import NotFittedError
from upgeo.eval.multitask.base import load_data
from upgeo.ml.multitask.base import MultiISpaceLearner, flatten_data,\
    asmtarray_object, compound_data
    
from scipy.optimize.minpack import check_gradient
from scipy.optimize.optimize import fminbound, fmin_cg, fmin_bfgs

from bisect import bisect
from scipy.optimize.lbfgsb import fmin_l_bfgs_b

from xml.etree import ElementTree

class BHCRegression(BaseEstimator, RegressorMixin):
    '''     
    Parameters
    ----------
    gamma: float
        The hyperparameter gamma defines a prior on all partitions of n tasks.
        The value of gamma is directly related to the expected number of clusters.
        

    alpha: float
        The hyperparameter alpha defines the precision for the normal prior of 
        regression weights.

    beta: float
        The hyperparameter beta defines the precision of normal distribution.

    Attributes
    ----------
    _task_tree: tree
        Hierarchical cluster structure of the different tasks.

    _is_init: bool
    '''
    
    __slots__ = ('_rho',            #cluster concentration hyperparameter
                 '_alpha',          #variance precision hyperparameter of regression weights
                 '_beta',           #variance precision hyperparameter of regression model
                 '_task_tree',      #the cluster tree of regression models
                 '_basis_function', #basis functions used in the regression models
                 '_use_meta_features',
                 '_is_init'
                 )        
    
    def __init__(self, rho, alpha=1.0, beta=1.0, basis_function=None, use_meta_features=False):
        '''
        '''
        if rho < 0:
            raise ValueError('rho must be non-negative.')
        
        self._rho = rho
        self._alpha = alpha
        self._beta = beta
        
        self._basis_function = basis_function
        self._use_meta_features = use_meta_features
        
        self._is_init = False

    def fit(self, X, Y, Z, model_matrix=None):
        '''
        @todo: - use a sparse matrix for distance computation (posteriors)
               - datatype checking + preprocessing
        '''
        X = asmtarray_object(X)
        Y = asmtarray_object(Y)
        Z = asmtarray_object(Z)
        
        if len(X) != len(Y) != len(Z):
            raise ValueError('number of task data X, Y and task ' + 
                             'description Z must be the same')
            
        X = self._preprocess_data(X)
        n = len(X)    
        
        if self._use_meta_features == True:
            #todo: problems occurs if different tasks have the same meta features
            XZ = compound_data(X,Z)
            
        #print 'init leaf clusters'
        #initialize the leafs of the cluster hierarchy by each task of its own
        task_cluster = [None]*n
        nodes = [None]*n
        for i in xrange(n):
            Xi = X[i]
            Yi = Y[i]
            Zi = Z[i]
            model = self.__make_regression_model(Xi, Yi)
            #print 'model={0}'.format(i)
            #print model.weights
            #print model.intercept
            if self._use_meta_features:
                Xi = XZ[i]
                
            task_cluster[i] = _ClusterNode.make_leaf(i, Xi, Yi, model, 
                                                     Zi, self._rho)
            nodes[i] = MutableBTree(task_cluster[i])
            #print 'intercept={0}, weight={1}'.format(model.intercept, model.weights)
            
        #print 'init matrices'
        #initialize the model matrix and the probability matrix of merged 
        #hypothesis
        if model_matrix == None:
            model_matrix = np.empty([n,n], dtype=object)
            prob_matrix = np.ones([n,n], dtype=float)*-np.inf
                
            for i in xrange(n):
                cluster_i = task_cluster[i]
                for j in xrange(i+1, n):
                    cluster_j = task_cluster[j]
                    covars = np.r_[cluster_i.covariates, cluster_j.covariates]
                    targets = np.r_[cluster_i.targets, cluster_j.targets]
                    
                    model = self.__make_regression_model(covars, targets)
                    cluster = _ClusterNode.make_internal(cluster_i, cluster_j, 
                                                         model, self._rho)
                    
                    #print "i={0}, j={1}".format(i, j)
                    #print model.weights
                    #print model.intercept
                    model_matrix[i,j] = model_matrix[j,i] = model
                    prob_matrix[i,j] = prob_matrix[j,i] = cluster.post
        else:
            prob_matrix = np.ones([n,n], dtype=float)*-np.inf
            for i in xrange(n):
                cluster_i = task_cluster[i]
                for j in xrange(i+1, n):
                    cluster_j = task_cluster[j]
                    model = model_matrix[i,j]
                    cluster = _ClusterNode.make_internal(cluster_i, cluster_j, 
                                                         model, self._rho)
                    
                    prob_matrix[i,j] = prob_matrix[j,i] = cluster.post
        
        print 'INIT'
        t1 = time.clock()
        reg_time = 0
        g = 0
        #print prob_matrix
        #main-loop of the hierarchical task clustering process
        last_ins = -1 #could be removed, because the last cluster node (root) is at pos 0
        n_cluster = n
        while n_cluster > 1:
            #print 'clustering'
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
            #covars = np.r_[X[p], X[q]]
            #targets = np.r_[Y[p], Y[q]]
            
            #print 'coordinates ({0},{1}), prob={2}'.format(p,q, max_prob)
            #print 'matrix'
            #print prob_matrix    
            
            cluster = _ClusterNode.make_internal(cluster_p, cluster_q, 
                                                 model_matrix[p,q], self._rho)
            
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
                    
                    covars = np.r_[cluster.covariates, cluster_i.covariates] #same as np.r_[X[p],X[q],X[i]]
                    targets = np.r_[cluster.targets, cluster_i.targets] #same as np.r_[Y[p],Y[q],Y[i]]
                    
                    g += 1
                    t2 = time.clock()
                    model = self.__make_regression_model(covars, targets)
                    reg_time += time.clock()-t2
                    tmp_cluster = _ClusterNode.make_internal(cluster, cluster_i, 
                                                             model, self._rho)
                    
                    
                    model_matrix[p,i] = model_matrix[i,p] = model
                    model_matrix[q,i] = model_matrix[i,q] = None
                    prob_matrix[p,i] = prob_matrix[i,p] = tmp_cluster.post
                    prob_matrix[q,i] = prob_matrix[i,q] = -1
                    
            n_cluster -= 1
            last_ins = p
              
        #print 'CLUSTERED = {0}, {1}, {2}'.format(g, time.clock()-t1, reg_time)
        
        self._task_tree = nodes[last_ins]
        self.__compute_cluster_weights(self._task_tree)      
        self._is_init = True   
        
    def predict_by_task(self, X, tasks, method='maxpath'):
        '''
        @todo: - more generic parametrization for invoking the different prediction methods
        '''
        self._init_check()
        X = asmtarray_object(X)
        
        if len(X) != len(tasks):
            raise ValueError('number of task data X and tasks must be equal.')
        
        X = self._preprocess_data(X)
        n = len(X)
        
        if method == 'path':
            pred_method = self.__predict_task_path
        elif method == 'maxpath':
            pred_method = self.__predict_task_maxpath
        elif method == 'cutpath':
            pred_method = self.__predict_task_cutpath
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        Y = np.empty(n, dtype='object')
        for i in xrange(n):
            yhat = pred_method(X[i], tasks[i])
            Y[i] = yhat
            
        return Y
        
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
        for node in path:
            total_weight += node.cargo.weight
            
        print 'weighting scheme'
        for node in path:
            cluster = node.cargo
            print 'weight={0}, norm_weight={1}, post={2}, model={3}, int={4}, shape={5}'.format(cluster.weight, cluster.weight/total_weight, cluster.post, cluster.model.weights, cluster.model.intercept, cluster.model.X.shape)
            
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        for node in path:
            model = node.cargo.model
            yhat += node.cargo.weight*model.predict(X)/total_weight
        
        return yhat
        

    def __predict_task_maxpath(self, X, task):
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
            
        X = self._preprocess_data(X)
        n = len(X)
            
        if method == 'path':
            pred_method = self.__predict_neighbor_path
        elif method == 'maxpath':
            pred_method = self.__predict_max_path
        elif method == 'path2':
            pred_method = self.__predict_cut_neighbor_path
        elif method == 'internal_path':
            pred_method = self.__predict_internal_path
        elif method == 'subtree':
            pred_method = self.__predict_neighbor_subtree
        elif method == 'cut_path':
            pred_method = self.__predict_cut_path
        elif method == 'complete':
            pred_method = self.__predict_complete_tree
        else:
            raise TypeError('Unknown prediction method %s' % method)
        
        Y = np.empty(n, dtype='object')
        for i in xrange(n):
            yhat = pred_method(X[i], Z[i], k)
            Y[i] = yhat
            
        return Y

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
            
        cluster_element.attrib['descr'] = str(cluster.descriptor)
        
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
    
    def _get_basis_function(self):
        '''
        '''
        return self._basis_function
    
    basis_function = property(fget=_get_basis_function)
    
    def _get_likelihood(self):
        '''
        '''
        self._init_check()
        cluster = self._task_tree.cargo
        return cluster.marg
    
    likelihood = property(fget=_get_likelihood)
    
    def _preprocess_data(self, X):
        if self._basis_function != None:
            n = len(X)
            Xret = np.empty(n, dtype='object')
            for i in xrange(n):
                Xret[i] = self._basis_function(X[i])
        else:
            Xret = X
        return Xret
    
    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')
    
    def __predict_complete_tree(self, X, z, k):
        '''
        @todo: - should the weights normalized
        '''
        tree = self._task_tree
        n = X.shape[0]
        yhat = np.zeros(n)
        
        total_weight = 0
        for cluster in tree:
            total_weight += cluster.weight
        #print 'tot_weight={0}'.format(total_weight)
        #iterating of the elements
        for cluster in tree:
            model = cluster.model
            weight = cluster.weight
            yhat += weight*model.predict(X)/total_weight
        
        return yhat
    
    def __predict_neighbor_path(self, X, z, k):
        '''
        @todo: - check whether the treenodes putting into the cluster_set are correctly 
                 inserted, that means no collosion occurs. Is the id() function sufficient
                 to compute the hash value of the object? What is better, using treenode or
                 cluster nodes?
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.descriptor for node in leaves])
        knn_tasks, dist = neighbor_struct.query_knn(z, k)
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #print 'meta task={0}'.format(z)
        for i in knn_tasks:
            print 'matched task={0}, dist={1}'.format(leaves[i].cargo.descriptor, dist)
        
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
            print 'weight={0}, norm_weight={1}, post={2}, model={3}, int={4}, shape={5}'.format(cluster.weight, cluster.weight/total_weight, cluster.post, cluster.model.weights, cluster.model.intercept, cluster.model.X.shape)
        
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over treenode objects
        
        #print self._task_tree
        
        if self._use_meta_features == True:
            Xz = compound_data([X], [z])
            Xz = Xz[0]
        for cluster in cluster_set:
            model = cluster.cargo.model
            if self._use_meta_features == True and not cluster.isatom():
                yhat += cluster.cargo.weight*model.predict(Xz)/total_weight
            else:
                yhat += cluster.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    def __predict_cut_neighbor_path(self, X, z, k, r=0.5):
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.descriptor for node in leaves])
        knn_tasks, dist = neighbor_struct.query_knn(z, k)
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #print 'meta task={0}'.format(z)
        #for i in knn_tasks:
        #    print 'matched task={0}, dist={1}'.format(leaves[i].cargo.descriptor, dist)
        
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
        
        if self._use_meta_features == True:
            Xz = compound_data([X], [z])
            Xz = Xz[0]
        for cluster in cluster_set:
            if cluster.cargo.post >= np.log(r):
                model = cluster.cargo.model
                if self._use_meta_features == True and not cluster.isatom():
                    yhat += cluster.cargo.weight*model.predict(Xz)/total_weight
                else:
                    yhat += cluster.cargo.weight*model.predict(X)/total_weight
        
        return yhat

    
    def __predict_max_path(self, X, z, k):
        '''
        '''
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.descriptor for node in leaves])
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
    
    def __predict_cut_path(self, X, z, k):
        '''
        The path is cutted below the common ancestor of the k-matched leaves.
        '''
        #print "Z={0}".format(z)
        tree = self._task_tree
        leaves = np.array(tree.leaves()) #changed because multiple indexing
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.descriptor for node in leaves])
        knn_tasks,_ = neighbor_struct.query_knn(z, k)
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #construct the neighborhood set of the given task z, which is defined
        #by the subtree of the common ancestor of all nearest neighbors
        neighbors = leaves[knn_tasks]
        ancestor = common_ancestor(neighbors)
        cluster_path = [node.cargo for node in path_to_root(ancestor)] 
        
        #compute the total weight for normalizing
        #iterating over cluster_node
        total_weight = 0
        for cluster in cluster_path:
            total_weight += cluster.weight
            
        #for cluster in cluster_path:
        #    print 'weight={0}, norm_weight={1}, post={2}, model={3}, int={4}, shape={5}'.format(cluster.weight, cluster.weight/total_weight, cluster.post, cluster.model.weights, cluster.model.intercept, cluster.model.X.shape)
        
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks
        #iterating over cluster_node 
        for cluster in cluster_path:
            model = cluster.model
            yhat += cluster.weight*model.predict(X)/total_weight
        
        return yhat
    
    def __predict_internal_path(self, X, z, k):
        tree = self._task_tree
        #nodes = [node for node in tree]
        nodes = np.asarray(list(tree.subtree()))
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.descriptor for node in nodes])
        knn_tasks,_ = neighbor_struct.query_knn(z, k)
        
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #construct an unique neighborhood set of tasks taking into account for 
        #prediction. this set is defined by all nodes in the paths from the 
        #nearest neighbors to the root node
        neighbors = nodes[knn_tasks]
        cluster_set = set()
        for node in neighbors:
            cluster_set.update(path_to_root(node))


        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        for cluster in cluster_set:
            total_weight += cluster.cargo.weight
        
        #iterating over treenode objects 
        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks 
        for cluster in cluster_set:
            model = cluster.cargo.model
            yhat += cluster.cargo.weight*model.predict(X)/total_weight
        
        return yhat
    
    def __predict_neighbor_subtree(self, X, Z, k):
        '''
        '''
        tree = self._task_tree
        leaves = np.asarray(tree.leaves())
        
        neighbor_struct = NaiveMatrixSearch([node.cargo.descriptor for node in leaves])
        knn_tasks,_ = neighbor_struct.query_knn(Z, k)
        
        n = X.shape[0]
        yhat = np.zeros(n)
        
        #construct the neighborhood set of the given task z, which is defined
        #by the subtree of the common ancestor of all nearest neighbors
        neighbors = leaves[knn_tasks]
        subtree = common_ancestor(neighbors)

        #compute the total weight for normalizing
        #iterating over treenode objects
        total_weight = 0
        for cluster in subtree:
            total_weight += cluster.weight

        #estimate the functional value by iterating of the prediction values
        #of the neighborhood tasks 
        for cluster in subtree:
            model = cluster.model
            yhat += cluster.weight*model.predict(X)/total_weight
                
        return yhat
    
    def __make_regression_model(self, X, y):
        model = FastBayesRegression(self._alpha, self._beta, weight_bias=True)
        model.fit(X,y)
        return model
    
    def __compute_cluster_weights(self, tree):
        '''
        @todo: - check for numerical instablities
        '''
        for node in tree.subtree():
            cluster = node.cargo
            cluster.weight = np.exp(cluster.post) 
            cluster.inv_weight = 1.0-cluster.weight
            #if cluster.inv_weight == 0.0:
            #    cluster.inv_weight += 1e-16
            if not node.isroot():
                parent = node.parent
                super_cluster = parent.cargo
                inv_weight = super_cluster.inv_weight
                cluster.weight *= inv_weight
                cluster.inv_weight *= inv_weight


class ItGradBHCRegression(BaseEstimator, RegressorMixin):
    '''
    '''
    __slots__ = ('_bhc_model',      #optimized bhc regression model 
                 '_gradient_tree',  #
                 '_rho0',
                 '_rho'         
                 '_alpha',
                 '_alpha0',
                 '_beta',
                 '_beta0',
                 '_max_it',
                 '_likel_tol',
                 '_basis_function',
                 '_use_meta_features',
                 '_is_init'
                 )
    
    def __init__(self, rho0, alpha0=2, beta0=1, basis_function=None, use_meta_features=False,
                 likel_tol=10e-4, max_it=100):
        self._rho0 = rho0
        self._alpha0 = alpha0
        self._beta0 = beta0
        
        self._likel_tol = likel_tol
        self._max_it = max_it
        
        self._basis_function = basis_function
        self._use_meta_features = use_meta_features
        self._is_init = False
    
    def fit(self, X, Y, Z):
        '''
        '''
        X = asmtarray_object(X)
        Y = asmtarray_object(Y)
        Z = asmtarray_object(Z)
        
        if len(X) != len(Y) != len(Z):
            raise ValueError('number of tasks must be same in X,Y,Z.')
        
        X = self._preprocess_data(X)
        
        #Estimate hyperparameters alpha and beta by fitting the linear model
        #to the whole data
        covars = flatten_data(X)
        targets = flatten_data(Y)
        reg_model = EBChenRegression()
        reg_model.fit(covars, targets)
        
        alpha = reg_model.alpha
        beta = reg_model.beta
        print 'alpha={0}, beta={1}'.format(alpha, beta)
        
        
        #optimize the hyperparameter rho by an iterative procedure where tn the
        #first step the best tree structure is find for a fixed rho and in the 
        #second step the hyperparameter is optimized by a gradient descent mehtod
        rho = np.atleast_1d(np.asarray(self._rho0))
        i = 0
        bhc_model = None
        likel = np.inf*-1
        while i < self._max_it:
            #@todo: - use tree likelihood as criterion
            print 'step: {0}, likel={1}, rho={2}'.format(i, likel, rho) 
            #estimate the new tree of the optimized rho value
            model_new = self._make_bhc_model(X, Y, Z, rho, alpha, beta)
            #print model_new._task_tree
            #print model_new._task_tree
            #optimize the hyperparameter phi via conjugate gradient descent.
            #the tree structure is fixed
            #@todo: replace the implicit approxomative gradient by an explicit one
            likel_fun = lambda x : BHCLikelihood(model_new._task_tree)(x)*-1
            #opt_result = fmin_cg(likel_fun, rho, full_output=True, disp=False)
            #rho_new = opt_result[0]
            #likel_new = opt_result[1]
            rho_new, likel_new, _ = fmin_l_bfgs_b(likel_fun, rho, approx_grad=True, bounds=[(0,None)])
            #print likel_new
            print likel_new
            likel_new *= -1
            if likel_new-likel < self._likel_tol:
                break
            
            likel = likel_new
            rho = rho_new
            bhc_model = model_new
            
            i += 1
        
        self._update_tree(rho, bhc_model._task_tree)
        
        self._bhc_model = bhc_model
        
        #print bhc_model._task_tree
        #print rho
        #print i
        
        self._alpha = alpha
        self._beta = beta
        self._rho = rho
        
        self._is_init = True
    
    def predict(self, X, Z, method='maxpath', k=1):
        '''
        '''
        self._init_check()
        X = asmtarray_object(X)
        Z = asmtarray_object(Z)
        X = self._preprocess_data(X)
        Y = self._bhc_model.predict(X, Z, method, k)
        return Y
    
    def predict_by_task(self, X, tasks, method='path'):
        '''
        '''
        self._init_check()
        X = asmtarray_object(X)
        X = self._preprocess_data(X)
        tasks = asmtarray_object(tasks)
        Y = self._bhc_model.predict_by_task(X, tasks, method)
        return Y
    
    
    def _make_bhc_model(self, X, Y, Z, rho, alpha, beta):
        '''
        '''
        model = BHCRegression(rho, alpha, beta, use_meta_features=self._use_meta_features)
        model.fit(X, Y, Z)
        return model
    
    def _update_tree(self, rho, treenode):
        self.__update_params(rho, treenode)
        self.__update_weights(treenode)
    
    def __update_params(self, rho, treenode):
        '''
        '''
        if treenode.isatom():
            cluster = treenode.cargo
            cluster.norm_const = math.log(rho)+sps.gammaln(cluster.ntasks)
        else:
            self.__update_params(rho, treenode.left)
            self.__update_params(rho, treenode.right)
            
            cluster = treenode.cargo
            left = treenode.left.cargo
            right = treenode.right.cargo
            
            cluster = treenode.cargo
            
            model = cluster.model
            n = cluster.ntasks
        
            #compute the cluster prior 'pi'. Because numerical issues, the norm_const
            #is compute by the sum of individual logs.
            nom = math.log(rho)+sps.gammaln(n) 
            cluster.norm_const = sumln([nom, left.norm_const+right.norm_const]) 
            cluster.pi = nom - cluster.norm_const
        
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = left.norm_const + right.norm_const - cluster.norm_const
            a = cluster.pi + model.log_evidence
            b = npi+left.marg+right.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + model.log_evidence - cluster.marg
            
    def __update_weights(self, tree):
        '''
        @todo: - check for numerical instablities
        '''
        for node in tree.subtree():
            cluster = node.cargo
            cluster.weight = np.exp(cluster.post) 
            cluster.inv_weight = 1.0-cluster.weight
            #if cluster.inv_weight == 0.0:
            #    cluster.inv_weight += 1e-16
            if not node.isroot():
                parent = node.parent
                super_cluster = parent.cargo
                inv_weight = super_cluster.inv_weight
                cluster.weight *= inv_weight
                cluster.inv_weight *= inv_weight

    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')
        
    def _preprocess_data(self, X):
        if self._basis_function != None:
            n = len(X)
            Xret = np.empty(n, dtype='object')
            for i in xrange(n):
                Xret[i] = self._basis_function(X[i])
        else:
            Xret = X
        return Xret
            
class _ClusterNode:
    
    __slots__ = ('task_id',     #task identifier 
                 'covariates',  
                 'targets', 
                 'post',        
                 'marg', 
                 'weight',
                 'inv_weight' 
                 'pi', 
                 'descriptor', 
                 'norm_const', 
                 'model',
                 'ntasks',
                 'size')

    @staticmethod
    def make_leaf(id, covariates, targets, model, descriptor, rho):
        '''
        This factory method creates a leaf _ClusterNode at which the node 
        parameters instantiated from scratch.
        '''
        n = 1
        #n = len(covariates)
        norm_const = math.log(rho)+sps.gammaln(n)
        marg = model.log_evidence
        node = _ClusterNode(covariates, targets, model, descriptor, n, marg, 
                            norm_const=norm_const, id=id)
        return node
    
    @staticmethod
    def make_internal(left, right, model, rho):
        '''
        This factory method creates an internal _ClusterNode, which is the 
        parent of the given left and right _ClusterNode. The instantiated 
        parameters of the created node are the merged one of both children.
        The concrete initialization depends on the parameters. The weight
        of the new created internal node is given as parameter.
        
        '''
        covariates = np.r_[left.covariates, right.covariates]
        targets = np.r_[left.targets, right.targets]
        n = left.ntasks+right.ntasks
        #n = len(covariates)
        
        #compute the cluster prior 'pi'. Because numerical issues, the norm_const
        #is compute by the sum of individual logs.
        nom = math.log(rho)+sps.gammaln(n) 
        norm_const = sumln([nom, left.norm_const+right.norm_const]) 
        pi = nom - norm_const
        
        #compute the log probability weight of the merged hypothesis p(H|D)
        npi = left.norm_const + right.norm_const - norm_const
        marg = sumln([pi + model.log_evidence, npi+left.marg+right.marg])
        post = pi + model.log_evidence - marg
        
        #the descriptor is the mean of the both child nodes.
        descr = (left.descriptor+right.descriptor) / 2.0
        
        #print 'pi={0}, marg={1}, post={2}'.format(pi, marg, post)
        node = _ClusterNode(covariates, targets, model, descr, n, 
                            marg, post, pi, norm_const)
        return node
    
    #@change: post and pi are initialized by default by the log values
    def __init__(self, covariates, targets, model, descriptor=None, ntasks=1, 
                 marg=0.0, post=0.0, pi=0.0, norm_const=0.0, id=None):
        self.task_id = id
        self.covariates = covariates
        self.targets = targets
        self.model = model
        self.descriptor = descriptor
        self.post = post
        self.pi = pi
        self.norm_const = norm_const
        self.marg = marg
        self.ntasks = ntasks
        self.weight = np.nan
        self.inv_weight = np.nan
        
        self.size = len(covariates)
        
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
        copy_node = _ClusterNode(self.covariates, self.targets, self.model, 
                                 self.descriptor, self.ntasks, self.marg, 
                                 self.post, self.pi, self.norm_const)
        return copy_node

class _GradientNode:
    '''
    '''
    
    __slots__ = ('marg_grad', #gradient  of the marginal data likelihood  P(D|T) 
                 'hypo_grad'  #gradient of the merging hypothesis p(H)
                 'norm_grad' #gradient of the normalizing constant d
                 )
    
    def __init__(self, marg_grad, hypo_grad, norm_grad):
        self.marg_grad = marg_grad
        self.hypo_grad = hypo_grad
        self.norm_grad = norm_grad
        
    def __str__(self):
        str = 'marg={0}, hypo={1}, norm={2}'.format(self.marg_grad, self.hypo_grad, 
                                                    self.norm_grad)
        return str
    
class BHCLikelihood(object):
    '''
    @todo: - build log gradient to permit negative values of rho
    '''
    __slots__ = ('_task_tree',
                 '_likel_tree', 
                 '_grad_tree',
                 '_last_rho'
                 )
    
    def __init__(self, task_tree):
        self._task_tree = task_tree 
        self._last_rho = np.inf
    
    def __call__(self, rho):
        '''
        '''
        if rho > 0:
            if rho != self._last_rho:
                self._likel_tree = self._estimate_likelihood(rho, self._task_tree)
                self._grad_tree = self._make_gradient_tree(rho, self._task_tree)
                self._last_rho = rho 
            
            likel = self._likel_tree.cargo.marg
        else:
            print 'fucker'
            likel = np.inf*-1
       
        
        return likel#*-1
    
    def gradient(self, rho):
        '''
        '''
        if rho <= 0:
            rho = 1e-20

        if rho != self._last_rho:
            self._likel_tree = self._estimate_likelihood(rho, self._task_tree)
            self._grad_tree = self._make_gradient_tree(rho, self._task_tree)
            self._last_rho = rho
        
        grad = self._grad_tree.cargo.marg_grad
        #print self._grad_tree
        return grad
    
    def _estimate_likelihood(self, rho, treenode):
        '''
        '''
        if treenode.isatom():
            cluster = treenode.cargo.copy()
            cluster.norm_const = math.log(rho)+sps.gammaln(cluster.ntasks)
            
            likel_tree = MutableBTree(cluster)
        else:
            left = self._estimate_likelihood(rho, treenode.left)
            right = self._estimate_likelihood(rho, treenode.right)
            
            cluster = treenode.cargo.copy()
            cluster_l = left.cargo
            cluster_r = right.cargo
            
            model = cluster.model
            n = cluster.ntasks
        
            #compute the cluster prior 'pi'. Because numerical issues, the norm_const
            #is compute by the sum of individual logs.
            nom = math.log(rho)+sps.gammaln(n) 
            cluster.norm_const = sumln([nom, cluster_l.norm_const+cluster_r.norm_const]) 
            cluster.pi = nom - cluster.norm_const
        
            #compute the log probability weight of the merged hypothesis p(H|D)
            npi = cluster_l.norm_const + cluster_r.norm_const - cluster.norm_const
            a = cluster.pi + model.log_evidence
            b = npi+cluster_l.marg+cluster_r.marg
            cluster.marg = sumln([a, b])
            cluster.post = cluster.pi + model.log_evidence - cluster.marg
            
            likel_tree = MutableBTree(cluster, None, left, right)
            
        return likel_tree
        
    def _make_gradient_tree(self, rho, treenode):
        '''
        - the gradient cannot be computed simply with log values, because the gradient
          of the normalization term could be negative. alternatively, each gradient node
          can hold a sign variable, to note whether the gradient is negative or positive.
        '''
        if treenode.isatom():
            gradient = _GradientNode(0.0, 0.0, 0.0) #check the initial gradients
            grad_tree = MutableBTree(gradient)
        else:
            grad_node_l = self._make_gradient_tree(rho, treenode.left)
            grad_node_r =  self._make_gradient_tree(rho, treenode.right)
            
            gradient_l = grad_node_l.cargo
            gradient_r = grad_node_r.cargo
            
            cluster = treenode.cargo
            cluster_l = treenode.left.cargo
            cluster_r = treenode.right.cargo
            
            model = cluster.model
            
            n = cluster.ntasks
            norm_grad = sumln([sps.gammaln(n), gradient_l.norm_grad+cluster_r.norm_const,
                               gradient_r.norm_grad+cluster_l.norm_const])
            #hypo_grad = diffln([cluster.pi-np.log(rho), cluster.pi+norm_grad-cluster.norm_const])
            a = cluster.pi-np.log(rho)
            b = cluster.pi+norm_grad-cluster.norm_const
            hypo_grad = np.exp(a) - np.exp(b) #if numerical problems occure use sign method
            
            #todo: - check, if npi = log(1-pi) is computed correctly
            npi = np.exp(cluster_l.norm_const + cluster_r.norm_const - cluster.norm_const)  
            #marg_grad = sumln([hypo_grad+model.log_evidence, 
            #                   npi+gradient_l.marg_grad+cluster_r.marg,
            #                   npi+gradient_r.marg_grad+cluster_l.marg])
            #marg_grad = diffln([marg_grad, hypo_grad+cluster_l.marg+cluster_r.marg])
            
            marg_grad = hypo_grad*np.exp(model.log_evidence)
            marg_grad -= hypo_grad*np.exp(cluster_l.marg)*np.exp(cluster_r.marg)
            marg_grad += npi*gradient_l.marg_grad*np.exp(cluster_r.marg)
            marg_grad += npi*gradient_r.marg_grad*np.exp(cluster_l.marg)
            
            gradient = _GradientNode(marg_grad, hypo_grad, norm_grad)
            grad_tree = MutableBTree(gradient, None, grad_node_l, grad_node_r)
            
        return grad_tree
    
def plot_bhc_likel_old(X, Y, Z, alpha, beta, basis_fun=None, 
                   rho_min=0, rho_max=2147483647, num=200, log_space=False):
    '''
    @todo: - preprocess the data in X with basis_functions can speedup computational time
    '''
    #generating a sequence of rho values
    if log_space == False:
        rhos = np.linspace(rho_min, rho_max, num)
    else:
        #start = np.log(rho_min)/np.log(10)
        #stop = np.log(rho_max)/np.log(10)
        rhos = np.logspace(rho_min, rho_max, num) 
        
    likels = np.empty(num)
    i = 0
    while i < num:
        model = BHCRegression(rhos[i], alpha, beta, basis_fun)
        model.fit(X, Y, Z)
        likels[i] = model.likelihood
        #print likels[i]
        i += 1
        
    plt.plot(rhos, likels, 'b')
    plt.show()
    #return p

def _post(pi, model_likel, marg):
    res = pi*model_likel/marg
    return res

def _marg(pi, model_likel, margl, margr):
    res = pi*model_likel + (1-pi)*margl*margr
    return res

def _norm(alpha, n, dl, dr):
    d = alpha*sps.gamma(n) + dl*dr
    return d

def _pi(alpha, n, dl, dr):
    d = _norm(alpha, n, dl, dr)
    res = alpha*sps.gamma(n)/d
    return res

def _postln(pi, model_likel, marg):
    '''
    all parameters in log-scale
    '''

    res = pi+model_likel-marg
    return res

def _margln(pi, model_likel, margl, margr):
    '''
    all parameters in log-scale
    todo: - problems with log-sum (1-pi)??????
    '''
    a = pi+model_likel
    b = diffln([0, pi])+margl+margr#check
    res = sumln([a,b])
    return res

def _normln(alpha, n, dl, dr):
    '''
    dl and dr in log scale
    '''
    d = sumln([np.log(alpha)+sps.gammaln(n), dl+dr])
    return d

def _piln(alpha, n, dl, dr):
    '''
    dl and dr in log scale
    '''
    d = _normln(alpha, n, dl, dr)
    res = np.log(alpha)+sps.gammaln(n)-d
    return res
    

    
if __name__ == '__main__':
    
    #filename = '/home/marcel/datasets/multilevel/ilea/exam_london.csv'
    #task_key = 'school'
    #task_fields = ['school_mixed', 'school_boys', 'school_girls', 
    #               'intake_score', 'school_vrb']
    #target_field = 'exam_score'
    
    filename = '/home/marcel/datasets/multilevel/ilea/school_norm_vrb.csv'
    task_key = 'school'
    task_fields = ['fsm', 'school_vr1', 'school_mixed', 'school_male', 'school_female', 
                   'sdenom_maintain', 'sdenom_coe', 'sdenom_rc']
    target_field = 'exam_score'
    
    X,Y,Z = load_data(filename, task_key, task_fields, target_field)
    test_x = X[64]
    test_y = Y[64]
    test_z = Z[64] 
    
    X = X[0:20]
    Y = Y[0:20]
    Z = Z[0:20]
    covars = flatten_data(X)
    targets = flatten_data(Y)
    
    reg_model = EMBayesRegression()
    reg_model.fit(covars, targets)
    alpha = reg_model.alpha
    beta = reg_model.beta

    bhc_reg = BHCRegression(22.95, alpha, beta)
    bhc_reg.fit(X, Y, Z)
    print bhc_reg._task_tree
    
    likel_fun = BHCLikelihood(bhc_reg._task_tree)
    #likel_fun = BHCLikelihood(bhc_reg._task_tree)
    print 'likelihoods'
    print likel_fun(30)
    print likel_fun.gradient(0.1)
    print likel_fun(0.4)
    print likel_fun.gradient(0.4)
    print likel_fun(0.5)
    print likel_fun.gradient(0.5)
    print likel_fun(5)
    print likel_fun.gradient(5)
    print likel_fun(10)
    print likel_fun.gradient(10)
    print likel_fun(20)
    print likel_fun.gradient(20)
    print likel_fun(50)
    print likel_fun.gradient(50)
    print likel_fun(100)
    print likel_fun.gradient(100)
    
    print likel_fun(1e100)
    #print check_gradient(likel_fun, likel_fun.gradient, 100)
    #print fmin_cg(likel_fun, 10, retall=True, full_output=True)
    #print likel_fun._task_tree
    likel_fun = lambda x : BHCLikelihood(bhc_reg._task_tree)(x)*-1
    
    #xopt, yopt,_,_,_ = fmin_cg(likel_fun, 10, full_output=True, disp=True)
    #print xopt
    print 'fmin_cg={0}'.format(fmin_cg(likel_fun, 0.1))
    #print 'fmin_bfgs={0}'.format(fmin_bfgs(likel_fun, 0.1))
    #print bisect(likel_fun, 0.1e-14, 1e10)
    print fmin_l_bfgs_b(likel_fun, [0.1], approx_grad=True, bounds=[(0,None)])
    print 'fminbound={0}'.format(fminbound(likel_fun, 1e-10, 1e+10))
    
    bhc_reg = BHCRegression(21.25, alpha, beta)
    bhc_reg.fit(X, Y, Z)
    yhat = bhc_reg.predict([test_x], [test_z])
    print mspe(test_y, yhat[0])
    
    bhc_reg.marshal_xml('bhc_model.xml')
    
    bhc_reg = ItGradBHCRegression(10)
    bhc_reg.fit(X, Y, Z)
    print bhc_reg._rho    
    print bhc_reg._alpha
    print bhc_reg._beta
    print bhc_reg._bhc_model._task_tree
    
    
    print distance_matrix(Z)
    
    