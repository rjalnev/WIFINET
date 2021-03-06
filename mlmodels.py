import pickle
import time
import os
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

class KNN():
    '''K-Nearest Neighbor using euclidean distance metric and max processors by default. Specify number of neighbors as input.'''
    def __init__(self, num_neighbors, metric = 'euclidean', n_jobs = -1):
        self.num_neighbors = num_neighbors
        self.metric = metric
        self.model = KNeighborsClassifier(n_neighbors = num_neighbors, metric = metric, n_jobs = n_jobs)
        
    def fit(self, X, Y):
        print('Training KNN model for Neighbors = {} using the {} metric ...'.format(self.num_neighbors, self.metric))
        start = time.time()
        self.model.fit(X, Y)
        ptime = round(time.time() - start, 2)
        print('Training took {} seconds.'.format(ptime))
         
    def save(self, path = './model/knn.pkl'):
        directory, _ = os.path.split(path)
        if not os.path.exists(directory): os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def predict(self, x):
        print('Predicting class for {} samples with {} neighbors ...'.format(x.shape[0], self.num_neighbors))
        start = time.time()
        pred = self.model.predict_proba(x)
        ptime = round(time.time() - start, 2)
        return pred, ptime
        
        
class SVM():
    '''Support Vector Machine with Probability Calibration'''
    def __init__(self, verbose = 0):
        self.model = CalibratedClassifierCV(LinearSVC(dual = False, verbose = verbose))
        
    def fit(self, X, Y):
        print('Training SVM model ...')
        start = time.time()
        self.model.fit(X, Y)
        ptime = round(time.time() - start, 2)
        print('Training took {} seconds.'.format(ptime))
         
    def save(self, path = './model/svm.pkl'):
        directory, _ = os.path.split(path)
        if not os.path.exists(directory): os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def predict(self, x):
        print('Predicting class for {} samples ...'.format(x.shape[0]))
        start = time.time()
        pred = self.model.predict_proba(x)
        ptime = round(time.time() - start, 2)
        return pred, ptime
        
        
class Tree():
    '''Decision Tree Classifier with random state set by default. Specify number of minimum leaves.'''
    def __init__(self, n_leaves):
        self.n_leaves = n_leaves
        self.model = DecisionTreeClassifier(random_state = 42, min_samples_leaf = n_leaves)
        
    def fit(self, X, Y):
        print('Training Tree model with a min of {} leaves ...'.format(self.n_leaves))
        start = time.time()
        self.model.fit(X, Y)
        ptime = round(time.time() - start, 2)
        print('Training took {} seconds.'.format(ptime))
         
    def save(self, path = './model/tree.pkl'):
        directory, _ = os.path.split(path)
        if not os.path.exists(directory): os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def predict(self, x):
        print('Predicting class for {} samples with {} leaves ...'.format(x.shape[0], self.n_leaves))
        start = time.time()
        pred = self.model.predict_proba(x)
        ptime = round(time.time() - start, 2)
        return pred, ptime
        
        
class Forest():
    '''Random Forest with max features set to None, random state set, and max processors by default. Specify the number of trees.'''
    def __init__(self, n_trees, n_jobs = -1, verbose = 0):
        self.n_trees = n_trees
        self.model = RandomForestClassifier(n_estimators = n_trees, max_features = None, random_state = 42, verbose = verbose, n_jobs = n_jobs)
        
    def fit(self, X, Y):
        print('Training Forest model with {} trees ...'.format(self.n_trees))
        start = time.time()
        self.model.fit(X, Y)
        ptime = round(time.time() - start, 2)
        print('Training took {} seconds.'.format(ptime))
         
    def save(self, path = './model/forest.pkl'):
        directory, _ = os.path.split(path)
        if not os.path.exists(directory): os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def predict(self, x):
        print('Predicting class for {} samples ...'.format(x.shape[0]))
        start = time.time()
        pred = self.model.predict_proba(x)
        ptime = round(time.time() - start, 2)
        return pred, ptime
        
        
class LogReg():
    '''Logistic Regression with random state set, max processors, and sag solver by default.'''
    def __init__(self, n_jobs = -1, verbose = 0):
        self.model = LogisticRegression(dual = False, random_state = 42, verbose = verbose, n_jobs = n_jobs, solver = 'sag')
        
    def fit(self, X, Y):
        print('Training Logistic Regression model ...')
        start = time.time()
        self.model.fit(X, Y)
        ptime = round(time.time() - start, 2)
        print('Training took {} seconds.'.format(ptime))
         
    def save(self, path = './model/logreg.pkl'):
        directory, _ = os.path.split(path)
        if not os.path.exists(directory): os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def predict(self, x):
        print('Predicting class for {} samples ...'.format(x.shape[0]))
        start = time.time()
        pred = self.model.predict_proba(x)
        ptime = round(time.time() - start, 2)
        return pred, ptime
        
        
class GNB():
    '''Guassian Naive Bayes with all default parameters.'''
    def __init__(self):
        self.model = GaussianNB()
        
    def fit(self, X, Y):
        print('Training Gaussian Naive Bayes model ...')
        start = time.time()
        self.model.fit(X, Y)
        ptime = round(time.time() - start, 2)
        print('Training took {} seconds.'.format(ptime))
         
    def save(self, path = './model/gnb.pkl'):
        directory, _ = os.path.split(path)
        if not os.path.exists(directory): os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def predict(self, x):
        print('Predicting class for {} samples ...'.format(x.shape[0]))
        start = time.time()
        pred = self.model.predict_proba(x)
        ptime = round(time.time() - start, 2)
        return pred, ptime
        
        
class AdaBoost():
    '''AdaBoost Classifier with random state set by default. Specify number of estimators.'''
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.model = AdaBoostClassifier(n_estimators = n_estimators, random_state = 42)
        
    def fit(self, X, Y):
        print('Training AdaBoost Classifier model with {} estimators...'.format(self.n_estimators))
        start = time.time()
        self.model.fit(X, Y)
        ptime = round(time.time() - start, 2)
        print('Training took {} seconds.'.format(ptime))
         
    def save(self, path = './model/ada.pkl'):
        directory, _ = os.path.split(path)
        if not os.path.exists(directory): os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def predict(self, x):
        print('Predicting class for {} samples ...'.format(x.shape[0]))
        start = time.time()
        pred = self.model.predict_proba(x)
        ptime = round(time.time() - start, 2)
        return pred, ptime