import pickle
import time
import os
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

class KNN():
    ''''''
    def __init__(self, num_neighbors, metric = 'euclidean', n_jobs = -1):
        ''''''
        self.num_neighbors = num_neighbors
        self.metric = metric
        self.results = {'time': None, 'pred_acc': None, 'pred_time': None}
        self.model = KNeighborsClassifier(n_neighbors = num_neighbors, metric = metric, n_jobs = n_jobs)
        
    def fit(self, X, Y):
        ''''''
        print('Training KNN model for Neighbors = {} using the {} metric ...'.format(self.num_neighbors, self.metric))
        start = time.time()
        self.model.fit(X, Y)
        self.results['time'] = round(time.time() - start, 2)
        print('Training took {} seconds.'.format(self.results['time']))
         
    def save(self, path = './model/knn.pkl'):
        '''''' 
        directory, _ = os.path.split(path)
        if not os.path.exists(directory): os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(cls, path):
        ''''''
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def predict(self, x):
        ''''''
        print('Predicting class for {} samples with {} neighbors ...'.format(x.shape[0], self.num_neighbors))
        start = time.time()
        self.results['pred_acc'] = self.model.predict(x)
        self.results['pred_time'] = round(time.time() - start, 2)
        return self.results['pred_acc'], self.results['pred_time']
        
        
class SVM():
    ''''''
    def __init__(self, verbose = 0):
        ''''''
        self.results = {'time': None, 'pred_acc': None, 'pred_time': None}
        self.model = LinearSVC(dual = False, verbose = verbose)
        
    def fit(self, X, Y):
        ''''''
        print('Training SVM model ...')
        start = time.time()
        self.model.fit(X, Y)
        self.results['time'] = round(time.time() - start, 2)
        print('Training took {} seconds.'.format(self.results['time']))
         
    def save(self, path = './model/svm.pkl'):
        '''''' 
        directory, _ = os.path.split(path)
        if not os.path.exists(directory): os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(cls, path):
        ''''''
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def predict(self, x):
        ''''''
        print('Predicting class for {} samples ...'.format(x.shape[0]))
        start = time.time()
        self.results['pred_acc'] = self.model.predict(x)
        self.results['pred_time'] = round(time.time() - start, 2)
        return self.results['pred_acc'], self.results['pred_time']
        
        
class Tree():
    ''''''
    def __init__(self):
        ''''''
        self.results = {'time': None, 'pred_acc': None, 'pred_time': None}
        self.model = DecisionTreeClassifier(random_state = 42)
        
    def fit(self, X, Y):
        ''''''
        print('Training Tree model ...')
        start = time.time()
        self.model.fit(X, Y)
        self.results['time'] = round(time.time() - start, 2)
        print('Training took {} seconds.'.format(self.results['time']))
         
    def save(self, path = './model/tree.pkl'):
        '''''' 
        directory, _ = os.path.split(path)
        if not os.path.exists(directory): os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(cls, path):
        ''''''
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def predict(self, x):
        ''''''
        print('Predicting class for {} samples ...'.format(x.shape[0]))
        start = time.time()
        self.results['pred_acc'] = self.model.predict(x)
        self.results['pred_time'] = round(time.time() - start, 2)
        return self.results['pred_acc'], self.results['pred_time']
        
        
class Forest():
    ''''''
    def __init__(self, n_trees = 25, n_jobs = -1, verbose = 0):
        ''''''
        self.n_trees = n_trees
        self.results = {'time': None, 'pred_acc': None, 'pred_time': None}
        self.model = RandomForestClassifier(n_estimators = n_trees, max_features = None, random_state = 42, verbose = verbose, n_jobs = n_jobs)
        
    def fit(self, X, Y):
        ''''''
        print('Training Forest model with {} trees ...'.format(self.n_trees))
        start = time.time()
        self.model.fit(X, Y)
        self.results['time'] = round(time.time() - start, 2)
        print('Training took {} seconds.'.format(self.results['time']))
         
    def save(self, path = './model/forest.pkl'):
        '''''' 
        directory, _ = os.path.split(path)
        if not os.path.exists(directory): os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(cls, path):
        ''''''
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def predict(self, x):
        ''''''
        print('Predicting class for {} samples ...'.format(x.shape[0]))
        start = time.time()
        self.results['pred_acc'] = self.model.predict(x)
        self.results['pred_time'] = round(time.time() - start, 2)
        return self.results['pred_acc'], self.results['pred_time']
        
        
class LogReg():
    ''''''
    def __init__(self, n_jobs = -1, verbose = 0):
        ''''''
        self.results = {'time': None, 'pred_acc': None, 'pred_time': None}
        self.model = LogisticRegression(dual = False, random_state = 42, verbose = verbose, n_jobs = n_jobs, solver = 'sag')
        
    def fit(self, X, Y):
        ''''''
        print('Training Logistic Regression model ...')
        start = time.time()
        self.model.fit(X, Y)
        self.results['time'] = round(time.time() - start, 2)
        print('Training took {} seconds.'.format(self.results['time']))
         
    def save(self, path = './model/logreg.pkl'):
        '''''' 
        directory, _ = os.path.split(path)
        if not os.path.exists(directory): os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(cls, path):
        ''''''
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def predict(self, x):
        ''''''
        print('Predicting class for {} samples ...'.format(x.shape[0]))
        start = time.time()
        self.results['pred_acc'] = self.model.predict(x)
        self.results['pred_time'] = round(time.time() - start, 2)
        return self.results['pred_acc'], self.results['pred_time']
        
        
class GNB():
    ''''''
    def __init__(self):
        ''''''
        self.results = {'time': None, 'pred_acc': None, 'pred_time': None}
        self.model = GaussianNB()
        
    def fit(self, X, Y):
        ''''''
        print('Training Gaussian Naive Bayes model ...')
        start = time.time()
        self.model.fit(X, Y)
        self.results['time'] = round(time.time() - start, 2)
        print('Training took {} seconds.'.format(self.results['time']))
         
    def save(self, path = './model/gnb.pkl'):
        '''''' 
        directory, _ = os.path.split(path)
        if not os.path.exists(directory): os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(cls, path):
        ''''''
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def predict(self, x):
        ''''''
        print('Predicting class for {} samples ...'.format(x.shape[0]))
        start = time.time()
        self.results['pred_acc'] = self.model.predict(x)
        self.results['pred_time'] = round(time.time() - start, 2)
        return self.results['pred_acc'], self.results['pred_time']
        
        
class AdaBoost():
    ''''''
    def __init__(self, n_estimators = 50):
        ''''''
        self.n_estimators = n_estimators
        self.results = {'time': None, 'pred_acc': None, 'pred_time': None}
        self.model = AdaBoostClassifier(n_estimators = n_estimators, random_state = 42)
        
    def fit(self, X, Y):
        ''''''
        print('Training AdaBoost Classifier model with {} estimators...'.format(self.n_estimators))
        start = time.time()
        self.model.fit(X, Y)
        self.results['time'] = round(time.time() - start, 2)
        print('Training took {} seconds.'.format(self.results['time']))
         
    def save(self, path = './model/ada.pkl'):
        '''''' 
        directory, _ = os.path.split(path)
        if not os.path.exists(directory): os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(cls, path):
        ''''''
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def predict(self, x):
        ''''''
        print('Predicting class for {} samples ...'.format(x.shape[0]))
        start = time.time()
        self.results['pred_acc'] = self.model.predict(x)
        self.results['pred_time'] = round(time.time() - start, 2)
        return self.results['pred_acc'], self.results['pred_time']