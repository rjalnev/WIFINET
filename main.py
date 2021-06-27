import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tensorflow messages
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1' #prevent error 200 in console

import numpy as np
import time

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from random import seed as set_py_seed
from numpy.random import seed as set_np_seed
from tensorflow.random import set_seed as set_tf_seed

from utils import play_sound, calc_accuracy
from wifinet import WIFINet
from ann import ANN
from cnn import CNN
from mlmodels import KNN, SVM, Tree, Forest, LogReg, GNB, AdaBoost

def main():
    ''''''
    train, test, train_labels, test_labels = load_data() #load data
    #train, test, train_labels, test_labels = np.random.randint(0, 90, size=(1000, 20000, 1)), np.random.randint(0, 90, size=(100, 20000, 1)), np.random.randint(0, 7, size=(1000,)), np.random.randint(0, 7, size=(100,)) #fake data for faster testing
    
    #train 6 wifinet models using various sample lengths
    slen = [5000, 10000, 15000, 20000, 25000, 30000]
    for i in range(len(slen)):
        wnet = WIFINet((slen[i], 1), (7, ), dilation_depth = 4, num_filters = 32, O1 = 32, O2 = 10, D1 = 1024, D2 = 128) #create model
        opt = Adam(learning_rate = 1e-4, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = False) #setup optimizer
        wnet.fit(train[:, 0:slen[i], :], to_categorical(train_labels), validation_data = (test[:, 0:slen[i], :], to_categorical(test_labels)),
                 epochs = 10, batch_size = 32, optimizer = opt, save = True, model_name = 'wifinet_L{}'.format(slen[i]), hist_name = 'wifinet_history_L{}'.format(slen[i])) #fit model
    
    #train 3 additional wifinet models of various depths
    slen = 20000
    depth = [2, 8, 10]
    for i in range(len(depth)):
        wnet = WIFINet((slen, 1), (7, ), dilation_depth = depth[i], num_filters = 32, O1 = 32, O2 = 10, D1 = 1024, D2 = 128) #create model
        opt = Adam(learning_rate = 1e-4, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = False) #setup optimizer
        wnet.fit(train[:, 0:slen, :], to_categorical(train_labels), validation_data = (test[:, 0:slen, :], to_categorical(test_labels)),
                 epochs = 10, batch_size = 32, optimizer = opt, save = True, model_name = 'wifinet_D{}'.format(depth[i]), hist_name = 'wifinet_history_D{}'.format(depth[i])) #fit model

    #train 6 ANNs of various depths
    depth = [[1024, 128], [2048, 128], [4096, 128], [1024, 512, 128], [2048, 512, 128], [4096, 512, 128]]
    for i in range(len(depth)):
        ann = ANN((slen, ), (7, ), depth = depth[i]) #create model
        opt = Adam(learning_rate = 1e-4, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = False) #setup optimizer
        ann.fit(train[:, 0:slen, :], to_categorical(train_labels), validation_data = (test[:, 0:slen, :], to_categorical(test_labels)),
                epochs = 10, batch_size = 32, optimizer = opt, save = True, model_name = 'ann_D{}'.format(depth[i]), hist_name = 'ann_history_D{}'.format(depth[i])) #fit model
    
    #train 3 CNNs of various depths
    depth = [[1, 2, 1], [1,3,2], [2, 5, 3]]
    for i in range(len(depth)):
        cnn = CNN((slen, 1), (7, ), depth = depth[i], filters = [16, 32, 64], kernels = [3, 5, 3], D1 = 1024, D2 = 128) #create model
        opt = Adam(learning_rate = 1e-4, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = False) #setup optimizer
        cnn.fit(train[:, 0:slen, :], to_categorical(train_labels), validation_data = (test[:, 0:slen, :], to_categorical(test_labels)),
                epochs = 10, batch_size = 32, optimizer = opt, save = True, model_name = 'cnn_D{}'.format(depth[i]), hist_name = 'cnn_history_D{}'.format(depth[i])) #fit model
    
    #train 12 KNN models of various number of neighbors
    neighbors = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
    for i in range(len(neighbors)):
        knn = KNN(num_neighbors = neighbors[i])
        knn.fit(np.squeeze(train)[:, 0:slen], train_labels)
        knn.save(path = './model/knn_{}.pkl'.format(neighbors[i]))
    
    #run prediction for all KNN models on 1000 samples only, too long otherwise even on 24 cores for this many models
    num_samples = 1000
    for i in range(len(neighbors)):
        knn = KNN.load('./model/knn_{}.pkl'.format(neighbors[i]))
        pred, time = knn.predict(np.squeeze(test)[0:num_samples, 0:slen])
        print('The prediction took {} seconds to predict {} samples.'.format(time, num_samples))
        print('Accuracy: {}%'.format(calc_accuracy(pred, test_labels[0:num_samples])))
    
    #train SVM model and run prediction.
    svm = SVM()
    svm.fit(np.squeeze(train)[:, 0:slen], train_labels)
    svm.save()
    pred, time = svm.predict(np.squeeze(test)[:, 0:slen])
    print('The prediction took {} seconds.'.format(time))
    print('Accuracy: {}%'.format(calc_accuracy(pred, test_labels)))
    
    #train tree model and run prediction
    tree = Tree()
    tree.fit(np.squeeze(train)[:, 0:slen], train_labels)
    tree.save()
    pred, time = tree.predict(np.squeeze(test)[:, 0:slen])
    print('The prediction took {} seconds.'.format(time))
    print('Accuracy: {}%'.format(calc_accuracy(pred, test_labels)))
    
    #train 15 forest models with cvarious numbers of trees and run prediction
    trees = [1, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300]
    for i in range(len(trees)):
        forest = Forest(n_trees = trees[i])
        forest.fit(np.squeeze(train)[:, 0:slen], train_labels)
        forest.save(path = './model/forest_{}.pkl'.format(trees[i]))
        pred, time = forest.predict(np.squeeze(test)[:, 0:slen])
        print('The prediction took {} seconds.'.format(time))
        print('Accuracy: {}%'.format(calc_accuracy(pred, test_labels)))
    
    #train logistic regression model and run prediction
    logreg = LogReg()
    logreg.fit(np.squeeze(train)[:, 0:slen], train_labels)
    logreg.save()
    pred, time = logreg.predict(np.squeeze(test)[:, 0:slen])
    print('The prediction took {} seconds.'.format(time))
    print('Accuracy: {}%'.format(calc_accuracy(pred, test_labels)))
    
    #train guassian naive bayes model and run prediction
    gnb = GNB()
    gnb.fit(np.squeeze(train)[:, 0:slen], train_labels)
    gnb.save()
    pred, time = gnb.predict(np.squeeze(test)[:, 0:slen])
    print('The prediction took {} seconds.'.format(time))
    print('Accuracy: {}%'.format(calc_accuracy(pred, test_labels)))
    
    #train 13 adaboost ensemble model and run prediction
    ests = [1, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300]
    for i in range(len(ests)):
        ada = AdaBoost(n_estimators = ests[i])
        ada.fit(np.squeeze(train)[:, 0:slen], train_labels)
        ada.save(path = './model/ada_{}.pkl'.format(ests[i]))
        pred, time = ada.predict(np.squeeze(test)[:, 0:slen])
        print('The prediction took {} seconds.'.format(time))
        print('Accuracy: {}%'.format(calc_accuracy(pred, test_labels)))
    

def load_data(normalize = True, type = 1):
    ''''''
    path = ['data/data.npy', 'data/labels.npy']
    print('Loading data from {} and {} ...'.format(path[0], path[1]))
    data = np.load(path[0])[:, 0, :] #load data and keep only I component
    info = np.load(path[1]) #load labels
    
    #normalize data
    if normalize:
        if type == 1: #offset by threshold and scale to [-1, 1] while preserving zero location
            threshold, abs_max = 59.0, np.max(np.abs(data))
            print('Normalizing data using threshold offset {} and absolute max value {} ...'.format(threshold, abs_max))
            data = (data + threshold) / abs_max
        elif type == 2: #offset by threshold and an scale to [0, 1]
            threshold, x_min, x_max = 59.0, np.min(data), np.max(data)
            print('Normalizing data using threshold offset {}, min value {}, and max value {} ...'.format(threshold, x_min, x_max))
            data = ((data + threshold) - x_min) / (x_max - x_min)
        elif type == 3: #offset by threshold and scale to [-1, 1]
            threshold, abs_max = 59.0, np.max(np.abs(data))
            print('Normalizing data using threshold offset {}, min value {}, and max value {} ...'.format(threshold, x_min, x_max))
            data = (2 * ((data + threshold) - x_min) / (x_max - x_min)) - 1
    
    #expand dims
    data = np.expand_dims(data, axis = 2)
    #split dataset into train and test
    labels = info[:, 0]
    train, test, train_labels, test_labels = train_test_split(data, labels, test_size = 0.20, random_state = 42, shuffle = True, stratify = labels)
    
    print('Train Shape:', train.shape, 'Train Labels Shape:', train_labels.shape, 'Test Shape:', test.shape, 'Test Labels Shape:', test_labels.shape)
    return train, test, train_labels, test_labels


def set_seeds(seed=42):
    ''''''
    os.environ['PYTHONHASHSEED'] = str(seed) #set python hash seed
    os.environ['TF_CUDNN_DETERMINISM'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    set_py_seed(seed) #set python seed
    set_np_seed(seed) #set numpy seed
    set_tf_seed(seed) #set tensorflow seed


if __name__ == '__main__':
    #try:
    set_seeds()
    main()
    #    play_sound()
    #except Exception as err:
    #    print('exception occurred ...')
    #    print(err)
    #    play_sound()