import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tensorflow messages
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1' #prevent error 200 in console

import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from wifinet import WIFINet

def main():
    ''''''
    
    #load data
    path = ['data/data.npy', 'data/labels.npy']
    print('Loading data from {} ... '.format(path[0]))
    data = np.load(path[0])[:, 0, :].astype('float') #load data and keep only I component
    print('Loading labels from {} ... '.format(path[1]))
    info = np.load(path[1]) #load labels
    
    #normalize data and expand dims
    threshold, abs_max = 59.0, np.max(np.abs(data))
    print('Normalizing data... threshold offset: {}, absolute max value: {}'.format(threshold, abs_max))
    #x_min, x_max = np.min(data), np.max(data)
    #data = ((data + threshold) - x_min) / (x_max - x_min) #offset by threshold and an scale to [0, 1]
    #data = (2 * ((data + threshold) - x_min) / (x_max - x_min)) - 1 #offset by threshold and scale to [-1, 1]
    data = (data + threshold) / abs_max #offset by threshold and scale to [-1, 1] while preserving zero location
    data = np.expand_dims(data, axis = 2)
    
    #split dataset into train and test
    labels = info[:, 0]
    train, test, train_labels, test_labels = train_test_split(data, labels, test_size = 0.20, random_state = 42, shuffle = True, stratify = labels)
    del data, labels #free up memory
    
    #one_hot encode labels
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    
    print('Train Shape:', train.shape, 'Train Labels Shape:', train_labels.shape, 'Test Shape:', test.shape, 'Test Labels Shape:', test_labels.shape)
    
    wnet = WIFINet((15000, 1), (7, ), dilation_depth = 10, num_filters = 40, load = False) #create model
    opt = Adam(learning_rate = 1e-5, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = False) #setup optimizer
    wnet.fit(train, train_labels, validation_data = (test, test_labels), epochs = 10, batch_size = 32, optimizer = opt, save = False) #fit model

if __name__ == '__main__':
    main()