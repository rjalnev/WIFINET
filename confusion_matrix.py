import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tensorflow messages

import numpy as np
import time
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from wifinet import WIFINet


def load_and_save_test():
    #load data
    path = ['data/data.npy', 'data/labels.npy']
    print('Loading data from {} ... '.format(path[0]))
    data = np.load(path[0])[:, 0, :].astype('float') #load data and keep only I component
    print('Loading labels from {} ... '.format(path[1]))
    labels = np.load(path[1]) #load labels
    
    #normalize data and expand dims
    threshold, abs_max = 59.0, 85.0
    print('Normalizing data... threshold offset: {}, absolute max value: {}'.format(threshold, abs_max))
    data = (data + threshold) / abs_max #offset by threshold and scale to [-1, 1] while preserving zero location
    data = np.expand_dims(data, axis = 2)
    
    
    #split dataset into train and test
    _, test, _, test_labels = train_test_split(data, labels, test_size = 0.20, random_state = 42, shuffle = True, stratify = labels[:, 0])
    del data, labels #free up memory
    
    #predict and time
    wnet = WIFINet((15000, 1), (5, ), dilation_depth = 10, num_filters = 40, load = True) #create model
    
    start_time = time.time()
    predict = wnet.predict(test)
    elapsed_time = np.asarray([time.time() - start_time]) #elapsed_time
    
    #save the npz
    path = 'data/test.npz'
    keywords = {"data": test, "labels": test_labels, "predict": predict, "time": elapsed_time}
    np.savez(path, **keywords)
    print('Prediction took {} seconds. Test data, labels, and prediction saved to {} ...'.format(elapsed_time, path))
        

def main():
    ''''''
    
    #load_and_save_test()
    
    test = np.load('data/test.npz')
    print('Data Shape:', test['data'].shape, 'Labels Shape:', test['labels'].shape, 'Predict Shape:', test['predict'].shape)
    
    pred_labels = np.argmax(test['predict'], axis = 1)
    print('Accuracy: {}%\n'.format(np.mean(pred_labels == test['labels'][:, 0]) * 100))
    print('Prediction Time: {}'.format(test['time']))
    
    cf_matrix = confusion_matrix(test['labels'][:, 0], pred_labels, normalize = 'true')
    print(np.round(cf_matrix, 4))
    
    df_cm = pd.DataFrame(cf_matrix,
                         index = ['AX', 'AC', 'N', 'AX / AC', 'AX / N'],
                         columns = ['AX', 'AC', 'N', 'AX / AC', 'AX / N'])
    
    ax = sn.heatmap(df_cm, annot=True, cmap='Greens', annot_kws={"size": 22})
    #plt.xticks(rotation=45)
    #plt.yticks(rotation=45)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)
    plt.xlabel('Predicted', labelpad=20, fontsize=24)
    plt.ylabel('True', labelpad=20, fontsize=24)
    plt.show()

if __name__ == '__main__':
    main()
    
    
    
    


