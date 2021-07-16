import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tensorflow messages

import numpy as np
from wifinet import WIFINet

def main():
    '''Test the WIFIMet model on a sample test set included in the repo.
       Test set is 1000 random samples from validation set that has already been preprocessed and normalized.'''
    #load the trained model, requires model and .csv history
    print('Loading model ...')
    wnet = WIFINet.load(model_path = 'model\wifinet.h5', hist_path = 'model\wifinet_history.csv')
    
    #load the small test data that has already been preprocessed and normalized, 1000 random samples from full val set
    data = np.load('data/test.npz')

    #make prediction and print accuracy
    pred, time = wnet.predict(data['test'])
    print('Prediction took {} seconds to predict {} samples ...'.format(time, data['test'].shape[0]))
    acc = np.sum(np.argmax(pred, axis = 1) == data['test_labels']) / pred.shape[0] * 100
    print('Accuracy: {:.4f}%'.format(acc))

if __name__ == '__main__':
    main()