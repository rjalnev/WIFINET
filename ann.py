import os
import numpy as np
import time

#TensorFlow 2.0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Dropout
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

class ANN():
    '''Basic ANN model.
         input_shape: Shape of input, i.e. (20000, 1)
         output_shape: Shape of output, i.e. the number of class, (7, )
         depth: Number of dense layers given as list of sizes, i.e. [2048, 512, 128]'''
    def __init__(self, input_shape = None, output_shape = None, depth = None):
        '''Creates a new model and sets epoch to 0. Inits to None if using load to load a model.'''
        if input_shape is None:
            self.model = None
            self.current_epoch = None
        else:
            self.model = self.build_model(input_shape, output_shape, depth)
            self.current_epoch = 0
        
    def build_model(self, input_shape, output_shape, depth):
        '''Builds the model.'''
        #input layers
        x = Input(shape = input_shape, name = 'input')
        out = x
        
        #dense layers
        for i in range(len(depth)):
            out = Dense(depth[i], name = 'dense_{}'.format(i+1), activation='relu')(out)
            out = Dropout(0.5, name = 'dropout_{}'.format(i+1))(out)
        
        #output layers
        out = Dense(output_shape[0], name = 'predicted_signal', activation = 'softmax')(out)
        
        #model
        model = Model(x, out)
        model.summary()
        return model

    def fit(self, X, Y, validation_data = None, epochs = 10, batch_size = 32, optimizer = 'adam', verbose = 1,
            save = False, directory = './model/', model_name = 'cnn', hist_name = 'cnn_history'):
        '''Trains the model and returns history. If save = True then saves best model and training history.'''
        if save: # set callback functions if saving model
            if not os.path.exists(directory): os.makedirs(directory)
            mpath = directory + model_name + ".h5"
            hpath = directory + hist_name + '.csv'
            if validation_data is None:
                checkpoint = ModelCheckpoint(filepath = mpath, monitor = 'loss', verbose = 0, save_best_only = True)
            else:
                checkpoint = ModelCheckpoint(filepath = mpath, monitor = 'val_loss', verbose = 0, save_best_only = True)
            cvs_logger = CSVLogger(hpath, separator = ',', append = True)
            callbacks = [cvs_logger, checkpoint]
        else:
            callbacks = None
        
        self.model.compile(optimizer, 'categorical_crossentropy', ['accuracy']) #compile model
        start = time.time()
        hist = self.model.fit(X, Y, validation_data = validation_data,
                              epochs = epochs, batch_size = batch_size, shuffle = True,
                              initial_epoch = self.current_epoch,
                              callbacks = callbacks,
                              verbose = verbose ) #fit model
        print('Training took {} seconds.'.format(round(time.time() - start, 2)))
        self.current_epoch += epochs
        return hist
        
    @classmethod
    def load(cls, model_path = './model/cnn.h5', hist_path = './model/cnn_history.csv'):
        '''Loads a pretrained model. Can continue training if desired.'''
        _cls = cls.__new__(cls)
        _cls.model = load_model(model_path)
        hist_shape = np.genfromtxt(hist_path, delimiter = ',', skip_header = 1).shape
        _cls.current_epoch = hist_shape[0] if len(hist_shape) > 1 else 1
        return _cls
        
    def predict(self, x):
        '''Predicts class for x and returns prediction and time.'''
        print('Predicting class for {} samples ...'.format(x.shape[0]))
        start = time.time()
        return self.model.predict(x), round(time.time() - start, 2)