import os
import numpy as np
import time

#TensorFlow 2.0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Add, Multiply, Conv1D, AveragePooling1D, Dropout
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
#from tensorflow.keras import backend as K
#from tensorflow.keras import metrics
#from tensorflow.keras import optimizers

class WIFINet():
    ''''''
    def __init__(self, input_shape = None, output_shape = None, dilation_depth = None, num_filters = None, O1 = None, O2 = None, D1 = None, D2 = None):
        ''''''
        if input_shape is None:
            self.model = None
            self.current_epoch = None
        else:
            self.model = self.build_model(input_shape, output_shape, dilation_depth, num_filters, O1, O2, D1, D2)
            self.current_epoch = 0

    def residual_block(self, x, i, num_filters):
        ''''''
        sigm = Conv1D(num_filters,
                      2,
                      name = 'gate_sigm_{}'.format(2 ** i),
                      padding = 'causal',
                      activation = 'sigmoid',
                      dilation_rate = 2 ** i
                      )(x)
        tanh = Conv1D(num_filters,
                      2,
                      name = 'filter_tanh_{}'.format(2 ** i),
                      padding = 'causal',
                      activation = 'tanh',
                      dilation_rate = 2 ** i
                      )(x)
        mult = Multiply(name = 'gate_filter_multiply_{}'.format(i))([tanh, sigm])
        skip = Conv1D(num_filters, 1, name = 'skip_{}'.format(i))(mult)
        res = Add(name = 'residual_block_{}'.format(i))([skip, x])
        return res, skip
        
    def build_model(self, input_shape, output_shape, dilation_depth, num_filters, O1, O2, D1, D2):
        ''''''
        #input layers
        x = Input(shape = input_shape, name = 'input')
        
        #residual layers
        out = Conv1D(num_filters, 1, name = 'conv_1', dilation_rate = 1, padding = 'causal')(x)
        skip_connections = []
        for i in range(1, dilation_depth + 1):
            out, skip = self.residual_block(out, i, num_filters)
            skip_connections.append(skip)
        out = Add(name = 'skip_connections')(skip_connections)
        
        #output layers
        out = Activation('relu')(out)
        out = Conv1D(num_filters, O1, name = 'conv_2', dilation_rate = 1, padding = 'same', activation = 'relu')(out)
        out = AveragePooling1D(O2, name = 'pooling', padding = 'same')(out)
        out = Flatten(name = 'flatten')(out)
        out = Dense(D1, name = 'dense_1', kernel_initializer = 'he_uniform', activation = 'relu')(out)
        out = Dropout(0.5, name = 'dropout_1')(out)
        out = Dense(D2, name = 'dense_2', kernel_initializer = 'he_uniform', activation = 'relu')(out)
        out = Dropout(0.5, name = 'dropout_2')(out)
        out = Dense(output_shape[0], name = 'predicted_signal', activation = 'softmax')(out)
        
        #model
        model = Model(x, out)
        model.summary()
        return model

    def fit(self, X, Y, validation_data = None, epochs = 10, batch_size = 32, optimizer = 'adam', verbose = 1, 
            save = False, directory = './model/', model_name = 'wifinet', hist_name = 'wifinet_history'):
        ''''''
        if save: # set callback functions if saving model
            if not os.path.exists(directory): os.makedirs(directory)
            mpath = directory + model_name + '.h5'
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
    def load(cls, model_path = './model/wifinet.h5', hist_path = './model/wifinet_history.csv'):
        _cls = cls.__new__(cls)
        _cls.model = load_model(model_path)
        hist_shape = np.genfromtxt(hist_path, delimiter = ',', skip_header = 1).shape
        _cls.current_epoch = hist_shape[0] if len(hist_shape) > 1 else 1
        return _cls
        
    def predict(self, x):
        ''''''
        print('Predicting class for {} samples ...'.format(x.shape[0]))
        start = time.time()
        return self.model.predict(x), round(time.time() - start, 2)