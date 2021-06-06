import os
import numpy as np

#TensorFlow 2.0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Add, Multiply, Conv1D, AveragePooling1D, Concatenate
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
#from tensorflow.keras import backend as K
#from tensorflow.keras import metrics
#from tensorflow.keras import optimizers

class WIFINet():
    ''''''
    def __init__(self, input_shape, output_shape, dilation_depth, num_filters, verbose = 1, load = False, directory = './model/'):
        ''''''
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dilation_depth = dilation_depth
        self.num_filters = num_filters
        self.verbose = verbose

        if load:
            self.model = load_model(directory + 'wifinet.h5')
            self.current_epoch = np.genfromtxt(directory + 'wifinet_history.csv', delimiter = ',', skip_header = 1).shape[0]
        else:
            self.model = self.build_model()
            self.current_epoch = 0

    def residual_block(self, x, i):
        ''''''
        sigm = Conv1D(self.num_filters,
                      2,
                      name = 'gate_sigm_{}'.format(2 ** i),
                      padding = 'causal',
                      activation = 'sigmoid',
                      dilation_rate = 2 ** i
                      )(x)
        tanh = Conv1D(self.num_filters,
                      2,
                      name = 'filter_tanh_{}'.format(2 ** i),
                      padding = 'causal',
                      activation = 'tanh',
                      dilation_rate = 2 ** i
                      )(x)
        mult = Multiply(name = 'gate_filter_multiply_{}'.format(i))([tanh, sigm])
        skip = Conv1D(self.num_filters, 1, name = 'skip_{}'.format(i))(mult)
        res = Add(name = 'residual_block_{}'.format(i))([skip, x])
        return res, skip
        
        
    def build_model(self):
        ''''''
        #input layers
        x = Input(shape = self.input_shape, name = 'input')
        
        #residual layers
        out = Conv1D(self.num_filters, 2, name = 'conv_1', dilation_rate = 1, padding = 'causal')(x)
        skip_connections = []
        for i in range(1, self.dilation_depth + 1):
            out, skip = self.residual_block(out, i)
            skip_connections.append(skip)
        out = Add(name = 'skip_connections')(skip_connections)
        
        #output layers
        out = Activation('relu')(out)
        out = Conv1D(self.num_filters, 50, name = 'conv_2', dilation_rate = 1, padding = 'same', activation = 'relu')(out)
        out = AveragePooling1D(50, name = 'avg_pool', padding = 'same')(out)
        out = Flatten(name = 'flatten')(out)
        out = Dense(4000, name = 'dense_1', activation = 'relu', kernel_initializer = 'he_uniform')(out)
        out = Dense(40, name = 'dense_2', activation = 'relu', kernel_initializer = 'he_uniform')(out)
        out = Dense(self.output_shape[0], name = 'predicted_signal', activation = 'softmax')(out)
        
        #model
        model = Model(x, out)
        model.summary()
        
        return model


    def fit(self, X, Y, validation_data = None, epochs = 10, batch_size = 32, optimizer = 'adam', save = False, directory = './model/'):
        ''''''
        if save: # set callback functions if saving model
            if not os.path.exists(directory): os.makedirs(directory)
            mpath = directory + "wifinet.h5"
            hpath = directory + 'wifinet_history.csv'
            if validation_data is None:
                checkpoint = ModelCheckpoint(filepath = mpath, monitor = 'loss', verbose = 0, save_best_only = True)
            else:
                checkpoint = ModelCheckpoint(filepath = mpath, monitor = 'val_loss', verbose = 0, save_best_only = True)
            cvs_logger = CSVLogger(hpath, separator = ',', append = True)
            callbacks = [cvs_logger, checkpoint]
        else:
            callbacks = None
        
        self.model.compile(optimizer, 'categorical_crossentropy', ['accuracy']) #compile model
        self.model.fit( X, Y, validation_data = validation_data,
                        epochs = epochs, batch_size = batch_size, shuffle = True,
                        initial_epoch = self.current_epoch,
                        callbacks = callbacks,
                        verbose = self.verbose ) #fit model
                        
        self.current_epoch += epochs
        return

    def predict(self, x):
        ''''''
        return self.model.predict(x)