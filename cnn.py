import os
import numpy as np

#TensorFlow 2.0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Conv1D, AveragePooling1D
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

class CNN():
    ''''''
    def __init__(self, input_shape, output_shape, depth, filters, kernels, load = False, directory = './model/'):
        ''''''
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.depth = depth
        self.filters = filters
        self.kernels = kernels

        if load:
            self.model = load_model(directory + 'cnn.h5')
            self.current_epoch = np.genfromtxt(directory + 'cnn_history.csv', delimiter = ',', skip_header = 1).shape[0]
        else:
            self.model = self.build_model()
            self.current_epoch = 0
        
        
    def build_model(self):
        ''''''
        #input layers
        x = Input(shape = self.input_shape, name = 'input')
        out = x
        
        #conv layers
        for i in range(len(self.depth)):
            for j in range(self.depth[i]):
                out = Conv1D(self.filters[i], self.kernels[i], name = 'conv_{}.{}'.format(i+1, j+1), dilation_rate = 1, padding = 'same', activation='relu')(out)
            out = AveragePooling1D(2, name = 'avg_pool_{}'.format(i+1), padding = 'same')(out)
        
        #output layers
        out = Flatten(name = 'flatten')(out)
        out = Dense(1028, name = 'dense_1', activation = 'relu', kernel_initializer = 'he_uniform')(out)
        out = Dense(512, name = 'dense_2', activation = 'relu', kernel_initializer = 'he_uniform')(out)
        out = Dense(self.output_shape[0], name = 'predicted_signal', activation = 'softmax')(out)
        
        #model
        model = Model(x, out)
        model.summary()
        
        return model


    def fit(self, X, Y, validation_data = None, epochs = 10, batch_size = 32, optimizer = 'adam', save = False, directory = './model/'):
        ''''''
        if save: # set callback functions if saving model
            if not os.path.exists(directory): os.makedirs(directory)
            mpath = directory + "cnn.h5"
            hpath = directory + 'cnn_history.csv'
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
                        verbose = 1 ) #fit model
                        
        self.current_epoch += epochs
        return

    def predict(self, x):
        ''''''
        return self.model.predict(x)