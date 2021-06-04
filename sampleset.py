import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tensorflow messages

import numpy as np
from utils import save_samples, plot

def main():
    ''''''
    # save_samples(data, info, idx_array)
    
    data = np.load('data/sample_plots.npz')
    
    plot(data['data'], data['info'])
    
    #normalize data
    threshold, abs_max = 59.0, 85
    print('Normalizing data... threshold offset: {}, absolute max value: {}'.format(threshold, abs_max))
    norm = (data['data'] + threshold) / abs_max #offset by threshold and scale to [-1, 1] while preserving zero location
    
    plot(norm, data['info'])

if __name__ == '__main__':
    main()
    
    
    
    


