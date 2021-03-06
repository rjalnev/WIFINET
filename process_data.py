import numpy as np

def process_raw_data(size = 10000, normalize = False, seed = None):
    '''Loop through all raw data files, sample the data, generate labels, and save as numpy array.'''
    np.random.seed(seed = seed) #set numpy seed
    standards = ['AX', 'AC', 'N', 'AX_AC', 'AX_N', 'AC_N', 'AX_AC_N']
    num_standards = len(standards)
    num_tests = np.asarray([5, 5, 5, 25, 25, 25, 27])
    num_repeats = np.asarray([5, 5, 5, 5, 5, 5, 5])
    num_samples = np.asarray([675, 675, 675, 135, 135, 135, 125])
    total_samples = np.sum(num_tests * num_repeats * num_samples)
    random_spacing = np.asarray([570000, 570000, 570000, 2900000, 2900000, 2900000, 3100000])
    
    path = 'data/info.csv' #generate path for info
    info = np.loadtxt(open(path), delimiter = ',', dtype = np.float32) #load test info
    
    IQ = np.empty((total_samples, 2, size), dtype = np.float32)
    labels = np.empty((total_samples, 8), dtype = np.float32)
    
    idx = 0
    sample_index = 0
    #loop through all of the binary files for each standard
    for i in range(0, num_standards):
        for j in range(0, num_tests[i]):
            for k in range(0, num_repeats[i]):
            
                path = 'data/raw/power/{}/{:03}{:02}.bin'.format(standards[i],
                                                                 np.sum(num_tests[0:i]) + j + 1,
                                                                 k + 1) #generate path to data file
                print('Processing binary file: {}'.format(path))
                data = np.fromfile(path, dtype = 'int8') #load data
                
                #sample signal and generate labels
                IQ_, labels_ = sample_IQ_signal( data,
                                                 info[idx, :],
                                                 num_samples = num_samples[i],
                                                 size = size,
                                                 random_spacing = random_spacing[i],
                                                 trim_range = slice(100000000, 500000000) )
                
                #append data and labels to parent arrays
                IQ[sample_index:sample_index + num_samples[i], :, :] = IQ_
                labels[sample_index:sample_index + num_samples[i]] = labels_
                
                sample_index += num_samples[i]
                idx += 1 #increment info index
    
    #save arrays to disk
    np.save('data/data.npy', IQ)
    np.save('data/labels.npy', labels)
    
    np.random.seed(seed = None) #reset numpy seed


def sample_IQ_signal(signal, info, num_samples, size, random_spacing, trim_range):
    '''Separate IQ data, trim off begining and end, sample the data, and return sampled data with labels.'''
    IQ = np.zeros((num_samples, 2, size), dtype = np.float32) #allocate memory for data
    labels = np.zeros((num_samples, 4), dtype = np.float32) #allocate memory for labels

    I, Q = [signal[idx::2] for idx in range(2)] #separate interweaved data into I and Q
    I, Q = I[trim_range], Q[trim_range] #trim off beginnning and end of data

    #get [num_samples] number of samples of size [size] with random space in between of max size [random_spacing]
    for i in range(0, num_samples):
        idx = i * size + np.random.randint(0, random_spacing)
        IQ[i, 0, :] = I[idx:idx + size]
        IQ[i, 1, :] = Q[idx:idx + size]
    
    #generate labels from info
    labels = np.repeat(info.reshape(1, 8), repeats = num_samples, axis = 0)
   
    return IQ, labels


def main():
    '''Process the raw data and save as numpy arrays.'''
    process_raw_data(size = 30000, seed = 42)


if __name__ == '__main__':
    main()