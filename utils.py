import numpy as np
import matplotlib.pyplot as plt

def generate_index_array(sample_size):
    ''''''
    start = np.asarray([[3000 * 0, 3000 * 1, 3000 * 2, 3000 * 3, 3000 * 4],
                        [3000 * 5, 3000 * 6, 3000 * 7, 3000 * 8, 3000 * 9],
                        [3000 * 10, 3000 * 11, 3000 * 12, 3000 * 13, 3000 * 14],
                        [3000 * 15, 3000 * 16, 3000 * 17, 3000 * 18, 3000 * 19],
                        [3000 * 20, 3000 * 21, 3000 * 22, 3000 * 23, 3000 * 24]])
                       
    idx_array = np.zeros((5, 5), dtype=np.int)
    
    start_idx = 0
    for i in range(5):
        for j in range(5):
            idx_array[i, j] = np.random.randint(low = start[i, j], high = start[i, j] + 3000)
            
    return idx_array


def plot_samples(I, info, ROWS = 5, COLUMNS = 5, idx_array = None):
    ''''''
    standards = ['AX', 'AC', 'N', 'AX_AC', 'AX_N']
    interval = int(I.shape[0] / ROWS)
    s = np.arange(0, I.shape[1])
    start = [0, interval, interval * 2, interval * 3, interval * 4]
    end = [interval, interval * 2, interval * 3, interval * 4, interval * 5]
    
    assert idx_array.shape == (ROWS, COLUMNS)

    fig, axes = plt.subplots(ROWS, COLUMNS, figsize = (10,10))
    for j in range(ROWS):
        for k in range(COLUMNS):
            if idx_array is None:
                i = np.random.randint(low = start[j], high = end[j])
            else:
                i = idx_array[j][k]
            axes[j][k].plot(s, I[i], linewidth = 0.1)
            axes[j][k].set_title('{} - Limit: {}  Actual: {}  DC: {}'.format(standards[int(info[i, 0])], info[i, 1], info[i, 2], info[i, 3]))
            
    plt.show()
    
def save_samples(I, info, idx_array):
    ''''''
    
    #load data
    path = ['data/data.npy', 'data/labels.npy']
    print('Loading data from {} ... '.format(path[0]))
    data = np.load(path[0])[:, 0, :].astype('float') #load data and keep only I component
    print('Loading labels from {} ... '.format(path[1]))
    info = np.load(path[1]) #load labels

    idx_array = generate_index_array(data.shape[0]) #get random sample indexes
    I = I[idx_array, :]
    info = info[idx_array, :]
    
    #save the npz
    path = 'data/sample_plots.npz'
    keywords = {"data": I, "info": info}
    np.savez(path, **keywords)
    print(I.shape, info.shape)
    print('Saved to {} ...'.format(path))

    
    

def plot(I, info):
    ''''''
    ROWS, COLUMNS = 5, 5
    standards = ['AX', 'AC', 'N', 'AX_AC', 'AX_N']
    s = np.arange(0, I.shape[2])

    fig, axes = plt.subplots(ROWS, COLUMNS, figsize = (10, 10))
    for j in range(ROWS):
        for k in range(COLUMNS):
            axes[j][k].plot(s, I[j, k, :], linewidth = 0.1)
            axes[j][k].set_title('{} - Limit: {}'.format(standards[int(info[j, k, 0])], info[j, k, 1]), fontsize=10)
    plt.tight_layout(pad=0.4, w_pad=0.2, h_pad=1.0)
    plt.show()