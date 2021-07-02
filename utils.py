import numpy as np
import matplotlib.pyplot as plt
import winsound

def generate_index_array():
    '''Generates an array of random indexes that encompasses all classes and throughputs.'''
    start = np.arange(35).reshape((7, 5)) * 3375
    idx_array = np.zeros((7, 5), dtype = np.int)
    start_idx = 0
    for i in range(7):
        for j in range(5):
            idx_array[i, j] = np.random.randint(low = start[i, j], high = start[i, j] + 3375)
            
    return idx_array


def plot_samples(data, labels, ROWS = 7, COLUMNS = 5, idx_array = None):
    '''Plot random samples. If idx_array is given plot the samples by the given indexes.'''
    standards = ['AX', 'AC', 'N', 'AX_AC', 'AX_N', 'AC_N', 'AX_AC_N']
    interval = int(data.shape[0] / ROWS)
    s = np.arange(0, data.shape[1])
    start = [0, interval, interval * 2, interval * 3, interval * 4]
    end = [interval, interval * 2, interval * 3, interval * 4, interval * 5]
    
    assert idx_array.shape == (ROWS, COLUMNS)

    fig, axes = plt.subplots(ROWS, COLUMNS, figsize = (25, 35))
    for j in range(ROWS):
        for k in range(COLUMNS):
            if idx_array is None:
                i = np.random.randint(low = start[j], high = end[j])
            else:
                i = idx_array[j][k]
            axes[j][k].plot(s, data[i], linewidth = 0.1)
            title = '{} - Mbps:'.format(standards[int(labels[i, 0])])
            title = title + ' {:.1f}'.format(labels[i, 2]) if labels[i, 2] > 0 else title
            title = title + ' | {:.1f}'.format(labels[i, 4]) if labels[i, 4] > 0 else title
            title = title + ' | {:.1f}'.format(labels[i, 6]) if labels[i, 6] > 0 else title
            axes[j][k].set_title(title)
            axes[j][k].set_xticks([0, 10000, 20000, 30000])
            ymin = np.min(data[i])
            if ymin  < -1.0: #not normalized data
                axes[j][k].set_yticks([0, -20, -40, -60, -80])
            elif ymin >= 0: #norm type 2
                axes[j][k].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            else: #norm types 1 or 3
                axes[j][k].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            
    plt.show()
    

def play_sound(freq = 1000, duration = 3000):
    '''Plays a sounds with frequency in Hz and duration in ms.'''
    winsound.Beep(freq, duration)
    

def calc_accuracy(pred, actual):
    ''''''
    return np.round(np.sum(pred == actual) / pred.shape[0] * 100, 2)