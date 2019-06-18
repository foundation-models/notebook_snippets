import numpy as np
import pandas as pd

def normalize(data):
    return (data - data.mean())/(data.max() - data.min())
    
class data_reader():  
    def __init__(self, filename, columns=None, label_index=0, window_size=10, batchsize=32, random=True):
        # process the data into a matrix, and return the lenght
        print("Warning: Data passed should be normalized!")
        self.frac = 0.65
        self.random = random
        print('reading data from file', filename)
        df = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines=False, index_col=False)
        print('Raw data', df.shape)
        print(df.columns)
        print(columns)
        print(["NBP (Mean)", "Minute Volume"])
        self.data = df[["NBP (Mean)", "Minute Volume"]].dropna().values
        print('Dropna with selected columns', self.data.shape)
        print(self.data[0:3,:])
        print(data.head(3))
        
        self.process(window_size)
        self.columns = columns
        self.batchsize = batchsize
        self.label_index = label_index

        self.pointer = 0
        self.epoch = 0
        
    def process(self, window_size):
        # Generate the data matrix
        normalize(self.data)        
        length = self.data.shape[0]
        sliding_window_data = np.zeros((length-window_size, window_size))
        sliding_window_label = np.zeros((length-window_size, 1))
        for counter in range(length-window_size):
            sliding_window_data[counter, :] = self.data[counter: counter+window_size, 1]
            sliding_window_label[counter, :] = self.data[counter+window_size, 0]
        # Random shuffle
        length = sliding_window_data.shape[0]
        idx = np.random.choice(length, length, replace=False)
        if not self.random:
            idx = np.arange(length)
        self.val_idx = idx[int(self.frac*length):]

        shuf_data = sliding_window_data[idx, :]
        shuf_label = sliding_window_label[idx, :]
        self.shuf_data = shuf_data
        self.shuf_label = shuf_label
        self.train = sliding_window_data
        self.label = sliding_window_label

        self.train_data = shuf_data[:int(self.frac*length), :]
        self.train_label = shuf_label[:int(self.frac*length), :]
        self.train_size = int(self.frac*length)

        self.val_data = shuf_data[int(self.frac*length):, :]
        self.val_label = shuf_label[int(self.frac*length):, :]
        self.val_size = int((1-self.frac)*length)

        return None

    def get_next_train_batch(self):
        # getting the next train batch
        if self.pointer + self.batchsize >= self.train_size:
            end = self.train_size
            start = self.pointer
            self.pointer = 0
            self.epoch += 1
        else:
            end = self.pointer + self.batchsize
            start = self.pointer
            self.pointer += self.batchsize
        X = np.expand_dims(self.train_data[start:end, :], axis=-1)
        Y = self.train_label[start:end, :]
        return X, Y

    def get_val(self):
        X = np.expand_dims(self.val_data, axis=-1)

        return X, self.val_label[:]

    def get_whole(self):
        # get whole, for validation set
        X = np.expand_dims(self.data[:, :], axis=-1)
        Y = self.label[:, :]
        return X, Y

    def reset(self):
        self.pointer = 0
        self.epoch = 0

    def get_epoch(self):
        # return the current epoch
        return self.epoch
    
    def get_shuff_train_label(self):
        X = np.expand_dims(self.shuf_data, axis=-1)
        Y = self.shuf_label
        return X, Y
