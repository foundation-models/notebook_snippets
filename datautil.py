import numpy as np
import pandas as pd

class data_reader():
    def __init__(self, filename, columns=None, window_size=10, batchsize=32, random=True):
        # process the data into a matrix, and return the lenght
        print("Warning: Data passed should be normalized!")
        self.frac = 0.65
        self.random = random
        print('reading data from file', filename)
        df = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines=False, index_col=False)
        print('Raw data', df.shape)
        df = df[self.columns].dropna().as_matrix()
        print('Dropna with selected columns', df.shape)
        print(df.head(3))
        
        self.process(df, window_size)
        self.columns = columns
        self.batchsize = batchsize
        self.dataframe = df

        self.pointer = 0
        self.epoch = 0
        
    def normalize(self, data):
        return (data - data.mean())/(data.max() - data.min())
        
    def process(self, window_size):
        # Generate the data matrix
        normalize(self.df)        
        length = self.df.shape[0]
        data = np.zeros((length-window_size, window_size))
        label = np.zeros((length-window_size, 1))
        for counter in range(length-window_size):
            data[counter, :] = self.df[counter: counter+window_size, 1]
            label[counter, :] = self.df[counter+window_size, 0]
        # Random shuffle
        length = data.shape[0]
        idx = np.random.choice(length, length, replace=False)
        if not self.random:
            idx = np.arange(length)
        self.val_idx = idx[int(self.frac*length):]

        shuf_data = data[idx, :]
        shuf_label = label[idx, :]
        self.shuf_data = shuf_data
        self.shuf_label = shuf_label
        self.data =data
        self.label = label

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
