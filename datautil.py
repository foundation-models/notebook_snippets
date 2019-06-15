import numpy as np
import pandas as pd

class data_reader():
    def __init__(self, filename, l=10, batchsize=32, random=True):
        # process the data into a matrix, and return the lenght
        print("Warning: Data passed should be normalized!")
        self.frac = 0.65
        self.random = random
        df = pd.read_csv(filename)
        self.process(df, l)
        self.batchsize = batchsize

        self.pointer = 0
        self.epoch = 0
    def process(self, df, l):
        # Generate the data matrix
        print(df.head(3))
        df = df[["NBP (Mean)", "Minute Volume"]].dropna().as_matrix()
        length = df.shape[0]
        data = np.zeros((length-l, l))
        label = np.zeros((length-l, 1))
        for counter in range(length-l):
            data[counter, :] = df[counter: counter+l, 1]
            label[counter, :] = df[counter+l, 0]
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
