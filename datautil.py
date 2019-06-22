import numpy as np
import pandas as pd

def normalize(dataframe):
    print('before normalize: ', dataframe.head(2))
    result = (dataframe - dataframe.mean())/(dataframe.max() - dataframe.min())
    print('after normalize: ', result.head(2))
    return result
    
class data_reader():  
    def __init__(self, filename, time_column=None, feature_column=None, label_column=None, window_size=10, random_shuffle=True):
        # process the data into a matrix, and return the lenght
        print("Warning: Data passed should be normalized!")
        self.frac = 0.65
        self.random = random_shuffle
        print('reading data from file', filename)
        df = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines=False, index_col=False)
        print('Raw data', df.shape)
        cols = np.array([time_column, feature_column, label_column])
        columns = cols[cols != np.array(None)]
        dataframe = df.dropna() if columns.size == 0 else df[columns].dropna()
        print('Dropna with selected columns', dataframe.shape)
        scaledDataFrame = normalize(dataframe) 
            
        self.dataframe = dataframe # origina
        self.scaledDataFrame = scaledDataFrame
        print(self.scaledDataFrame.head(3))
        self.time_column = time_column
        self.feature_column = feature_column
        self.label_column = label_column
        

        
    def scaledBackDataFrame(self, data, size, column_name):
        df = pd.DataFrame(index=self.dataframe.iloc[0:size].index)
        df[self.time_column] = self.dataframe[self.time_column][0:size]
        df[column_name] = data
        print('before normalize: ', df.head(2))
        result = df * (self.dataframe.max() - self.dataframe.min()) + self.dataframe.mean()
        print('after normalize: ', result.head(2))
        return result
        
    def process(self, window_size):
        # Generate the data matrix      
        length0 = self.scaledDataFrame.shape[0]
        sliding_window_data = np.zeros((length0-window_size, window_size))
        sliding_window_label = np.zeros((length0-window_size, 1))
        
        for counter in range(length0-window_size):
            print(counter, self.scaledDataFrame[self.label_column][counter])
            sliding_window_label[counter, :] = self.scaledDataFrame[self.label_column][counter+window_size]
          
        if(self.feature_column is not None):
            for counter in range(length0-window_size):
                sliding_window_data[counter, :] = self.scaledDataFrame[self.feature_column][counter: counter+window_size]
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
        X = np.expand_dims(self.train[:, :], axis=-1)
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
