'''
* This file define the format of input dataset
'''

import numpy as np

class NASData(object):
    def __init__(self, name):
        self.name = name
        self.data = []
        pass

    # next_batch return x, y data with size batch_size
    def next_batch(self, batch_size):
        batch_x = []
        batch_y = []

        return batch_x, batch_y

# NASData provide train and test data which has next_batch function
class NASDataSet(object):
    def __init__(self):
        self.train = NASData("Train")
        self.test = NASData("Test")

    # read data to train and test data, return data length
    def read_data(self, filepath):
        len_train_data = 0
        len_test_data = 0
        return len_train_data, len_test_data

