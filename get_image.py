import numpy as np
import h5py

class data:

    def __init__(self, train_image, test_image, split):
        self.train_data, self.train_label = self.unpack_image(train_image, 0)
        self.test_data, self.test_label = self.unpack_image(test_image, 1)
        #train_length = int(np.shape(train_data)[0] * split)
        #self.train_data = train_data[:train_length]
        #self.train_label = train_label[:train_length]
        #self.validation_data = train_data[train_length:]
        #self.validation_label = train_label[train_length:]

    def unpack_image(self, path, train_or_test):
        data = h5py.File(path)
        if (train_or_test == 0):
            ret_image = np.array(data['train_set_x'])
            ret_label = np.array(data['train_set_y'])
        else:
            ret_image = np.array(data['test_set_x'])
            ret_label = np.array(data['test_set_y'])
        return (ret_image / 255, ret_label)
