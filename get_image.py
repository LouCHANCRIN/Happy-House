import numpy as np
import h5py
import matplotlib.pyplot as plt
class data:

    def __init__(self, train_image, test_image, split):
        self.train_data, self.train_label = self.unpack_image(train_image, 0)
        label = self.train_label
        data = self.train_data
        self.data_augmentation(data, label)
        np.random.seed(7)
        np.random.shuffle(self.train_data)
        np.random.seed(7)
        np.random.shuffle(self.train_label)
        self.test_data, self.test_label = self.unpack_image(test_image, 1)

    def reverse(self, data, i, line, col, chanel):
        tmp2 = [0] * line
        for j in range(0, line):
            tmp3 = [0] * col
            for k in range(0, col):
                tmp4 = [0] * chanel
                for l in range(0, chanel):
                    x = data[i][j][col - 1 - k][l]
                    tmp4[l] = x
                tmp3[k] = tmp4
            tmp2[j] = tmp3
        return (tmp2)

    def data_augmentation(self, data, label):
        # mirroring, rotation
        train_data2 = []
        nb_image, line, col, chanel = np.shape(data)
        tmp = [0] * nb_image
        for i in range(0, nb_image):
            tmp[i] = self.reverse(data, i, line, col, chanel)
        tmp = np.reshape(tmp, (nb_image, line, col, chanel))
        self.train_data = np.concatenate((self.train_data, tmp), axis=0)
        self.train_label = np.concatenate((self.train_label, label), axis=0)

    def unpack_image(self, path, train_or_test):
        data = h5py.File(path)
        if (train_or_test == 0):
            ret_image = np.array(data['train_set_x'])
            label_value = np.array(data['train_set_y'])
        else:
            ret_image = np.array(data['test_set_x'])
            label_value = np.array(data['test_set_y'])
        ret_label = [0] * np.shape(ret_image)[0]
        for i in range(0, np.shape(ret_image)[0]):
            ret = [0] * 2
            ret[label_value[i]] = 1
            ret_label[i] = ret
        return (ret_image / 255, ret_label)
