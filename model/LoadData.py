import os
import sys
import numpy as np
import scipy.io
import torch
import torch.utils.data as data

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)


class LoadData(data.Dataset):
    def __init__(self, data_dir):
        super(LoadData, self).__init__()
        data_names = os.listdir(data_dir)
        self.data_path = []
        for index in range(len(data_names)):
            self.data_path.append(os.path.join(data_dir, data_names[index]))

    def __getitem__(self, index):
        data = scipy.io.loadmat(self.data_path[index])['Y'].astype(np.float64)
        target = scipy.io.loadmat(self.data_path[index])['X0_240'].astype(
            np.float64)
        return data, target

    def __len__(self, data_names):
        return len(data_names)

    def get_library(self, lib_path, lib_symbol):
        library = scipy.io.loadmat(lib_path)[lib_symbol]
        return torch.as_tensor(library)


def get_loaded_data(data_dir, lib_path, lib_symbol):
    train_data = LoadData(data_dir)
    lib_data = train_data.get_library(lib_path, lib_symbol)
    y = train_data.get_y()
    return y, lib_data
