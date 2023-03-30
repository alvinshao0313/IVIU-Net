import os
import sys
import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

sys.path.append("/opt/data/private/IVIU_Net")
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

class LoadData(Dataset):
    def __init__(self, data_dir):
        super(LoadData, self).__init__()
        self.data_names = os.listdir(data_dir)
        self.data_path = []
        for index in range(len(self.data_names)):
            self.data_path.append(
                os.path.join(data_dir, self.data_names[index]))

    def __getitem__(self, index):
        data = scipy.io.loadmat(self.data_path[index])['Y']
        target = scipy.io.loadmat(self.data_path[index])['X0_240']
        return data, target

    def __len__(self):
        return len(self.data_names)

    def get_library(self, lib_path, lib_symbol):
        library = scipy.io.loadmat(lib_path)[lib_symbol].astype(np.float64)
        return torch.as_tensor(library).float()


def get_loaded_data(batch_size, data_dir, lib_path, lib_symbol):
    data = LoadData(data_dir)
    loaded_data = DataLoader(data, batch_size)
    lib = data.get_library(lib_path, lib_symbol)
    return loaded_data, lib


# def main():
#     run = get_loaded_data(1, data_dir="IVIU_Net/data/DC1/DC1_75s_40dB_train",
#     lib_path='IVIU_Net/data/Library/USGS_Lib_240.mat',
#     lib_symbol='Lib_240')
#     max_batch = len(run)
#     iterator = iter(run)
#     for i in range(max_batch):
#         x, y = next(iterator)
#         print(x, y)

# if __name__=='__main__':
#     main()