import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset
import pickle as pkl
import torch.nn as nn

from torch.utils.data.dataset import T_co

from config import SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP


class OpportunityDataset(Dataset):
    def __init__(self, file_path, train_or_test="train"):
        with open(file_path, "rb") as f:
            data = pkl.load(f)
            (train_data, train_labels), (test_data, test_labels) = data
        if train_or_test == "train":
            self.sensor_data = train_data.astype("float32")
            self.labels = train_labels.astype("int32")

            (self.sensor_data, self.labels) = self.open_sliding_window_over_data()
        elif train_or_test == "test":
            self.sensor_data = test_data.astype("float32")
            self.labels = test_labels.astype("int32")

            (self.sensor_data, self.labels) = self.open_sliding_window_over_data()
        else:
            raise ValueError("Invalid argument, use train or test")

    def open_sliding_window_over_data(self, window_size=SLIDING_WINDOW_LENGTH, step_size=SLIDING_WINDOW_STEP):
        print('opening sliding window of data')
        flatted_sensor_data = self.sensor_data.ravel()
        flatted_labels = self.labels.ravel()

        data_x = sliding_window_view(flatted_sensor_data, window_size*113)
        data_x = data_x[::step_size*113]
        data_y = sliding_window_view(flatted_labels, window_size)[::step_size]
        data_x = data_x[:len(data_y)]

        data_x = np.reshape(data_x, (-1, window_size, 113))
        data_y = np.reshape(data_y, (-1, window_size))

        data_y = np.asarray([i[-1] for i in data_y])
        n_values = 18
        data_y = np.eye(n_values)[data_y]

        data_y = data_y.astype("int64")

        return data_x, data_y

    def __len__(self):
        return len(self.sensor_data)

    def __getitem__(self, index) -> T_co:
        return self.sensor_data[index], self.labels[index]
