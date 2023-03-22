import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset
import pickle as pkl

from torch.utils.data.dataset import T_co


class OpportunityDataset(Dataset):
    def __init__(self, file_path, sliding_window_length, sliding_window_step, train_or_test="train"):
        with open(file_path, "rb") as f:
            data = pkl.load(f)
            (train_data, train_labels), (test_data, test_labels) = data
        if train_or_test == "train":
            self.sensor_data = train_data.astype("float32")
            self.labels = train_labels.astype("float32")

            (self.sensor_data, self.labels) = self.open_sliding_window_over_data(sliding_window_length,
                                                                                 sliding_window_step)
        elif train_or_test == "test":
            self.sensor_data = test_data.astype("float32")
            self.labels = test_labels.astype("float32")

            (self.sensor_data, self.labels) = self.open_sliding_window_over_data(sliding_window_length,
                                                                                 sliding_window_step)
        else:
            raise ValueError("Invalid argument, use train or test")

    def open_sliding_window_over_data(self, window_size, step_size):
        print('opening sliding window of data')
        flatted_sensor_data = self.sensor_data.ravel()
        flatted_labels = self.labels.ravel()

        data_x = sliding_window_view(flatted_sensor_data, window_size*113)
        data_x = data_x[::step_size*113]
        data_y = sliding_window_view(flatted_labels, window_size)[::step_size]
        data_x = data_x[:len(data_y)]

        data_x = np.reshape(data_x, (-1, window_size, 113))
        data_y = np.reshape(data_y, (-1, window_size))
        return data_x, data_y

    def __len__(self):
        return len(self.sensor_data)

    def __getitem__(self, index) -> T_co:
        return self.sensor_data[index], self.labels[index]
