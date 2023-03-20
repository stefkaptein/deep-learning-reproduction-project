from torch.utils.data import Dataset
import pickle as pkl

from torch.utils.data.dataset import T_co


class OpportunityDataset(Dataset):
    def __init__(self, file_path, train_or_test="train"):
        with open(file_path, "rb") as f:
            data = pkl.load(f)
            (train_data, train_labels), (test_data, test_labels) = data
        if train_or_test == "train":
            self.sensor_data = train_data.astype("float32")
            self.labels = train_labels.astype("float32")
        elif train_or_test == "test":
            self.sensor_data = test_data.astype("float32")
            self.labels = test_labels.astype("float32")
        else:
            raise ValueError("Invalid argument, use train or test")

    def __len__(self):
        return len(self.sensor_data)

    def __getitem__(self, index) -> T_co:
        return self.sensor_data[index], self.labels[index]
