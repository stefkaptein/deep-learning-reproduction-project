import pickle as pkl

import numpy as np

import torch
import torch.nn as nn

INPUT_CHANNELS = 64

CONV_HIDDEN_CHANNELS = 64

FILTER_SIZE = 5

LSTM_HIDDEN_CHANNELS = 128

NUM_SENSOR_CHANNELS = 113

NUM_CLASSES = 18

DROP_RATE = 0.5


def train(train_loader, net, optimizer, criterion):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    correct = 0
    total = 0

    # iterate through batches
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return avg_loss / len(train_loader), 100 * correct / total


def test(test_loader, net, criterion):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data loader for test set.
        net: Neural network model.
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    correct = 0
    total = 0

    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # iterate through batches
        for data in test_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # keep track of loss and accuracy
            avg_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return avg_loss / len(test_loader), 100 * correct / total


def load_data():
    with open("data/pre-processed.pkl", "rb") as f:
        dataset = pkl.load(f)
        (train_data, train_labels), (test_data, test_labels) = dataset
    return train_data, train_labels, test_data, test_labels


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cl2 = nn.Conv1d(INPUT_CHANNELS, CONV_HIDDEN_CHANNELS, FILTER_SIZE)
        self.cl3 = nn.Conv1d(CONV_HIDDEN_CHANNELS, CONV_HIDDEN_CHANNELS, FILTER_SIZE)
        self.cl4 = nn.Conv1d(CONV_HIDDEN_CHANNELS, CONV_HIDDEN_CHANNELS, FILTER_SIZE)
        self.cl5 = nn.Conv1d(CONV_HIDDEN_CHANNELS, CONV_HIDDEN_CHANNELS, FILTER_SIZE)
        self.dropout = nn.Dropout(DROP_RATE)
        self.rec6 = nn.LSTM(CONV_HIDDEN_CHANNELS * NUM_SENSOR_CHANNELS, LSTM_HIDDEN_CHANNELS)
        self.rec7 = nn.LSTM(LSTM_HIDDEN_CHANNELS, LSTM_HIDDEN_CHANNELS)
        self.fc8 = nn.Linear(LSTM_HIDDEN_CHANNELS, NUM_CLASSES)
        self.softmax = nn.Softmax(LSTM_HIDDEN_CHANNELS)

    def forward(self, x):
        x = self.cl2(x)
        x = self.cl3(x)
        x = self.cl4(x)
        x = self.cl5(x)
        x = x.view(-1, CONV_HIDDEN_CHANNELS * NUM_SENSOR_CHANNELS)
        x = nn.ReLU(x)
        x = self.dropout(x)
        x = self.rec6(x)
        x = nn.ReLU(x)
        x = self.dropout(x)
        x = self.rec7(x)
        x = self.dropout(x)
        x = nn.ReLU(x)
        x = self.fc8(x)
        x = self.softmax(x)
        return x

    def backward(self, x):
        x = self.fc8(x)
        x = self.dropout(x)
        x = nn.ReLU(x)
        x = self.rec7(x)
        x = self.dropout(x)
        x = nn.ReLU(x)
        x = self.rec6(x)
        x = self.dropout(x)
        x = nn.ReLU(x)
        x = x.view(-1, CONV_HIDDEN_CHANNELS * NUM_SENSOR_CHANNELS)
        x = self.cl5(x)
        x = self.cl4(x)
        x = self.cl3(x)
        x = self.cl2(x)
        return x


if __name__ == "__main__":
    load_data()
