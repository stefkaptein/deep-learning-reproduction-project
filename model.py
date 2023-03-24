import torch
import torch.nn as nn

from config import NUM_CLASSES, LSTM_HIDDEN_CHANNELS, \
    NUM_SENSOR_CHANNELS, CONV_HIDDEN_CHANNELS, DROP_RATE, FILTER_SIZE


class DeepConvLSTM(nn.Module):
    def __init__(self):
        super(DeepConvLSTM, self).__init__()
        self.cl2 = nn.Conv2d(1, CONV_HIDDEN_CHANNELS, (FILTER_SIZE, 1))
        self.cl3 = nn.Conv2d(CONV_HIDDEN_CHANNELS, CONV_HIDDEN_CHANNELS, (FILTER_SIZE, 1))
        self.cl4 = nn.Conv2d(CONV_HIDDEN_CHANNELS, CONV_HIDDEN_CHANNELS, (FILTER_SIZE, 1))
        self.cl5 = nn.Conv2d(CONV_HIDDEN_CHANNELS, CONV_HIDDEN_CHANNELS, (FILTER_SIZE, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(DROP_RATE)
        self.rec6 = nn.LSTM(CONV_HIDDEN_CHANNELS * NUM_SENSOR_CHANNELS, LSTM_HIDDEN_CHANNELS, batch_first=True)
        self.rec7 = nn.LSTM(LSTM_HIDDEN_CHANNELS, LSTM_HIDDEN_CHANNELS, batch_first=True)
        self.fc8 = nn.Linear(LSTM_HIDDEN_CHANNELS, NUM_CLASSES)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        b, w, s = x.shape
        x = x.reshape((b, 1, w, s))

        x = nn.functional.relu(self.cl2(x))
        x = nn.functional.relu(self.cl3(x))
        x = nn.functional.relu(self.cl4(x))
        x = nn.functional.relu(self.cl5(x))

        x = x.transpose(1, 2)
        x = torch.flatten(x, start_dim=2)

        x = self.dropout(x)
        x = self.rec6(x)[0]
        x = torch.tanh(x)

        x = self.dropout(x)
        x = self.rec7(x)[0]
        x = torch.tanh(x)

        x = x.reshape((-1, 128))
        x = self.fc8(x)
        x = self.softmax(x)
        x = x.reshape((b, 8, NUM_CLASSES)).select(1, -1)

        return x
