import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from opportunity_dataset import OpportunityDataset

CONV_HIDDEN_CHANNELS = 64

NUM_FILTERS = 64

FILTER_SIZE = 5

LSTM_HIDDEN_CHANNELS = 128

NUM_SENSOR_CHANNELS = 113

INPUT_CHANNELS = NUM_SENSOR_CHANNELS

NUM_CLASSES = 18

DROP_RATE = 0.5

# Training parameters
LEARNING_RATE = 10 ** -3
EPOCHS = 3

BATCH_SIZE = 100
DECAY = 0.9

SLIDING_WINDOW_LENGTH = 24
SLIDING_WINDOW_STEP = 12


def try_gpu():
    """
    If GPU is available, return torch.device as cuda:0; else return torch.device
    as cpu.
    """
    if torch.cuda.is_available():
        print("GPU Available")
        device = torch.device('cuda:0')
    else:
        print("GPU NOT Available")
        device = torch.device('cpu')
    return device


def evaluate_accuracy(data_loader, net, device=torch.device('cpu')):
    """Evaluate accuracy of a model on the given data set."""
    net.eval()  # make sure network is in evaluation mode

    # init
    acc_sum = torch.tensor([0], dtype=torch.float32, device=device)
    n = 0

    for X, y in data_loader:
        # Copy the data to device.
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y = y.long()
            acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))
            n += y.shape[0]  # increases with the number of samples in the batch
    return acc_sum.item() / n


def train(train_loader, test_loader, net, optimizer, criterion):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        test_loader: Data loader for test set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """

    train_losses = []
    train_accs = []
    test_accs = []

    # iterate through batches
    device = try_gpu()

    for epoch in range(EPOCHS):

        # Network in training mode and to device
        net.train()
        net.to(device)

        # Training loop
        for i, (x_batch, y_batch) in enumerate(train_loader):
            # Set to same device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Set the gradients to zero
            optimizer.zero_grad()

            # Perform forward pass
            y_pred = net(x_batch)

            # Compute the loss
            loss = criterion(y_pred, y_batch)
            train_losses.append(loss)

            # Backward computation and update
            loss.backward()
            optimizer.step()

        # Compute train and test error
        train_acc = 100 * evaluate_accuracy(train_loader, net.to('cpu'))
        test_acc = 100 * evaluate_accuracy(test_loader, net.to('cpu'))

        # Development of performance
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # Print performance
        print('Epoch: {:.0f}'.format(epoch + 1))
        print('Accuracy of train set: {:.00f}%'.format(train_acc))
        print('Accuracy of test set: {:.00f}%'.format(test_acc))
        print('')


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


class DeepConvLSTM(nn.Module):
    def __init__(self):
        super(DeepConvLSTM, self).__init__()
        self.cl2 = nn.Conv2d(1, CONV_HIDDEN_CHANNELS, (5, 1), groups=1)
        self.cl3 = nn.Conv2d(CONV_HIDDEN_CHANNELS, CONV_HIDDEN_CHANNELS, (5, 1), groups=1)
        self.cl4 = nn.Conv2d(CONV_HIDDEN_CHANNELS, CONV_HIDDEN_CHANNELS, (5, 1), groups=1)
        self.cl5 = nn.Conv2d(CONV_HIDDEN_CHANNELS, CONV_HIDDEN_CHANNELS, (5, 1), groups=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(DROP_RATE)
        self.rec6 = nn.LSTM(CONV_HIDDEN_CHANNELS * NUM_SENSOR_CHANNELS, LSTM_HIDDEN_CHANNELS)
        self.rec7 = nn.LSTM(LSTM_HIDDEN_CHANNELS, LSTM_HIDDEN_CHANNELS)
        self.fc8 = nn.Linear(LSTM_HIDDEN_CHANNELS, NUM_CLASSES)
        self.softmax = nn.Softmax(LSTM_HIDDEN_CHANNELS)

    def forward(self, x):
        print("Initial shape [%s]" % str(x.shape))
        b, w, s = x.shape
        x = x.reshape((b, 1, w, s))
        print("Shape after transpose [%s]" % str(x.shape))
        x = self.cl2(x)
        print("Shape after first conv [%s]" % str(x.shape))
        x = self.cl3(x)
        print("Shape after second conv [%s]" % str(x.shape))
        x = self.cl4(x)
        print("Shape after third conv [%s]" % str(x.shape))
        x = self.cl5(x)
        print("Shape after last conv [%s]" % str(x.shape))
        x = nn.functional.relu(x)
        x = self.dropout(x)
        print("Shape before LSTM [%s]" % str(x.shape))
        x = self.rec6(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.rec7(x)
        x = self.dropout(x)
        x = nn.functional.relu(x)
        x = self.fc8(x)
        x = self.softmax(x)
        return x


def init_params(params_iter):
    for param in params_iter:
        nn.init.normal_(param)


if __name__ == "__main__":
    print("loading data")
    training_dataloader = DataLoader(
        OpportunityDataset("data/pre-processed.pkl", SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, train_or_test="train"),
        batch_size=100)
    test_dataloader = DataLoader(
        OpportunityDataset("data/pre-processed.pkl", SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, train_or_test="test"),
        batch_size=100)
    print("data loaded")

    net = DeepConvLSTM()

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)
    init_params(net.parameters())
    train(training_dataloader, test_dataloader, net, optimizer, loss_criterion)
