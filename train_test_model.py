import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import DECAY, LEARNING_RATE, EPOCHS, SAVE_MODEL_NAME, BATCH_SIZE
from model import DeepConvLSTM
from opportunity_dataset import OpportunityDataset

DEVICE = torch.device('cuda:0')


def train(train_loader, model, optimizer, criterion):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data iterator for training set.
        model: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    correct = 0
    total = 0

    # Switch to train mode
    model.train()

    # Iterate through batches
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Keep track of loss and accuracy
        avg_loss += loss
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return avg_loss / len(train_loader), 100 * correct / total


def test(test_loader, model, criterion):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data iterator for test set.
        model: Neural network model.
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    correct = 0
    total = 0

    # Switch to evaluation mode
    model.eval()

    test_pred = np.empty((0))
    test_true = np.empty((0))
    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # Iterate through batches
        for i, (inputs, labels) in enumerate(test_loader):

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(inputs)

            outputs2 = torch.max(outputs, dim=1)[1].int()
            test_pred = np.append(test_pred, outputs2.cpu(), axis=0)
            test_true = np.append(test_true, labels.cpu(), axis=0)

            loss = criterion(outputs, labels)

            # Keep track of loss and accuracy
            avg_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("\tTest fscore:\t{:.4f} ".format(metrics.f1_score(test_true, test_pred, average='weighted')))
    return avg_loss / len(test_loader), 100 * correct / total


def init_params(params_iter):
    for name, param in params_iter:
        if 'weight' in name:
            nn.init.orthogonal_(param)


def run():
    # Create a writer to write to Tensorboard
    writer = SummaryWriter()

    # Create classifier model
    model = DeepConvLSTM()
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, alpha=DECAY, weight_decay=0.0001)

    init_params(model.named_parameters())

    print("Loading and applying sliding window over data")
    training_dataloader = DataLoader(
        OpportunityDataset("data/pre-processed.pkl", train_or_test="train"),
        shuffle=True,
        batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(
        OpportunityDataset("data/pre-processed.pkl", train_or_test="test"),
        shuffle=True,
        batch_size=BATCH_SIZE)
    print("Data loaded and sliding window applied")

    # Training loop
    for epoch in range(EPOCHS):
        print(f'\n Epoch {epoch}')

        # Train on data
        train_loss, train_acc = train(training_dataloader,
                                      model,
                                      optimizer,
                                      criterion)

        # Test on data
        test_loss, test_acc = test(test_dataloader,
                                   model,
                                   criterion)

        print(f"Train loss: {train_loss}")
        print(f"Test loss: {test_loss}")

        print(f"Train Accuracy: {train_acc}")
        print(f"Test Accuracy: {test_acc}")

        # Write metrics to Tensorboard
        writer.add_scalars('Loss', {
            'Train': train_loss,
            'Test': test_loss
        }, epoch)
        writer.add_scalars('Accuracy', {
            'Train': train_acc,
            'Test': test_acc
        }, epoch)
        writer.flush()

    print('\nFinished.')
    writer.close()

    torch.save(model.state_dict(), SAVE_MODEL_NAME)


if __name__ == "__main__":
    run()
