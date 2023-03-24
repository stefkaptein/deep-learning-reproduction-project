import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    for i, (x_batch, y_batch) in enumerate(train_loader):

        optimizer.zero_grad()

        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        output = model(x_batch)
        loss = criterion(output, y_batch)

        loss.backward()

        optimizer.step()

        # Keep track of loss and accuracy
        avg_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

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

    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # Iterate through batches
        for i, (x_batch, y_batch) in enumerate(test_loader):

            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # Forward pass
            output = model(x_batch)
            loss = criterion(output, y_batch)

            # Keep track of loss and accuracy
            avg_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    return avg_loss / len(test_loader), 100 * correct / total


def init_params(params_iter):
    for param in params_iter:
        nn.init.normal_(param)


def run():
    # Create a writer to write to Tensorboard
    writer = SummaryWriter()

    # Create classifier model
    model = DeepConvLSTM()
    model = model.to(DEVICE)

    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)

    init_params(model.parameters())

    print("Loading and applying sliding window over data")
    training_dataloader = DataLoader(
        OpportunityDataset("data/pre-processed.pkl", train_or_test="train"),
        batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(
        OpportunityDataset("data/pre-processed.pkl", train_or_test="test"),
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

        print(f"Train Accuracy: {train_loss}")
        print(f"Test Accuracy: {test_loss}")

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
