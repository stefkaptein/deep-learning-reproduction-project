from torch import nn
from torch.utils.data import DataLoader

from model import DeepConvLSTM
import torch

from opportunity_dataset import OpportunityDataset
from train_test_model import test, DEVICE

SAVE_MODEL_NAME = "DeepConvLSTM_Opportunity_Model-LR-0001.pt"


if __name__ == "__main__":
    print('helloooo')
    model = DeepConvLSTM()
    print("Model created")

    model.load_state_dict(torch.load(SAVE_MODEL_NAME))

    model.to(DEVICE)
    print("Model loaded")

    test_dataloader = DataLoader(
        OpportunityDataset("data/pre-processed.pkl", train_or_test="test"))
    print("Data loaded and sliding window applied")

    criterion = nn.CrossEntropyLoss()

    # Test on data
    print("Testing model")
    test_loss, test_acc = test(test_dataloader,
                               model,
                               criterion)

