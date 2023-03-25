# Deep Learning Reproduction Project
Reproduction of the 'Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition' paper.

[Paper link](https://www.mdpi.com/1424-8220/16/1/115/html)

[Dataset download link](https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip)

## Requirements

## Dataset
The zip downloaded from the [link above](https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip) should be extracted in the `data` directory,  in the root directory of the project. The path to the dataset should be: 
`~/data/OpportunityUCIDataset/dataset/`.

## Running tensorboard

```commandline
tensorboard --logdir runs/
```

## What needed change

In the paper they mention 10e-3 lr, but this is too high and they actually
use 10e-4. 10e-3 is too high and will cause overshooting? so there is never any improvement
We found this out by checking their own training code.

Also we found that the paper uses l2 regularization, but they don't mention it in the paper. They
use a value of 0.0001 for this. This is called the weight_decay in the pytorch code for RMSprop.

Lastly, in the paper the do mention weight_decay, but this is understood wrong by them. Instead
the weight_decay is actually the rho of the RMSprop algorithm. In pytorch this is called alpha.