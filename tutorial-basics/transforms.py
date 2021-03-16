import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(), # converts PIL / numpy ndarray into normalized FloatTensor
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    # one-hot encoding: Lambda applies user defined lambda function
    # dim, torch.tensor(y)=index where is being written, value that is written
)