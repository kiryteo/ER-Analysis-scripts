from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch


class Skeldataset(Dataset):
    def __init__(self, ):
        pass


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        return image, label


