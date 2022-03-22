from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class Skeldataset(Dataset):
    def __init__(self, imgdir, pgtdir, mpdir):
        self.imgdir = Path(imgdir)
        self.pgtdir = Path(pgtdir)
        self.mpdir = Path(mpdir)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.
        return image, pgt, mplabel


