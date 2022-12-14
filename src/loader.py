"""
Dataloader for skeleton extraction unet experiments
"""

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

        # imgdir = 'data/images'
        self.ids = glob.glob(f'{imgdir}/climp/*') + glob.glob(f'{imgdir}/control/*') + glob.glob(f'{imgdir}/rtn/*')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = Image.open(name)
        pgt_file_name = name.split('/')
        pgt_file = f'{pgt_file_name[0]}/PGT/{pgt_file_name[2]}' + pgt_file_name[3].split('.')[0] + '_erode_enhance_projection.png'

        pgt_file = Image.open(pgt_file)
        sernum = int(name.split('/')[-1].split('_')[0][-3:])
        mplabel = name.split('/')[0] + '/MPSkel/' + name.split('/')[2] + str(sernum) + '.png'
        return img_file, pgt_file, mplabel


