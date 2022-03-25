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
import glob
from skimage import io

import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset

home = os.path.expanduser('~')

class Skeldataset(Dataset):
    def __init__(self, imgdir, transform=None):
        self.imgdir = Path(imgdir)
        # self.pgtdir = Path(pgtdir)
        # self.mpdir = Path(mpdir)

        # imgdir = 'data/images'
        self.ids = glob.glob(imgdir + '/climp/*') + glob.glob(imgdir + '/control/*') + glob.glob(imgdir + '/rtn/*')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = Image.open(name)
        img_file = io.imread(name)
        img_file = np.expand_dims(img_file, axis=0)
        pgt_file_name = name.split('/')
        pgt_file = home + '/unet-exp/data/PGT/' + pgt_file_name[6] + '/' + pgt_file_name[7].split('.')[0] + '_erode_enhance_projection.png'
        # pgt_file = pgt_file_name[0] + '/PGT/' + pgt_file_name[2] + '/' + pgt_file_name[3].split('.')[0] + '_erode_enhance_projection.png'
        # pgt_file = Image.open(pgt_file)
        pgt_file = io.imread(pgt_file)
        pgt_file = np.expand_dims(pgt_file, axis=0)
        sernum = int(name.split('/')[-1].split('_')[0][-3:])
        # mplabel = name.split('/')[0] + '/MPSkel/' + name.split('/')[2] + '/' + str(sernum) + '.png'
        mplabel = home + '/unet-exp/data/MPSkel/' + name.split('/')[6] + '/' + str(sernum) + '.png'
        # mplabel = Image.open(mplabel)
        mplabel = io.imread(mplabel)
        mplabel = np.expand_dims(mplabel, axis=0)

        composite_input = np.concat((img_file, pgt_file), axis=1)

        return composite_input, mplabel

# imgdir = '/localhome/asa420/unet-exp/data/images'
# ids = glob.glob(imgdir + '/climp/*') + glob.glob(imgdir + '/control/*') + glob.glob(imgdir + '/rtn/*')
# img = Image.open(ids[0])
# print(torch.as_tensor(img).shape)

# train_dataloader = DataLoader(Skeldataset, batch_size=32, shuffle=False)

transform = transforms.Compose(
    [transforms.ToTensor(),
     ]
)

skdata = Skeldataset(imgdir='/localhome/asa420/unet-exp/data/images', transform=transform)


# import matplotlib.pyplot as plt
# #for i in range(len(skdata)):
# for i in range(5):
#     sample = skdata[i]
#     # print(sample)
#     fig, ax = plt.subplots(1, 3)
#     ax[0].imshow(sample[0])
#     ax[1].imshow(sample[1])
#     ax[2].imshow(sample[2])
#     plt.show()
#     # sample[2].show()
#     #print(i, sample)
#
#     # plt.imshow()
tloader = DataLoader(skdata, batch_size=32, shuffle=True)

dataiter = iter(tloader)
im, pg, mp = dataiter.next()

print(im.shape)

