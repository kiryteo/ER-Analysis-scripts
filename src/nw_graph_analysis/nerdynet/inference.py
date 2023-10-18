import contextlib
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from model import NerdyNet
from PIL import Image

from skimage import restoration
from skimage.filters import threshold_otsu
import copy
import imageio
import os


home = os.path.expanduser('~')

sted_data_prefix = f'{home}/MIAL/data/sted-data/vess_enh_unet'

groups = ['climp', 'control', 'rtn']

in_channels = 1
out_channels = 1

model = NerdyNet(in_channels, out_channels)
model.load_state_dict(torch.load('model.pth'))

for group, i in itertools.product(groups, range(1, 17)):
    with contextlib.suppress(Exception):
        imgpath = f'{sted_data_prefix}/{group}/images/sted_{group}{i}_er_mean.png'

        image = Image.open(imgpath)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        # Forward pass through the model
        model.eval()
        with torch.no_grad():
            output = model(image.cuda())

        # Convert the output to probabilities by applying the sigmoid activation
        output_probs = torch.sigmoid(output)

        # Convert tensors to numpy arrays for visualization
        image_np = image.squeeze().numpy()
        output_probs_np = output_probs.cpu().squeeze().numpy()

        norm = (output_probs_np - output_probs_np.min()) / (output_probs_np.max() - output_probs_np.min())

        rb = restoration.rolling_ball(norm)
        subt = norm - rb

        thr = threshold_otsu(subt)

        subt2 = copy.deepcopy(subt)

        subt2[subt < thr] = 0.
        subt2[subt >= thr] = 255.

        imageio.imsave(f'{sted_data_prefix}/{group}/nerdynet_v2/sted_{group}{i}_er_mean_pred.png', subt2)