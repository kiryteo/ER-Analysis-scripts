import tifffile
import imageio
from skimage.io import imread
from skimage import exposure
import ks_multithresh
import matplotlib.pyplot as plt

from os.path import basename
import os

home = os.path.expanduser('~')

# fname = home + '/MIAL/data-Feb4-Guang/Climp1/Series033_decon/Series033_decon_ch02.tif'
# file = imread(fname)

fname = '/localhome/asa420/MIAL/data/CROP-n/Plos_data_crops/RTN_crops/Series006_decon_ch02-3.tif'
file = imread(fname)
img_adapteq = exposure.equalize_adapthist(file)#, clip_limit=0.015)
imageio.imsave('/localhome/asa420/MIAL/data/CROP-n/Plos_data_crops/RTN_crops/Series006_decon_ch02-3_adapt.png', img_adapteq)

# plt.imshow(img_adapteq)
# plt.show()

exit()

print(basename(fname))
# print(file.shape)
# beads, beads_km = ks_multithresh.test_thresholds(file)
# ks_multithresh.test_thresholds(file)
# plt.imshow(beads_km)
# plt.show()
# plt.imshow(beads)
# plt.show()
