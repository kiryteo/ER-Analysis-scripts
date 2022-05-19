import imageio
import numpy as np
import skimage
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt

img = imageio.imread('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/Decon/Series002_decon_converted/std/Series002_decon_converted_t00_ch00_std.png')

er = skimage.morphology.erosion(img)

bin_er = pcv.threshold.binary(gray_img=er, threshold=12, max_value=255)

plt.imshow(bin_er)
plt.show()