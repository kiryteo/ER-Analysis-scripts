import imageio
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.filters import threshold_local

er_mean = imageio.imread('/localhome/asa420/MIAL/data/confocal-data/Climp/er_mean/climp1_er_mean.png')

er_mean = er_mean / 2

for i in range(100):
    er = imageio.imread(f'/localhome/asa420/MIAL/data/confocal-data/Climp/std/C1_decon_t0{i:02d}_ch00_std.png')

    er = er / 2

    er = (er + er_mean)

    print(er.max())

    # plt.imshow(er)
    # plt.show()
    aop = skimage.morphology.area_opening(er, area_threshold=2)
    erod = skimage.morphology.erosion(aop)
    aop = skimage.morphology.area_opening(erod, area_threshold=2)
    cl = skimage.morphology.area_closing(aop, area_threshold=32)
    aop = skimage.morphology.area_opening(cl, area_threshold=2)
    loc = threshold_local(aop, 3)

    plt.imshow(loc)
    plt.show()