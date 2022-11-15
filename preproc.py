import imageio
import skimage
from skimage.filters import threshold_local
import matplotlib.pyplot as plt
import numpy as np
import copy
from skimage.color import rgb2gray
import cv2
import scipy.ndimage

def preprocess_samples():
    img = imageio.imread('/localhome/asa420/Downloads/Oligos_PLP1_abcam_MBP_DAPI_40x_z0_Composite.png')
    # std_img = ((img) / (img.max() - img.min())) * 255
    # plt.imshow(img)
    # plt.show()
    # exit()
    # img = rgb2gray(img)
    aop = skimage.morphology.area_opening(img, area_threshold=2)
    # plt.imshow(aop)
    # plt.show()
    # exit()

    erod = skimage.morphology.erosion(aop)
    nerod = copy.deepcopy(erod)
    # plt.imshow(erod)
    # plt.show()
    nerod[np.where(erod)]= img[np.where(erod)]

    nimg = img - nerod

    # plt.imshow(nimg)
    # plt.show()

    # exit()
    aop = skimage.morphology.area_opening(nimg, area_threshold=2)

    # plt.imshow(aop)
    # plt.show()
    #
    # exit()

    cl = skimage.morphology.area_closing(aop, area_threshold=32)
    # plt.imshow(cl)
    # plt.show()
    #
    # exit()
    aop = skimage.morphology.area_opening(cl, area_threshold=2)
    # plt.imshow(aop)
    # plt.show()

    # print(aop.shape)
    aop = rgb2gray(aop)
    # plt.imshow(aop)
    # plt.show()

    # exit()
    loc = threshold_local(aop, 3)
    # plt.imshow(loc)
    # plt.show()
    # exit()

    loc = threshold_local(loc, 3)
    cv2.imwrite('/localhome/asa420/Downloads/Oligos_preproc.png', loc*255)

# preprocess_samples()
# img = imageio.imread('/localhome/asa420/Downloads/Oligos_PLP1_abcam_MBP_DAPI_40x_z0_Composite.png')
# print(img.min())
# print(img.max())

# img = imageio.imread('/localhome/asa420/Downloads/Oligos_preproc_enhance.png')
# aop = skimage.morphology.area_opening(img, area_threshold=8)

from skimage.filters.rank import median
from skimage.morphology import disk

# aop = scipy.ndimage.median_filter(img, size=4)

# fig, axes = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
# ax = axes.ravel()

# ax[0].imshow(skimage.morphology.area_opening(img, area_threshold=12))
# ax[1].imshow(median(img, disk(2)))

# aop = median(img, disk(1))
# aop = threshold_local(aop, 3)

# cv2.imwrite('/localhome/asa420/Downloads/Oligos_preproc_enh_proc.png', aop)

# plt.imshow(aop)
# plt.show()

from plantcv import plantcv as pcv

nimg = imageio.imread('/localhome/asa420/Downloads/Oligos_preproc_enhance.png')
sk = pcv.morphology.skeletonize(nimg)

# op = median(sk, disk(1))
# op = skimage.morphology.area_opening(sk, area_threshold=1)
# op = skimage.morphology.area_opening(op, area_threshold=1)
# op = skimage.morphology.area_closing(op, area_threshold=12)

cv2.imwrite('/localhome/asa420/Downloads/Oligos_preproc_enh_skel.png', sk)
# plt.imshow(op)
# plt.show()

