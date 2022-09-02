import imageio
import skimage
import os
from skimage.filters import threshold_local
import cv2
from plantcv import plantcv as pcv


def preprocess_samples(group):
    path_pref = '/localhome/asa420/MIAL/data/confocal_movies/' + group + '/files/'
    new_pref = '/localhome/asa420/MIAL/data/confocal_movies/' + group + '/new_op_jul/preproc_mcherry/'

    for i in range(1, 27):
        os.makedirs(new_pref + 'A%s' % f'{i}')
        for j in range(100):
            img = imageio.imread(path_pref + 'A%s_decon_t0%s_ch01.tif' % (f'{i}', f'{j:02d}'))
            std_img = ((img) / (img.max() - img.min())) * 255
            aop = skimage.morphology.area_opening(std_img, area_threshold=2)
            erod = skimage.morphology.erosion(aop)
            aop = skimage.morphology.area_opening(erod, area_threshold=2)
            cl = skimage.morphology.area_closing(aop, area_threshold=32)
            aop = skimage.morphology.area_opening(cl, area_threshold=2)
            loc = threshold_local(aop, 3)
            loc = threshold_local(loc, 3)
            cv2.imwrite(new_pref + 'A%s/A%s_decon_t0%s_ch01_proc_op5.png' % (f'{i}', f'{i}', f'{j:02d}'), loc)


# preprocess_samples('ATL')

def get_skel():
    pref = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/'
    for i in range(1, 27):
        os.makedirs(pref + 'skel_mcherry/A%s' % f'{i}')
        for j in range(100):
            img = imageio.imread(
                pref + 'preproc_mcherry/A%s/A%s_decon_t0%s_ch01_proc_enhance.png' % (f'{i}', f'{i}', f'{j:02d}'))
            sk = pcv.morphology.skeletonize(img)
            cv2.imwrite(pref + 'skel_mcherry/A%s/A%s_decon_t0%s_ch01_skel.png' % (f'{i}', f'{i}', f'{j:02d}'), sk)


# get_skel()
# exit()

img_ch0 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/preproc/A1/A1_decon_t000_ch00_proc.png')
skel_ch0 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t000_ch00_skel.png')

img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/preproc_mcherry/A1/A1_decon_t000_ch01_proc.png')
skel = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel_mcherry/A1/A1_decon_t000_ch01_skel.png')
# op = skimage.morphology.area_opening(img, area_threshold=2)

# cc = skimage.measure.label(img)
# props = skimage.measure.regionprops(cc)

# print(props[0]['area'])

# l = []
# for i, val in enumerate(props):
#     if props[i]['area'] < 4:
#         l.append(i)
#
#
#
# print(len(props))

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,3))
plt.title('ATL Series 1 t=0')
plt.axis('off')
r,c = 1, 4

fig.add_subplot(r, c, 1)
plt.title('EGFP')
plt.axis('off')
plt.imshow(img_ch0)

fig.add_subplot(r, c, 2)
plt.title('MCherry')
plt.axis('off')
plt.imshow(img)

fig.add_subplot(r, c, 3)
plt.title('EGFP skel')
plt.axis('off')
plt.imshow(skel_ch0)

fig.add_subplot(r, c, 4)
plt.title('MCherry skel')
plt.axis('off')
plt.imshow(skel)

plt.savefig('ATL_S1_2channel_pt2', bbox_inches='tight')
plt.close()
# plt.show()