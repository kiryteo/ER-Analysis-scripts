import skimage
from skimage.filters import threshold_local, threshold_otsu, unsharp_mask, frangi, meijering, sato, butterworth
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from plantcv import plantcv as pcv
import glob

def preprocess_samples(group):
    path_pref = '/localhome/asa420/MIAL/data/confocal_movies/' + group + '/files/'
    new_pref = '/localhome/asa420/MIAL/data/confocal_movies/' + group + '/new_op_jul/preproc/'


    for i in range(27, 32):
        os.makedirs(new_pref + 'C%s'%f'{i}')
        for j in range(100):
            img = imageio.imread(path_pref + 'C%s_decon_t0%s_ch00.tif'%(f'{i}',f'{j:02d}'))
            std_img = ((img) / (img.max() - img.min())) * 255
            aop = skimage.morphology.area_opening(std_img, area_threshold=2)
            erod = skimage.morphology.erosion(aop)
            aop = skimage.morphology.area_opening(erod, area_threshold=2)
            cl = skimage.morphology.area_closing(aop, area_threshold=32)
            aop = skimage.morphology.area_opening(cl, area_threshold=2)
            loc = threshold_local(aop, 3)
            loc = threshold_local(loc, 3)
            cv2.imwrite(new_pref + 'C%s/C%s_decon_t0%s_ch00_proc.png'%(f'{i}',f'{i}',f'{j:02d}'), loc)


# preprocess_samples('Climp')
# exit()


def get_skel():
    pref = '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/'
    for i in range(1, 30):
        os.makedirs(pref + 'skel/R%s'%f'{i}')
        for j in range(100):
            img = imageio.imread(pref + 'preproc/R%s/R%s_decon_t0%s_ch00_proc_enhance.png'%(f'{i}', f'{i}', f'{j:02d}'))
            sk = pcv.morphology.skeletonize(img)
            cv2.imwrite(pref + 'skel/R%s/R%s_decon_t0%s_ch00_skel.png'%(f'{i}',f'{i}',f'{j:02d}'), sk)


# get_skel()
# exit()


def Agg(path):
    for i in range(1, 32):
        files = glob.glob(path + 'C%s/*'%(f'{i}'))
        imgstack = []
        img_mean = np.zeros((128, 128))
        for each in files:
            img = imageio.imread(each)
            img_mean += img
            imgstack.append(img)
        cv2.imwrite('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/C%s_mean.png'%f'{i}', img_mean/len(files))
        # pcv.print_image(img=img_mean/len(files), filename=home + '/Desktop/Climp/' + 'C%s-mean-brpts.png'%(f'{i}'))
        # # imageio.imwrite(home + '/Desktop/ATL/' + 'A%s-mean.png'%(f'{i}'), img_mean/len(files))
        # new = np.stack(imgstack, axis=2)
        # maximg = np.amax(new, axis=2)
        # pcv.print_image(img=maximg, filename=home + '/Desktop/Climp/' + 'C%s-max-brpts.png'%(f'{i}'))
        # imageio.imwrite(home + '/Desktop/ATL/' + 'A%s-max.png'%(f'{i}'), maximg)

# Agg('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/skel/')

exit()

# plt.imshow(mei)
# plt.imshow(fr)
fig = plt.figure(figsize=(8, 6))
# plt.title('A1_decon_t001_ch00_frame')
plt.axis('off')
r, c = 2, 3
#

fig.add_subplot(r, c, 1)
plt.imshow(aop)
plt.axis('off')

fig.add_subplot(r, c, 2)
plt.imshow(loc)
plt.axis('off')

fig.add_subplot(r, c, 3)
plt.imshow(bu1)
plt.axis('off')
# plt.title('Input')

fig.add_subplot(r, c, 4)
plt.imshow(bu2)
plt.axis('off')
# plt.title('sharp')

fig.add_subplot(r, c, 5)
plt.imshow(bu3)
plt.axis('off')

fig.add_subplot(r, c, 6)
plt.imshow(bu4)
plt.axis('off')

# plt.imshow(bu)

# fig.add_subplot(r, c, 5)
# plt.imshow(bu5)
# plt.axis('off')
# # plt.title('Input')
#
# fig.add_subplot(r, c, 6)
# plt.imshow(bu6)
# plt.axis('off')
# # plt.title('sharp')
#
# fig.add_subplot(r, c, 7)
# plt.imshow(bu7)
# plt.axis('off')
#
# fig.add_subplot(r, c, 8)
# plt.imshow(bu8)
# plt.axis('off')

plt.show()

# cv2.imwrite('cv2_res.png', img8)

# print(img.max())
# print(img.min())
#
#
exit()
