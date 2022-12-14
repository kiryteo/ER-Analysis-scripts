import skimage
from skimage.filters import threshold_local
import os
import imageio
import cv2
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt

# img = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_t0_ch0.png')
# aop = skimage.morphology.area_opening(img, area_threshold=2)
# erod = skimage.morphology.erosion(aop)
# aop = skimage.morphology.area_opening(erod, area_threshold=2)
# # cl = skimage.morphology.area_closing(aop, area_threshold=32)
# # aop = skimage.morphology.area_opening(cl, area_threshold=2)
# loc = threshold_local(aop, 3)
# loc = threshold_local(loc, 3)
# cv2.imwrite('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_t0_ch0_loc2_noclaop.png', loc)

# img = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_t0_ch0_loc2_enhance.png')
# sk = pcv.morphology.skeletonize(mask=img)
# cv2.imwrite('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_t0_ch0_loc2_enhance_skel.png', sk)

img = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_t0_ch0_loc2_enhance_skel.png')
# aop = skimage.morphology.area_opening(img, area_threshold=2)
aop = skimage.morphology.remove_small_objects(img, 32)
plt.imshow(aop)
plt.show()

exit()

def preprocess_samples(group):
    path_pref = f'/localhome/asa420/MIAL/data/live-cell-movies/{group}/Decon/'
    # new_pref = '/localhome/asa420/MIAL/data/confocal_movies/' + group + '/new_op_jul/preproc/'
    new_pref = f'/localhome/asa420/MIAL/data/live-cell-movies/{group}/Decon/'

    for i in range(2, 3):
        os.makedirs(f'{new_pref}C{i}')
        for j in range(100):
            img = imageio.imread(f'{path_pref}Series{i:03d}_decon_converted/files/Series{i:03d}_decon_converted_t{j:02d}_ch00.tif')

            std_img = ((img) / (img.max() - img.min())) * 255
            aop = skimage.morphology.area_opening(std_img, area_threshold=2)
            erod = skimage.morphology.erosion(aop)
            aop = skimage.morphology.area_opening(erod, area_threshold=2)
            cl = skimage.morphology.area_closing(aop, area_threshold=32)
            aop = skimage.morphology.area_opening(cl, area_threshold=2)
            loc = threshold_local(aop, 3)
            loc = threshold_local(loc, 3)
            cv2.imwrite(f'{new_pref}Series{i:03d}_decon_converted/new_op_sept/preproc/C{i}/C{i}_decon_t0{j:02d}_ch00_proc.png', loc)


preprocess_samples('COSKDEL')


