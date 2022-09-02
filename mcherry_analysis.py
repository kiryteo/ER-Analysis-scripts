import imageio
import skimage
import os
from skimage.filters import threshold_local
import cv2


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
            cv2.imwrite(new_pref + 'A%s/A%s_decon_t0%s_ch01_proc.png' % (f'{i}', f'{i}', f'{j:02d}'), loc)


preprocess_samples('ATL')

