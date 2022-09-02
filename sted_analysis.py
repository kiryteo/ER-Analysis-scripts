import skimage
from skimage.filters import threshold_local
import os
import imageio
import cv2


def preprocess_samples(group):
    path_pref = '/localhome/asa420/MIAL/data/live-cell-movies/' + group + '/Decon/'
    # new_pref = '/localhome/asa420/MIAL/data/confocal_movies/' + group + '/new_op_jul/preproc/'
    new_pref = '/localhome/asa420/MIAL/data/live-cell-movies/' + group + '/Decon/'

    for i in range(2, 3):
        os.makedirs(new_pref + 'C%s' % f'{i}')
        for j in range(100):
            img = imageio.imread(path_pref + 'Series%s_decon_converted/files/Series%s_decon_converted_t%s_ch00.tif' % (f'{i:03d}', f'{i:03d}', f'{j:02d}'))
            std_img = ((img) / (img.max() - img.min())) * 255
            aop = skimage.morphology.area_opening(std_img, area_threshold=2)
            erod = skimage.morphology.erosion(aop)
            aop = skimage.morphology.area_opening(erod, area_threshold=2)
            cl = skimage.morphology.area_closing(aop, area_threshold=32)
            aop = skimage.morphology.area_opening(cl, area_threshold=2)
            loc = threshold_local(aop, 3)
            loc = threshold_local(loc, 3)
            cv2.imwrite(new_pref + 'Series%s_decon_converted/new_op_sept/preproc/C%s/C%s_decon_t0%s_ch00_proc.png' % (f'{i:03d}', f'{i}', f'{i}', f'{j:02d}'), loc)


preprocess_samples('COSKDEL')