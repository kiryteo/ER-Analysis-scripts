import skan
from skan import draw
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from plantcv import plantcv as pcv
import scipy
from scipy import ndimage

mean_img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A1_mean.png')

# thresh = threshold_otsu(mean_img)
# bin_img = mean_img > thresh
#
# skel = pcv.morphology.skeletonize(mask=bin_img)

# for i in range(100):
#     img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0%s_ch00.png'%f'{i:02d}')
#
#     # op = 0.5 * img + 0.5 * mean_img
#     # imageio.imwrite('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/A1_skel_mean_overlay/A1_decon_t0%s_ch00_skel.png'%f'{i:02d}', op)
#     fig, ax = plt.subplots()
#     op = draw.overlay_skeleton_2d(img, skel, dilate=0, axes=ax)
#     # plt.imshow(op)
#     # plt.show()
#     plt.savefig('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/A1_skel_mean_overlay/A1_decon_t0'
#                     '%s_ch00_skel.png'%f'{i:02d}', bbox_inches='tight', pad_inches=0)
#     plt.close()


# proc = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/preproc/A1/A1_decon_t000_ch00_proc.png')

skel = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t000_ch00_skel.png')

# proc_n = np.logical_not(skel)

op = ndimage.distance_transform_edt(skel)
plt.imshow(op)
plt.show()

