import imageio

import imageio
from skimage import morphology
from skimage import segmentation
import numpy as np

from skimage.filters import threshold_otsu

from scipy import ndimage as ndi

# blur_img = 


import numpy as np

def blur_op(group):
    for frame in range(1, 17):
        try:
            # img = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/{group}/{group.lower()}{frame}_er_mean.png')

            img = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/skel/sted_{group.lower()}{frame}_proc_skel.png')

            n = 128

            data = []

            for i in range(0, n, 4):
                for j in range(0, n, 4):
                    data.append(np.mean(img[i:i+4, j:j+4]))

            data = np.array(data)
            data = data.reshape(32, 32)

            data = morphology.dilation(data)

            thr = threshold_otsu(data)

            data[data < thr] = 0.
            data[data >= thr] = 255.

            # imageio.imsave(f'/localhome/asa420/MIAL/data/sted-data/{group}/er_mean_blur/{group.lower()}{frame}_er_mean_blur.png', data)
            imageio.imsave(f'/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/blur_masks/sted_{group.lower()}{frame}_er_mean_blur_mask.png', data)
        except:
            pass


blur_op('climp')
blur_op('control')
blur_op('rtn')


