import imageio
from plantcv import plantcv as pcv
import skimage
from skimage import measure
import glob
import numpy as np
import cv2
import pickle

er_files = glob.glob('/localhome/asa420/MIAL/data/selective_analysis/climp/climp_s1/files/*')
skel_files = glob.glob('/localhome/asa420/MIAL/data/selective_analysis/climp/climp_s1/skel/*')


def get_props(skel):
    brpts = pcv.morphology.find_branch_pts(skel_img=skel)
    dil_brpts = skimage.morphology.binary_dilation(brpts).astype('int')
    tubules = skel - dil_brpts
    tubules = (tubules == 255).astype('int')
    lab_tubules = skimage.measure.label(tubules)
    dil_tubules = skimage.morphology.dilation(lab_tubules)
    # coords = np.where(dil_tubules > 0)
    props = skimage.measure.regionprops(dil_tubules)

    return props


dirs_path = '/localhome/asa420/MIAL/data/selective_analysis/'

data = {}

# grp = ''

grp = 'climp'

for grp in ['climp', 'ctrl', 'rtn']:
    data = {}
    series_names = glob.glob(dirs_path + grp + '/*')
    for series in series_names:
        ser_files = glob.glob(series + '/skel/*')
        for file in ser_files:
            skel = imageio.imread(file)
            props = get_props(skel)
            data[file] = props
    dname = grp + '.pkl'
    with open(dname, 'wb') as fl:
        pickle.dump(data, fl)

# for i in range(1, len(np.unique(dil_tubules))):
#     a, b = np.where(dil_tubules==i)[0], np.where(dil_tubules==i)[1]
#     if len(a) > 10:
# print(er_data_std[a, b].shape)

# if len(a) > 10:
#     dt[i] = {}
#     dt[i]['X'] = a
#     dt[i]['Y'] = b

# import matplotlib.pyplot as plt
# plt.hist(l)
# plt.show()
