"""
Image intensity histogram
To check the variation in intensity in movies
"""

import imageio
import matplotlib.pyplot as plt
import glob
import numpy as np
import os

def hist_creator(group, num):
    if group == 'climp':
        dirpath = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELCLIMP/'
    elif group == 'ctrl':
        dirpath = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/'
    else:
        dirpath = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/'

    prefix = f'{dirpath}Decon/Series0{num:02d}_decon_converted/std/'
    histprf = f'{dirpath}Decon/Series0{num:02d}_decon_converted/hist/'

    if not os.path.exists(histprf):
        os.makedirs(histprf)

    for i in range(100):
        fname = f'{prefix}Series0{num:02d}_decon_converted_t{i:02d}_ch00_std.png'
        img = imageio.imread(fname)
        val = img.flatten()
        # v = val[np.where(val>0)]
        plt.figure()
        b, bins, patches = plt.hist(val, 64)
        plt.xlim([0, 255])
        plt.ylim([0, 10000])
        # plt.show()
        plt.title(f'STED_Control_Series0{num:02d}_t{i:02d}_ch00')
        plt.xlabel('Intensity')
        plt.ylabel('Num of pixels')
        plt.savefig(f'{histprf}Series0{num:02d}_decon_converted_t{i:02d}_hist.png', bbox_inches='tight')

hist_creator('climp', 5)
