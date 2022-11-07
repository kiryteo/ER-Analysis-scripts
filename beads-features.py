"""
Extract high-intensity bead-like structures
from the ER networks.
"""

#start with matlab bwconncomp

# use watershed/ conn comp to label
#then get the props


import numpy as np
import glob
import os
import ks_multithresh
import cv2
import scipy.io

home = os.path.expanduser('~')

# path to data
prefix = home + '/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/'

# prefix = home + '/MIAL/data-Feb4-Guang/RTN2'

def get_beads(group, modality):

    # path to npy files with images for all modalities
    files = glob.glob(prefix + group + '/*')

    # path to output dir specific to the group
    beads_dir = prefix + group + '-beads/'

    for file in files:
        beads, fname = beads_multithresh(file, beads_dir)

        get_beads_images(beads, fname)
        matname = fname.split('.')[0] + '.mat'

        get_beads_mat(beads, matname)

def beads_multithresh(file, beads_dir):
    np_data = np.load(file)
    name = file.split('/')[-1].split('.')[0]

    if modality == 'confocal':
        data = np_data[0]
        fname = beads_dir + name + '_beads-conf.png'
    elif modality == 'sted':
        data = np_data[1]
        fname = beads_dir + name + '_beads-sted.png'
    elif modality == 'synth':
        data = np_data[2]
        fname = beads_dir + name + '_beads-synth.png'

    beads = ks_multithresh.test_thresholds(data)
    return beads, fname

def get_beads_images(beads, fname):
    cv2.imwrite(fname, beads)

def get_beads_mat(beads, matname):
    scipy.io.savemat(matname, {'npy': beads})

# get_beads('climp', 'confocal')
# get_beads('climp', 'sted')
# get_beads('climp', 'synth')
# get_beads('control', 'confocal')
# get_beads('control', 'sted')
# get_beads('control', 'synth')
# get_beads('rtn', 'confocal')
# get_beads('rtn', 'sted')
# get_beads('rtn', 'synth')
