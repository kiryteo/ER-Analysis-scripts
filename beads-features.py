# get beads output

#start with matlab bwconncomp

# use watershed/ conn comp to label
#then get the props


import numpy as np
import glob
import os
import ks_multithresh
import cv2

home = os.path.expanduser('~')

# path to data
prefix = home + '/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/'

# define global names for dir names
# dirname = prefix + 'climp/'
# climp_beads_dir = prefix + 'climp-beads/'
# ctrl_beads_dir = prefix + 'control-beads/'
# rtn_beads_dir = prefix + 'rtn-beads/'

def get_beads(group, modality):

    # path to npy files with images for all modalities
    files = glob.glob(prefix + group + '/*')

    # path to output dir specific to the group
    beads_dir = prefix + group + '-beads/'

    for file in files:
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
        cv2.imwrite(fname, beads)

# get_beads('climp', 'confocal')
get_beads('climp', 'sted')
get_beads('climp', 'synth')
get_beads('control', 'confocal')
get_beads('control', 'sted')
get_beads('control', 'synth')
get_beads('rtn', 'confocal')
get_beads('rtn', 'sted')
get_beads('rtn', 'synth')
