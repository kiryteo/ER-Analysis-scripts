# get beads output

#start with matlab bwconncomp

# use watershed/ conn comp to label
#then get the props


import numpy as np
import glob
import os
import ks_multithresh
import cv2

prefix = '/localhome/asa420/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/'

dirname = prefix + 'climp/'
climp_beads_dir = prefix + 'climp-beads/'
ctrl_beads_dir = prefix + 'control-beads/'
rtn_beads_dir = prefix + 'rtn-beads/'

files = glob.glob(dirname + '*')

for file in files:
    data = np.load(file)
    conf = data[0]
    sted = data[1]
    synth = data[2]

    name = file.split('/')[-1].split('.')[0]
    conf_name = name + '_beads-conf.png'
    sted_name = name + '_beads-sted.png'
    synth_name = name + '_beads-synth.png'

    conf_beads = ks_multithresh.test_thresholds(conf)
    sted_beads = ks_multithresh.test_thresholds(sted)
    synth_beads = ks_multithresh.test_thresholds(synth)

    cv2.imwrite(conf_name, conf_beads)
    cv2.imwrite(sted_name, sted_beads)
    cv2.imwrite(synth_name, synth_beads)
