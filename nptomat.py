import glob
import scipy
import scipy.io
import numpy as np
import os

home = os.path.expanduser('~')

for each in glob.glob(home + '/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/rtn/*'):
    fl = np.load(each, allow_pickle=True)
    pth = home + '/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/rtn-mat/'

# for RTN
    name = each.split('/')[-1].split('.')
    name = name[0] + name[1]
    name = pth + name + '.mat'
    scipy.io.savemat(name, {'npy': fl})
