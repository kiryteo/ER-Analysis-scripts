import glob
import scipy
import scipy.io
import numpy as np

for each in glob.glob('/localhome/asa420/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/rtn/*'):
    fl = np.load(each, allow_pickle=True)
    pth = '/localhome/asa420/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/rtn-mat/'

# for RTN
    name = each.split('/')[-1].split('.')
    name = name[0] + name[1]
    name = pth + name + '.mat'
    scipy.io.savemat(name, {'npy': fl})
