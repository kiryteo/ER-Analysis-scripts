import os
import glob
import imageio
import numpy as np
import scipy.io
from scipy import stats
from plantcv import plantcv as pcv

home = os.path.expanduser('~')
# path = home + '/MIAL/live-cell-movies/COSKDELRTN/COSKDELRTN/Decon/'
path = home + '/Desktop/RTN/skel/'

def Agg(path):
    for i in range(1, 30):
        files = glob.glob(path + 'R%s/*'%(f'{i}'))
        imgstack = []
        img_mean = np.zeros((128, 128))
        for each in files:
            img = imageio.imread(each)
            img_mean += img
            imgstack.append(img)
        pcv.print_image(img=img_mean/len(files), filename=home + '/Desktop/RTN/' + 'R%s-mean.png'%(f'{i}'))
        # imageio.imwrite(home + '/Desktop/ATL/' + 'A%s-mean.png'%(f'{i}'), img_mean/len(files))
        new = np.stack(imgstack, axis=2)
        maximg = np.amax(new, axis=2)
        pcv.print_image(img=maximg, filename=home + '/Desktop/RTN/' + 'R%s-max.png'%(f'{i}'))
        # imageio.imwrite(home + '/Desktop/ATL/' + 'A%s-max.png'%(f'{i}'), maximg)


def Aggregate(path):
    for i in range(12, 13):
        # dirc = glob.glob(path + 'Series%s_decon_converted/std/*'%(f'{i:03d}'))
        dirc = glob.glob(path + 'Series%s_decon_converted/brpts/*'%(f'{i:03d}'))
        imgstack = []
        img_avg = np.zeros((128, 128))

        for each in dirc:
            img = imageio.imread(each)
            img_avg += img
            imgstack.append(img)
        #
        # # imageio.imwrite('RTN-series%s-avg.png'%(i), img_avg/len(dirc))
        # imageio.imwrite(path + 'Ctrl-brpts%s-avg.png'%(i), img_avg/len(dirc))

        new = np.stack(imgstack, axis=2)
        maximg = np.amax(new, axis=2)
        scipy.io.savemat(path + 'RTN-brpts%s-max.mat'%(i), {'npy':maximg})
        # imageio.imwrite(path + 'Ctrl-brpts%s-max.png'%(i), maximg)

        # modeimg = stats.mode(new, axis=None)

        # medimg = np.median(new, axis=2)
        # imageio.imwrite('RTN-series%s-max.png'%(i), maximg)

        # imageio.imwrite('RTN-series%s-median.png'%(i), medimg)
        # imageio.imwrite(path + 'CLimp-brpts%s-median.png'%(i), medimg)

Agg(path)
# Aggregate()
