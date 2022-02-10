import os
import imageio
import glob
import numpy as np
from scipy import stats



# Series001_decon_converted


def Aggregate():
    path = '/localhome/asa420/MIAL/live-cell-movies/COSKDEL/COSKDEL/Decon/'
    for i in range(2, 17):
        # dirc = glob.glob(path + 'Series%s_decon_converted/std/*'%(f'{i:03d}'))
        dirc = glob.glob(path + 'Series%s_decon_converted/brpts/*'%(f'{i:03d}'))
        imgstack = []
        img_avg = np.zeros((128, 128))

        # print(dirc)
        for each in dirc:
            img = imageio.imread(each)
            img_avg += img
            imgstack.append(img)

        # imageio.imwrite('RTN-series%s-avg.png'%(i), img_avg/len(dirc))
        imageio.imwrite(path + 'Ctrl-brpts%s-avg.png'%(i), img_avg/len(dirc))

        new = np.stack(imgstack, axis=2)
        maximg = np.amax(new, axis=2)
        # modeimg = stats.mode(new, axis=None)
        # medimg = np.median(new, axis=2)
        # imageio.imwrite('RTN-series%s-max.png'%(i), maximg)
        imageio.imwrite(path + 'Ctrl-brpts%s-max.png'%(i), maximg)
        # print(np.array(modeimg))
        # imageio.imwrite('RTN-series%s-median.png'%(i), medimg)
        # imageio.imwrite(path + 'CLimp-brpts%s-median.png'%(i), medimg)

Aggregate()
