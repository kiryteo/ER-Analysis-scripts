import numpy as np
import scipy
import scipy.io

# f1 = scipy.io.loadmat('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/opflow_files/0_flow.mat')
#
# curl = f1['c'].flatten()
#
# print(type(curl))
#
import matplotlib.pyplot as plt
# plt.hist(curl)
# plt.show()

import glob

import numpy as np
import matplotlib.pyplot as plt

import skimage.io as io
from skimage import filters
from skimage.color import rgb2gray
from skimage.filters import window, difference_of_gaussians


def get_phase_correlation():
    im1 = rgb2gray(io.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t001_ch00.tif'))
    im2 = rgb2gray(io.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t002_ch00.tif'))

    im1w = im1*window('hann', im1.shape)
    im2w = im2*window('hann', im2.shape)

    f1 = np.fft.fft2(im1w)
    f2 = np.fft.fft2(im2w)

    cps = (f1*f2.conj()) / np.abs(f1*f2.conj())
    r = np.abs(np.fft.ifft2(cps))
    r = np.fft.fftshift(r)

    plt.imshow(r)
    plt.show()

    # [py,px] = np.argwhere(r==r.max())[0]
    #
    # cx,cy = 64,64
    # shift_x = cx - px
    # shift_y = cy - py
    #
    # print(f'Shift measured X:{shift_x}, Y:{shift_y}')


def get_curls(group):
    a = []

    files = glob.glob('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/opflow_files/*'%f'{group}')
    # print(files[0])
    for each in files:
        data = scipy.io.loadmat(each)
        curl = data['c'].flatten()
        a.extend(curl)

    return a

def get_div(group):
    a = []

    files = glob.glob('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/opflow_div/*'%f'{group}')
    # print(files[0])
    for each in files:
        data = scipy.io.loadmat(each)
        div = data['d'].flatten()
        a.extend(div)

    return a


## TODO: repair this function

def flow_mag_analysis():
    l = []
    # for i in range(99):
    #     data = scipy.io.loadmat('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/flow_mag/A1_%s_mag.mat'%f'{i}')
    #     l.append(data['mag'])
    #
    # l = np.array(l)
    # print(l.shape)
    # m = np.mean(l, axis=0)
    # # print(m.shape)
    # # plt.hist(m)
    # plt.imshow(m)
    # plt.show()
    dataX = scipy.io.loadmat('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/flow_mag/A1_10_magX.mat')
    dataY = scipy.io.loadmat('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/flow_mag/A1_10_magY.mat')
    mag = np.sqrt(dataX['Vx']**2 + dataY['Vy']**2)
    # print(mag.shape)

    plt.imshow(dataY['Vy'])
    # plt.imshow(mag)
    plt.show()

flow_mag_analysis()
exit()


def flow_curl_analysis():
    data = scipy.io.loadmat('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/opflow_curl/A1_0_flow.mat')
    print(data['c'])

# flow_curl_analysis()
# exit()


def plot_groups():
    atl = get_div('ATL')
    climp = get_div('Climp')
    ctrl = get_div('Control')
    rtn = get_div('RTN')

    import seaborn as sns

    # sns.distplot(atl, label='ATL')
    # sns.distplot(climp, label='Climp')
    # sns.distplot(ctrl, label='Control')
    # sns.distplot(rtn, label='RTN')
    plt.hist(atl, label='atl', alpha=0.25)
    plt.hist(climp, label='climp', alpha=0.25)
    plt.hist(ctrl, label='control', alpha=0.25)
    plt.hist(rtn, label='rtn', alpha=0.25)
    plt.suptitle('Vector Field divergence')
    plt.title('Distribution of vector field divergence based on optical flow between subsequent frames in a movie across different conditions')
    plt.xlabel('Divergence values')
    plt.legend()
    plt.show()

plot_groups()
exit()