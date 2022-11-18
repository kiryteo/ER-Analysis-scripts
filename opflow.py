import numpy as np
import scipy
import scipy.io
import pandas as pd

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
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import skimage.io as io
from skimage import filters
from skimage.color import rgb2gray
from skimage.filters import window, difference_of_gaussians

def junc_mag_disp():
    img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/er_mean_proc/rtn1_er_mean_proc.png')
    plt.imshow(img, cmap='gray')

    # fl = imageio.imread('R1_opflow.png')
    fl = imageio.imread('/localhome/asa420/ER-Analysis-data/R1_opflow.png')

    im1 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junctions/R1_junc_mean.png')
    im2 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junctions/R1_proc_junc_mean.png')

    l1 = np.where(im1!=0)
    l2 = np.where(im2!=0)

    x1 = l1[0]
    y1 = l1[1]

    x2 = l2[0]
    y2 = l2[1]

    high_mag = np.where(fl==255)
    mid_mag = np.where(fl==np.unique(fl)[1])
    hflx = high_mag[0]
    hfly = high_mag[1]

    mflx = mid_mag[0]
    mfly = mid_mag[1]
    # print(img.max())
    # print(np.unique(img))
    # exit()

    plt.plot(y1, x1, 'o', markerfacecolor='None', markeredgecolor='blue')
    plt.plot(y2, x2, 'o', markerfacecolor='None', markeredgecolor='red')
    plt.plot(hfly, hflx, 'x', markerfacecolor='None', markeredgecolor='yellow')
    plt.plot(mfly, mflx, 'x', markerfacecolor='None', markeredgecolor='green')

    plt.show()


def plot_junc_opflow():
    l = np.zeros((128, 128))
    for i in range(99):
        f1 = scipy.io.loadmat('/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junc_opflow/R1_0_opfl.mat')
        data = f1['m']
        l = l + data

    # a = np.mean(l, axis=1)
    a = l/99

    # a = (a - a.min()) / (a.max() - a.min())
    # a = a * 255.
    # cv2.imwrite('R1_opflow.png', a)
    # imageio.imsave('R1_opfl.png', a)
    plt.imshow(a)
    plt.show()


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
    for each in files[:100]:
        data = scipy.io.loadmat(each)
        curl = data['c'].flatten()
        a.extend(curl)

    return a


def get_nbrhood_mean(input, x, y):
    return (input[x-1, y-1] + input[x-1, y] + input[x-1, y+1] + input[x, y-1] + input[x, y] + input[x, y+1] + input[x+1, y-1] + input[x+1, y] + input[x+1, y+1]) / 9



def flow_analysis_er(group, prop):
    """

    @param group: group data to analyze
    @param prop: curl or diveregence
    @return: ER images flow outputs
    """

    pref = '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/intensity_based_flow_%s/'%(f'{group}', f'{prop}')

    l = []
    grp_num = {'ATL':27, 'Climp':32, 'Control':32, 'RTN':30}
    grp_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}
    for series in range(1, grp_num[group]):
        for frame in range(99):
            prop_data = scipy.io.loadmat(pref + '%s_%s_opfl_%s.mat'%(f'{grp_pref[group]}{series}', f'{frame}', f'{prop}'))['c']
            img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junctions/%s/%s_decon_t0%s_ch00_junc.png'%(f'{group}', f'{grp_pref[group]}{series}', f'{grp_pref[group]}{series}', f'{frame:02d}'))
            n = np.where(img!=0)
##################################
            temp = []
            for x, y in zip(n[0], n[1]):
                if x < 127 and x > 0 and y < 127 and y > 0:
                    nbr_mean = get_nbrhood_mean(prop_data, x, y)
                    temp.append(nbr_mean)
                else:
                    continue
                # print(nbr_mean)
                # exit()
                # l.extend(nbr_mean)
            l.extend(temp)
#####################################
            # dt = np.abs(curl_data[n])
            # # print(dt.flatten().shape)
            # # print(dt.flatten())
            # # exit()
            # l.extend(dt.flatten())
    #####################################
    # plt.hist(l)
    # sns.histplot(l)
    # plt.show()
    return l


atl = np.abs(flow_analysis_er('ATL', 'curl'))
climp = np.abs(flow_analysis_er('Climp', 'curl'))
ctrl = np.abs(flow_analysis_er('Control', 'curl'))
rtn = np.abs(flow_analysis_er('RTN', 'curl'))

df = pd.DataFrame()
df['Values'] = pd.Series(np.concatenate((atl, climp, ctrl, rtn)))
# df['ids'] = pd.Series(np.concatenate((np.arange(1, len(atl)+1), np.arange(1, len(climp)+1), np.arange(1, len(ctrl)+1), np.arange(1, len(rtn)+1))))
df['Group'] = pd.Series(np.concatenate((['ATL'] * len(atl), ['Climp'] * len(climp), ['Control'] * len(ctrl), ['RTN'] * len(rtn))))
# #
sns.violinplot(data=df, y='Group', x='Values')
# sns.scatterplot(data=df, x='ids', y='Values', hue='Group', style='Group')

# sns.distplot(atl, label='ATL', hist=False)
# sns.distplot(climp, label='Climp', hist=False)
# sns.distplot(ctrl, label='Control', hist=False)
# sns.distplot(rtn, label='RTN', hist=False)

# plt.hist(atl, label='ATL')
# plt.hist(climp, label='Climp')
# plt.hist(ctrl, label='Control')
# plt.hist(rtn, label='RTN')
plt.title('Vector field curl based on optical flow between subsequent frames at junction locations (3x3 neighbourhood mean) in a movie across conditions', fontsize=16)
# plt.title('Vector field divergence based on optical flow between subsequent frames at junction locations (single pixel) in a movie across different conditions', fontsize=16)

plt.xlabel('Curl values', fontsize=14)
# plt.legend()
plt.show()
exit()


def flow_magnitude_analysis_er(group):
    """

    @param group: group data to analyze
    @return: ER images flow outputs
    """

    pref = '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/intensity_based_flow/'%(f'{group}')

    l = []
    grp_num = {'ATL':27, 'Climp':32, 'Control':32, 'RTN':30}
    grp_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}
    for series in range(1, grp_num[group]):
        for frame in range(99):
            curl_data = scipy.io.loadmat(pref + '%s_%s_opfl.mat'%(f'{grp_pref[group]}{series}', f'{frame}'))['m']
            img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junctions/%s/%s_decon_t0%s_ch00_junc.png'%(f'{group}', f'{grp_pref[group]}{series}', f'{grp_pref[group]}{series}', f'{frame:02d}'))
            n = np.where(img!=0)
            temp = []
            for x, y in zip(n[0], n[1]):
                if x < 127 and x > 0 and y < 127 and y > 0:
                    nbr_mean = get_nbrhood_mean(curl_data, x, y)
                    temp.append(nbr_mean)
                else:
                    continue
                # print(nbr_mean)
                # exit()
                # l.extend(nbr_mean)
            l.extend(temp)
            # dt = curl_data[n]
            # print(dt.flatten().shape)
            # print(dt.flatten())
            # exit()
            # l.extend(dt.flatten())

    # plt.hist(l)
    # sns.histplot(l)
    # plt.show()
    return l


atl = flow_magnitude_analysis_er('ATL')
# print(len(atl))
# print(len(atl[0]))
# exit()

climp = flow_magnitude_analysis_er('Climp')
ctrl = flow_magnitude_analysis_er('Control')
rtn = flow_magnitude_analysis_er('RTN')



df = pd.DataFrame()
df['Values'] = pd.Series(np.concatenate((atl, climp, ctrl, rtn)))
# df['ids'] = pd.Series(np.concatenate((np.arange(1, len(atl)+1), np.arange(1, len(climp)+1), np.arange(1, len(ctrl)+1), np.arange(1, len(rtn)+1))))
df['Group'] = pd.Series(np.concatenate((['ATL'] * len(atl), ['Climp'] * len(climp), ['Control'] * len(ctrl), ['RTN'] * len(rtn))))
# #
sns.boxplot(data=df, y='Group', x='Values')
# sns.scatterplot(data=df, x='ids', y='Values', hue='Group', style='Group')

# sns.distplot(atl, label='ATL', hist=False)
# sns.distplot(climp, label='Climp', hist=False)
# sns.distplot(ctrl, label='Control', hist=False)
# sns.distplot(rtn, label='RTN', hist=False)

# plt.hist(atl, label='ATL')
# plt.hist(climp, label='Climp')
# plt.hist(ctrl, label='Control')
# plt.hist(rtn, label='RTN')
plt.title('Optical Flow magnitude between consecutive frames at junction locations (3x3 neighbourhood mean)', fontsize=16)
plt.xlabel('Flow magnitude values', fontsize=14)
# plt.legend()
plt.show()
exit()



def curl_analysis(group):
    """

    @param group: group data to analyze
    @return: Binary junction images flow outputs
    """
    pref = '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/opflow_files/'%(f'{group}')

    l = []
    grp_num = {'ATL':27, 'Climp':32, 'Control':30, 'RTN':30}
    grp_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}
    for series in range(1, grp_num[group]):
        for frame in range(99):
            curl_data = scipy.io.loadmat(pref + '%s_%s_flow.mat'%(f'{grp_pref[group]}{series}', f'{frame}'))['c']
            img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junctions/%s/%s_decon_t0%s_ch00_junc.png'%(f'{group}', f'{grp_pref[group]}{series}', f'{grp_pref[group]}{series}', f'{frame:02d}'))
            n = np.where(img!=0)
            dt = curl_data[n]
            # print(dt.flatten().shape)
            # print(dt.flatten())
            # exit()
            l.extend(dt.flatten())

    # plt.hist(l)
    # sns.histplot(l)
    # plt.show()
    return l

rtn = curl_analysis('RTN')
atl = curl_analysis('ATL')
climp = curl_analysis('Climp')
ctrl = curl_analysis('Control')

plt.hist(atl, label='ATL')
plt.hist(climp, label='Climp')
plt.hist(ctrl, label='Control')
plt.hist(rtn, label='RTN')

plt.legend()
plt.show()
exit()

file = scipy.io.loadmat('/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junc_opflow/R1_0_opfl.mat')
img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junctions/R1/R1_decon_t000_ch00_junc.png')
n = np.where(img!=0)
dt = file['m']
dtt = dt[n]
plt.hist(np.abs(dtt))
plt.show()
exit()

group = 'RTN'
files = glob.glob('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/opflow_files/*'%f'{group}')

print(files[0])
data = scipy.io.loadmat(files[0])
print(data['c'].shape)

img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junctions/R8/R8_decon_t095_ch00_junc.png')
n = np.where(img!=0)
dt = data['c']
dtt = dt[n]
# print(dtt)
plt.hist(np.abs(dtt))
plt.show()
exit()

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