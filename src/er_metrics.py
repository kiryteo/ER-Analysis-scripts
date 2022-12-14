import imageio
import skimage
from skimage.filters import threshold_otsu, threshold_local
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import seaborn as sns
import sknw
import networkx as nx
import numpy as np
from skimage import metrics
import scipy
from scipy import spatial
from scipy import ndimage
from skimage.measure import label
from mpl_toolkits import mplot3d
from scipy.ndimage import uniform_filter1d
from skan import draw
import itertools
import os
import cv2
from scipy.spatial import cKDTree
# from matplotlib.patches import Circle
from skimage import draw
import matplotlib.colors as mcolors
from matplotlib import cm

max_val = 999
confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'


def junc_spread_comparison(group):
    """
    Compare the junctions obtained via initial projection Vs.
    per frame junction projection
    @return:
    """
    img = imageio.imread(
        f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/atl1_er_mean_proc.png')
    im1 = imageio.imread(f'{confocal_data_path}{group}/new_op_jul/junctions/A1_junc_mean.png')
    im2 = imageio.imread(f'{confocal_data_path}{group}/new_op_jul/junctions/A1_proc_junc_mean.png')

    fig, ax = plt.subplots()
    r, c = 1, 2

    plt.axis('off')
    fig.add_subplot(r, c, 1)
    plt.title('Junction frame projection', fontsize=14)
    plt.imshow(img)
    plt.imshow(im1)

    fig.add_subplot(r, c, 2)
    plt.title('Junction based on mean projection of ER input frames', fontsize=14)
    plt.imshow(img)
    plt.imshow(im2)

    # plt.title('Junction detection methods comparison (based on input)', fontsize=12)
    plt.show()


EPS = np.finfo(float).eps


def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """

    jh = np.histogram2d(x, y)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                            output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    return ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2))) / np.sum(jh * np.log(jh))) - 1 if normalized else (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) - np.sum(s2 * np.log(s2)))


def junc_patch_mean(newps, img):
    """

    @param newps: List of nodes
    @param img: ER input sample
    @return: mean value of the junction neighbourhood patch (3x3)
    """
    junc_patches = []
    dt = {}
    for num, coordinate in enumerate(newps):
        x, y = newps[num]
        if x > 1 and x < 126 and y > 1 and y < 126:
            coord_vals = [img[x - 1, y], img[x + 1, y], img[x, y], img[x, y - 1], img[x, y + 1], img[x - 1, y - 1],
                          img[x - 1, y + 1], img[x + 1, y - 1],
                          img[x + 1, y + 1]]  # , img[x+2, y], img[x-2, y], img[x, y+2], img[x, y-2]]
            junc_mean = np.mean(coord_vals)
            # if junc_mean > thr:
            junc_patches.append(junc_mean)
            dt[(x, y)] = junc_mean
    return junc_patches, dt


def nearest_neighbors_kd_tree(x, y, k) :
    tree =scipy.spatial.cKDTree(y)
    ordered_neighbors = tree.query(x, k)[1]
    nearest_neighbor = np.empty((len(x),), dtype=np.intp)
    nearest_neighbor.fill(-1)
    used_y = set()
    for j, neigh_j in enumerate(ordered_neighbors) :
        for k in neigh_j :
            if k not in used_y :
                nearest_neighbor[j] = k
                used_y.add(k)
                break
    return nearest_neighbor





def get_mean_patch_intensity(er_input, a, b):
    return (er_input[a, b] + er_input[a - 1, b - 1] + er_input[a + 1, b + 1] + er_input[a, b + 1] + er_input[a, b - 1] + er_input[a + 1, b - 1] + er_input[a - 1, b + 1] + er_input[a + 1, b] + er_input[a - 1, b]) / 9


def get_junc_patch(er_input, a, b):
    patch = np.zeros((3, 3))
    pa = patch.flatten()
    vals = [er_input[a - 1, b - 1], er_input[a, b - 1], er_input[a + 1, b - 1], er_input[a - 1, b], er_input[a, b],
            er_input[a + 1, b], er_input[a - 1, b + 1], er_input[a, b + 1], er_input[a + 1, b + 1]]
    for i, each in enumerate(vals):
        pa[i] = each
    patch = np.resize(pa, (3, 3))
    return patch


# ref_input = imageio.imread(f'{confocal_data_path}ATL/files/A1_decon_t000_ch00.tif')
# patch = get_junc_patch(ref_input, 3, 79)


def get_junc_lists(group, num_series):
    junc_to_analyze = junc_area_locator(group, num_series)
    newdt = {}
    for k, v in junc_to_analyze.items():
        v_new = []

        j = 0
        for i in range(100):
            if j < len(v):
                data = v[j]
                if data[1] == i:
                    v_new.append(data)
                    j = j + 1
                else:
                    v_new.append(max_val)
            else:
                v_new.append(max_val)

        newdt[k] = v_new
    return newdt


def junc_intensity_variation(group, num_series):
    newdt = get_junc_lists(group, num_series)

    if group == 'Control':
        ref_input = imageio.imread(f'{confocal_data_path}Control/files/img_{num_series}_decon_t000.tif')

    else:
        ref_input = imageio.imread(f'{confocal_data_path}{group}/files/{group[0]}{num_series}_decon_t000_ch00.tif')


    ref_input = (ref_input - ref_input.min()) / (ref_input.max() - ref_input.min())

    mean_val_dt = {}
    for k, v in newdt.items():
        mean_val_dt[k] = []
        a, b = k[0], k[1]
        m_val = get_mean_patch_intensity(ref_input, a, b)

        if v[0] != max_val:
            a, b = v[0][0][0], v[0][0][1]
            m_val = get_mean_patch_intensity(ref_input, a, b)
        mean_val_dt[k].append(m_val)
        for i in range(1, 100):
            if group == 'Control':
                er_input = imageio.imread(f'{confocal_data_path}Control/files/img_{num_series}_decon_t0{i:02d}.tif')

            else:
                er_input = imageio.imread(f'{confocal_data_path}{group}/files/{group[0]}{num_series}_decon_t0{i:02d}_ch00.tif')


            er_input = (er_input - er_input.min()) / (er_input.max() - er_input.min())
            if v[i] is not max_val:
                a, b = v[i][0][0], v[i][0][1]
                m_val = get_mean_patch_intensity(er_input, a, b)
            else:
                # print(a, b)
                # m_val = get_mean_patch_intensity(er_input, a, b)
                m_val = get_mean_patch_intensity(er_input, k[0], k[1])
            mean_val_dt[k].append(m_val)
    return mean_val_dt


mean_val_dt = junc_intensity_variation('RTN', 1)

for k1, v1 in mean_val_dt.items():
    # plt.title('Junction ref location: %s'%f'{k1}')
    # plt.plot(v1, linestyle='--', marker='o')
    # grd = np.gradient(v1)
    # plt.plot(grd, linestyle='--', marker='o')
    # plt.show()
    f = np.abs(np.fft.fft(v1))
    k = np.fft.fftfreq(len(v1))
    plt.plot(k)
    plt.show()
    break

# plt.show()

exit()


# for i in range(100):
#     img = imageio.imread(f'{confocal_data_path}ATL/files/A1_decon_t0%s_ch00.tif'%f'{i:02d}')
#     img = (img - img.min()) / (img.max() - img.min())
#
#     plt.imshow(img, cmap='gray')
#     plt.plot(79, 3, 'b.')
#     plt.show()




def per_patch_pixel_fourier(group, channel):
    grp_dict = {}
    nd = {}
    sldt = []

    for num_series in range(1, 2):
        mean_img = f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

        newps = junction_flow(mean_img)

        sl = []
        for frame in range(100):
            if group == 'Control':
                path = f'{confocal_data_path}{group}/files/img_{num_series}_decon_t0{frame:02d}.tif'

            else:
                path = f'{confocal_data_path}{group}/files/{group[0]}{num_series}_decon_t0{frame:02d}_ch0{channel}.tif'

            img = imageio.imread(path)
            img = (img - img.min()) / (img.max() - img.min())

            # thr = threshold_otsu(img)

            # im1_patch_vals = get_junc_patches(newps, img)
            im1_patch_vals = junc_patch_mean(newps, img)
            # print(max(im1_patch_vals))
            # print(min(im1_patch_vals))
            sl.append(im1_patch_vals)
                    # sldt.append(dt)

        # for each in sl:
        #     print(len(each))

        # slt shape: (9, num_patches, 100)
        slt = np.array(sl).T
        ht = slt.flatten()

        lval = np.quantile(ht, 0.05)
        hval = np.quantile(ht, 0.95)

        print(lval)
        print(hval)

        plt.hist(ht)
        plt.show()

        exit()

        #        sldt -> dict with key per junction and its mean patch value
        # sldt_ar = np.array(sldt).T
        # klist = sldt_ar[0].keys()

        # for k in klist:

        # for fr in range(len(slt)):
        #     plt.plot(slt[fr])
        #     plt.title('')
        #     plt.show()

        exit()

        # slt shape -> (93, 100)

    #     grp_dict[num_series] = slt
    #
    # return grp_dict


# atl_egfp = per_patch_pixel_fourier('ATL', 0)
per_patch_pixel_fourier('ATL', 0)

exit()


# atl_mc = per_patch_pixel_fourier('ATL', 1)
# climp_egfp = per_patch_pixel_fourier('Climp', 0)
# climp_mc = per_patch_pixel_fourier('Climp', 1)
# rtn_egfp = per_patch_pixel_fourier('RTN', 0)
# rtn_mc = per_patch_pixel_fourier('RTN', 1)
#

def inter_channel_correlation(c1, c2):
    data = []
    for (v1, v2) in zip(c1.values(), c2.values()):
        data.extend(np.corrcoef(mv1, mv2)[0, 1] for mv1, mv2 in zip(v1, v2))
    return data


atl_data = inter_channel_correlation(atl_egfp, atl_mc)
climp_data = inter_channel_correlation(climp_egfp, climp_mc)
rtn_data = inter_channel_correlation(rtn_egfp, rtn_mc)

sns.distplot(atl_data, label='ATL')
sns.distplot(climp_data, label='Climp')
sns.distplot(rtn_data, label='RTN')
plt.legend()
plt.title('Per patch correlation coefficient between EGFP and mCherry channels', fontsize=16)
plt.xlabel('Correlation coefficient', fontsize=12)
plt.show()

exit()

# for p in range(5):
#     grp_list.extend(uniform_filter1d(slt[p], 5))

# return  grp_list
# for p in range(5):
#     plt.plot(uniform_filter1d(slt[p], 5))
#
# plt.xlabel("Timesteps", fontsize=16)
# plt.title('%s condition Series 1, mean patch intensity'%f'{group}')
# plt.show()
#
#     ssl = []
#     for each in slt:
#         ssl.append(np.var(each))
#
#     grp_list.extend(ssl)
#
# return grp_list
# print(ssl.max())
# print(max(ssl), len(ssl))
# sns.displot(ssl, kind='kde', bw_adjust=0.25)

# plt.xlabel('Timestep', fontsize=16)
# plt.ylabel()
# plt.title()
# plt.show()
# exit()

# plt.ylabel('Intensity values', fontsize=16)
# plt.xlabel('Timestep', fontsize=16)
#
# plt.title('Intensity values per patch in RTN Series 1 movie (total 93 patches)')
# plt.show()
# exit()


# Obtain variance per pixel in the patch
# over 100 frames so we get 9 values per patch and then
# get the index of dispersion per patch

# ser_list = []
# for junc_num, junc in enumerate(slt):
#     ser_list.append(np.var(junc) / np.mean(junc))

# print(len(ser_list))
# for tf in range(len(sl)):
#     # a = slt.T[patch_num]
#     a = slt[tf]
#
#     # print(a.shape)
#     # exit()
#     ser_list.append(np.var(a) / np.mean(a))
# sns.distplot(a, hist=False)
# plt.plot(a)

# ser_list.append(np.abs(np.fft.fft(a)))
# plt.xlabel()

# label = 'movie num: ' + str(num_series)
# plt.plot(ser_list, label=label)
# # grp_list.extend(ser_list)
# plt.ylabel('Index of dispersion values', fontsize=16)
# plt.xlabel('Timestep', fontsize=16)
# plt.title('Index of dispersion for all patches (junctions) in a movie over time - RTN group', fontsize=20)
# plt.legend()
# plt.show()
# print(grp_list)
# return grp_list


atl_list = per_patch_pixel_fourier('ATL', 0)
climp_list = per_patch_pixel_fourier('Climp', 0)
ctrl_list = per_patch_pixel_fourier('Control', 0)
rtn_list = per_patch_pixel_fourier('RTN', 0)

# print(atl_list)
import pandas as pd

df = pd.DataFrame()

df['patch_intensity_vals'] = pd.Series(atl_list + climp_list + ctrl_list + rtn_list)
df['group'] = pd.Series()

exit()


# ATL_list = per_patch_pixel_fourier('ATL', 0)
# Climp_list = per_patch_pixel_fourier('Climp', 0)
# Ctrl_list = per_patch_pixel_fourier('Control', 0)
# RTN_list = per_patch_pixel_fourier('RTN', 0)
#
# sns.distplot(ATL_list, hist=False, label='ATL')
# sns.distplot(Climp_list, hist=False, label='Climp')
# sns.distplot(Ctrl_list, hist=False, label='Control')
# sns.distplot(RTN_list, hist=False, label='RTN')
#
# plt.legend()
# plt.xlabel('Variance')
# plt.title('')
# plt.show()

# exit()

def per_patch_pixel_variance(group):
    grp_list = []
    for num_series in range(1, 25):
        mean_img = f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

        newps = junction_flow(mean_img)

        sl = []

        for frame in range(100):
            if group == 'Control':
                path = f'{confocal_data_path}{group}/files/img_{num_series}_decon_t0{frame:02d}.tif'

            else:
                path = f'{confocal_data_path}{group}/files/{group[0]}{num_series}_decon_t0{frame:02d}_ch00.tif'

            img = imageio.imread(path)
            img = (img - img.min()) / (img.max() - img.min())

            im1_patch_vals = get_junc_patches(newps, img)
            sl.append(im1_patch_vals)

        # slt shape: (9, num_patches, 100)
        slt = np.array(sl)

        for patch_num in range(len(sl[0])):
            a = slt[:, patch_num, :]
        vl = [np.var(each) for each in a.T]
        ser_list = [np.var(vl) / np.mean(vl)]
        grp_list.extend(ser_list)

    return grp_list


# atl_list = per_patch_pixel_variance('ATL')
# climp_list = per_patch_pixel_variance('Climp')
# ctrl_list = per_patch_pixel_variance('Control')
# rtn_list = per_patch_pixel_variance('RTN')

# print(atl_list)
# print(len(atl_list[0]))
# sns.boxplot(atl_list)
# sns.boxplot(climp_list)
# sns.boxplot(ctrl_list)
# sns.boxplot(rtn_list)
# plt.show()

exit()


def per_patch_if_corr():
    grp_list = []
    for num_series in range(1, 25):
        mean_img = f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

        newps = junction_flow(mean_img)

        sl = []

        for frame in range(100):
            if group == 'Control':
                path = f'{confocal_data_path}{group}/files/img_{num_series}_decon_t0{frame:02d}.tif'

            else:
                path = f'{confocal_data_path}{group}/files/{group[0]}{num_series}_decon_t0{frame:02d}_ch00.tif'

            img = imageio.imread(path)
            img = (img - img.min()) / (img.max() - img.min())

            im1_patch_vals = get_junc_patches(newps, img)
            sl.append(im1_patch_vals)

            # slt shape: (9, num_patches, 100)
            slt = np.array(sl)

            # Obtain variance per pixel in the patch
            # over 100 frames so we get 9 values per patch and then
            # get the index of dispersion per patch

            ser_list = []
            for patch_num in range(len(sl[0])):
                a = slt[:, patch_num, :]
                vl = [np.var(each) for each in a.T]
                ser_list.append(np.var(vl) / np.mean(vl))

            grp_list.extend(ser_list)

    return grp_list


# op = np.abs(np.fft.fft(a))

# print(op)
# plt.plot(op)
# plt.show()

# print(a)
# print(im1_patch_vals)
# im1_patch_vals[1] = np.reshape(im1_patch_vals[1], (3,3))
# # print(im1_patch_vals[0])
# print(len(im1_patch_vals))
# print(len(im1_patch_vals[0]))
#
# op = np.abs(np.fft.fft2(im1_patch_vals[1]))
# # plt.imshow(op)
# plt.plot(op)
# plt.show()

exit()


def get_group_dif(group, metric):
    grp_list = []
    global met_val
    prefix = confocal_data_path
    if group == 'Control':
        for series_num in range(1, 25):
            newps = junction_flow(f'{confocal_data_path}Control/new_op_jul/Ctrl_mean_proj/Ct{series_num}_mean.png')

            ser_list = []
            for i in range(99):
                i1 = imageio.imread(f'{confocal_data_path}Control/files/img_{series_num}_decon_t0{i:02d}.tif')

                i2 = imageio.imread(f'{confocal_data_path}Control/files/img_{series_num}_decon_t0{i + 1:02d}.tif')


                i1 = (i1 - i1.min()) / (i1.max() - i1.min())
                i2 = (i2 - i2.min()) / (i2.max() - i2.min())

                img1_patch_vals = get_junc_patches(newps, i1)
                img2_patch_vals = get_junc_patches(newps, i2)

                l = []
                for i in range(len(img1_patch_vals)):
                    if metric == 'ssim':
                        met_val = metrics.structural_similarity(np.array(img1_patch_vals[i]),
                                                                np.array(img2_patch_vals[i]))
                    elif metric == 'cos_sim':
                        met_val = 1 - spatial.distance.cosine((img1_patch_vals[i]), (img2_patch_vals[i]))
                    elif metric == 'norm_cc':
                        i1 = np.array(img1_patch_vals[i])
                        i2 = np.array(img1_patch_vals[i])
                        met_val = scipy.signal.correlate(i1, i2)
                    elif metric == 'nmi':
                        met_val = mutual_information_2d(img1_patch_vals[i], img2_patch_vals[i])
                    elif metric == 'IF_corr':
                        valnum = np.corrcoef(img1_patch_vals[i], img2_patch_vals[i])
                        met_val = valnum[0, 1]

                    l.append(met_val)
                ser_list.extend(l)
            grp_list.extend(ser_list)
    else:
        for series_num in range(1, 25):
            newps = junction_flow(f'{confocal_data_path}{group}/new_op_jul/{group}_mean_proj/{group[0]}{series_num}_mean.png')

            ser_list = []
            for i in range(98):
                i1 = imageio.imread(f'{confocal_data_path}{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch00.tif')

                i2 = imageio.imread(f'{confocal_data_path}{group}/files/{group[0]}{series_num}_decon_t0{i + 1:02d}_ch00.tif')


                i1 = (i1 - i1.min()) / (i1.max() - i1.min())
                i2 = (i2 - i2.min()) / (i2.max() - i2.min())

                img1_patch_vals = get_junc_patches(newps, i1)
                img2_patch_vals = get_junc_patches(newps, i2)

                l = []
                for i in range(len(img1_patch_vals)):
                    if metric == 'ssim':
                        met_val = metrics.structural_similarity(np.array(img1_patch_vals[i]),
                                                                np.array(img2_patch_vals[i]))
                    elif metric == 'cos_sim':
                        met_val = 1 - spatial.distance.cosine((img1_patch_vals[i]), (img2_patch_vals[i]))
                    elif metric == 'norm_cc':
                        i1 = np.array(img1_patch_vals[i])
                        i2 = np.array(img1_patch_vals[i])
                        met_val = scipy.signal.correlate(i1, i2)
                    elif metric == 'nmi':
                        met_val = mutual_information_2d(img1_patch_vals[i], img2_patch_vals[i])
                    elif metric == 'IF_corr':
                        valnum = np.corrcoef(img1_patch_vals[i], img2_patch_vals[i])
                        met_val = valnum[0, 1]

                    l.append(met_val)
                ser_list.extend(l)
            grp_list.extend(ser_list)

    return grp_list


rtn = get_group_dif('RTN', 'IF_corr')
atl = get_group_dif('ATL', 'IF_corr')
climp = get_group_dif('Climp', 'IF_corr')
ctrl = get_group_dif('Control', 'IF_corr')

import pandas as pd

df = pd.DataFrame()
df['IF_correlation_vals'] = pd.Series(np.concatenate((atl, climp, ctrl, rtn)))
a = ['ATL'] * len(atl)
cl = ['Climp'] * len(climp)
ct = ['Control'] * len(ctrl)
r = ['RTN'] * len(rtn)
l2 = pd.Series(np.concatenate((a, cl, ct, r)))
df['group'] = l2

# sns.distplot(rtn, hist=False, label='RTN')
# sns.distplot(atl, hist=False, label='ATL')
# sns.distplot(climp, hist=False, label='Climp')
# sns.distplot(ctrl, hist=False, label='Control')

sns.violinplot(data=df, x='IF_correlation_vals', y='group')

# plt.legend()
plt.title('Junction area (3x3 patch) intensity variation over time for all movies across groups')
# plt.xlabel('Interframe correlation between consecutive frames')

# atl = get_group_cos_sim('ATL')
# climp = get_group_cos_sim('Climp')
# sns.distplot(rtn)
# sns.distplot(atl)
# sns.distplot(climp)
# plt.xlim((0,1))
plt.show()

exit()

dt = atl_list[0] < np.quantile(atl_list[0], 0.95)

# sns.distplot(atl_list[0], hist=False)
# sns.distplot(atl_list[1], hist=False)
# sns.distplot(atl_list[2], hist=False)
# sns.distplot(atl_list[3], hist=False)
# sns.distplot(atl_list[4], hist=False)
# sns.distplot(atl_list[5], hist=False)
# sns.distplot(atl_list[6], hist=False)
# sns.distplot(atl_list[7], hist=False)
# sns.distplot(atl_list[8], hist=False)
sns.distplot(dt)
plt.show()

# valnum = np.corrcoef(img1, im2)
# corr_vals.append(valnum[0, 1])


exit()
