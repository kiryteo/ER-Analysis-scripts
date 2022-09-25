import imageio
import skimage
from skimage.filters import threshold_otsu
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

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
              / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return mi


def junction_flow(mean_img):
    mean_proj_img = imageio.imread(mean_img)

    # perform thresholding + binarization + skel
    thresh = threshold_otsu(mean_proj_img)
    bin_img = mean_proj_img > thresh

    skel = pcv.morphology.skeletonize(mask=bin_img)

    # Build graph from the skeleton
    g = sknw.build_sknw(skel, iso=False)
    G = nx.Graph()

    G.add_nodes_from(g.nodes)
    G.add_edges_from(g.edges)

    nodes = g.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])

    # nx.draw_networkx(G, pos=pos, with_labels=True, node_size=10)
    degree_list = G.degree

    # get all the nodes with degree greater than 2
    newps = []
    for i, val in enumerate(degree_list):
        if val[1] > 2:
            newps.append(ps[i])

    # Dilate junctions and remove them to get individual tubules
    # dil_brpts = pcv.dilate(gray_img=brpts_img, ksize=3, i=1)

    return newps


def get_junction_image(newps):
    brpts_img = np.zeros((128, 128))
    for each in newps:
        brpts_img[each[0], each[1]] = 255.
    return brpts_img



# patch1_vals -> list of intensity values per patch

def dil_junctions():
    global newps
    prefix = '/localhome/asa420/MIAL/data/confocal_movies/'
    for series_num in range(1, 2):
        newps = junction_flow(prefix + 'ATL/new_op_jul/ATL_mean_proj/A%s_mean.png' % f'{series_num}')
    return newps


def get_junc_patches(newps, img):
    img_patches = []
    for num, coordinate in enumerate(newps):
        x, y = newps[num]
        if x > 1 and x < 126 and y > 1 and y < 126:
            coord_vals = [img[x-1,y], img[x+1,y], img[x,y], img[x,y-1], img[x,y+1], img[x-1,y-1], img[x-1,y+1], img[x+1,y-1], img[x+1,y+1]]#, img[x+2, y], img[x-2, y], img[x, y+2], img[x, y-2]]
            img_patches.append(coord_vals)
    return img_patches


mean_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A1_mean.png'
newps = junction_flow(mean_img)

sl = []
for i in range(99):
    img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0%s_ch00.tif'%f'{i:02d}')
    img = (img - img.min())/(img.max() - img.min())

    im1_patch_vals = get_junc_patches(newps, img)
    sl.append(im1_patch_vals)

slt = np.array(sl)

a = slt[:,0,0]

# print(a.shape)

op = np.abs(np.fft.fft(a))

print(op)
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
    prefix = '/localhome/asa420/MIAL/data/confocal_movies/'
    if group == 'Control':
        for series_num in range(1, 25):
            newps = junction_flow(prefix + 'Control/new_op_jul/Ctrl_mean_proj/Ct%s_mean.png'%f'{series_num}')
            ser_list = []
            for i in range(98):
                i1 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_%s_decon_t0%s.tif'%(f'{series_num}', f'{i:02d}'))
                i2 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_%s_decon_t0%s.tif'%(f'{series_num}', f'{i+1:02d}'))

                i1 = (i1 - i1.min())/(i1.max() - i1.min())
                i2 = (i2 - i2.min())/(i2.max() - i2.min())

                img1_patch_vals = get_junc_patches(newps, i1)
                img2_patch_vals = get_junc_patches(newps, i2)

                l = []
                for i in range(len(img1_patch_vals)):
                    if metric == 'ssim':
                        met_val = metrics.structural_similarity(np.array(img1_patch_vals[i]), np.array(img2_patch_vals[i]))
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
    else:
        for series_num in range(1, 25):
            newps = junction_flow(prefix + '%s/new_op_jul/%s_mean_proj/%s_mean.png' % (f'{group}', f'{group}',f'{group[0]}{series_num}'))
            ser_list = []
            for i in range(98):
                i1 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/files/%s_decon_t0%s_ch00.tif'%(f'{group}', f'{group[0]}{series_num}', f'{i:02d}'))
                i2 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/files/%s_decon_t0%s_ch00.tif'%(f'{group}', f'{group[0]}{series_num}', f'{i+1:02d}'))

                i1 = (i1 - i1.min())/(i1.max() - i1.min())
                i2 = (i2 - i2.min())/(i2.max() - i2.min())

                img1_patch_vals = get_junc_patches(newps, i1)
                img2_patch_vals = get_junc_patches(newps, i2)

                l = []
                for i in range(len(img1_patch_vals)):
                    if metric == 'ssim':
                        met_val = metrics.structural_similarity(np.array(img1_patch_vals[i]), np.array(img2_patch_vals[i]))
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