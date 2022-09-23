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


def get_if_corr(img1_patch_vals, img2_patch_vals):
    """

    @param img1_patch_vals:
    @param img2_patch_vals:
    @return: inter-frame correlation value for each patch
    """
    for i in range(len(img1_patch_vals)):
        valnum = np.corrcoef(img1_patch_vals[i], img2_patch_vals[i])
        return valnum[0, 1]


def ssim_patch(img1_patch_vals, img2_patch_vals):
    for i in range(len(img1_patch_vals)):
        ssim = metrics.structural_similarity(np.array(img1_patch_vals[i]), np.array(img2_patch_vals[i]))
        print(ssim)
        break


def cos_sim():
    for i in range(len(img1_patch_vals)):
        cosim = 1 - spatial.distance.cosine((img1_patch_vals[i]), (img2_patch_vals[i]))
        print(cosim)
        break

# cos_sim()

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

# def mutual_information(hgram):
#     pxy = hgram / float(np.sum(hgram))
#     px = np.sum(pxy, axis=1) # marginal for x over y
#     py = np.sum(pxy, axis=0) # marginal for y over x
#     px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
#     # Now we can do the calculation using the pxy, px_py 2D arrays
#     nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
#     return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def nmi():
    for i in range(len(img1_patch_vals)):
        nmi_val = mutual_information_2d((img1_patch_vals[i]), (img2_patch_vals[i]))
        print(nmi_val)
        break


# def cross_corr():
#     for i in range(len(img1_patch_vals)):
#         n1 = np.linalg.norm(img1_patch_vals[i])
#         n1_data = img1_patch_vals[i] / n1
#         n2 = np.linalg.norm(img2_patch_vals[i])
#         n2_data = img2_patch_vals[i] / n2
#         cc = np.correlate(n1_data, n2_data, mode='full')
#         print(cc)
#         break

# cross_corr()

# for each in img1_patch_vals:
#     print(np.var(each))


# def norm_cc():
#     for i in range(len(img1_patch_vals)):
#         i1 = np.array(img1_patch_vals[i])
#         i2 = np.array(img2_patch_vals[i])
#         dist_euclidean = np.sqrt(sum((i1 - i2)**2)) / i1.size
#         dist_manhattan = sum(abs(i1 - i2)) / i1.size
#         dist_ncc = sum( (i1 - np.mean(i1)) * (i2 - np.mean(i2)) ) / (
#         (i1.size - 1) * stdev(i1) * stdev(i2) )
#         print(dist_ncc)
#         break



def norm_cc():
    for i in range(len(img1_patch_vals)):
        i1 = np.array(img1_patch_vals[i])
        i2 = np.array(img2_patch_vals[i])
        print(scipy.signal.correlate(i1, i2))
        break


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





# i1 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t000_ch00.tif')
# i2 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t001_ch00.tif')
#
# i1 = (i1 - i1.min())/(i1.max() - i1.min())
# i2 = (i2 - i2.min())/(i2.max() - i2.min())



# patch1_vals -> list of intensity values per patch

def dil_junctions():
    global brpts, newps
    prefix = '/localhome/asa420/MIAL/data/confocal_movies/'
    for series_num in range(1, 2):
        newps = junction_flow(prefix + 'ATL/new_op_jul/ATL_mean_proj/A%s_mean.png' % f'{series_num}')
    return newps


newps = dil_junctions()


def get_junc_patches(newps, img):
    img_patches = []
    for num, coordinate in enumerate(newps):
        x, y = newps[num]
        coord_vals = [img[x-1,y], img[x+1,y], img[x,y], img[x,y-1], img[x,y+1], img[x-1,y-1], img[x-1,y+1], img[x+1,y-1], img[x+1,y+1]]
        img_patches.append(coord_vals)
    return img_patches


def get_group_IF_correlation():

    for num_series in range(27):

    ATL1_list = []
    for i in range(98):
        i1 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0%s_ch00.tif'%f'{i:02d}')
        i2 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0%s_ch00.tif'%f'{i+1:02d}')

        i1 = (i1 - i1.min())/(i1.max() - i1.min())
        i2 = (i2 - i2.min())/(i2.max() - i2.min())

        img1_patch_vals = get_junc_patches(newps, i1)
        img2_patch_vals = get_junc_patches(newps, i2)

        # print(img1_patch_vals)
        # print(img2_patch_vals)

        # print(len(img1_patch_vals))
        # print(img1_patch_vals)

        # val = get_if_corr(img1_patch_vals, img2_patch_vals)
        l = []
        for i in range(len(img1_patch_vals)):
            valnum = np.corrcoef(img1_patch_vals[i], img2_patch_vals[i])
            l.append(valnum[0, 1])

        ATL1_list.append(l)
    return ATL1_list


atl_list = get_group_IF_correlation()
sns.distplot(atl_list)
plt.show()

# valnum = np.corrcoef(img1, im2)
# corr_vals.append(valnum[0, 1])



exit()