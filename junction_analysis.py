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



def junc_spread_comparison():
    """
    Compare the junctions obtained via initial projection Vs.
    per frame junction projection
    @return:
    """
    img = imageio.imread(
        '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc.png')
    im1 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junctions/A1_junc_mean.png')
    im2 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junctions/A1_proc_junc_mean.png')

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


def junc_spread_display(group, num_series):
    im1 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junctions/%s_junc_mean.png' % (
        f'{group}', f'{group[0]}{num_series}'))
    im2 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junctions/%s_proc_junc_mean.png' % (
        f'{group}', f'{group[0]}{num_series}'))

    l1 = np.where(im1 != 0)
    l2 = np.where(im2 != 0)

    x1 = l1[0]
    y1 = l1[1]

    x2 = l2[0]
    y2 = l2[1]

    fl = imageio.imread('R1_opflow.png')
    stable_pts = np.where(fl == 255)
    moving_pts = np.where()

    # n1 = []
    # for a,b in zip(x1, y1):
    #     n1.append([a, b])
    #
    # n2 = []
    # for a,b in zip(x2, y2):
    #     n2.append([a, b])

    # print(n1)
    # print(n2)
    img = imageio.imread(
        '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/er_mean_proc/%s_er_mean_proc.png' % (
            f'{group}', f'{group.lower()}{num_series}'))
    plt.imshow(img)
    # plt.scatter(y1, x1, color='red')
    # plt.scatter(y2, x2, color='blue')

    plt.plot(y1, x1, 'o', markerfacecolor='None', markeredgecolor='blue')
    plt.plot(y2, x2, 'o', markerfacecolor='None', markeredgecolor='red')

    plt.suptitle(
        '%s series %s, Blue spots: junctions from a), Red spots: junctions from b)' % (f'{group}', f'{num_series}'))
    plt.title('a) Projection from all junction frames b) Projection at the initial stage (ER input)')
    plt.show()

    # plt.plot(im1, 'r.')
    # plt.plot(im2, 'b.')
    # plt.show()


# fig, ax = plt.subplots()
# r, c = 1, 2
#
# ax.add_subplot(r,c,1)
# plt.imshow(im1)

# junc_spread_display('Climp', 2)
# exit()

def skel_to_graph(skel):
    g = sknw.build_sknw(skel, iso=False)
    G = nx.Graph()

    G.add_nodes_from(g.nodes)
    G.add_edges_from(g.edges)

    nodes = g.nodes()
    degree_list = G.degree
    return nodes, degree_list

def junction_flow(mean_img):
    """

    @param mean_img: Input mean projection skel image (ndarray, binary)
    @return: list of Nodes with degree > 2
    """

    # if mean_img is obtained via skel mean projection
    # step, then use otsu else directly read the image

    # mean_proj_img = imageio.imread(mean_img)
    #
    # # perform thresholding + binarization + skel
    # thresh = threshold_otsu(mean_proj_img)
    # bin_img = mean_proj_img > thresh

    # skel = pcv.morphology.skeletonize(mask=mean_proj_img)

    skel = imageio.imread(mean_img)

    # Build graph from the skeleton
    nodes, degree_list = skel_to_graph(skel)
    ps = np.array([nodes[i]['o'] for i in nodes])

    # get all the nodes with degree greater than 2
    newps = []
    for i, val in enumerate(degree_list):
        if val[1] > 2:
            newps.append(ps[i])

    # Dilate junctions and remove them to get individual tubules
    # dil_brpts = pcv.dilate(gray_img=brpts_img, ksize=3, i=1)

    return newps


def per_frame_junc_projection(group, num_series):
    junc_mean = np.zeros((128, 128))
    for i in range(100):
        img = imageio.imread(
            '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junctions/%s/%s_decon_t0%s_ch00_junc.png' % (
                f'{group}', f'{group[0]}{num_series}', f'{group[0]}{num_series}', f'{i:02d}'))
        junc_mean += img

    cv2.imwrite('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junctions/%s_junc_mean.png' % (
        f'{group}', f'{group[0]}{num_series}'), junc_mean / 100)


# for i in range(1, 30):
#     per_frame_junc_projection('RTN', i)
#
# exit()

def init_proc_projection(group, num_series):
    sk = imageio.imread(
        '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/er_mean_proc/%s_er_mean_proc_enhance_skel.png' % (
            f'{group}', f'{group.lower()}{num_series}'))

    nodes, degree_list = skel_to_graph(sk)
    ps = np.array([nodes[i]['o'] for i in nodes])

    # get all the nodes with degree greater than 2
    newps = []
    for j, val in enumerate(degree_list):
        if val[1] > 2:
            newps.append(ps[j])

    brpts_img = np.zeros((128, 128))
    # brpts_img[newps] = 1.
    for each in newps:
        brpts_img[each[0], each[1]] = 255.

    cv2.imwrite('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junctions/%s_proc_junc_mean.png' % (
        f'{group}', f'{group[0]}{num_series}'), brpts_img)


def mean_frame_validation(group, total_series):
    if group == 'Control':
        pref = 'Ct'
    else:
        pref = group[0]
    for num_ser in range(1, total_series + 1):
        os.makedirs(
            '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junctions/%s' % (f'{group}', f'{pref}{num_ser}'))
        for frame in range(100):
            sk = imageio.imread(
                '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/skel/%s/%s_decon_t0%s_ch00_skel.png' % (
                    f'{group}', f'{pref}{num_ser}', f'{pref}{num_ser}', f'{frame:02d}'))

            nodes, degree_list = skel_to_graph(sk)
            ps = np.array([nodes[i]['o'] for i in nodes])

            # get all the nodes with degree greater than 2
            newps = []
            for j, val in enumerate(degree_list):
                if val[1] > 2:
                    newps.append(ps[j])

            brpts_img = np.zeros((128, 128))
            for each in newps:
                brpts_img[each[0], each[1]] = 255.

            imageio.imsave(
                '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junctions/%s/%s_decon_t0%s_ch00_junc.png' % (
                    f'{group}', f'{pref}{num_ser}', f'{pref}{num_ser}', f'{frame:02d}'), brpts_img)


# mean_frame_validation('RTN', 29)
# exit()


def get_junction_image(newps):
    """

    @param newps: List of nodes
    @return: Image with nodes -> 1, else 0
    """
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
    """

    @param newps: List of nodes
    @param img: ER input sample
    @return: junction neighbourhood patch (3x3)
    """
    img_patches = []
    for num, coordinate in enumerate(newps):
        x, y = newps[num]
        if x > 1 and x < 126 and y > 1 and y < 126:
            coord_vals = [img[x - 1, y], img[x + 1, y], img[x, y], img[x, y - 1], img[x, y + 1], img[x - 1, y - 1],
                          img[x - 1, y + 1], img[x + 1, y - 1],
                          img[x + 1, y + 1]]  # , img[x+2, y], img[x-2, y], img[x, y+2], img[x, y-2]]
            img_patches.append(coord_vals)
    return img_patches


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


def per_patch_variation(group, channel):
    """

    @param group: Select the condition for analysis
    @param channel: Select the protein channel for analysis
    @return: Metric output for variation over time
    """
    for num_series in range(1, 2):
        mean_img = '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/er_mean_proc/%s_er_mean_proc_enhance_skel.png' % (
            f'{group}', f'{group.lower()}{num_series}')
        newps = junction_flow(mean_img)

        sl = []

        for frame in range(100):
            if group == 'Control':
                path = '/localhome/asa420/MIAL/data/confocal_movies/%s/files/img_%s_decon_t0%s.tif' % (
                    f'{group}', f'{num_series}', f'{frame:02d}')
            else:
                path = '/localhome/asa420/MIAL/data/confocal_movies/%s/files/%s_decon_t0%s_ch0%s.tif' % (
                    f'{group}', f'{group[0]}{num_series}', f'{frame:02d}', f'{channel}')
            img = imageio.imread(path)
            img = (img - img.min()) / (img.max() - img.min())

            # im1_patch_vals = get_junc_patches(newps, img)
            im1_patch_vals = junc_patch_mean(newps, img)
            sl.append(im1_patch_vals)

        slt = np.array(sl)
        print(slt.shape)
        print(len(sl))
        print(len(sl[0]))
        exit()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        X = slt
        Y = slt.T


def patch_variation_viz():
    mean_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'

    newps = junction_flow(mean_img)
    sl = []
    dtl = []

    for i in range(100):
        path = '/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0%s_ch00.tif' % f'{i:02d}'
        img = imageio.imread(path)
        img = (img - img.min()) / (img.max() - img.min())

        im1_patch_vals, dt = junc_patch_mean(newps, img)
        sl.append(im1_patch_vals)
        dtl.append(dt)

    # slt = np.array(sl).T
    # print(slt.shape)

    coord_dt = {}
    for each in dtl:
        for k, v in each.items():
            if k in coord_dt.keys():
                coord_dt[k].append(v)
            else:
                coord_dt[k] = []
                coord_dt[k].append(v)

    return coord_dt


# coord_dt = patch_variation_viz()

def junc_intensity_plot_creator(coord_dt):
    for k, v in coord_dt.items():
        plt.plot(v)
        plt.xlabel('Timeframe')
        plt.ylabel('Intensity values')
        plt.title('ATL 1, Junction coordinates: %s intensity variation' % f'{k[1], k[0]}')
        plt.savefig('ATL_1_Junction_%s.png' % f'{k[1], k[0]}', bbox_inches='tight', pad_inches=0.2)
        plt.close()


# def nearest_neighbors_kd_tree(x, y, k) :
#     tree =scipy.spatial.cKDTree(y)
#     ordered_neighbors = tree.query(x, k)[1]
#     nearest_neighbor = np.empty((len(x),), dtype=np.intp)
#     nearest_neighbor.fill(-1)
#     used_y = set()
#     for j, neigh_j in enumerate(ordered_neighbors) :
#         for k in neigh_j :
#             if k not in used_y :
#                 nearest_neighbor[j] = k
#                 used_y.add(k)
#                 break
#     return nearest_neighbor


def seq_fourier_analysis(group, num_series):
    l = []
    for i in range(100):
        img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/files/%s_decon_t0%s_ch00.tif' % (
            f'{group}', f'{group[0]}{num_series}', f'{i:02d}'))
        stimg = (img - img.min()) / (img.max() - img.min())
        l.extend(stimg)

    l = np.reshape(l, (100, 128, 128))

    f = np.fft.fftn(l - np.mean(l))
    fabs = np.abs(f)
    fviz = np.fft.fftshift(fabs)

    # print(fviz.shape)
    # print(np.sum(fviz, axis=0).shape)
    k = []
    for each in fviz:
        k.append(np.sum(each))

    return k


# atl = seq_fourier_analysis('ATL', 1)
# climp = seq_fourier_analysis('Climp', 1)
# rtn = seq_fourier_analysis('RTN', 1)
#
# plt.plot(atl, label='ATL')
# plt.plot(climp, label='Climp')
# plt.plot(rtn, label='RTN')
# plt.legend()
# plt.show()
# exit()


def junc_area_locator(group, num_series):
    """

    @param group: group to be analyzed
    @param num_series: sequence number
    @return: dt, dictionary with matched junctions per reference junction
    """
    # mean_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'

    mean_img = '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/er_mean_proc/%s_er_mean_proc_enhance_skel.png' % (
        f'{group}', f'{group.lower()}{num_series}')

    # Get junction coordinates from projection frame
    newps = junction_flow(mean_img)

    nps = []
    for each in newps:
        nps.append([each[0], each[1]])

    nps = np.array(nps)
    # print(nps.shape)

    # sort the array for nearest neighbour matching per frame
    # nps_sorted is the mean projection (reference) junction list
    nps_sorted = sorted(nps, key=lambda t: t[0])

    dt = {}
    for frame in range(100):

        if group == 'Control':
            pref = 'Ct'
        else:
            pref = group[0]

        sk_img = '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/skel/%s/%s_decon_t0%s_ch00_skel.png' % (
            f'{group}', f'{pref}{num_series}', f'{pref}{num_series}', f'{frame:02d}')

        sk_newps = junction_flow(sk_img)

        sk_nps = []
        for each in sk_newps:
            sk_nps.append([each[0], each[1]])

        sk_nps = np.array(sk_nps)

        # sk_nps_sorted is the per frame junction list
        sk_nps_sorted = sorted(sk_nps, key=lambda t: t[0])

        # matching of junction candidates
        for elem in nps_sorted:
            for sk_elem in sk_nps_sorted:
                dst = np.sqrt((sk_elem[0] - elem[0]) ** 2 + (sk_elem[1] - elem[1]) ** 2)
                if dst < 3:
                    if (elem[0], elem[1]) in dt.keys():
                        dt[(elem[0], elem[1])].append([sk_elem, frame, dst])
                    else:
                        dt[(elem[0], elem[1])] = []
                        dt[(elem[0], elem[1])].append([sk_elem, frame, dst])

    return dt


def get_all_junc(group, num_series):
    """

    @param group: group to be analyzed
    @param num_series: sequence number
    @return: dt, dictionary with matched junctions per reference junction
    """
    # mean_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'

    mean_img = '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/er_mean_proc/%s_er_mean_proc_enhance_skel.png' % (
        f'{group}', f'{group.lower()}{num_series}')

    # Get junction coordinates from projection frame
    newps = junction_flow(mean_img)

    nps = []
    for each in newps:
        nps.append([each[0], each[1]])

    skdata = []
    for frame in range(100):

        if group == 'Control':
            pref = 'Ct'
        else:
            pref = group[0]

        sk_img = '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/skel/%s/%s_decon_t0%s_ch00_skel.png' % (
            f'{group}', f'{pref}{num_series}', f'{pref}{num_series}', f'{frame:02d}')

        sk_newps = junction_flow(sk_img)

        sk_nps = []
        for each in sk_newps:
            sk_nps.append([each[0], each[1]])

        # sk_nps = np.array(sk_nps)
        skdata.extend(sk_nps)

    return nps, skdata


def refine_junc_dt(dt, min_presence=50):
    dt_refined = {}
    for k, v in dt.items():
        if len(v) > min_presence:
            dt_refined[k] = v

    return dt_refined


def get_junction_types(nps, lab):
    label_vals = {}

    assigned_components = []

    for each in nps:
        if lab[each[0], each[1]] != 0:
            if (lab[each[0], each[1]]) in label_vals.keys():
                label_vals[(lab[each[0], each[1]])].append([each[0], each[1]])
                assigned_components.append(lab[each[0], each[1]])
            else:
                label_vals[(lab[each[0], each[1]])] = []
                label_vals[(lab[each[0], each[1]])].append([each[0], each[1]])
                assigned_components.append(lab[each[0], each[1]])
    return label_vals, assigned_components


def get_uncertain_junctions(lab, skdata, num_components, assigned_components):
    unassigned_components = [x for x in num_components if x not in assigned_components]

    unassigned_cc_dict = {}

    for each in skdata:
        cc_label = lab[each[0], each[1]]
        if cc_label != 0 and cc_label in unassigned_components:
            if cc_label in unassigned_cc_dict.keys():
                unassigned_cc_dict[(lab[each[0], each[1]])].append([each[0], each[1]])
            else:
                unassigned_cc_dict[(lab[each[0], each[1]])] = []
                unassigned_cc_dict[(lab[each[0], each[1]])].append([each[0], each[1]])

    return unassigned_cc_dict


def fuz_isolated_junctions(group, series_num):

    nps, skdata = get_all_junc(group, series_num)

    nps = np.array(nps)
    skdata = np.array(skdata)

    spread_img = np.zeros((128, 128))
    for each in skdata:
        spread_img[each[0], each[1]] = 255.

    lab = label(spread_img)

    num_components = np.unique(lab)

    label_vals, assigned_components = get_junction_types(nps, lab)

    unassigned_cc_dict = get_uncertain_junctions(lab, skdata, num_components, assigned_components)

    isolated_junc = []
    fuzzy_junc = []
    unknown_junc = []
    for k, v in label_vals.items():
        if k != 0:
            if len(v) == 1:
                isolated_junc.append(v[0])
            else:
                fuzzy_junc.append(v)

    for k, v in unassigned_cc_dict.items():
        unknown_junc.append(v)

    iso = np.array(isolated_junc)

    fuz = list(itertools.chain.from_iterable(fuzzy_junc))
    fuz = np.array(fuz)

    unk = list(itertools.chain.from_iterable(unknown_junc))
    unk = np.array(unk)

    # img = imageio.imread(
    #     '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc.png')

    for i in range(100):
        plt.axis('off')
        if group == 'Control':
            img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/files/img_%s_decon_t0%s.tif'%(f'{group}', f'{series_num}', f'{i:02d}'))
        else:
            img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/files/%s_decon_t0%s_ch00.tif'%(f'{group}', f'{group[0]}{series_num}', f'{i:02d}'))
        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img, cmap='gray')
        plt.plot(iso[:, 1], iso[:, 0], 'o', markerfacecolor='None', markeredgecolor='red')
        if len(fuz) > 0:
            plt.plot(fuz[:, 1], fuz[:, 0], 'o', markerfacecolor='None', markeredgecolor='blue')
        plt.plot(unk[:, 1], unk[:, 0], 'o', markerfacecolor='None', markeredgecolor='green')
        if group == 'Control':
            plt.savefig('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junc_types_movies/Ct%s_decon_t0%s_ch00.png'%(f'{group}', f'{series_num}', f'{i:02d}'), bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junc_types_movies/%s_decon_t0%s_ch00.png'%(f'{group}', f'{group[0]}{series_num}', f'{i:02d}'), bbox_inches='tight', pad_inches=0)
        plt.close()


# for i in range(1, 27):
#     fuz_isolated_junctions('ATL', i)

# for i in range(6, 32):
#     fuz_isolated_junctions('Control', i)

# for i in range(1, 32):
#     fuz_isolated_junctions('Climp', i)
#
# for i in range(24, 30):
#     fuz_isolated_junctions('RTN', i)
#
# exit()


def gmovie_creator(group, num_series):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('%s_s%s_junction_types.mp4'%(f'{group}', f'{num_series}'), fourcc, 1.5, (369, 369))
    if group == 'Control':
        pref = 'Ct'
    else:
        pref = group[0]
    for frame in range(100):
        img = cv2.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junc_types_movies/%s_decon_t0%s_ch00.png'%(f'{group}', f'{pref}{num_series}', f'{frame:02d}'))
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

#
# for i in range(1, 27):
#     gmovie_creator('ATL', i)
#
# for i in range(1, 32):
#     gmovie_creator('Climp', i)

# for i in range(23, 30):
#     gmovie_creator('RTN', i)

# for i in range(1, 32):
#     gmovie_creator('Control', i)
#
# exit()

from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.figure import figaspect

# w, h = figaspect(1)
#
# fig, ax = plt.subplots(figsize=(8,8))
#
# plt.show()
#
# exit()

# def forceAspect(ax,aspect=1):
#     im = ax.get_images()
#     extent = im[0].get_extent()
#     ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


vor = Voronoi(nps)

# fig, ax = plt.subplots()
fig = voronoi_plot_2d(vor)
# plt.figure(num=1, figsize=(8,8))

# plt.title('Voronoi for ATL Series 10, Blue points: Reference junctions, Red points: per frame junctions')
# img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/er_mean/%s_er_mean.png'%(f'{group}', f'{group.lower()}{num_series}'))

plt.plot(nps[:, 0], nps[:, 1], 'o', markerfacecolor='None', markeredgecolor='blue', mew=3)
plt.plot(skdata[:, 0], skdata[:, 1], 'o', markerfacecolor='None', markeredgecolor='red')
# plt.imshow()
# plt.gca().set_aspect(1)
# plt.axis('scaled')

# plt.savefig('Voro_ATL_10_new.png', bbox_inches='tight')

f = plt.gcf()
f.set_size_inches(8, 8)

plt.show()

exit()

# dt = junc_area_locator('Climp', 1)
# dt = refine_junc_dt(dt_init, 10)

# print(dt.keys())

# print(dt[(31, 12)])
# print(len(dt[(31, 12)]))

# exit()
# print(dt[(59, 53)])
# print(len(dt[(59, 53)]))
#
# print(dt[(76, 102)])
# print(len(dt[(76, 102)]))
#
# print(dt[(78, 125)])
# print(len(dt[(78, 125)]))
#
# exit()
#
# l1 = []
# l2 = []

# for k, v in dt.items():
#     j = 0
#     idl = []
#     n = []
#     for i in range(100):
#         if v[j][1] == i:
#             n.append(v[j][2])
#             idl.append(i)
#             if j < len(v)-1:
#                 j = j+1
#         else:
#             n.append(-1)
#             idl.append(i)
#     l1.append(n)
#     l2.append(idl)


# print(l1)
# print(l2)

# for each in l1:
#     print(len(l1))

# exit()

# l = []
# for k, v in dt.items():
#     n = []
#     # j = 0
#     for i in range(len(v)):
#         if i in v[i][1]:
#             n.append(v[i][2])
#         else:
#             n.append(-1)
#     l.append(n)
#
# print(l[2])
#
# plt.plot(l[15])
# plt.show()

# exit()

# dt = refine_junc_dt(dt_init)
# print(np.var(dt[(3, 79)][2]))
# print(list(dt.keys()))
# exit()
# t = dt[(22, 124)]
# ll = []
# for each in t:
#     ll.append(each[2])
#
# print(ll)
# print(np.var(ll))
# exit()

n1 = []
n2 = []
n3 = []
n5 = []
n6 = []
for k, v in dt.items():
    n1.append([k[0], k[1]])
    n4 = []
    for each in v:
        n2.append([each[0][0], each[0][1]])
        n4.append(each[2])
    n5.append(np.var(n4))
    n6.append(len(v))
    n3.append([np.var(n4)] * len(v))

med = np.median(n5)

n1 = np.array(n1)
n2 = np.array(n2)

n3arr = np.array(n3)

# print(n1)


# print(vor.point_region)

exit()

import itertools

n3val = list(itertools.chain.from_iterable(n3))


# med = np.median(n3val)

# print(n1)
# print(n2)

# print(np.median(n3val))
# print(np.mean(n3val))
# print(n3val)
# sns.distplot(n3val)
# plt.show()


def plot_junc_spread(group, n1, n2, num_series):
    fig = plt.gcf()
    ax = fig.gca()
    # gr = cm.Greens(np.linspace(n3arr.min()[0], n3arr.max()[0], num=len(n3)))
    # mcmap = mcolors.LinearSegmentedColormap.from_list('mcmap', gr)
    img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/er_mean/%s_er_mean.png' % (
        f'{group}', f'{group.lower()}{num_series}'))
    plt.imshow(img, cmap='gray', interpolation='none')
    # plt.plot(n2[:, 1], n2[:, 0], 'b.')
    plt.scatter(n2[:, 1], n2[:, 0], c=n3val, cmap='Blues', marker='o')
    # plt.colorbar()
    plt.plot(n1[:, 1], n1[:, 0], 'o', markerfacecolor='None', markeredgecolor='red', mew=1.5)  # , ms=4)
    # plt.plot(n1[:, 1], n1[:, 0], 'r.')
    # c = Circle((n1[0, 1], n1[0, 0]), radius=3, linewidth=2, facecolor='none', edgecolor='green', alpha=0.7)
    # ax.add_patch(c)
    # plt.plot(n2[:, 1], n2[:, 0], 'o', markerfacecolor='blue', markeredgecolor='blue')

    for i, each in enumerate(n1):
        c1 = plt.Circle((n1[i, 1], n1[i, 0]), 3, color='r', fill=False, linestyle='--')
        ax.add_patch(c1)
        # if n5[i] > med:
        s = '(' + '%.2f' % n5[i] + ',' + str(n6[i]) + ')'
        # s = '(' + str(n1[i,1]) + ',' + str(n1[i,0]) + ',' + '%.2f'%n5[i] + ',' + str(n6[i]) + ')'
        ax.text(n1[i, 1], n1[i, 0], s, c='yellow')

    plt.axis('off')
    plt.suptitle('Climp series %s junctions movement variance' % f'{num_series}')
    plt.title('Variance of list with distances for matched junctions per frame w.r.t. reference frame junctions')
    # plt.savefig('RTN1_junc_spread.png', bbox_inches='tight', pad_inches=0)
    plt.show()


plot_junc_spread('Climp', n1, n2, 3)
exit()


def junction_var_median():
    var_vals = []
    for each in n3:
        var_vals.append(each[0])

    med = np.median(var_vals)
    mn = np.mean(var_vals)

    for i, each in enumerate(var_vals):
        if each < mn:
            # print(n1[i])
            print(len(dt[(n1[i][0], n1[i][1])]))


# hlist = []
# X = []
# Y = []
# for k, v in dt.items():
#     l = []
#     for each in v:
#         l.append(each[2])
#     va = np.var(l)
#     # hlist.append(l)
#     X.append(len(v))
#     Y.append(va)
#     # hlist.append([len(v), va])
#     # plt.title('Climp Series 1, junction: %s matches: %s, variance: %s'%(f'{k}', f'{len(v)}', f'{va:.2f}'))
#     # plt.hist(l)
#     # plt.show()
#
# # print(hlist)
#
# plt.scatter(X, Y)
# plt.title('Relation between variance of distance and number of matched junctions')
# plt.xlabel('# Matched junctions')
# plt.ylabel('Distance Variance values')
# plt.show()

# sns.distplot(hlist)
# plt.show()
# for each in hlist:
#     plt.hist(each, alpha=0.4)
#     plt.show()

# plt.title('Combined plot with histograms for junction spread within threshold 9')
# plt.show()

exit()

# nps, dt_refined = refine_junc_dt(dt)
# print(nps)

l = list(dt_refined.keys())[20]
# print(l)

import statsmodels.api as sm

X = []
Y = []
for each in dt_refined[l]:
    X.append(each[0][0])
    Y.append(each[0][1])

minx = min(X)
maxx = max(X)

x = np.arange(minx, maxx, 1)
y = 0.0378 * x

print(X)
print(Y)

plt.scatter(X, Y)
# plt.plot(y, 'r')
plt.show()

# X = sm.add_constant(X)
#
# res = sm.OLS(X, Y).fit()
# print(res.summary())

exit()

l = list(dt_refined.keys())
ll = []
for each in l:
    ll.append([each[0], each[1]])
ll = np.array(ll)

# print(nps[:, 1])
# print(ll[:, 1])

# print(np.array(dt_refined.keys())[0])
# print(np.array(dt_refined.keys())[:, 1])

plt.imshow(imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean/atl1_er_mean.png'),
           cmap='gray')
plt.plot(nps[:, 1], nps[:, 0], 'r.')
plt.plot(ll[:, 1], ll[:, 0], 'b.')
plt.savefig('refined_atl1_er_junctions_50.png', bbox_inches='tight', pad_inches=0)
plt.close()
# plt.show()

# plt.plot(list(dt_refined.keys())[:, 1])
exit()


def get_mean_patch_intensity(er_input, a, b):
    m_val = (er_input[a, b] + er_input[a - 1, b - 1] + er_input[a + 1, b + 1] + er_input[a, b + 1] + er_input[
        a, b - 1] + er_input[a + 1, b - 1] + er_input[a - 1, b + 1] + er_input[a + 1, b] + er_input[a - 1, b]) / 9
    return m_val


def get_junc_patch(er_input, a, b):
    patch = np.zeros((3, 3))
    pa = patch.flatten()
    vals = [er_input[a - 1, b - 1], er_input[a, b - 1], er_input[a + 1, b - 1], er_input[a - 1, b], er_input[a, b],
            er_input[a + 1, b], er_input[a - 1, b + 1], er_input[a, b + 1], er_input[a + 1, b + 1]]
    for i, each in enumerate(vals):
        pa[i] = each
    patch = np.resize(pa, (3, 3))
    return patch


# ref_input = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t000_ch00.tif')
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
        ref_input = imageio.imread(
            '/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_%s_decon_t000.tif' % f'{num_series}')
    else:
        ref_input = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/files/%s_decon_t000_ch00.tif' % (
            f'{group}', f'{group[0]}{num_series}'))

    ref_input = (ref_input - ref_input.min()) / (ref_input.max() - ref_input.min())

    mean_val_dt = {}
    for k, v in newdt.items():
        mean_val_dt[k] = []
        a, b = k[0], k[1]
        m_val = get_mean_patch_intensity(ref_input, a, b)

        if v[0] == max_val:
            mean_val_dt[k].append(m_val)
        else:
            a, b = v[0][0][0], v[0][0][1]
            m_val = get_mean_patch_intensity(ref_input, a, b)
            mean_val_dt[k].append(m_val)

        for i in range(1, 100):
            if group == 'Control':
                er_input = imageio.imread(
                    '/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_%s_decon_t0%s.tif' % (
                        f'{num_series}', f'{i:02d}'))
            else:
                er_input = imageio.imread(
                    '/localhome/asa420/MIAL/data/confocal_movies/%s/files/%s_decon_t0%s_ch00.tif' % (
                        f'{group}', f'{group[0]}{num_series}', f'{i:02d}'))

            er_input = (er_input - er_input.min()) / (er_input.max() - er_input.min())
            if v[i] is not max_val:
                a, b = v[i][0][0], v[i][0][1]
                m_val = get_mean_patch_intensity(er_input, a, b)
                mean_val_dt[k].append(m_val)
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
#     img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0%s_ch00.tif'%f'{i:02d}')
#     img = (img - img.min()) / (img.max() - img.min())
#
#     plt.imshow(img, cmap='gray')
#     plt.plot(79, 3, 'b.')
#     plt.show()


def junction_location_plotter(group, num_series):
    # mean_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'

    mean_img = '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/er_mean_proc/%s_er_mean_proc_enhance_skel.png' % (
        f'{group}', f'{group.lower()}{num_series}')

    newps = junction_flow(mean_img)

    nps = []
    for each in newps:
        nps.append([each[0], each[1]])

    nps = np.array(nps)

    # print(nps[46, 1], nps[46, 0])
    #
    # exit()
    # plt.imshow(mean_proj_img, cmap='gray')

    # for (s,e) in g.edges():
    #     ps = g[s][e]['pts']
    #     plt.plot(ps[:,1], ps[:,0], 'green')
    # for dr in range(len(nps)):
    #     os.makedirs('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/A1_junc_viz/junc%s'%f'{dr+1}')

    for frame in range(100):

        # ER Input image
        if group == 'Control':
            path = '/localhome/asa420/MIAL/data/confocal_movies/%s/files/img_%s_decon_t0%s.tif' % (
                f'{group}', f'{num_series}', f'{frame:02d}')
            pref = 'Ct'
        else:
            path = '/localhome/asa420/MIAL/data/confocal_movies/%s/files/%s_decon_t0%s_ch00.tif' % (
                f'{group}', f'{group[0]}{num_series}', f'{frame:02d}')  # , f'{channel}')
            pref = group[0]
        img = imageio.imread(path)
        img = (img - img.min()) / (img.max() - img.min())

        # ER Skel image
        # sk_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t0%s_ch00_skel.png'%f'{i:02d}'
        sk_img = '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/skel/%s/%s_decon_t0%s_ch00_skel.png' % (
            f'{group}', f'{pref}{num_series}', f'{pref}{num_series}', f'{frame:02d}')

        sk_newps = junction_flow(sk_img)

        sk_nps = []
        for each in sk_newps:
            sk_nps.append([each[0], each[1]])

        sk_nps = np.array(sk_nps)

        # img = img * 255.
        # # plt.plot(ps[:, 1], ps[:, 0], 'y.')
        # for j in range(47, 48):
        # plt.figure(figsize=(128/77, 128/77))
        plt.imshow(img, cmap='gray')
        # plt.imshow(imageio.imread(img), cmap='gray')
        plt.axis('off')
        # plt.title('t=%s'%f'{i}')
        plt.plot(nps[:, 1], nps[:, 0], 'r.')
        plt.plot(sk_nps[:, 1], sk_nps[:, 0], 'b.')
        # y = nps[0, 1]
        # x = nps[0, 0]
        # cv2.rectangle(img, (x-1, y-1), (x+1, y+1), (0, 0, 255), 2)

        plt.savefig('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/%s_junc_viz/%s_t%s.png' % (
            f'{group}', f'{group.lower()}', f'{pref}{num_series}', f'{frame:02d}'), bbox_inches='tight', pad_inches=0)

        plt.close()


# junction_location_plotter('RTN', 1)
# exit()


def crop_img():
    mean_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'

    newps = junction_flow(mean_img)

    nps = []
    for each in newps:
        nps.append([each[0], each[1]])

    nps = np.array(nps)
    for i in range(100):
        img = imageio.imread(
            '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/A1_junc_viz/j48_skel/ATL1_t%s.png' % f'{i:02d}')

        # plt.imshow(img)
        # plt.show()

        # y = nps[74,1]
        # x = nps[74,0]
        # #
        # # # print(y, x)
        cimg = img[137 - 30:137 + 30, 143 - 30:143 + 30]

        plt.axis('off')
        plt.title('t=%s' % f'{i}')
        plt.imshow(cimg, interpolation='nearest', aspect='auto')
        plt.savefig(
            '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/A1_junc_viz/j48_skel/crops/ATL1_t%s.png' % f'{i:02d}',
            bbox_inches='tight', pad_inches=0)

        # imageio.imsave('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/A1_junc_viz/j47_skel/crops/ATL1_t%s.png'%f'{i:02d}', cimg)
        plt.close()


crop_img()
exit()


def per_patch_pixel_fourier(group, channel):
    grp_dict = {}
    nd = {}
    for num_series in range(1, 2):
        mean_img = '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/er_mean_proc/%s_er_mean_proc_enhance_skel.png' % (
            f'{group}', f'{group.lower()}{num_series}')
        newps = junction_flow(mean_img)

        sl = []
        sldt = []

        for frame in range(100):
            if group == 'Control':
                path = '/localhome/asa420/MIAL/data/confocal_movies/%s/files/img_%s_decon_t0%s.tif' % (
                    f'{group}', f'{num_series}', f'{frame:02d}')
            else:
                path = '/localhome/asa420/MIAL/data/confocal_movies/%s/files/%s_decon_t0%s_ch0%s.tif' % (
                    f'{group}', f'{group[0]}{num_series}', f'{frame:02d}', f'{channel}')
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
        for mv1, mv2 in zip(v1, v2):
            data.append(np.corrcoef(mv1, mv2)[0, 1])

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
        mean_img = '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/er_mean_proc/%s_er_mean_proc_enhance_skel.png' % (
            f'{group}', f'{group.lower()}{num_series}')
        newps = junction_flow(mean_img)

        sl = []

        for frame in range(100):
            if group == 'Control':
                path = '/localhome/asa420/MIAL/data/confocal_movies/%s/files/img_%s_decon_t0%s.tif' % (
                    f'{group}', f'{num_series}', f'{frame:02d}')
            else:
                path = '/localhome/asa420/MIAL/data/confocal_movies/%s/files/%s_decon_t0%s_ch00.tif' % (
                    f'{group}', f'{group[0]}{num_series}', f'{frame:02d}')
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
        vl = []
        for each in a.T:
            vl.append(np.var(each))

        ser_list.append(np.var(vl) / np.mean(vl))

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
        mean_img = '/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/er_mean_proc/%s_er_mean_proc_enhance_skel.png' % (
            f'{group}', f'{group.lower()}{num_series}')
        newps = junction_flow(mean_img)

        sl = []

        for frame in range(100):
            if group == 'Control':
                path = '/localhome/asa420/MIAL/data/confocal_movies/%s/files/img_%s_decon_t0%s.tif' % (
                    f'{group}', f'{num_series}', f'{frame:02d}')
            else:
                path = '/localhome/asa420/MIAL/data/confocal_movies/%s/files/%s_decon_t0%s_ch00.tif' % (
                    f'{group}', f'{group[0]}{num_series}', f'{frame:02d}')
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
                vl = []
                for each in a.T:
                    vl.append(np.var(each))

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
    prefix = '/localhome/asa420/MIAL/data/confocal_movies/'
    if group == 'Control':
        for series_num in range(1, 25):
            newps = junction_flow(prefix + 'Control/new_op_jul/Ctrl_mean_proj/Ct%s_mean.png' % f'{series_num}')
            ser_list = []
            for i in range(99):
                i1 = imageio.imread(
                    '/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_%s_decon_t0%s.tif' % (
                        f'{series_num}', f'{i:02d}'))
                i2 = imageio.imread(
                    '/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_%s_decon_t0%s.tif' % (
                        f'{series_num}', f'{i + 1:02d}'))

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
    else:
        for series_num in range(1, 25):
            newps = junction_flow(
                prefix + '%s/new_op_jul/%s_mean_proj/%s_mean.png' % (f'{group}', f'{group}', f'{group[0]}{series_num}'))
            ser_list = []
            for i in range(98):
                i1 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/files/%s_decon_t0%s_ch00.tif' % (
                    f'{group}', f'{group[0]}{series_num}', f'{i:02d}'))
                i2 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/files/%s_decon_t0%s_ch00.tif' % (
                    f'{group}', f'{group[0]}{series_num}', f'{i + 1:02d}'))

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
