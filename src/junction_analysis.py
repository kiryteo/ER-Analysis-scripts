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
from skimage.measure import label, regionprops
from skimage import measure
from mpl_toolkits import mplot3d
from scipy.ndimage import uniform_filter1d
from skan import draw
import itertools
import pandas as pd
import os
import cv2
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import cKDTree

from skimage import draw
import matplotlib.colors as mcolors
from matplotlib import cm

import plotly
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA

max_val = 999
confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'


def get_std_img(path):
    img = imageio.imread(path)
    return (img - img.min()) / (img.max() - img.min())

# skel = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_t0_ch0_loc2_enhance_skel.png')
#
# graph = sknw.build_sknw(skel, iso=False)


# img = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure2/A1_decon_t000_ch00_skel.png')
# plt.axis('off')
# plt.imshow(img, cmap='gray')
#
# plt.savefig('A1_t0_skel_dpi700.png', bbox_inches='tight', pad_inches=0, dpi=700)
# plt.close()
# # plt.show()
# exit()




def junc_spread_comparison():
    """
    Compare the junctions obtained via initial projection Vs.
    per frame junction projection
    @return:
    """
    img = imageio.imread(confocal_data_path + 'ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc.png')
    im1 = imageio.imread(confocal_data_path + 'ATL/new_op_jul/junctions/A1_junc_mean.png')
    im2 = imageio.imread(confocal_data_path + 'ATL/new_op_jul/junctions/A1_proc_junc_mean.png')

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
    """

    @param group:
    @param num_series:
    @return: display mean proj frame junctions (red spots) + per frame skel junctions (blue spots)
    """
    init_mean_proj_img = imageio.imread((confocal_data_path + f'{group}/new_op_jul/junctions/{group[0]}{num_series}_junc_mean.png'))

    junc_mean_proj_img = imageio.imread((confocal_data_path + f'{group}/new_op_jul/junctions/{group[0]}{num_series}_proc_junc_mean.png'))


    init_proj_img_coords = np.where(init_mean_proj_img != 0)
    junc_proj_img_coords = np.where(junc_mean_proj_img != 0)

    x1, y1 = init_proj_img_coords[0], init_proj_img_coords[1]
    x2, y2 = junc_proj_img_coords[0], junc_proj_img_coords[1]

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
    img = imageio.imread((confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc.png'))

    plt.imshow(img)
    # plt.scatter(y1, x1, color='red')
    # plt.scatter(y2, x2, color='blue')

    plt.plot(y1, x1, 'o', markerfacecolor='None', markeredgecolor='blue')
    plt.plot(y2, x2, 'o', markerfacecolor='None', markeredgecolor='red')

    plt.suptitle(f'{group} series {num_series}, Blue spots: junctions from a), Red spots: junctions from b)')

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

    node_set = g.nodes()

    G.add_nodes_from(node_set)
    G.add_edges_from(g.edges)

    degree_list = G.degree
    return node_set, degree_list


# sk = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A1/A1_decon_t000_ch00_skel.png')
# node_set, degree_list = skel_to_graph(sk)
# print(node_set[0])
#
# # ps = np.array([node_set[node]['o'] for node in node_set])
# # print(node_set)
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


def get_junctions(mean_img):
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
    node_set, degree_list = skel_to_graph(skel)
    node_coords = np.array([node_set[node]['o'] for node in node_set])

    return [node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2]


def per_frame_junc_projection(group, num_series):
    junc_mean = np.zeros((128, 128))
    for i in range(100):
        img = imageio.imread((confocal_data_path + f'{group}/new_op_jul/junctions/{group[0]}{num_series}/{group[0]}{num_series}_decon_t0{i:02d}_ch00_junc.png'))

        junc_mean += img

    cv2.imwrite((confocal_data_path + f'{group}/new_op_jul/junctions/{group[0]}{num_series}_junc_mean.png'), junc_mean / 100)


def init_proc_projection(group, num_series):
    sk = imageio.imread((confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'))


    node_set, degree_list = skel_to_graph(sk)
    node_coords = np.array([node_set[node]['o'] for node in node_set])

    # get all the nodes with degree greater than 2
    newps = [node_coords[j] for j, val in enumerate(degree_list) if val[1] > 2]
    brpts_img = get_junction_image(newps)

    cv2.imwrite((confocal_data_path + f'{group}/new_op_jul/junctions/{group[0]}{num_series}_proc_junc_mean.png'), brpts_img)


def mean_frame_validation(group, total_series):
    group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}
    for num_ser in range(1, total_series + 1):
        os.makedirs(confocal_data_path + f'{group}/new_op_jul/junctions/{group_pref[group]}{num_ser}')

        for frame in range(100):
            sk = imageio.imread((confocal_data_path + f'{group}/new_op_jul/skel/{group_pref[group]}{num_ser}/{group_pref[group]}{num_ser}_decon_t0{frame:02d}_ch00_skel.png'))


            node_set, degree_list = skel_to_graph(sk)
            node_coords = np.array([node_set[node]['o'] for node in node_set])

            # get all the nodes with degree greater than 2
            newps = [node_coords[j] for j, val in enumerate(degree_list) if val[1] > 2]
            brpts_img = get_junction_image(newps)

            imageio.imsave((confocal_data_path + f'{group}/new_op_jul/junctions/{pref}{num_ser}/{pref}{num_ser}_decon_t0{frame:02d}_ch00_junc.png'), brpts_img)


# patch1_vals -> list of intensity values per patch

def dil_junctions():
    global newps
    for series_num in range(1, 2):
        newps = get_junctions(confocal_data_path + f'ATL/new_op_jul/ATL_mean_proj/A{series_num}_mean.png')

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
        mean_img = confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

        newps = get_junctions(mean_img)

        sl = []

        for frame in range(100):
            if group == 'Control':
                path = confocal_data_path + f'{group}/files/img_{num_series}_decon_t0{frame:02d}.tif'

            else:
                path = confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t0{frame:02d}_ch0{channel}.tif'


            img = get_std_img(path)

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
    mean_img = confocal_data_path + 'ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'

    newps = get_junctions(mean_img)
    sl = []
    dtl = []

    for i in range(100):
        path = confocal_data_path + f'ATL/files/A1_decon_t0{i:02d}_ch00.tif'
        img = get_std_img(path)

        im1_patch_vals, dt = junc_patch_mean(newps, img)
        sl.append(im1_patch_vals)
        dtl.append(dt)

    # slt = np.array(sl).T
    # print(slt.shape)

    coord_dt = {}
    for each in dtl:
        for k, v in each.items():
            if k not in coord_dt.keys():
                coord_dt[k] = []
            coord_dt[k].append(v)
    return coord_dt


# coord_dt = patch_variation_viz()

def junc_intensity_plot_creator(coord_dt):
    for k, v in coord_dt.items():
        plt.plot(v)
        plt.xlabel('Timeframe')
        plt.ylabel('Intensity values')
        plt.title(f"ATL 1, Junction coordinates: {k[1]['<unknown expression ERROR>', 0]} intensity variation")

        plt.savefig(f"ATL_1_Junction_{k[1]['<unknown expression ERROR>', 0]}.png", bbox_inches='tight', pad_inches=0.2)

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
        if group == 'Control':
            path = confocal_data_path + f'{group}/files/img_{num_series}_decon_t0{i:02d}.tif'

        else:
            path = confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t0{i:02d}_ch00.tif'

        img = get_std_img(path)
        l.extend(img)

    l = np.reshape(l, (100, 128, 128))

    f = np.fft.fftn(l - np.mean(l))
    fabs = np.abs(f)
    fviz = np.fft.fftshift(fabs)

    return [np.sum(each) for each in fviz]


def seq_movie_fourier_analysis(group, num_series):
    l = []
    for num, i in itertools.product(range(1, num_series+1), range(100)):
        path = confocal_data_path + f'{group}/files/img_{num}_decon_t0{i:02d}.tif' if group == 'Control' else confocal_data_path + f'{group}/files/{group[0]}{num}_decon_t0{i:02d}_ch00.tif'

        img = get_std_img(path)
        l.extend(img)

    # if group == 'ATL':
    #     l = np.reshape(l, (2600, 128, 128))
    # elif group == 'Climp' or group == 'Control':
    #     l = np.reshape(l, (3100, 128, 128))
    # else:
    #     l = np.reshape(l, (2900, 128, 128))
    l = np.reshape(l, (2600, 128, 128))

    f = np.fft.fftn(l - np.mean(l))
    fabs = np.abs(f)
    fviz = np.fft.fftshift(fabs)

    return [np.sum(each) for each in fviz]


def seq_movie_fourier_analysis_runner():
    atl = seq_movie_fourier_analysis('ATL', 26)
    climp = seq_movie_fourier_analysis('Climp', 26)
    control = seq_movie_fourier_analysis('Control', 26)
    rtn = seq_movie_fourier_analysis('RTN', 26)

    # import scipy.io
    # from scipy.io import savemat
    #
    # atldt = {}
    # atldt['atl'] = atl
    # cldt = {}
    # cldt['climp'] = climp
    # ctdt = {}
    # ctdt['control'] = control
    # rtndt = {}
    # rtndt['rtn'] = rtn
    #
    # savemat('atl.mat', atldt)
    # savemat('climp.mat', cldt)
    # savemat('control.mat', ctdt)
    # savemat('rtn.mat', rtndt)


    # climp = []
    # control = []
    # rtn = []

    # for i in range(1, 27):
    #     atl_k = seq_fourier_analysis('ATL', i)
    #     atl.extend(atl_k)

    # print(len(atl))
    # print(atl[0])
    # exit()


    # atl = list(itertools.chain.from_iterable(atl))

    # for i in range(1, 32):
    #     climp_k = seq_fourier_analysis('Climp', i)
    #     climp.extend(climp_k)
    #
    # # climp = list(itertools.chain.from_iterable(climp))
    #
    # for i in range(1, 32):
    #     ctrl_k = seq_fourier_analysis('Control', i)
    #     control.extend(ctrl_k)
    #
    # # control = list(itertools.chain.from_iterable(control))
    #
    # for i in range(1, 30):
    #     rtn_k = seq_fourier_analysis('RTN', i)
    #     rtn.extend(rtn_k)

    # rtn = list(itertools.chain.from_iterable(rtn))

    # atl = sorted(atl)
    # climp = sorted(climp)
    # control = sorted(control)
    # rtn = sorted(rtn)

    plt.plot(atl, label='ATL')
    plt.plot(climp, label='Climp')
    plt.plot(control, label='Control')
    plt.plot(rtn, label='RTN')
    plt.legend()
    plt.title('EGFP frequency analysis across conditions')
    plt.show()


def junc_area_locator(group, num_series):
    """

    @param group: group to be analyzed
    @param num_series: sequence number
    @return: dt, dictionary with matched junctions per reference junction - nearest neighbour approach
    """
    # mean_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'
    group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}

    mean_img = confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'


    # Get junction coordinates from projection frame
    newps = get_junctions(mean_img)

    nps = [[each[0], each[1]] for each in newps]
    nps = np.array(nps)
    # print(nps.shape)

    # sort the array for nearest neighbour matching per frame
    # nps_sorted is the mean projection (reference) junction list
    nps_sorted = sorted(nps, key=lambda t: t[0])

    dt = {}
    for frame in range(100):

        sk_img = confocal_data_path + f'{group}/new_op_jul/skel/{group_pref[group]}{num_series}/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_skel.png'


        sk_newps = get_junctions(sk_img)

        sk_nps = [[each[0], each[1]] for each in sk_newps]
        sk_nnode_coords = np.array(sk_nps)

        # sk_nps_sorted is the per frame junction list
        sk_nps_sorted = sorted(sk_nps, key=lambda t: t[0])

        # matching of junction candidates
        for elem in nps_sorted:
            for sk_elem in sk_nps_sorted:
                dst = np.sqrt((sk_elem[0] - elem[0]) ** 2 + (sk_elem[1] - elem[1]) ** 2)
                if dst < 3:
                    if (elem[0], elem[1]) not in dt.keys():
                        dt[(elem[0], elem[1])] = []
                    dt[(elem[0], elem[1])].append([sk_elem, frame, dst])
    return dt


def get_all_junc(group, num_series):
    """

    @param group: group to be analyzed
    @param num_series: sequence number
    @return: nps (list) - provides all junctions with degree > 2 from the mean projection proc skeleton, skdata (list) - provides all junctions per skel frame
    """
    # mean_img = 'confocal_data_pathATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'

    group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}

    mean_img = confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'


    # Get junction coordinates from projection frame
    newps = get_junctions(mean_img)

    nps = [[each[0], each[1]] for each in newps]
    skdata = []
    for frame in range(100):

        sk_img = confocal_data_path + f'{group}/new_op_jul/skel/{group_pref[group]}{num_series}/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_skel.png'


        sk_newps = get_junctions(sk_img)

        sk_nps = [[each[0], each[1]] for each in sk_newps]
        # sk_nps = np.array(sk_nps)
        skdata.extend(sk_nps)

    return nps, skdata


# nps, skdata = get_all_junc('ATL', 2)
# print(len(skdata))
# print(len(skdata[0]))
# # print(skdata[0])
# # print(skdata[1])
# # print(skdata[100])
# # print(skdata[101])
# exit()

def refine_junc_dt(dt, min_presence=50):
    return {k: v for k, v in dt.items() if len(v) > min_presence}


def get_junction_types(nps, lab):
    """

    @param nps: (ndarray) reference junctions
    @param lab: (ndarray) connected components for junctions
    @return: label_vals (dict) provides corresponding reference junctions per cc_id, assigned_components (list) provides cc with at least 1 reference junction
    """
    label_vals = {}

    assigned_components = []

    for each in nps:
        if lab[each[0], each[1]] != 0:
            if lab[each[0], each[1]] not in label_vals.keys():
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
            if cc_label not in unassigned_cc_dict.keys():
                unassigned_cc_dict[(lab[each[0], each[1]])] = []
            unassigned_cc_dict[(lab[each[0], each[1]])].append([each[0], each[1]])
    return unassigned_cc_dict


def label_junctions(group, series_num):

    # fig, ax = plt.subplots()
    nps, skdata = get_all_junc(group, series_num)

    nps = np.array(nps)
    skdata = np.array(skdata)

    spread_img = np.zeros((128, 128))
    for each in skdata:
        spread_img[each[0], each[1]] = 255.


    # fig.add_subplot(1,2,1)
    # plt.imshow(spread_img)
    #
    labelled_img = label(spread_img, connectivity=2)
    # imageio.imsave('Climp12_junc_labelled.png', labelled_img)
    # fig.add_subplot(1,2,2)
    # plt.axis('off')
    # plt.imshow(labelled_img, cmap='gray')
    # plt.savefig('Climp12_junc_labelled.png', bbox_inches='tight', pad_inches=0, dpi=700)
    # plt.close()
    # plt.show()
    # exit()

    return nps, skdata, labelled_img

# label_junctions('ATL', 1)
# exit()


def viz_regionprops(labelled_img, spread_img):
    fig = px.imshow(spread_img, binary_string=True)
    fig.update_traces(hoverinfo='skip')

    props = regionprops(labelled_img, spread_img)
    properties = ['area', 'eccentricity', 'perimeter']

    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    for index in range(1, labelled_img.max()):
        label_i = props[index].label
        contour = measure.find_contours(labelled_img == label_i)[0]
        y, x = contour.T
        hoverinfo = ''.join(f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>' for prop_name in properties)

        fig.add_trace(go.Scatter(
            x=x, y=y, name=label_i,
            mode='lines', fill='toself', showlegend=False,
            hovertemplate=hoverinfo, hoveron='points+fills'))

    plotly.io.show(fig)


# for i in range(10):

def runner_viz_regionprops():
    img = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_junc_projection.png')
    # lab = label(img)
    lab = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/Climp12_junc_labelled_cc.png')

    img = np.stack((img, img, img, img), axis=2)

    print(img.shape)
    print(lab.shape)

    viz_regionprops(lab, img)

# runner_viz_regionprops()
# exit()



def separate_junc_cc(nps, skdata, labelled_img):
    regions = regionprops(labelled_img)

    # cc_list = []
    # for idx in range(1, labelled_img.max()):
    #     lab_i = props[idx].label

    # cc_area_dict = {}
    # for idx, props in enumerate(regions):
    #     cc_area_dict[idx] = props.area
        # cc_area_dict[idx] = [props.area, props.axis_major_length]

    num_components = np.unique(labelled_img)
    # print(num_components)

    label_vals, assigned_components = get_junction_types(nps, labelled_img)
    # print(label_vals)

    unassigned_cc_dict = get_uncertain_junctions(labelled_img, skdata, num_components, assigned_components)

    # return label_vals, cc_area_dict, unassigned_cc_dict
    return label_vals, unassigned_cc_dict


def get_junction_areas(label_vals, unassigned_cc_dict):
    isolated_junc = []
    isolated_junc_area = []
    fuzzy_junc = []
    fuzzy_junc_area = []
    for k, v in label_vals.items():
        if k != 0:
            if len(v) == 1:
                isolated_junc.append(v[0])
                # try:
                #     isolated_junc_area.append(cc_area_dict[k])
                # except:
                #     pass
            else:
                fuzzy_junc.append(v)
                # try:
                #     fuzzy_junc_area.append(cc_area_dict[k])
                # except:
                #     pass

    unknown_junc = [v for k, v in unassigned_cc_dict.items()]
    iso = np.array(isolated_junc)

    fuz = list(itertools.chain.from_iterable(fuzzy_junc))
    fuz = np.array(fuz)

    unk = list(itertools.chain.from_iterable(unknown_junc))
    unk = np.array(unk)

    # iso_area = list(itertools.chain.from_iterable(isolated_junc_area))
    # iso_area = np.array(isolated_junc_area)

    # fuz_area = list(itertools.chain.from_iterable(fuzzy_junc_area))
    # fuz_area = np.array(fuzzy_junc_area)


    return iso, fuz, unk#, iso_area, fuz_area


# nps, skdata, labelled_img = label_junctions('Climp', 1)
#
# label_vals, unassigned_cc_dict = separate_junc_cc(nps, skdata, labelled_img)
# iso, fuz, unk = get_junction_areas(label_vals, unassigned_cc_dict)
#


# print(iso_list)
# exit()


# def label_skdata_type(skdata, lab):
#     for each in skdata:
#         if each in iso_list:
#
#
#
# print(iso[:, 1])
#
# exit()

# def plot_junc_areas(group, series_num, iso, fuz, unk, labelled_img):
def plot_junc_areas(group, series_num, labelled_img, iso, fuz, skdata, iso_cc_coords, fuz_cc_coords, unk_cc_coords):
    regions = regionprops(labelled_img)
    for i in range(1):
        plt.axis('off')
        if group == 'Control':
            img = imageio.imread(confocal_data_path + f'{group}/files/img_{series_num}_decon_t0{i:02d}.tif')

        else:
            img = imageio.imread(confocal_data_path + f'{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch00.tif')

        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img, cmap='gray')
        # plt.plot(iso[:, 1], iso[:, 0], 's', markerfacecolor='None', markeredgecolor='red')

        # plt.plot(skdata[:, 1], skdata[:, 0], '.', markerfacecolor='None', markeredgecolor='green', mew=0.4)
        # for k, v in iso_cc_coords.items():
        #     plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='green', mew=0.4)
        #
        # for k, v in fuz_cc_coords.items():
        #     plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='yellow', mew=0.4)
        #
        # for k, v in unk_cc_coords.items():
        #     plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='pink', mew=0.4)
        #
        # if len(fuz) > 0:
        #     plt.plot(fuz[:, 1], fuz[:, 0], 'o', markerfacecolor='None', markeredgecolor='blue', mew=0.6)
        #
        # # plt.plot(unk[:, 1], unk[:, 0], 'o', markerfacecolor='None', markeredgecolor='green')
        # plt.plot(iso[:, 1], iso[:, 0], 'o', markerfacecolor='None', markeredgecolor='red', mew=0.6)

        for k, v in iso_cc_coords.items():
            plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='magenta', mew=0.35)

        for k, v in fuz_cc_coords.items():
            plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='blue', mew=0.35)

        for k, v in unk_cc_coords.items():
            plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='green', mew=0.5)

        if len(fuz) > 0:
            plt.plot(fuz[:, 1], fuz[:, 0], '.', markerfacecolor='None', markeredgecolor='white', mew=0.6)

        # plt.plot(unk[:, 1], unk[:, 0], 'o', markerfacecolor='None', markeredgecolor='green')
        # plt.plot(iso[:, 1], iso[:, 0], '.', markerfacecolor='None', markeredgecolor='yellow', mew=0.6)
        plt.plot(iso[:, 1], iso[:, 0], '.', markerfacecolor='None', markeredgecolor='yellow', mew=0.6)



        # plots contours
        for index in range(1, labelled_img.max()):
            label_i = regions[index].label
            contour = measure.find_contours(labelled_img == label_i, 0.8)[0]
            y, x = contour.T
            plt.plot(x, y, color='cyan')

        # plt.axis('off')
        # plt.savefig('Climp_series12_junc_representation_iso_fuz_unk_2_new_colors', bbox_inches='tight', pad_inches=0, dpi=700)
        # plt.close()

        plt.show()


def get_cc_ids(labelled_img, region):
    """

    @param labelled_img: Input with all CC areas
    @param region: (list) iso or fuz
    @return: isolated or fuzzy region CC ids list
    """

    num_cc = np.unique(labelled_img)
    # dict to store per component data
    dt = {each: [] for each in num_cc}
    for loc in region:
        locx, locy = loc[0], loc[1]
        cc_id = labelled_img[locx, locy]
        dt[cc_id] = loc

    dt_vals = dt.values()

    return [i for i, num in enumerate(dt_vals) if i > 0 and len(num) != 0]


# nps, skdata, labelled_img = label_junctions('Climp', 1)
#
# label_vals, unassigned_cc_dict = separate_junc_cc(nps, skdata, labelled_img)
# iso, fuz, unk = get_junction_areas(label_vals, unassigned_cc_dict)
#
# iso_cc = get_cc_ids(labelled_img, iso)
# fuz_cc = get_cc_ids(labelled_img, fuz)
# unk_cc = get_cc_ids(labelled_img, unk)
#
# iso_cc_coords = {each: np.where(labelled_img==each) for each in iso_cc}
# fuz_cc_coords = {each: np.where(labelled_img==each) for each in fuz_cc}
# unk_cc_coords = {each: np.where(labelled_img==each) for each in unk_cc}
#
# plot_junc_areas('ATL', 1, labelled_img, iso, fuz, skdata, iso_cc_coords, fuz_cc_coords, unk_cc_coords)
#
# exit()

def get_label_id(regions, iso, junc_id):
    for j in range(len(regions)):
        region_coords = regions[j].coords
        for each in region_coords:
            a, b = each[0], each[1]
            if a == iso[junc_id][0] and b == iso[junc_id][1]:
                return j

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def junction_crops_creator(group, ser_num, junc_id, channel):
    nps, skdata, labelled_img = label_junctions(group, ser_num)

    label_vals, unassigned_cc_dict = separate_junc_cc(nps, skdata, labelled_img)
    iso, fuz, unk = get_junction_areas(label_vals, unassigned_cc_dict)

    iso_cc = get_cc_ids(labelled_img, iso)
    iso_cc_coords = {each: np.where(labelled_img==each) for each in iso_cc}


    if channel == 'egfp':
        ch = 0
    else:
        ch = 1
    for i in range(100):
        file = get_std_img(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/{group[0]}{ser_num}_decon_t0{i:02d}_ch0{ch}.tif')

        # file = get_std_img(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junction_crops/S1_j15_29_74_CC/A{ser_num}_decon_t0{i:02d}_ch00.png')


        regions = regionprops(labelled_img)
        # print(regions[junc_id].label)
        #
        # exit()

        x, y = iso[junc_id][0], iso[junc_id][1]

        # fig = Figure()
        # canvas = FigureCanvas(fig)
        # ax = fig.gca()

        # contour code
        j = get_label_id(regions, iso, junc_id)
        label_i = regions[j].label
        contour = measure.find_contours(labelled_img == label_i, 0.8)[0]
        cntrY, cntrX = contour.T


        crp_file = file[x-5:x+5, y-5:y+5]
        # crp_mch = mch[x-5:x+5, y-5:y+5]
        plt.axis('off')
        # plt.imshow(crp_file, cmap='gray')
        # plt.imshow(file, cmap='gray')
        plt.plot(cntrX, cntrY, color='cyan')
        # fig = plt.gca()
        # crp = fig[x-5:x+5, y-5:y+5]
        plt.imshow(crp_file)
        # plt.title(f't={i}')
        # fig = plt.gca(figsize=(8,10))

        # plt.savefig(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/junction_crops/S{ser_num}_j{junc_id}_{x}_{y}_CC_crops/{group[0]}{ser_num}_decon_t0{i:02d}_ch0{ch}.png', bbox_inches='tight', pad_inches=0)
        # plt.close()
        plt.show()

        # imageio.imsave(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/junction_crops/S{ser_num}_junc{junc_id}_{x}_{y}/{group[0]}{ser_num}_decon_t0{i:02d}_ch00.png', crp_egfp)
        # imageio.imsave(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/junction_crops/S{ser_num}_junc{junc_id}_{x}_{y}/{group[0]}{ser_num}_decon_t0{i:02d}_ch01.png', crp_mch)


junction_crops_creator('ATL', 1, 15, 'egfp')
exit()

nps, skdata, labelled_img = label_junctions('ATL', 8)

label_vals, unassigned_cc_dict = separate_junc_cc(nps, skdata, labelled_img)
iso, fuz, unk = get_junction_areas(label_vals, unassigned_cc_dict)


iso_cc = get_cc_ids(labelled_img, iso)
fuz_cc = get_cc_ids(labelled_img, fuz)
unk_cc = get_cc_ids(labelled_img, unk)

iso_cc_coords = {each: np.where(labelled_img==each) for each in iso_cc}
fuz_cc_coords = {each: np.where(labelled_img==each) for each in fuz_cc}
unk_cc_coords = {each: np.where(labelled_img==each) for each in unk_cc}
#
#
# iso_list = []
# for each in iso:
#     iso_list.append([each[0], each[1]])

# plot_junc_areas('Climp', 12, iso, fuz, skdata, iso_cc_coords, fuz_cc_coords, unk_cc_coords)
# print(iso[55])




# nps, skdata, labelled_img = label_junctions('ATL', 1)
nps, skdata, labelled_img = label_junctions('Climp', 12)
# exit()

label_vals, unassigned_cc_dict = separate_junc_cc(nps, skdata, labelled_img)
iso, fuz, unk = get_junction_areas(label_vals, unassigned_cc_dict)

# mp_frame = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png')
mp_frame = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/er_mean/climp12_er_mean.png')

plt.imshow(mp_frame, cmap='gray')


# plt.plot(iso[:, 1], iso[:, 0], 'x', markerfacecolor='None', markeredgecolor='red', mew=0.6)

# exit()

g = sknw.build_sknw(mp_frame, iso=False)
G = nx.Graph()

G.add_nodes_from(g.nodes)
G.add_edges_from(g.edges)

ps = np.array([g.nodes[i]['o'] for i in g.nodes])
# print(ps)

# plt.plot(ps[:, 1], ps[:, 0], 'o', markerfacecolor='None', markeredgecolor='blue', mew=0.6)
#
# plt.show()
# exit()


iso_ps_ids = [i for i, val in enumerate(G.degree) if val[1] > 2]

# print(iso_ps_ids)
# exit()

# print(fuz)
print(len(G.nodes))
for each in iso_ps_ids:
    if g.nodes[each]['o'] in fuz:
        G.remove_node(each)
    # print(g.nodes[each]['o'])


gl = G.nodes
print(len(gl))

exit()

iso_ps = np.array([g.nodes[i]['o'] for i in gl])
# print(G.edges)
# G.remove_node()
# print(iso_ps)

plt.plot(iso_ps[:, 1], iso_ps[:, 0], 'o', markerfacecolor='blue', markeredgecolor='blue', mew=0.6)
plt.show()
exit()

nodes = g.nodes()
degree_list = G.degree





exit()

def per_frame_num_junctions(labelled_img, iso):

    iso_cc = get_cc_ids(labelled_img, iso)

    # get lists to store the count of junctions within CC per frame
    iso_junc_num = []
    fuz_junc_num = []

    for i in range(100):
        temp_iso = []
        temp_fuz = []
        junc_frame = imageio.imread(confocal_data_path + f'ATL/new_op_jul/junctions/A1/A1_decon_t0{i:02d}_ch00_junc.png')


        # get the junc locations
        locations = np.where(junc_frame > 0)

        for locx, locy in zip(locations[0], locations[1]):
            cc_id = labelled_img[locx, locy]
            if cc_id in iso_cc:
                temp_iso.append(cc_id)
            else:
                temp_fuz.append(cc_id)
        iso_junc_num.append(temp_iso)
        fuz_junc_num.append(temp_fuz)

    return iso_junc_num, fuz_junc_num

# iso_junc_num, fuz_junc_num = per_frame_num_junctions(labelled_img)
#
# # print(iso_junc_num)
#
# l = []
# for each in fuz_junc_num:
#     l.append(len(each))
#
# plt.plot(l)
# plt.show()


def calc_egfp_junction_intensity_nbrhood(group, series_num, channel):

    ch = 1 if channel=='mCherry' else 0
    sl = []
    for series_num in range(1, series_num+1):
        nps, skdata, labelled_img = label_junctions(group, series_num)
        label_vals, unassigned_cc_dict = separate_junc_cc(nps, skdata, labelled_img)
        iso, fuz, unk = get_junction_areas(label_vals, unassigned_cc_dict)

        ln = []
        for i in range(100):
            frame_data = []
            if group == 'Control':
                path = confocal_data_path + f'Control/files/img_{series_num}_decon_t0{i:02d}.tif'

            else:
                path = confocal_data_path + f'{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch0{ch}.tif'

            img = get_std_img(path)
            for each in iso:
                x, y = each[0], each[1]
                mean_val = (img[x-1, y-1] + img[x-1, y] + img[x-1, y+1] + img[x, y-1] + img[x, y] + img[x, y+1] + img[x+1, y-1] + img[x+1, y] + img[x+1, y+1]) / 9
                frame_data.append(mean_val)
            ln.extend(frame_data)
        ln = np.array(ln)

        sl.extend(ln)

    return sl


def junction_nbrhood_mean_plot():
    atl = calc_egfp_junction_intensity_nbrhood('ATL', 26, 'mCherry')
    climp = calc_egfp_junction_intensity_nbrhood('Climp', 31, 'mCherry')
    # # ctrl = calc_egfp_junction_intensity_nbrhood('Control', 31, 'EGFP')
    rtn = calc_egfp_junction_intensity_nbrhood('RTN', 29, 'mCherry')
    #
    # sns.distplot(atl, label='ATL', hist=False)
    # sns.distplot(climp, label='Climp', hist=False)
    # # sns.distplot(ctrl, label='Control', hist=False)
    # sns.distplot(rtn, label='RTN', hist=False)
    #

    df = pd.DataFrame()
    df['Values'] = pd.Series(np.concatenate((atl, climp, rtn)))
    # df['ids'] = pd.Series(np.concatenate((np.arange(1, len(atl)+1), np.arange(1, len(climp)+1), np.arange(1, len(ctrl)+1), np.arange(1, len(rtn)+1))))
    df['Group'] = pd.Series(np.concatenate((['ATL'] * len(atl), ['Climp'] * len(climp), ['RTN'] * len(rtn))))
    #
    sns.swarmplot(data=df, y='Group', x='Values')

    plt.suptitle('mCherry deposit in isolated reference junction 3x3 neighbourhood across conditions', fontsize=16)
    # # plt.title('Variance of junction CC mean per patch over 100 frames', fontsize=14)
    plt.title('Mean Intensity per junction neighbourhood', fontsize=14)
    # plt.xlabel('mCherry intensity mean values (3x3)', fontsize=12)
    # plt.legend()
    plt.show()

# exit()


def calc_egfp_deposit(group, channel, series_num, region):

    ch = 1 if channel=='mCherry' else 0
    sl = []
    for series_num in range(1, series_num+1):

        nps, skdata, labelled_img = label_junctions(group, series_num)
        label_vals, unassigned_cc_dict = separate_junc_cc(nps, skdata, labelled_img)
        iso, fuz, unk = get_junction_areas(label_vals, unassigned_cc_dict)
        # iso_cc = get_cc_ids(labelled_img, iso)
        if region == 'iso':
            region_cc = get_cc_ids(labelled_img, iso)
        else:
            region_cc = get_cc_ids(labelled_img, fuz)

        # dict to store the coords for each cc id
        # iso_cc_coords = {}
        # for each in iso_cc:
        #     iso_cc_coords[each] = np.where(labelled_img==each)

        region_cc_coords = {each: np.where(labelled_img==each) for each in region_cc}
        # for each cc, we obtain the index of dispersion per frame

        ln = []
        for i in range(100):
            if group == 'Control':
                path = confocal_data_path + f'Control/files/img_{series_num}_decon_t0{i:02d}.tif'

            else:
                path = confocal_data_path + f'{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch0{ch}.tif'

            # path = confocal_data_path + '%s/files/%s_decon_t0%s_ch0%s.tif'%(f'{group}', f'{group[0]}{series_num}', f'{i:02d}', f'{ch}')
            img = get_std_img(path)
            frame_data = [np.mean(img[v]) for v in region_cc_coords.values()]
            # ln.append(np.std(frame_data) / np.mean(frame_data))
            # ln.append(frame_data)
            ln.extend(frame_data)
        ln = np.array(ln)
        # m = np.std(ln, axis=0)

        # sl.extend(m)
        sl.extend(ln)
            # print(len(sl))
            # print(len(sl[0]))
            # print(sl[0])
            # exit()
    return sl


def junction_cc_mean_plot():

    atl = calc_egfp_deposit('ATL', 'mCherry', 26, 'iso')

    climp = calc_egfp_deposit('Climp', 'mCherry', 31, 'iso')
    # ctrl = calc_egfp_deposit('Control', 'EGFP', 31, 'iso')
    rtn = calc_egfp_deposit('RTN', 'mCherry', 29, 'iso')

    # print(len(atl))
    # print(len(climp))
    # # print(len(ctrl))
    # print(len(rtn))
    #
    # exit()

    sns.distplot(atl, label='ATL', hist=False)
    sns.distplot(climp, label='Climp', hist=False)
    # sns.distplot(ctrl, label='Control', hist=False)
    sns.distplot(rtn, label='RTN', hist=False)

    # sns.boxplot(atl, label='ATL')
    # sns.boxplot(climp, label='Climp')
    # sns.boxplot(ctrl, label='Control')
    # sns.boxplot(rtn, label='RTN')


    # df = pd.DataFrame()
    # df['Values'] = pd.Series(np.concatenate((atl, climp, ctrl, rtn)))
    # df['ids'] = pd.Series(np.concatenate((np.arange(1, len(atl)+1), np.arange(1, len(climp)+1), np.arange(1, len(ctrl)+1), np.arange(1, len(rtn)+1))))
    # df['Group'] = pd.Series(np.concatenate((['ATL'] * len(atl), ['Climp'] * len(climp), ['Control'] * len(ctrl), ['RTN'] * len(rtn))))
    # #
    # # sns.boxplot(data=df, y='Group', x='Values')
    # sns.scatterplot(data=df, x='ids', y='Values', hue='Group', style='Group')

    plt.suptitle('mCherry deposit in isolated region junction CCs across conditions', fontsize=16)
    # plt.title('Variance of junction CC mean per patch over 100 frames', fontsize=14)
    plt.title('Mean Intensity per junction CC patch', fontsize=14)
    # plt.xlabel('EGFP intensity mean values', fontsize=12)
    plt.legend()
    plt.show()

# sl = calc_egfp_deposit()
def get_junc_data_pca():
    pass
    # sl = np.array(sl)

    # print(len(sl))
    # print(sl.shape)
    # print(sl[0].shape)
    # print(sl[0])
    # print(len(sl[0][0]))
    # print(len(sl[0][1]))
    # exit()

    # pca = PCA(1)
    # pca.fit(sl)
    # evr = pca.explained_variance_ratio_
    # print(len(evr))
    # print(ln.shape)
    # plt.plot(ln)
    # plt.show()


def per_movie_num_junctions(group, num_series):
    grp_iso = []
    grp_fuz = []

    for i in range(1, num_series+1):
        nps, skdata, labelled_img = label_junctions(group, i)
        label_vals, unassigned_cc_dict = separate_junc_cc(nps, skdata, labelled_img)
        iso, fuz, unk = get_junction_areas(label_vals, unassigned_cc_dict)
        # grp_iso.append(len(iso))
        # grp_fuz.append(len(fuz))
        # grp_iso.append(len(iso_area))
        # grp_fuz.append(len(fuz_area))

    return grp_iso, grp_fuz


def plot_per_movie_junction_dist():
    ATL_iso, ATL_fuz = per_movie_num_junctions('ATL', 26)
    Climp_iso, Climp_fuz = per_movie_num_junctions('Climp', 31)
    Ctrl_iso, Ctrl_fuz = per_movie_num_junctions('Control', 31)
    RTN_iso, RTN_fuz = per_movie_num_junctions('RTN', 29)

    # sns.distplot(ATL_iso, hist=False, label='atl_iso')
    sns.distplot(ATL_fuz, hist=False, label='atl_fuz')
    # sns.distplot(Climp_iso, hist=False, label='climp_iso')
    sns.distplot(Climp_fuz, hist=False, label='climp_fuz')
    # sns.distplot(Ctrl_iso, hist=False, label='control_iso')
    sns.distplot(Ctrl_fuz, hist=False, label='control_fuz')
    # sns.distplot(RTN_iso, hist=False, label='rtn_iso')
    sns.distplot(RTN_fuz, hist=False, label='rtn_fuz')

    # and fuzzy region
    # (iso: isolated, fuz: fuzzy)
    plt.title('Distribution of fuzzy region junctions across conditions ', fontsize=16)
    plt.xlabel('Number of junctions (per movie)')
    plt.legend()

    plt.show()


def per_movie_junctions_area(group, num_series):
    grp_iso = []
    grp_fuz = []

    for i in range(1, num_series+1):
        nps, skdata, labelled_img = label_junctions(group, i)
        label_vals, unassigned_cc_dict = separate_junc_cc(nps, skdata, labelled_img)
        iso, fuz, unk = get_junction_areas(label_vals, unassigned_cc_dict)
        # grp_iso.append(len(iso))
        # grp_fuz.append(len(fuz))
        # grp_iso.extend(iso_area)
        # grp_fuz.extend(fuz_area)

    return grp_iso, grp_fuz


def plot_per_movie_junction_area_dist():
    ATL_iso, ATL_fuz = per_movie_junctions_area('ATL', 26)
    Climp_iso, Climp_fuz = per_movie_junctions_area('Climp', 31)
    Ctrl_iso, Ctrl_fuz = per_movie_junctions_area('Control', 31)
    RTN_iso, RTN_fuz = per_movie_junctions_area('RTN', 29)

    sns.distplot(ATL_iso, hist=False, label='atl_iso_area')
    # sns.distplot(ATL_fuz, hist=False, label='atl_fuz_area')
    sns.distplot(Climp_iso, hist=False, label='climp_iso_area')
    # sns.distplot(Climp_fuz, hist=False, label='climp_fuz_area')
    sns.distplot(Ctrl_iso, hist=False, label='control_iso_area')
    # sns.distplot(Ctrl_fuz, hist=False, label='control_fuz_area')
    sns.distplot(RTN_iso, hist=False, label='rtn_iso_area')
    # sns.distplot(RTN_fuz, hist=False, label='rtn_fuz_area')

    # and fuzzy region
    # (iso: isolated, fuz: fuzzy)
    # plt.title('Distribution of fuzzy region junctions across conditions ', fontsize=16)
    plt.title('Distribution of isolated region area across conditions ', fontsize=16)
    plt.xlabel('Region area values (per movie)')
    plt.legend()

    plt.show()
#
#
# plot_per_movie_junction_area_dist()
# exit()


# iso_junc = []
# fuz_junc = []
#
# for each in iso:
#     iso_junc.append([each[0], each[1]])
# for each in fuz:
#     fuz_junc.append([each[0], each[1]])
#
#
# per_frame_iso_area_dist = []
# per_frame_fuz_area_dist = []
# for i in range(100):
#     # l = []
#     junc_frame = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junctions/A2/A2_decon_t0%s_ch00_junc.png'%f'{i:02d}')
#     junctions = np.where(junc_frame > 0)
#     for x, y in zip(junctions[0], junctions[1]):
#         if [x, y] in iso_junc:
#             cc_id = labelled_img[x, y]
#             per_frame_iso_area_dist.append(cc_area_dict[cc_id])
#         elif [x, y] in fuz_junc:
#             cc_id = labelled_img[x, y]
#             per_frame_fuz_area_dist.append(cc_area_dict[cc_id])
#
# sns.distplot(per_frame_iso_area_dist, label='Iso')
# sns.distplot(per_frame_fuz_area_dist, label='Fuz')
# plt.legend()
# plt.show()
# exit()


def per_cc_analysis():
    nps, skdata, labelled_img = label_junctions('ATL', 1)

    print(np.unique(labelled_img))
    cc_dict = {num: [] for num in range(1, len(np.unique(labelled_img)))}
    exit()
    l = []
    for i in range(100):
        junc_frame = imageio.imread(confocal_data_path + f'ATL/new_op_jul/junctions/A1/A1_decon_t0{i:02d}_ch00_junc.png')

        junctions = np.where(junc_frame > 0)
        for x, y in zip(junctions[0], junctions[1]):
            cc_id = labelled_img[x, y]
            cc_dict[cc_id] = ...

        # exit()


def fuz_isolated_junctions(group, series_num):

    nps, skdata = get_all_junc(group, series_num)

    nps = np.array(nps)
    skdata = np.array(skdata)

    spread_img = np.zeros((128, 128))
    for each in skdata:
        spread_img[each[0], each[1]] = 255.

    # plt.imshow(spread_img)
    # plt.show()
    #
    # exit()

    labelled_img = label(spread_img, connectivity=2)

    # plt.imshow(labelled_img)
    # plt.show()
    #
    # exit()

    regions = regionprops(labelled_img)

    # cc_list = []
    # for idx in range(1, labelled_img.max()):
    #     lab_i = props[idx].label

    cc_area_dict = {idx: props.area for idx, props in enumerate(regions)}
    # print(cc_area_dict)

    # exit()

    num_components = np.unique(labelled_img)

    label_vals, assigned_components = get_junction_types(nps, labelled_img)

    unassigned_cc_dict = get_uncertain_junctions(labelled_img, skdata, num_components, assigned_components)

    isolated_junc = []
    isolated_junc_area = []
    fuzzy_junc = []
    fuzzy_junc_area = []
    for k, v in label_vals.items():
        if k != 0:
            if len(v) == 1:
                isolated_junc.append(v[0])
                isolated_junc_area.append(cc_area_dict[k])
            else:
                fuzzy_junc.append(v)
                fuzzy_junc_area.append(cc_area_dict[k])

    print(isolated_junc_area)
    print(fuzzy_junc_area)

    unknown_junc = [v for k, v in unassigned_cc_dict.items()]
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
            img = imageio.imread(confocal_data_path + f'{group}/files/img_{series_num}_decon_t0{i:02d}.tif')

        else:
            img = imageio.imread(confocal_data_path + f'{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch00.tif')

        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img, cmap='gray')
        plt.plot(iso[:, 1], iso[:, 0], 'o', markerfacecolor='None', markeredgecolor='red')
        if len(fuz) > 0:
            plt.plot(fuz[:, 1], fuz[:, 0], 'o', markerfacecolor='None', markeredgecolor='blue')
        plt.plot(unk[:, 1], unk[:, 0], 'o', markerfacecolor='None', markeredgecolor='green')
        plt.plot(skdata[:, 1], skdata[:, 0], 'x', markerfacecolor='None', markeredgecolor='yellow')
        plt.show()
        # if group == 'Control':
        #     plt.savefig('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junc_types_movies/Ct%s_decon_t0%s_ch00.png'%(f'{group}', f'{series_num}', f'{i:02d}'), bbox_inches='tight', pad_inches=0)
        # else:
        #     plt.savefig('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junc_types_movies/%s_decon_t0%s_ch00.png'%(f'{group}', f'{group[0]}{series_num}', f'{i:02d}'), bbox_inches='tight', pad_inches=0)
        # plt.close()


fuz_isolated_junctions('Control', 13)

# for i in range(1, 27):
#     fuz_isolated_junctions('ATL', i)

exit()

# for i in range(6, 32):
#     fuz_isolated_junctions('Control', i)

# for i in range(1, 32):
#     fuz_isolated_junctions('Climp', i)
#
# for i in range(24, 30):
#     fuz_isolated_junctions('RTN', i)
#
# exit()


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
    img = imageio.imread((confocal_data_path + f'{group}/new_op_jul/er_mean/{group.lower()}{num_series}_er_mean.png'))

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
    plt.suptitle(f'Climp series {num_series} junctions movement variance')
    plt.title('Variance of list with distances for matched junctions per frame w.r.t. reference frame junctions')
    # plt.savefig('RTN1_junc_spread.png', bbox_inches='tight', pad_inches=0)
    plt.show()


plot_junc_spread('Climp', n1, n2, 3)
exit()


def junction_var_median():
    var_vals = [each[0] for each in n3]
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
ll = [[each[0], each[1]] for each in l]
ll = np.array(ll)

# print(nps[:, 1])
# print(ll[:, 1])

# print(np.array(dt_refined.keys())[0])
# print(np.array(dt_refined.keys())[:, 1])

plt.imshow(imageio.imread(confocal_data_path + 'ATL/new_op_jul/er_mean/atl1_er_mean.png'),
           cmap='gray')
plt.plot(nps[:, 1], nps[:, 0], 'r.')
plt.plot(ll[:, 1], ll[:, 0], 'b.')
plt.savefig('refined_atl1_er_junctions_50.png', bbox_inches='tight', pad_inches=0)
plt.close()
# plt.show()

# plt.plot(list(dt_refined.keys())[:, 1])
exit()


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
        ref_input = imageio.imread(confocal_data_path + f'Control/files/img_{num_series}_decon_t000.tif')

    else:
        ref_input = imageio.imread((confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t000_ch00.tif'))


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
                er_input = imageio.imread((confocal_data_path + f'Control/files/img_{num_series}_decon_t0{i:02d}.tif'))

            else:
                er_input = imageio.imread((confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t0{i:02d}_ch00.tif'))


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


def junc_intensity_variation_runner():
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



# for i in range(100):
#     img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0%s_ch00.tif'%f'{i:02d}')
#     img = (img - img.min()) / (img.max() - img.min())
#
#     plt.imshow(img, cmap='gray')
#     plt.plot(79, 3, 'b.')
#     plt.show()


def junction_location_plotter(group, num_series):
    # mean_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'

    group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}
    mean_img = confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'


    newps = get_junctions(mean_img)

    nps = [[each[0], each[1]] for each in newps]
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
            path = confocal_data_path + f'{group}/files/img_{num_series}_decon_t0{frame:02d}.tif'

        else:
            path = confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t0{frame:02d}_ch00.tif'

        img = imageio.imread(path)
        img = (img - img.min()) / (img.max() - img.min())

        # ER Skel image
        # sk_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t0%s_ch00_skel.png'%f'{i:02d}'
        sk_img = confocal_data_path + f'{group}/new_op_jul/skel/{group_pref[group]}{num_series}/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_skel.png'


        sk_newps = get_junctions(sk_img)

        sk_nps = [[each[0], each[1]] for each in sk_newps]
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

        plt.savefig((confocal_data_path + f'{group}/new_op_jul/{group.lower()}_junc_viz/{group_pref[group]}{num_series}_t{frame:02d}.png'), bbox_inches='tight', pad_inches=0)


        plt.close()


# junction_location_plotter('RTN', 1)
# exit()


def crop_img():
    mean_img = confocal_data_path + 'ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'

    newps = get_junctions(mean_img)

    nps = [[each[0], each[1]] for each in newps]
    nps = np.array(nps)
    for i in range(100):
        img = imageio.imread(confocal_data_path + f'ATL/new_op_jul/A1_junc_viz/j48_skel/ATL1_t{i:02d}.png')


        # plt.imshow(img)
        # plt.show()

        # y = nps[74,1]
        # x = nps[74,0]
        # #
        # # # print(y, x)
        cimg = img[137 - 30:137 + 30, 143 - 30:143 + 30]

        plt.axis('off')
        plt.title(f't={i}')
        plt.imshow(cimg, interpolation='nearest', aspect='auto')
        plt.savefig(confocal_data_path + f'ATL/new_op_jul/A1_junc_viz/j48_skel/crops/ATL1_t{i:02d}.png', bbox_inches='tight', pad_inches=0)


        # imageio.imsave('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/A1_junc_viz/j47_skel/crops/ATL1_t%s.png'%f'{i:02d}', cimg)
        plt.close()


def per_patch_pixel_fourier(group, channel):
    grp_dict = {}
    nd = {}
    sldt = []

    for num_series in range(1, 2):
        mean_img = confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

        newps = get_junctions(mean_img)

        sl = []
        for frame in range(100):
            if group == 'Control':
                path = confocal_data_path + f'{group}/files/img_{num_series}_decon_t0{frame:02d}.tif'

            else:
                path = confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t0{frame:02d}_ch0{channel}.tif'

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


def inter_channel_correlation_runner():
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
        mean_img = confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

        newps = get_junctions(mean_img)

        sl = []

        for frame in range(100):
            if group == 'Control':
                path = confocal_data_path + f'{group}/files/img_{num_series}_decon_t0{frame:02d}.tif'

            else:
                path = confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t0{frame:02d}_ch00.tif'

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


def per_patch_if_corr():
    grp_list = []
    for num_series in range(1, 25):
        mean_img = confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

        newps = get_junctions(mean_img)

        sl = []

        for frame in range(100):
            if group == 'Control':
                path = confocal_data_path + f'{group}/files/img_{num_series}_decon_t0{frame:02d}.tif'

            else:
                path = confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t0{frame:02d}_ch00.tif'

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


def get_group_dif(group, metric):
    grp_list = []
    global met_val
    if group == 'Control':
        for series_num in range(1, 25):
            newps = get_junctions(confocal_data_path + f'Control/new_op_jul/Ctrl_mean_proj/Ct{series_num}_mean.png')

            ser_list = []
            for i in range(99):
                i1 = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_{series_num}_decon_t0{i:02d}.tif')

                i2 = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_{series_num}_decon_t0{i + 1:02d}.tif')


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
            newps = get_junctions(confocal_data_path + f'{group}/new_op_jul/{group}_mean_proj/{group[0]}{series_num}_mean.png')

            ser_list = []
            for i in range(98):
                i1 = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch00.tif')

                i2 = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/{group[0]}{series_num}_decon_t0{i + 1:02d}_ch00.tif')


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
