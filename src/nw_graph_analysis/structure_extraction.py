# All steps of the structure extraction from ER input samples
# Load the ER samples
# Std ER samples

from junction_analysis import *

import contextlib
import skimage
from skimage.filters import threshold_local
import os
import imageio
import cv2
import itertools
from plantcv import plantcv as pcv
import sknw
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_std_img(path):
    img = imageio.imread(path)
    return (img - img.min()) / (img.max() - img.min())


def preproc_individual_sample(img_path):
    """

    @param img_path: path to ER input sample
    @return: processed sample
    """
    img = imageio.imread(img_path)
    std_img = ((img) / (img.max() - img.min())) * 255
    img_aop = skimage.morphology.area_opening(std_img, area_threshold=2)
    img_erod = skimage.morphology.erosion(img_aop)
    img_aop = skimage.morphology.area_opening(img_erod, area_threshold=2)
    img_closing = skimage.morphology.area_closing(img_aop, area_threshold=32)
    img_aop = skimage.morphology.area_opening(img_closing, area_threshold=2)
    img_thr_loc = threshold_local(img_aop, 3)
    return threshold_local(img_thr_loc, 3)


def preproc_groups(path, group, num_series):
    """

    @param path: path to all ER input files
    @param group: group to process (ATL, Climp, Control, RTN)
    @param num_series: number of movies in the group
    """
    # path_pref = '/localhome/asa420/MIAL/data/live-cell-movies/' + group + '/Decon/'
    # # new_pref = '/localhome/asa420/MIAL/data/confocal_movies/' + group + '/new_op_jul/preproc/'
    # new_pref = '/localhome/asa420/MIAL/data/live-cell-movies/' + group + '/Decon/'

    # for i in range(2, 3):
    for _ in range(num_series):
        os.makedirs(f'{new_pref}C{i}')
        for _ in range(100):
            img_path = ...
            # img = imageio.imread(path_pref + 'Series%s_decon_converted/files/Series%s_decon_converted_t%s_ch00.tif' % (f'{i:03d}', f'{i:03d}', f'{j:02d}'))
            processed_sample = preproc_individual_sample(img_path)

            cv2.imwrite(f'{new_pref}Series{i:03d}_decon_converted/new_op_sept/preproc/C{i}/C{i}_decon_t0{j:02d}_ch00_proc.png', processed_sample)


# run Vessel2d.m to get the vessel enhancement output

def get_skeleton(img_path):
    """

    @param img_path: path to vessel enhancement output of preproc sample
    @return: extracted skeleton
    """
    vess_enhanced_sample = imageio.imread(img_path)
    return pcv.morphology.skeletonize(mask=vess_enhanced_sample)


def skel_to_graph(skel_img_path):
    """

    @param skel_img_path:
    @return:
    """
    skeleton_input = imageio.imread(skel_img_path)
    return sknw.build_sknw(skeleton_input, multi=True, iso=False)


def get_junctions(graph):
    """

    @param graph: Input graph to obtain the junctions
    @return: nodes (junctions) with degree > 2
    """
    # get all the nodes from the graph
    nodes_list = graph.nodes()
    node_coords = np.array([nodes_list[node]['o'] for node in nodes_list])

    # degree_list provides list of tuples with node id followed by its degree
    degree_list = graph.degree

    return [node_coords[node_num] for node_num, degree_val in enumerate(degree_list) if degree_val[1] > 2]


def get_relevant_nodes(junctions):
    nps = [[junction[0], junction[1]] for junction in junctions]
    nps = np.array(nps)
    return nps


def get_tubules(graph):
    """

    @param graph: Input graph to obtain the edges (tubules)
    @return: List of coordinates for all tubules
    """
    # store all the tubule coordinates (edges)
    tubule_coords_list = []

    # graph.edges provides list of tuples with start and end node of the edge
    edges_list = graph.edges()

    # get the list of edge coordinates list
    tubule_coords_list.extend(graph[start_node][end_node][0]['pts'] for start_node, end_node in edges_list)

    return tubule_coords_list


def get_relevant_tubules(graph, relevant_nodes):
    """

    @param graph: Input graph from skeleton
    @param relevant_nodes: nodes in the graph with degree > 2
    @return: tubules corresponding to nodes with degree > 2
    """

    node_set = graph.nodes()
    edge_set = graph.edges()

    # get the relevant nodes which provide start and end points
    # for corresponding edge
    relevant_node_list = []
    for r_node in relevant_nodes:
        for each in node_set:
            node_val = node_set[each]['o']
            # print(node_set[each]['o'])
            if r_node[0] == node_val[0] and r_node[1] == node_val[1]:
                relevant_node_list.append(each)

    return [graph[start_node][end_node][0]['pts'] for start_node, end_node in edge_set if start_node in relevant_node_list and end_node in relevant_node_list]


def plot_original_graph(skel_img_path):
    """

    @param skel_img_path: path the input skeleton
    """
    graph = skel_to_graph(skel_img_path)

    plt.axis('off')
    plt.imshow(imageio.imread(skel_img_path), cmap='gray')

    # draw node by o
    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])
    plt.plot(ps[:, 1], ps[:, 0], 'r.')

    # draw edges by pts
    for (start_node, end_node) in graph.edges():
        ps = graph[start_node][end_node][0]['pts']
        plt.plot(ps[:, 1], ps[:, 0], 'green')
        with contextlib.suppress(Exception):
            ps_multi = graph[start_node][end_node][1]['pts']
            plt.plot(ps_multi[:, 1], ps_multi[:, 0], 'cyan')
    plt.show()


# path = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t000_ch00_skel.png'
# plot_original_graph(path)
# exit()


def plot_relevant_graph(skel_img_path, relevant_nodes, relevant_edge_list):
    """

    @param relevant_nodes:
    @param relevant_edge_list:
    @return:
    """
    skel_img = imageio.imread(skel_img_path)

    plt.axis('off')
    plt.imshow(skel_img, cmap='gray')

    plt.plot(relevant_nodes[:, 1], relevant_nodes[:, 0], 'b.')

    for edge in relevant_edge_list:
        plt.plot(edge[:, 1], edge[:, 0], 'green')

    plt.show()


def plot_total_graph(skel_path, graph, relevant_nodes, relevant_edge_list):
    """

    @param graph: ER graph (obtained from skeleton)
    @param relevant_nodes: degree 3 and more nodes
    @param relevant_edge_list: edges specific to relevant nodes
    @return:
    """

    input = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/files/C1_decon_t050_ch00.tif')
    ip = (input - input.min())/(input.max() - input.min())
    plt.imshow(ip, cmap='gray')

    # img = imageio.imread(skel_path)
    # plt.axis('off')
    # plt.imshow(img, cmap='gray')

    # draw edges by pts
    for (start_node, end_node) in graph.edges():
        ps = graph[start_node][end_node][0]['pts']
        plt.plot(ps[:, 1], ps[:, 0], 'green')

    for each in relevant_edge_list:
        plt.plot(each[:, 1], each[:, 0], 'red', mew=2.8)

    # draw node by o
    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])
    # plt.plot(ps[:, 1], ps[:, 0], 'r.')
    plt.plot(ps[:, 1], ps[:, 0], 'o', markerfacecolor='yellow', markeredgecolor='yellow', mew=0.5, markersize=3)

    # plt.plot(relevant_nodes[:, 1], relevant_nodes[:, 0], 'b.')
    plt.plot(relevant_nodes[:, 1], relevant_nodes[:, 0], 'o', markerfacecolor='blue', markeredgecolor='blue', markersize=4)

    plt.show()


def runner(group, r_start, r_end):

    l = []

    pref = 'Ct' if group == 'Control' else group[0]
    for i, frame in itertools.product(range(r_start, r_end+1), range(100)):
        # input = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/{group[0]}{i}_decon_t0{frame:02d}_ch00.tif')
        # ip = (input - input.min())/(input.max() - input.min())

        path = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/{pref}{i}/{pref}{i}_decon_t0{frame:02d}_ch00_skel.png'
        graph = skel_to_graph(path)
        junctions = get_junctions(graph)
        relevant_nodes = get_relevant_nodes(junctions)
        relevant_edges = get_relevant_tubules(graph, relevant_nodes)
        l.append(len(relevant_edges))

    # plot_total_graph(path, graph, relevant_nodes, relevant_edges)
    return l


def rel_edges_length(group, r_start, r_end):
    l = []
    for i, frame in itertools.product(range(r_start, r_end+1), range(100)):
        # input = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/{group[0]}{i}_decon_t0{frame:02d}_ch00.tif')
        # ip = (input - input.min())/(input.max() - input.min())

        path = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/{group[0]}{i}/{group[0]}{i}_decon_t0{frame:02d}_ch00_skel.png'
        graph = skel_to_graph(path)
        junctions = get_junctions(graph)
        relevant_nodes = get_relevant_nodes(junctions)
        relevant_edges = get_relevant_tubules(graph, relevant_nodes)
        l.extend(len(each) for each in relevant_edges)

    return l


nps, skdata, labelled_img = label_junctions(group, ser_num)
label_vals, unassigned_cc_dict = separate_junc_cc(nps, skdata, labelled_img)
iso, fuz, unk = get_junction_areas(label_vals, unassigned_cc_dict)


def rel_edge_intensity(group, r_start, r_end):
    l_mean = []
    l_std = []
    pref = 'Ct' if group == 'Control' else group[0]

    for i, frame in itertools.product(range(r_start, r_end+1), range(100)):
        if group == 'Control':
            fname = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/img_{i}_decon_t0{frame:02d}.tif'
        else:
            fname = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/{group[0]}{i}_decon_t0{frame:02d}_ch00.tif'

        img = imageio.imread(fname)
        ip = (img - img.min())/(img.max() - img.min())
        path = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/{pref}{i}/{pref}{i}_decon_t0{frame:02d}_ch00_skel.png'
        graph = skel_to_graph(path)
        junctions = get_junctions(graph)
        relevant_nodes = get_relevant_nodes(junctions)
        relevant_edges = get_relevant_tubules(graph, relevant_nodes)
        for edge in relevant_edges:
            edge_intensity_data = [ip[coords[0], coords[1]] for coords in edge]
            # l_mean.append(np.mean(edge_intensity_data))
            l_std.append(np.std(edge_intensity_data))

    return l_std


atl_std = rel_edge_intensity('ATL', 1, 26)
cl_std = rel_edge_intensity('Climp', 1, 31)
rt_std = rel_edge_intensity('RTN', 1, 29)
ct_std = rel_edge_intensity('Control', 1, 31)

df = pd.DataFrame()

df['Tubule_intensity_std'] = pd.Series(np.concatenate((atl_std, cl_std, rt_std, ct_std)))
df['Group'] = pd.Series(np.concatenate((['ATL']*len(atl_std), ['Climp']*len(cl_std), ['RTN']*len(rt_std), ['Control']*len(ct_std))))

# sns.distplot(atl_mean, hist=False, label='atl')
# sns.distplot(cl_mean, hist=False, label='cl')
# sns.distplot(rt_mean, hist=False, label='rtn')
# sns.distplot(ct_mean, hist=False, label='ctrl')
sns.violinplot(data=df, y='Group', x='Tubule_intensity_std')
plt.title('Standard deviation per tubule intensity for tubules (edges) corresponding to nodes with degree greater '
          'than two', fontsize=20)
plt.xlabel('Intensity standard deviation', fontsize=18)
plt.ylabel('Group', fontsize=18)
# plt.legend()
plt.show()


exit()

atl = runner('ATL', 1, 26)
climp = runner('Climp', 1, 31)
rtn = runner('RTN', 1, 29)
ctrl = runner('Control', 1, 31)

df = pd.DataFrame()

df['Length'] = pd.Series(np.concatenate((atl, climp, rtn, ctrl)))
df['Group'] = pd.Series(np.concatenate((['ATL']*len(atl), ['Climp']*len(climp), ['RTN']*len(rtn), ['Control']*len(ctrl))))

# print(atl)
# print(climp)
# print(rtn)
# exit()

ax = sns.violinplot(data=df, y='Group', x='Length')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
# plt.title('Length of edges corresponding to nodes with degree greater than two (replicate 2)', fontsize=20)
plt.title('Count of edges corresponding to nodes with degree greater than two', fontsize=20)
plt.xlabel('Number of edges', fontsize=18)
plt.ylabel('Group', fontsize=18)
# sns.distplot(atl, hist=False, label='ATL')
# sns.distplot(climp, hist=False, label='Climp')
# sns.distplot(rtn, hist=False, label='RTN')
# plt.legend()
plt.show()
