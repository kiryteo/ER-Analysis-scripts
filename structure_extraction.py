# All steps of the structure extraction from ER input samples
# Load the ER samples
# Std ER samples


import skimage
from skimage.filters import threshold_local
import os
import imageio
import cv2
from plantcv import plantcv as pcv
import sknw
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def get_std_img(path):
    img = imageio.imread(path)
    std_img = (img - img.min()) / (img.max() - img.min())
    return std_img


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
    processed_sample = threshold_local(img_thr_loc, 3)
    return processed_sample


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
    for series in range(num_series):
        os.makedirs(new_pref + 'C%s' % f'{i}')
        for sample in range(100):
            img_path = ...
            # img = imageio.imread(path_pref + 'Series%s_decon_converted/files/Series%s_decon_converted_t%s_ch00.tif' % (f'{i:03d}', f'{i:03d}', f'{j:02d}'))
            processed_sample = preproc_individual_sample(img_path)

            cv2.imwrite(new_pref + 'Series%s_decon_converted/new_op_sept/preproc/C%s/C%s_decon_t0%s_ch00_proc.png' % (f'{i:03d}', f'{i}', f'{i}', f'{j:02d}'), processed_sample)


def get_skeleton(img_path):
    """

    @param img_path: path to vessel enhancement output of preproc sample
    @return: extracted skeleton
    """
    vess_enhanced_sample = imageio.imread(img_path)
    skeleton = pcv.morphology.skeletonize(mask=vess_enhanced_sample)
    return skeleton


def skel_to_graph(skel_img_path):
    """

    @param skel_img_path:
    @return:
    """
    skeleton_input = imageio.imread(skel_img_path)
    graph = sknw.build_sknw(skeleton_input, iso=False)
    return graph


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

    # select the nodes with degree > 2
    relevant_nodes = []
    for node_num, degree_val in enumerate(degree_list):
        if degree_val[1] > 2:
            relevant_nodes.append(node_coords[node_num])

    return relevant_nodes


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
    for (start_node, end_node) in edges_list:
        tubule_coords = graph[start_node][end_node]['pts']
        tubule_coords_list.append(tubule_coords)

    return tubule_coords_list


def get_relevant_tubules(graph, relevant_nodes):
    """

    @param graph:
    @param relevant_nodes:
    @return:
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

    # Check if both start and end node of the edge exists
    # in the relevant node list. Include edges accordingly
    relevant_edge_list = []
    for (start_node, end_node) in edge_set:
        if start_node in relevant_node_list and end_node in relevant_node_list:
            tubule_coords = graph[start_node][end_node]['pts']
            relevant_edge_list.append(tubule_coords)

    return relevant_edge_list


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
    plt.plot(ps[:,1], ps[:,0], 'r.')

    # draw edges by pts
    for (start_node, end_node) in graph.edges():
        ps = graph[start_node][end_node]['pts']
        plt.plot(ps[:,1], ps[:,0], 'green')

    plt.show()


def plot_relevant_