# All steps of the structure extraction from ER input samples
# Load the ER samples
# Std ER samples


import contextlib
import skimage
from skimage.filters import threshold_local
import os
import imageio
import cv2
import itertools
import copy
from plantcv import plantcv as pcv
import sknw
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.graph import route_through_array
from junction_analysis_modules import *
from collections import OrderedDict
import numpy as np
from sklearn.neighbors import NearestNeighbors
from statannotations.Annotator import Annotator

from junction_analysis_modules import JunctionAnalysis as JA


confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'


# group = 'ATL'
# series = 1
#
# path = f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{series}_er_mean_proc_enhance_skel.png'
#
# # path_frame = f'{confocal_data_path}{group}/new_op_jul/skel/{group[0]}{series}/{group[0]}{series}_decon_t006_ch00_skel.png'
#
# graph = sknw.build_sknw(imageio.imread(path), multi=True, iso=False)
# for each in graph.nodes():
#     print(each)
#
# for each in graph.nodes:
#     print(each)
#
#
# exit()


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

    # @param path: path to all ER input files
    # @param group: group to process (ATL, Climp, Control, RTN)
    # @param num_series: number of movies in the group
    """
    # path_pref = '/localhome/asa420/MIAL/data/live-cell-movies/' + group + '/Decon/'
    # # new_pref = '{confocal_data_path}' + group + '/new_op_jul/preproc/'
    # new_pref = '/localhome/asa420/MIAL/data/live-cell-movies/' + group + '/Decon/'

    # for i in range(2, 3):
    for _ in range(num_series):
        os.makedirs(f'{new_pref}C{i}')
        for _ in range(100):
            img_path = ...
            # img = imageio.imread(path_pref + 'Series%s_decon_converted/files/Series%s_decon_converted_t%s_ch00.tif' % (f'{i:03d}', f'{i:03d}', f'{j:02d}'))
            processed_sample = preproc_individual_sample(img_path)

            cv2.imwrite(
                f'{new_pref}Series{i:03d}_decon_converted/new_op_sept/preproc/C{i}/C{i}_decon_t0{j:02d}_ch00_proc.png',
                processed_sample)

# run Vessel2d.m to get the vessel enhancement output
def get_skeleton(img_path):
    """
    Extracts the skeleton from a vessel-enhanced image.

    @param img_path (str): The file path of the vessel-enhanced image.
    @return: extracted skeleton (numpy.ndarray)
    """
    # Load the vessel-enhanced image using the imageio library
    vess_enhanced_sample = imageio.imread(img_path)

    return pcv.morphology.skeletonize(mask=vess_enhanced_sample)

def skel_to_graph(skel_img_path):
    """
    @param skel_img_path:
    @return:
    """
    return sknw.build_sknw(imageio.imread(skel_img_path), multi=True, iso=False)

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

def get_relevant_tubules(self, graph, relevant_nodes):
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


def plot_total_graph(group, num_series, graph, relevant_nodes, relevant_edge_list):
    """

    @param graph: ER graph (obtained from skeleton)
    @param relevant_nodes: degree 3 and more nodes
    @param relevant_edge_list: edges specific to relevant nodes
    @return:
    """

    pref = 'Ct' if group == 'Control' else group[0]

    # projection frame analysis
    input = imageio.imread(
        f'{confocal_data_path}{group}/new_op_jul/er_mean/{group.lower()}{num_series}_er_mean.png')

    # per frame analysis
    # input = imageio.imread(f'{confocal_data_path}{group}/files/{pref}{num_series}_decon_t050_ch00.tif')
    # input = imageio.imread(f'{confocal_data_path}{group}/files/img_{num_series}_decon_t050.tif')
    # ip = (input - input.min())/(input.max() - input.min())

    ip = input
    plt.imshow(ip, cmap='gray')

    # img = imageio.imread(skel_path)
    # plt.axis('off')
    # plt.imshow(img, cmap='gray')

    # draw edges by pts
    for (start_node, end_node) in graph.edges():
        ps = graph[start_node][end_node][0]['pts']
        plt.plot(ps[:, 1], ps[:, 0], 'green')
        with contextlib.suppress(Exception):
            ps_multi = graph[start_node][end_node][1]['pts']
            plt.plot(ps_multi[:, 1], ps_multi[:, 0], 'green')

    for each in relevant_edge_list:
        plt.plot(each[:, 1], each[:, 0], 'red', mew=2.8)

    # draw node by o
    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])
    # plt.plot(ps[:, 1], ps[:, 0], 'r.')
    plt.plot(ps[:, 1], ps[:, 0], 'o', markerfacecolor='yellow', markeredgecolor='yellow', mew=0.5, markersize=3)

    # plt.plot(relevant_nodes[:, 1], relevant_nodes[:, 0], 'b.')
    plt.plot(relevant_nodes[:, 1], relevant_nodes[:, 0], 'o', markerfacecolor='blue', markeredgecolor='blue',
             markersize=4)
    plt.axis('off')
    # plt.savefig(f'graphs/{group}_{num_series}_edge_graph_projection', bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{group}_{num_series}_edge_graph_conn', bbox_inches='tight', pad_inches=0)
    plt.close()
    # plt.show()


def get_nbrs(nodes_array):
    """

    @param nodes_array: np array with nodes are [x, y] lists
    """

    # Create a NearestNeighbors object and fit the data
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(nodes_array)

    # Get the distances and indices of the nearest neighbors
    distances, indices = nbrs.kneighbors(nodes_array)

    # Print the indices of the nearest neighbors for each element
    # print(indices[:,1])
    return distances[:, 1], indices[:, 1]


def nearest_node(distances, nbrs):
    nbr_dict = {i: (nbrs[i], distances[i]) for i in range(len(nbrs))}

    # Create a dictionary of nearest neighbors for each node
    nn_dict = {}
    for node, (nbr, dist) in nbr_dict.items():
        if nbr not in nn_dict or nn_dict[nbr][1] > dist:
            nn_dict[nbr] = (node, dist)

    # print(nn_dict)
    # exit()

    return nbr_dict, nn_dict


# Find the closest neighbor of a given node in a graph
def get_closest_from_nbr(graph, node):
    # Get the list of neighbors and their distances to the node
    neighbors = list(graph.neighbors(node))
    distances = [graph.edges[node, neighbor, 0]['weight'] for neighbor in neighbors]

    # Find the index of the closest neighbor in the list of neighbors
    closest_neighbor_index = distances.index(min(distances))

    # Return the closest neighbor and its distance to the node
    return neighbors[closest_neighbor_index], min(distances)


# Get the coordinates of the path between a given node and its closest neighbor in a graph
def get_path_coords(er_input, cost_arr, g_nodes_array, node, fin_dict):
    # Get the starting and ending coordinates of the path
    start_coord = tuple(g_nodes_array[node][:2])
    end_coord = tuple(g_nodes_array[fin_dict[node][0]][:2])

    # Find the path coordinates using the `route_through_array()` function
    path_coords, _ = route_through_array(cost_arr, start=start_coord, end=end_coord, fully_connected=True)

    zero_signal_coords = sum(er_input[each] == 0 for each in path_coords)

    signal_coords = len(path_coords) - zero_signal_coords

    if zero_signal_coords > signal_coords:
        return None

    # Return the path coordinates as a NumPy array if the path is short enough, otherwise return None
    return np.array(path_coords) if len(path_coords) < 20 else None


import math


def connect_low_degree_nodes(temp_graph, node, fin_dict, path_coords):
    temp_graph.add_edge(node, fin_dict[node][0])
    total_distance = sum(
        math.sqrt((path_coords[i + 1][0] - path_coords[i][0]) ** 2 + (path_coords[i + 1][1] - path_coords[i][1]) ** 2)
        for i in range(len(path_coords) - 1))
    return {(node, fin_dict[node][0], 0): {'pts': path_coords, 'weight': total_distance}}


def remove_edge_if_exists(temp_graph, node, neighbor):
    if temp_graph.has_edge(node, neighbor) or temp_graph.has_edge(neighbor, node):
        temp_graph.remove_edge(node, neighbor)
        temp_graph.remove_nodes_from((node, neighbor))


def connect_nodes(er_input, temp_graph, n1, n2, fin_dict, cost_arr, g_nodes_array):
    if fin_dict[n1][0] != n2:

        # print(n1)
        path_coords = get_path_coords(er_input, cost_arr, g_nodes_array, n1, fin_dict)

        if path_coords is not None and not temp_graph.has_edge(n1, fin_dict[n1][0]):
            edge_data = connect_low_degree_nodes(temp_graph, n1, fin_dict, path_coords)

            nx.set_edge_attributes(temp_graph, edge_data)


def get_updated_degree_nodes(temp_graph):
    deg_one_nodes, deg_two_nodes, high_deg_nodes = [], [], []
    for node, degree in temp_graph.degree:
        if 'o' in temp_graph.nodes[node]:
            point = list(temp_graph.nodes[node]['o'])
            if degree == 1:
                deg_one_nodes.append(point)
            elif degree == 2:
                deg_two_nodes.append(point)
            else:
                high_deg_nodes.append(point)
    return np.array(deg_one_nodes), np.array(deg_two_nodes), np.array(high_deg_nodes)


def create_new_path(path_nbr1, path_nbr2):
    union_dict = OrderedDict.fromkeys(map(tuple, path_nbr1 + path_nbr2))

    return list(map(list, union_dict.keys()))


def get_new_edge_data(tgraph, nbr_a, nbr_b, path_a, path_b):
    path_coords = np.array(create_new_path(path_a, path_b))
    total_distance = sum(
        math.sqrt((path_coords[i + 1][0] - path_coords[i][0]) ** 2 + (path_coords[i + 1][1] - path_coords[i][1]) ** 2)
        for i in range(len(path_coords) - 1))
    tgraph.add_edge(nbr_a, nbr_b)
    return {(nbr_a, nbr_b, 0): {'pts': path_coords, 'weight': total_distance}}


def high_deg_connections(tgraph, node, nbr1, nbr2):
    path_nbr1 = [[int(x) for x in a] for a in tgraph[node][nbr1][0]['pts']]
    path_nbr2 = [[int(x) for x in a] for a in tgraph[node][nbr2][0]['pts']]

    if path_nbr1[0] == path_nbr2[0]:
        path_nbr2 = path_nbr2[::-1]
        edge_data = get_new_edge_data(tgraph, nbr2, nbr1, path_nbr2, path_nbr1)

    elif path_nbr1[0] == path_nbr2[-1]:
        edge_data = get_new_edge_data(tgraph, nbr2, nbr1, path_nbr2, path_nbr1)

    elif path_nbr1[-1] == path_nbr2[0]:
        edge_data = get_new_edge_data(tgraph, nbr1, nbr2, path_nbr1, path_nbr2)

    else:
        path_nbr2 = path_nbr2[::-1]
        edge_data = get_new_edge_data(tgraph, nbr1, nbr2, path_nbr1, path_nbr2)

    tgraph.remove_node(node)
    nx.set_edge_attributes(tgraph, edge_data)


def process_node(tgraph, node):
    nbrs = list(tgraph.neighbors(node))
    if len(nbrs) != 2:
        return

    nbr1, nbr2 = nbrs
    deg1, deg2 = tgraph.degree(nbr1), tgraph.degree(nbr2)


    # if deg1 >= 2 and deg2 >= 2:
    #     high_deg_connections(tgraph, node, nbr1, nbr2)
    # if (deg1 == 1 and deg2 >= 2) or (deg1 >= 2 and deg2 == 1):
    #     high_deg_connections(tgraph, node, nbr1, nbr2)
    if deg1 >= 1 and deg2 >= 1:
        high_deg_connections(tgraph, node, nbr1, nbr2)


def get_updated_neighbor_dict(graph):
    nodes = graph.nodes()

    g_nodes = np.array([graph.nodes[i]['o'] for i in graph.nodes])
    g_nodes_array = g_nodes.tolist()

    # closest point in graph which is connected by edge ('neighbor')
    closest_neighbor_dict = {}
    for each in nodes:
        nbr_id, dist = get_closest_from_nbr(graph, each)
        closest_neighbor_dict[each] = (nbr_id, dist)

    # closest point in graph that may or may not be connected by an edge
    distances, nbrs_all = get_nbrs(g_nodes)

    # nbr_dict: node id and nearest node id
    # nn_dict:
    nbr_dict, nn_dict = nearest_node(distances, nbrs_all)

    fin_dict = {}

    # for k, v in closest_neighbor_dict.items():
    #     if k in nn_dict:
    #         fin_dict[k] = nn_dict[k]

    for k, v in closest_neighbor_dict.items():
        if v[1] < nbr_dict[k][1]:
            fin_dict[k] = closest_neighbor_dict[k]
        else:
            fin_dict[k] = nbr_dict[k]

    return fin_dict, g_nodes_array


def graph_node_connector(group, series):
    global rel
    pref = 'Ct' if group == 'Control' else group[0]

    # projection frame analysis
    # path = '{confocal_data_path}Climp/new_op_jul/er_mean_proc/climp16_er_mean_proc_enhance_skel.png'
    path = f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{series}_er_mean_proc_enhance_skel.png'

    # path_frame = f'{confocal_data_path}{group}/new_op_jul/skel/{group[0]}{series}/{group[0]}{series}_decon_t006_ch00_skel.png'


    path_proc_enh = f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{series}_er_mean_proc_enhance.png'

    path_er = f'{confocal_data_path}{group}/new_op_jul/er_mean/{group.lower()}{series}_er_mean.png'

    path_er_proc = f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{series}_er_mean_proc.png'

    graph = skel_to_graph(path)

    # all junction from the graph with degree > 2
    junc_analysis = JA(confocal_data_path)
    junctions = junc_analysis.get_junctions(graph)

    relevant_nodes = np.array(junctions)

    fin_dict, g_nodes_array = get_updated_neighbor_dict(graph)

    temp_graph = copy.deepcopy(graph)

    # er_proc = imageio.imread(path_er_proc)
    # er_proc_bg = np.where(er_proc==0)

    er_input = imageio.imread(path_er)
    cost_arr = np.ones((128, 128))
    # cost_arr[er_proc_bg] = 0

    for node in dict(graph.degree()):

        # access the first element of graph.neighbors
        neighbor = next(iter(graph.neighbors(node)))

        connect_nodes(er_input, temp_graph, node, neighbor, fin_dict, cost_arr, g_nodes_array)

    tgraph = copy.deepcopy(temp_graph)
    for node in temp_graph.nodes():
        process_node(tgraph, node)

    tgraph2 = copy.deepcopy(tgraph)
    for node in tgraph.nodes():
        process_node(tgraph2, node)


    # for (st, end) in tgraph2.edges():
    #     print(st, end)
    #
    #
    exclude_edges = []
    for (node1, node2) in tgraph2.edges():
        if tgraph2.degree(node1) == 1 or tgraph2.degree(node2) == 1:
            exclude_edges.append((node1, node2))
    #
    # print(exclude_edges)
    # exit()

    ### Plotting the updated graph
    deg_one_nodes, deg_two_nodes, high_deg_nodes = get_updated_degree_nodes(tgraph2)

    er_mean = imageio.imread(
        f'{confocal_data_path}{group}/new_op_jul/er_mean/{group.lower()}{series}_er_mean.png')
    plt.imshow(er_mean, cmap='gray')

    for (start_node, end_node) in tgraph2.edges():
        if tgraph2[start_node][end_node][0]:
            ps = tgraph2[start_node][end_node][0]['pts']
            if (start_node, end_node) not in exclude_edges:
                plt.plot(ps[:, 1], ps[:, 0], 'red')
            else:
                plt.plot(ps[:, 1], ps[:, 0], 'green')
        # elif temp_graph[start_node][end_node][1]:
        #     ps = tgraph2[start_node][end_node][1]['pts']
        #     plt.plot(ps[:, 1], ps[:, 0], 'red')

    if len(deg_one_nodes) != 0:
        plt.plot(deg_one_nodes[:, 1], deg_one_nodes[:, 0], 'o', markerfacecolor='yellow', markeredgecolor='yellow',
                 mew=0.5, markersize=3)

    # if len(deg_two_nodes) != 0:
    #     plt.plot(deg_two_nodes[:, 1], deg_two_nodes[:, 0], 'o', markerfacecolor='magenta', markeredgecolor='magenta',
    #              mew=0.5, markersize=3)

    if len(high_deg_nodes) != 0:
        plt.plot(high_deg_nodes[:, 1], high_deg_nodes[:, 0], 'o', markerfacecolor='blue', markeredgecolor='blue',
                 mew=0.5, markersize=3)

    # plt.axis('off')
    # plt.savefig(f'graphs/connected/repair/{group}_{series}_edge_graph_projection_connected_final_v2', bbox_inches='tight', pad_inches=0)
    # plt.savefig(f'graphs/connected/{group}_{series}_v2-2', bbox_inches='tight', pad_inches=0)
    # plt.close()
    # #
    plt.show()
    # return tgraph2





# for i in range(1, 27):
#     graph_node_connector('ATL', i)
#
# exit()


def node_connector(path_er, path_frame):

    graph = skel_to_graph(path_frame)

    # all junction from the graph with degree > 2
    junc_analysis = JA(confocal_data_path)
    junctions = junc_analysis.get_junctions(graph)

    relevant_nodes = np.array(junctions)

    fin_dict, g_nodes_array = get_updated_neighbor_dict(graph)

    temp_graph = copy.deepcopy(graph)

    # er_proc = imageio.imread(path_er_proc)
    # er_proc_bg = np.where(er_proc==0)

    er_input = imageio.imread(path_er)
    cost_arr = np.ones((128, 128))
    # cost_arr[er_proc_bg] = 0

    for node in dict(graph.degree()):
        # access the first element of graph.neighbors
        neighbor = next(iter(graph.neighbors(node)))

        connect_nodes(er_input, temp_graph, node, neighbor, fin_dict, cost_arr, g_nodes_array)

    tgraph = copy.deepcopy(temp_graph)
    for node in temp_graph.nodes():
        process_node(tgraph, node)

    tgraph2 = copy.deepcopy(tgraph)
    for node in tgraph.nodes():
        process_node(tgraph2, node)

    # deg_one_nodes, deg_two_nodes, high_deg_nodes = get_updated_degree_nodes(tgraph2)

    return tgraph2


# atl_t0 = node_connector('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/std_egfp/A1_decon_t000_ch00_std.png', '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t000_ch00_skel.png')
#
# atl_t1 = node_connector('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/std_egfp/A1_decon_t001_ch00_std.png', '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t001_ch00_skel.png')
#
# # t0 = skel_to_graph('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t000_ch00_skel.png')
# # t1 = skel_to_graph('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t001_ch00_skel.png')
#
# d1 = nx.optimize_graph_edit_distance(atl_t0, atl_t1)
# for v in d1:
#     print(v)
# # print(d1)
#
# exit()

# graph = graph_node_connector('ATL', 1)
# exit()
# print(graph.degree)
# print(graph.nodes)
# # graph_node_connector('ATL', 3)
# exit()


def runner(group, r_start, r_end):
    l = []

    for series in range(r_start, r_end+1):

        path_er = f'{confocal_data_path}{group}/new_op_jul/er_mean/{group.lower()}{series}_er_mean.png'

        path = f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{series}_er_mean_proc_enhance_skel.png'

        graph = node_connector(path_er, path)
        junc_analysis = JA(confocal_data_path)
        junctions = junc_analysis.get_junctions(graph)
        relevant_nodes = np.array(junctions)
        relevant_edges = get_relevant_tubules(graph, relevant_nodes)

        # print(relevant_edges)

        l.append(len(relevant_edges))
    return l


from statannot import add_stat_annotation

# atl = runner('ATL', 1, 10)
# climp = runner('Climp', 1, 10)
# rtn = runner('RTN', 1, 10)
# ctrl = runner('Control', 1, 10)
#
# atl_series = pd.Series(atl, name='ATL')
# climp_series = pd.Series(climp, name='Climp')
# rtn_series = pd.Series(rtn, name='RTN')
# ctrl_series = pd.Series(ctrl, name='Control')
#
# df = pd.concat([atl_series, climp_series, rtn_series, ctrl_series], axis=1)
#
#
# df_long = pd.melt(df, var_name='Group', value_name='Length')
#
#
#
#
# print(df_long)
# exit()
#
# #
# box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('ATL', 'Control'), ('Climp', 'RTN'), ('Climp', 'Control'), ('RTN', 'Control')]
#
# ax = sns.boxplot(data=df_long, y='Group', x='Length')
# # ax = sns.violinplot(data=df_long, y='Length', x='Group')
# ax.set_xscale('log')


# test_results = add_stat_annotation(ax, data=df_long, y="Group", x="Length", box_pairs=[('ATL', 'Climp'), ('ATL', 'RTN'), ('ATL', 'Control'), ('Climp', 'RTN'), ('Climp', 'Control'), ('RTN', 'Control')], test='Mann-Whitney', text_format='star', loc='outside')

# annot = Annotator(ax, box_pairs, data=df_long, x='Length', y='Group')
# annot.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annot.apply_and_annotate()
#
#
# plt.title('Count of edges corresponding to nodes with degree greater than two', fontsize=20)
# plt.xlabel('Number of edges', fontsize=18)
# plt.ylabel('Group', fontsize=18)
# plt.show()
# exit()


# atl = runner('ATL', 1, 2)
# climp = runner('Climp', 1, 2)
# rtn = runner('RTN', 1, 2)
# ctrl = runner('Control', 1, 2)
#
# df = pd.DataFrame()
#
# df['Length'] = pd.Series(np.concatenate((atl, climp, rtn, ctrl)))
# df['Group'] = pd.Series(np.concatenate((['ATL']*len(atl), ['Climp']*len(climp), ['RTN']*len(rtn), ['Control']*len(ctrl))))
#
# box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('ATL', 'Control'), ('Climp', 'RTN'), ('Climp', 'Control'), ('RTN', 'Control')]
#
# # print(atl)
# # print(climp)
# # print(rtn)
# # exit()
#
# ax = sns.violinplot(data=df, y='Group', x='Length')
# # ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
# # plt.title('Length of edges corresponding to nodes with degree greater than two (replicate 2)', fontsize=20)
# # ax.set_yscale('log')
#
# # statannot.add_stat_annotation(ax, x='Length', y='Group', data=df, box_pairs=box_pairs, test='Mann-Whitney', text_format='simple', loc='outside', verbose=2, fontsize='large')
#
# annot = Annotator(ax, box_pairs, data=df, x='Length', y='Group')
# annot.new_plot(ax=ax, pairs=box_pairs, plot='violinplot', data='df', x='Length', y='Group')
# annot.configure(test='Mann-Whitney', loc='inside')
# annot.annotate()
# # annot.apply_and_annotate()
#
# plt.title('Count of edges corresponding to nodes with degree greater than two', fontsize=20)
# # plt.xlabel('Number of edges', fontsize=18)
# # plt.ylabel('Group', fontsize=18)
# # sns.distplot(atl, hist=False, label='ATL')
# # sns.distplot(climp, hist=False, label='Climp')
# # sns.distplot(rtn, hist=False, label='RTN')
# # plt.legend()
# plt.show()
#
# exit()


def graph_plotter(group, series):
    pref = 'Ct' if group == 'Control' else group[0]

    # path = f'{confocal_data_path}{group}/new_op_jul/skel/{pref}{series}/{pref}{series}_decon_t050_ch00_skel.png'

    # projection frame analysis
    # path = '{confocal_data_path}Climp/new_op_jul/er_mean_proc/climp16_er_mean_proc_enhance_skel.png'


    # path = f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{series}_er_mean_proc_enhance_skel.png'

    path_frame = f'{confocal_data_path}{group}/new_op_jul/skel/{group[0]}{series}/{group[0]}{series}_decon_t006_ch00_skel.png'

    # plt.imshow(imageio.imread(path), cmap='gray')

    # graph = skel_to_graph(path_frame)
    graph = graph_node_connector(group, series)

    junc_analysis = JA(confocal_data_path)
    junctions = junc_analysis.get_junctions(graph)
    relevant_nodes = np.array(junctions)

    ps = np.array([graph.nodes[i]['o'] for i in graph.nodes])

    # pspp = []
    # rel = []
    #
    # # get ps and relevant_nodes in correct format
    # for each in ps:
    #     pspp.append([each[0], each[1]])
    # for each in relevant_nodes:
    #     rel.append([each[0], each[1]])
    #
    # # get only 1, 2 degree nodes (yellow spots)
    # low_deg_nodes = [x for x in pspp if x not in rel]
    #
    # # get nearest neighbour distances and indices for the 1, 2 degree nodes
    # distances, nbrs = get_nbrs(low_deg_nodes)
    #
    # edge_len_list = []
    #
    # new_node_indices = []
    # for (i, a), (j, b) in zip(enumerate(distances), enumerate(edge_len_list)):
    #     if a < b:
    #         new_node_indices.append(i)
    #
    # # print(new_node_indices)
    # # print(nbrs)
    # # print(low_deg_nodes)
    # # print(len(nbrs))
    # # print(len(low_deg_nodes))
    #
    # # print(graph.edges)
    # for each in new_node_indices:
    #     ldi_start = each
    #     ldi_end = nbrs[each]
    #
    #     u = pspp.index(low_deg_nodes[ldi_start])
    #     v = pspp.index(low_deg_nodes[ldi_end])
    #
    #     try:
    #         sh_path = nx.shortest_path(graph, u, v)
    #
    #         st_coord = (pspp[u][0], pspp[u][1])
    #         end_coord = (pspp[v][0], pspp[v][1])
    #
    #         new_path = [graph.nodes[n]['o'] for n in sh_path]
    #
    #         if graph.has_edge(u, v) or graph.has_edge(v, u):
    #             continue
    #         else:
    #             graph.add_edge(u, v)
    #
    #             npath = []
    #
    #             for ele in new_path:
    #                 npath.append(list(ele))
    #
    #             # npath = [(21, 59), (29, 74), (35, 76), (38, 75), (40, 67), (39, 56), (32, 50), (27, 58)]
    #
    #             edge_data = {(u, v, 0): {'pts': np.array(npath)}}
    #
    #             nx.set_edge_attributes(graph, edge_data)
    #     except:
    #         pass
    #
    # # print(graph.edges[(22, 29, 0)]['pts'])
    # # print(graph.edges)
    # plt.imshow(imageio.imread(path), cmap='gray')
    #
    # for (start_node, end_node) in graph.edges():
    #     ps = graph[start_node][end_node][0]['pts']
    #     plt.plot(ps[:, 1], ps[:, 0], 'green')
    #
    # plt.show()
    #
    # exit()

    relevant_edges = get_relevant_tubules(graph, relevant_nodes)

    plot_total_graph(group, series, graph, relevant_nodes, relevant_edges)


# graph_plotter('ATL', 1)
# exit()
# for i in range(1, 30):
#     graph_plotter('RTN', i)
# exit()


def rel_edges_length(group, r_start, r_end):
    l = []
    for i, frame in itertools.product(range(r_start, r_end + 1), range(100)):
        # input = imageio.imread(f'{confocal_data_path}{group}/files/{group[0]}{i}_decon_t0{frame:02d}_ch00.tif')
        # ip = (input - input.min())/(input.max() - input.min())

        path = f'{confocal_data_path}{group}/new_op_jul/skel/{group[0]}{i}/{group[0]}{i}_decon_t0{frame:02d}_ch00_skel.png'
        graph = skel_to_graph(path)
        junc_analysis = JA(confocal_data_path)
        junctions = junc_analysis.get_junctions(graph)
        relevant_nodes = np.array(junctions)
        relevant_edges = get_relevant_tubules(graph, relevant_nodes)
        l.extend(len(each) for each in relevant_edges)

    return l


# nps, skdata, labelled_img = label_junctions(group, ser_num)
# label_vals, unassigned_cc_dict = separate_junc_cc(nps, skdata, labelled_img)
# iso, fuz, unk = get_junction_areas(label_vals, unassigned_cc_dict)


def rel_edge_intensity(group, r_start, r_end):
    l_mean = []
    l_std = []
    pref = 'Ct' if group == 'Control' else group[0]

    for i, frame in itertools.product(range(r_start, r_end + 1), range(100)):
        if group == 'Control':
            fname = f'{confocal_data_path}{group}/files/img_{i}_decon_t0{frame:02d}.tif'
        else:
            fname = f'{confocal_data_path}{group}/files/{group[0]}{i}_decon_t0{frame:02d}_ch00.tif'

        img = imageio.imread(fname)
        ip = (img - img.min()) / (img.max() - img.min())
        path = f'{confocal_data_path}{group}/new_op_jul/skel/{pref}{i}/{pref}{i}_decon_t0{frame:02d}_ch00_skel.png'
        graph = skel_to_graph(path)
        junc_analysis = JA(confocal_data_path)
        junctions = junc_analysis.get_junctions(graph)
        relevant_nodes = np.array(junctions)
        relevant_edges = get_relevant_tubules(graph, relevant_nodes)
        for edge in relevant_edges:
            edge_intensity_data = [ip[coords[0], coords[1]] for coords in edge]
            # l_mean.append(np.mean(edge_intensity_data))
            l_std.append(np.std(edge_intensity_data))

    return l_std


import statannot


def rel_edge_count():
    a1 = runner('ATL', 1, 26)
    # a2 = runner('ATL', 11, 20)
    # a3 = runner('ATL', 21, 26)
    c1 = runner('Climp', 1, 31)
    # c2 = runner('Climp', 11, 20)
    # c3 = runner('Climp', 21, 31)
    r1 = runner('RTN', 1, 29)
    # r2 = runner('RTN', 11, 20)
    # r3 = runner('RTN', 21, 29)
    ct1 = runner('Control', 1, 31)
    # ct2 = runner('Control', 11, 20)
    # ct3 = runner('Control', 21, 31)

    df = pd.DataFrame()

    # df['Count'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3, ct1, ct2, ct3)))
    df['Count'] = pd.Series(np.concatenate((a1, c1, r1, ct1)))
    # df['Group'] = pd.Series(np.concatenate((['ATL']*len(a1), ['ATL']*len(a2), ['ATL']*len(a3), ['Climp']*len(c1), ['Climp']*len(c2), ['Climp']*len(c3), ['RTN']*len(r1), ['RTN']*len(r2), ['RTN']*len(r3), ['Control']*len(r3), ['Control']*len(r3), ['Control']*len(r3))))
    df['Group'] = pd.Series(
        np.concatenate((['ATL'] * len(a1), ['Climp'] * len(c1), ['RTN'] * len(r1), ['Control'] * len(ct1))))

    # df['Replicate'] = pd.Series(np.concatenate((['R1']*len(a1), ['R2']*len(a2), ['R3']*len(a3), ['R1']*len(c1),['R2']*len(c2),['R3']*len(c3),['R1']*len(r1),['R2']*len(r2),['R3']*len(r3),['R1']*len(ct1),['R2']*len(ct2),['R3']*len(ct3))))

    # ax = sns.boxplot(data=df, x='Replicate', y='Count', hue='Group', dodge=True)
    ax = sns.boxplot(data=df, x='Group', y='Count')
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    # sns.boxplot(data=df, x='Group', y='CC_area', hue='replicate', color='white', dodge=True)

    plt.yscale('log')

    # box_pairs = [(('R1', 'ATL'), ('R1', 'Climp')), (('R1', 'ATL'), ('R1', 'RTN')), (('R1', 'ATL'), ('R1', 'Control')), (('R1', 'Climp'), ('R1', 'RTN')), (('R1', 'Climp'), ('R1', 'Control')), (('R1', 'RTN'), ('R1', 'Control')), (('R2', 'ATL'), ('R2', 'Climp')), (('R2', 'ATL'), ('R2', 'RTN')), (('R2', 'ATL'), ('R2', 'Control')), (('R2', 'Climp'), ('R2', 'RTN')), (('R2', 'Climp'), ('R2', 'Control')), (('R2', 'RTN'), ('R2', 'Control')), (('R3', 'ATL'), ('R3', 'Climp')), (('R3', 'ATL'), ('R3', 'RTN')), (('R3', 'ATL'), ('R3', 'Control')), (('R3', 'Climp'), ('R3', 'RTN')), (('R3', 'Climp'), ('R3', 'Control')), (('R3', 'RTN'), ('R3', 'Control'))]
    box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('ATL', 'Control'), ('Climp', 'RTN'), ('Climp', 'Control'),
                 ('RTN', 'Control')]

    # statannot.add_stat_annotation(ax, x='Replicate', y='Count', hue='Group', data=df, box_pairs=box_pairs, test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')
    statannot.add_stat_annotation(ax, x='Group', y='Count', data=df, box_pairs=box_pairs, test='Mann-Whitney',
                                  text_format='simple', loc='inside', verbose=2, fontsize='large')

    # region_name = 'Isolated' if region == 'iso' else 'Fuzzy'

    # plt.suptitle(f'{region_name} CC area across conditions', fontsize=20)
    # plt.suptitle('')
    plt.title('Count of edges (tubules) corresponding to nodes (junctions) with degree greater than two', fontsize=20)
    plt.grid(True)
    plt.ylabel('Number of edges (tubules), log scale', fontsize=18)
    # plt.xlabel('Replicate', fontsize=18)
    plt.xlabel('Group', fontsize=18)
    plt.show()


# rel_edge_count()
# exit()


# atl_std = rel_edge_intensity('ATL', 1, 26)
# cl_std = rel_edge_intensity('Climp', 1, 31)
# rt_std = rel_edge_intensity('RTN', 1, 29)
# ct_std = rel_edge_intensity('Control', 1, 31)
#
# df = pd.DataFrame()
#
# df['Tubule_intensity_std'] = pd.Series(np.concatenate((atl_std, cl_std, rt_std, ct_std)))
# df['Group'] = pd.Series(
#     np.concatenate((['ATL'] * len(atl_std), ['Climp'] * len(cl_std), ['RTN'] * len(rt_std), ['Control'] * len(ct_std))))
#
# # sns.distplot(atl_mean, hist=False, label='atl')
# # sns.distplot(cl_mean, hist=False, label='cl')
# # sns.distplot(rt_mean, hist=False, label='rtn')
# # sns.distplot(ct_mean, hist=False, label='ctrl')
# sns.violinplot(data=df, y='Group', x='Tubule_intensity_std')
# plt.title('Standard deviation per tubule intensity for tubules (edges) corresponding to nodes with degree greater '
#           'than two', fontsize=20)
# plt.xlabel('Intensity standard deviation', fontsize=18)
# plt.ylabel('Group', fontsize=18)
# # plt.legend()
# plt.show()
#
# exit()
