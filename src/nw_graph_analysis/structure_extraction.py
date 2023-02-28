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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.measure import label, regionprops
from skimage.graph import route_through_array
from junction_analysis_modules import *
import numpy as np
from sklearn.neighbors import NearestNeighbors

confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'


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
    # # new_pref = '/localhome/asa420/MIAL/data/confocal_movies/' + group + '/new_op_jul/preproc/'
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
    return sknw.build_sknw(imageio.imread(skel_img_path), multi=True, iso=False)


def get_junctions(graph):
    """

    @param graph: Input graph to obtain the junctions
    @return: nodes (junctions) with degree > 2
    """
    # get all the nodes from the graph
    node_coords = np.array([graph.nodes[node]['o'] for node in graph.nodes()])

    # graph.degree provides list of tuples with node id followed by its degree

    return [node_coords[node_num] for node_num, degree_val in enumerate(graph.degree) if degree_val[1] > 2]


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

    return [graph[start_node][end_node][0]['pts'] for start_node, end_node in edge_set if
            start_node in relevant_node_list and end_node in relevant_node_list]


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
        f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/er_mean/{group.lower()}{num_series}_er_mean.png')

    # per frame analysis
    # input = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/{pref}{num_series}_decon_t050_ch00.tif')
    # input = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/img_{num_series}_decon_t050.tif')
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
    # plt.close()
    plt.show()


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


def close_neighbor(distances, nbrs):
    nbr_dict = {i: (nbrs[i], distances[i]) for i in range(len(nbrs))}

    # For each key, check its appearance as nearest neighbor
    nn_check_dict = {k: [i for i, x in enumerate(nbr_dict.values()) if x[0] == k] for k in nbr_dict}

    # In case of multiple NN candidates, select the closest
    candidate_nn = {k: min(((v[i], distances[v[i]]) for i in range(len(v))), key=lambda x: x[1]) for k, v in nn_check_dict.items() if v}

    # Reformat the dict
    ncp2 = {k: v for k, v in candidate_nn.items() if v[0] in candidate_nn}

    return nbr_dict, ncp2


# def closest_neighbor(distances, nbrs):
#     global nbr_dict
#
#     # node and its (NN, distance)
#     nbr_dict = {i: (nbrs[i], distances[i]) for i in range(len(nbrs))}
#
#     # print(nbr_dict)
#     keys = list(nbr_dict.keys())
#     vals = list(nbr_dict.values())
#
#     # for each key, get all other keys where it appears as nearest nbr
#     new = {}
#     for k in keys:
#         new[k] = []
#         for v in vals:
#             if v[0] == k:
#                 new[k].append(vals.index(v))
#
#     ncp = copy.deepcopy(new)
#
#     for k, v in new.items():
#         # print(len(v))
#         min_dist = 1000000
#         if len(v) > 1:
#             num = 0
#             for idx, vl in enumerate(v):
#                 if distances[vl] < min_dist:
#                     min_dist = distances[vl]
#                     num = idx
#             ncp[k] = [v[num], min_dist]
#         elif len(v) == 0:
#             del ncp[k]
#
#     # print(ncp)
#
#     ncp2 = copy.deepcopy(ncp)
#
#     for k, v in ncp.items():
#         if v[0] not in ncp.keys():
#             del ncp2[k]
#
#     return nbr_dict, ncp2



def graph_plotter1(group, series):
    global rel
    pref = 'Ct' if group == 'Control' else group[0]

    # path = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/{pref}{series}/{pref}{series}_decon_t050_ch00_skel.png'

    # projection frame analysis
    # path = '/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/er_mean_proc/climp16_er_mean_proc_enhance_skel.png'
    path = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/er_mean_proc/{group.lower()}{series}_er_mean_proc_enhance_skel.png'

    path_proc = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/er_mean_proc/{group.lower()}{series}_er_mean_proc_enhance.png'

    path_er = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/er_mean/{group.lower()}{series}_er_mean.png'

    graph = skel_to_graph(path)

    # all junction from the graph
    junctions = get_junctions(graph)

    # nodes with degree > 2
    relevant_nodes = get_relevant_nodes(junctions)

    # relevant_edges = get_relevant_tubules(graph, relevant_nodes)
    # print(relevant_edges)

    # draw node by o
    nodes = graph.nodes()

    g_nodes = np.array([graph.nodes[i]['o'] for i in graph.nodes])

    # edge start and end points
    # neighbors_dict = {i: list(graph.neighbors(i))[0] for i in graph.nodes}
    degree_list = list(graph.degree())


    distances, nbrs_all = get_nbrs(g_nodes)
    nbr_dict, ncp2 = close_neighbor(distances, nbrs_all)

    # print(nbr_dict)
    # print(ncp2)
    #
    # exit()

    degree_dict = {each[0]: each[1] for each in degree_list}

    temp_graph = copy.deepcopy(graph)


    skel = imageio.imread(path)

    for node, node_degree in degree_dict.items():
        if node_degree == 1:

            # access the first element of graph.neighbors
            neighbor = next(iter(graph.neighbors(node)))

            # If the neighbor also has degree 1 and the edge between the two exists in temp_graph, remove the edge and nodes
            if degree_dict.get(neighbor, 0) == 1 and temp_graph.has_edge(node, neighbor):
                temp_graph.remove_edge(node, neighbor)
                temp_graph.remove_nodes_from((node, neighbor))

        # If a node has degree > 2 and all its neighbors have degree 1, remove the node and its neighbors
        elif node_degree > 2 and all(degree_dict.get(n, 0) == 1 for n in graph.neighbors(node)):
            coords = temp_graph.nodes[node]['o']
            x, y = coords[0], coords[1]

            # If the node is in relevant_nodes, delete it
            idx = np.where((relevant_nodes == [x, y]).all(axis=1))[0][0] if (x, y) in relevant_nodes else -1
            relevant_nodes = np.delete(relevant_nodes, idx, axis=0)

            # Remove the node and its neighbors from temp_graph
            nbrs_list = list(temp_graph.neighbors(node))
            temp_graph.remove_nodes_from(nbrs_list + [node])

    # exit()
    tgraph = copy.deepcopy(temp_graph)
    for node, node_degree in degree_dict.items():
        if node_degree == 1:
            neighbor = next(iter(graph.neighbors(node)))
            if degree_dict.get(neighbor, 0) > 2 and temp_graph.has_edge(node, neighbor) and ncp2[node][0] == neighbor:
                continue



    #
    # for node, node_degree in degree_dict.items():
    #     if node_degree == 1:
    #         neighbor = list(graph.neighbors(node))[0]
    #         if degree_dict[neighbor] > 2:
    #             # if nn is neighbor - dont do anything
    #             # if nn is 1/ 2 degree

    # exit()

    er_mean = imageio.imread(
        f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/er_mean/{group.lower()}{series}_er_mean.png')
    plt.imshow(er_mean, cmap='gray')

    # plt.imshow(imageio.imread(path), cmap='gray')

    # k = []
    for (start_node, end_node) in temp_graph.edges():
        ps = graph[start_node][end_node][0]['pts']
        plt.plot(ps[:, 1], ps[:, 0], 'green')
        # k.append(ps)

    ps = np.array([temp_graph.nodes[i]['o'] for i in temp_graph.nodes])
    # plt.plot(ps[:, 1], ps[:, 0], 'r.')
    plt.plot(ps[:, 1], ps[:, 0], 'o', markerfacecolor='yellow', markeredgecolor='yellow', mew=0.5, markersize=3)

    # plt.plot(relevant_nodes[:, 1], relevant_nodes[:, 0], 'b.')
    plt.plot(relevant_nodes[:, 1], relevant_nodes[:, 0], 'o', markerfacecolor='blue', markeredgecolor='blue',
             markersize=4)

    plt.axis('off')
    plt.savefig(f'graphs/{group}_{series}_edge_graph_projection_updated_f27_v3', bbox_inches='tight', pad_inches=0)
    plt.close()

    # plt.show()

    exit()

    g_nodes_array = [[each[0], each[1]] for each in g_nodes]
    rel_nodes_array = [[each[0], each[1]] for each in relevant_nodes]
    deg_one_nodes = []

    # get only 1, 2 degree nodes (yellow spots)
    low_deg_nodes = [x for x in g_nodes_array if x not in rel_nodes_array]

    # print(g_nodes_array)
    # print(low_deg_nodes)
    # print(g_nodes_array[3] == low_deg_nodes[3])
    exit()

    # get nearest neighbour distances and indices for the 1, 2 degree nodes
    distances, nbrs = get_nbrs(low_deg_nodes)

    print(distances)

    # print(nbrs)

    # exit()

    edge_len_list = []

    for each in low_deg_nodes:
        l = list(graph.neighbors(g_nodes_array.index(each)))
        if len(l) == 1:
            # if nx.is_simple_path(graph, [pspp.index(each), l[0]]) or nx.is_simple_path(graph, [l[0], pspp.index(each)]):
            edge_len = len(graph[g_nodes_array.index(each)][l[0]][0]['pts'])
            edge_len_list.append(edge_len)
        else:
            edge_len_1 = len(graph[g_nodes_array.index(each)][l[0]][0]['pts'])
            edge_len_2 = len(graph[g_nodes_array.index(each)][l[1]][0]['pts'])
            if edge_len_1 < edge_len_2:
                edge_len_list.append(edge_len_1)
            else:
                edge_len_list.append(edge_len_2)

    # print(edge_len_list)
    # print(len(edge_len_list))

    new_node_indices = [i for (i, a), (j, b) in zip(enumerate(distances), enumerate(edge_len_list)) if a < b]

    # print(new_node_indices)
    # exit()

    skel = imageio.imread(path_er)

    for k, v in ncpnn.items():
        st = g_nodes_array.index(low_deg_nodes[k])
        end = g_nodes_array.index(low_deg_nodes[v[0]])

        st_coord = (g_nodes_array[st][0], g_nodes_array[st][1])
        end_coord = (g_nodes_array[end][0], g_nodes_array[end][1])
        # for each in new_node_indices:
        #     ldi_start = each
        #     ldi_end = nbrs[each]
        #
        #     u = pspp.index(low_deg_nodes[ldi_start])
        #     v = pspp.index(low_deg_nodes[ldi_end])

        # st_coord = (pspp[u][0], pspp[u][1])
        # end_coord = (pspp[v][0], pspp[v][1])

        path_coords, _ = route_through_array(skel, start=st_coord, end=end_coord)
        path_coords = np.array(path_coords)

        if graph.has_edge(st, end) or graph.has_edge(end, st):
            continue
        graph.add_edge(st, end)

        # npath = [(21, 59), (29, 74), (35, 76), (38, 75), (40, 67), (39, 56), (32, 50), (27, 58)]

        # edge_data = {(u, v, 0): {'pts': path_coords}}
        edge_data = {(st, end, 0): {'pts': path_coords}}

        nx.set_edge_attributes(graph, edge_data)

            # try:
            #     path_coords, _ = route_through_array(skel, start=st_coord, end=end_coord)
            #     path_coords = np.array(path_coords)
            #
            #     # print(path_coords)
            #     # exit()
            #
            #     new_edge = (u, v, {'pts': path_coords})
            #
            #     if graph.has_edge(u, v) or graph.has_edge(v, u):
            #         continue
            #     else:
            #         graph.add_edge(*new_edge)
            #
            #         # npath = []
            #         #
            #         # for ele in new_path:
            #         #     npath.append(list(ele))
            #         #
            #         # # npath = [(21, 59), (29, 74), (35, 76), (38, 75), (40, 67), (39, 56), (32, 50), (27, 58)]
            #         #
            #         # edge_data = {(u, v, 0): {'pts': np.array(npath)}}
            #         #
            #         # nx.set_edge_attributes(graph, edge_data)
            # except:
            #     pass

    # print(graph.edges)
    # exit()

    # print(graph.edges[(22, 29, 0)]['pts'])
    # print(graph.edges)
    er_mean = imageio.imread(
        f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/er_mean/{group.lower()}{series}_er_mean.png')
    plt.imshow(er_mean, cmap='gray')

    # plt.imshow(imageio.imread(path), cmap='gray')

    # k = []
    for (start_node, end_node) in graph.edges():
        ps = graph[start_node][end_node][0]['pts']
        plt.plot(ps[:, 1], ps[:, 0], 'green')
        # k.append(ps)

    ps = np.array([nodes[i]['o'] for i in nodes])
    # plt.plot(ps[:, 1], ps[:, 0], 'r.')
    plt.plot(ps[:, 1], ps[:, 0], 'o', markerfacecolor='yellow', markeredgecolor='yellow', mew=0.5, markersize=3)

    # plt.plot(relevant_nodes[:, 1], relevant_nodes[:, 0], 'b.')
    plt.plot(relevant_nodes[:, 1], relevant_nodes[:, 0], 'o', markerfacecolor='blue', markeredgecolor='blue',
             markersize=4)

    # print(k)
    # exit()
    # l0 = [x[0] for x in k]
    # l1 = [x[1] for x in k]

    # for (start_node, end_node) in graph.edges():
    #     ps = graph[start_node][end_node][0]['pts']
    #     plt.plot(ps[:, 1], ps[:, 0], 'green')
    #     with contextlib.suppress(Exception):
    #         ps_multi = graph[start_node][end_node][1]['pts']
    #         plt.plot(ps_multi[:, 1], ps_multi[:, 0], 'green')
    # plt.plot(l1, l0, 'green')
    plt.axis('off')
    plt.savefig(f'graphs/{group}_{series}_edge_graph_projection_updated_new_er', bbox_inches='tight', pad_inches=0)
    plt.close()
    # plt.show()


graph_plotter1('ATL', 13)
exit()




def runner(group, r_start, r_end):
    l = []

    pref = 'Ct' if group == 'Control' else group[0]
    for i, frame in itertools.product(range(r_start, r_end + 1), range(100)):
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


import statannot
from statannotations.Annotator import Annotator


# atl = runner('ATL', 1, 2)
# climp = runner('Climp', 1, 2)
# rtn = runner('RTN', 1, 2)
# ctrl = runner('Control', 1, 2)
#
# atl_series = pd.Series(atl, name='ATL')
# climp_series = pd.Series(climp, name='Climp')
# rtn_series = pd.Series(rtn, name='RTN')
# ctrl_series = pd.Series(ctrl, name='Control')
#
# df = pd.concat([atl_series, climp_series, rtn_series, ctrl_series], axis=1)
#
# df_long = pd.melt(df, var_name='Group', value_name='Length')
#
# box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('ATL', 'Control'), ('Climp', 'RTN'), ('Climp', 'Control'), ('RTN', 'Control')]
#
# ax = sns.violinplot(data=df_long, y='Group', x='Length')
# ax.set_xscale('log')
#
# annot = Annotator(ax, box_pairs, data=df_long, x='Length', y='Group')
# annot.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annot.apply_and_annotate()
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

    # path = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/{pref}{series}/{pref}{series}_decon_t050_ch00_skel.png'

    # projection frame analysis
    # path = '/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/er_mean_proc/climp16_er_mean_proc_enhance_skel.png'
    path = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/er_mean_proc/{group.lower()}{series}_er_mean_proc_enhance_skel.png'

    # plt.imshow(imageio.imread(path), cmap='gray')

    graph = skel_to_graph(path)
    # print(graph[0][15][0]['pts'])
    # exit()

    # for (start, end) in graph.edges():
    #     # print(start, end)
    #     print(list(graph[start][end].keys()))
    #     # edlen = len(graph[start][end][0]['pts'])
    #     # print(edlen)
    # # print(graph[5])
    # exit()

    junctions = get_junctions(graph)
    # print(junctions)
    # exit()
    relevant_nodes = get_relevant_nodes(junctions)

    ps = np.array([graph.nodes[i]['o'] for i in graph.nodes])
    # print(ps)
    #
    # exit()
    # print(junctions)
    # print(relevant_nodes)
    # print(ps - relevant_nodes)
    # print(len(ps))
    # print(len(relevant_nodes))
    # yellow_nodes = [x for x in ps if x not in relevant_nodes]
    # print(len(yellow_nodes))

    # print(ps)
    # print(relevant_nodes)

    pspp = []
    rel = []

    # get ps and relevant_nodes in correct format
    for each in ps:
        pspp.append([each[0], each[1]])
    for each in relevant_nodes:
        rel.append([each[0], each[1]])

    # get only 1, 2 degree nodes (yellow spots)
    low_deg_nodes = [x for x in pspp if x not in rel]

    # junc_data = [x.tolist() for x in junctions]
    # nbrs = get_nbrs(junc_data)

    # get nearest neighbour distances and indices for the 1, 2 degree nodes
    distances, nbrs = get_nbrs(low_deg_nodes)

    edge_len_list = []

    for each in low_deg_nodes:
        l = list(graph.neighbors(pspp.index(each)))
        if len(l) == 1:
            # if nx.is_simple_path(graph, [pspp.index(each), l[0]]) or nx.is_simple_path(graph, [l[0], pspp.index(each)]):
            edge_len = len(graph[pspp.index(each)][l[0]][0]['pts'])
            edge_len_list.append(edge_len)
        else:
            edge_len_1 = len(graph[pspp.index(each)][l[0]][0]['pts'])
            edge_len_2 = len(graph[pspp.index(each)][l[1]][0]['pts'])
            if edge_len_1 < edge_len_2:
                edge_len_list.append(edge_len_1)
            else:
                edge_len_list.append(edge_len_2)

    # print(edge_len_list)
    # print(len(edge_len_list))

    new_node_indices = []
    for (i, a), (j, b) in zip(enumerate(distances), enumerate(edge_len_list)):
        if a < b:
            new_node_indices.append(i)

    # print(new_node_indices)
    # print(nbrs)
    # print(low_deg_nodes)
    # print(len(nbrs))
    # print(len(low_deg_nodes))

    # print(graph.edges)
    for each in new_node_indices:
        ldi_start = each
        ldi_end = nbrs[each]

        u = pspp.index(low_deg_nodes[ldi_start])
        v = pspp.index(low_deg_nodes[ldi_end])

        try:
            sh_path = nx.shortest_path(graph, u, v)

            st_coord = (pspp[u][0], pspp[u][1])
            end_coord = (pspp[v][0], pspp[v][1])

            new_path = [graph.nodes[n]['o'] for n in sh_path]

            if graph.has_edge(u, v) or graph.has_edge(v, u):
                continue
            else:
                graph.add_edge(u, v)

                npath = []

                for ele in new_path:
                    npath.append(list(ele))

                # npath = [(21, 59), (29, 74), (35, 76), (38, 75), (40, 67), (39, 56), (32, 50), (27, 58)]

                edge_data = {(u, v, 0): {'pts': np.array(npath)}}

                nx.set_edge_attributes(graph, edge_data)
        except:
            pass

    # print(graph.edges[(22, 29, 0)]['pts'])
    # print(graph.edges)
    plt.imshow(imageio.imread(path), cmap='gray')

    for (start_node, end_node) in graph.edges():
        ps = graph[start_node][end_node][0]['pts']
        plt.plot(ps[:, 1], ps[:, 0], 'green')

    plt.show()

    exit()

    relevant_edges = get_relevant_tubules(graph, relevant_nodes)

    plot_total_graph(group, series, graph, relevant_nodes, relevant_edges)


graph_plotter('ATL', 1)
# for i in range(1, 30):
#     graph_plotter('RTN', i)
exit()


def rel_edges_length(group, r_start, r_end):
    l = []
    for i, frame in itertools.product(range(r_start, r_end + 1), range(100)):
        # input = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/{group[0]}{i}_decon_t0{frame:02d}_ch00.tif')
        # ip = (input - input.min())/(input.max() - input.min())

        path = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/{group[0]}{i}/{group[0]}{i}_decon_t0{frame:02d}_ch00_skel.png'
        graph = skel_to_graph(path)
        junctions = get_junctions(graph)
        relevant_nodes = get_relevant_nodes(junctions)
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
            fname = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/img_{i}_decon_t0{frame:02d}.tif'
        else:
            fname = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/{group[0]}{i}_decon_t0{frame:02d}_ch00.tif'

        img = imageio.imread(fname)
        ip = (img - img.min()) / (img.max() - img.min())
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


rel_edge_count()
exit()

atl_std = rel_edge_intensity('ATL', 1, 26)
cl_std = rel_edge_intensity('Climp', 1, 31)
rt_std = rel_edge_intensity('RTN', 1, 29)
ct_std = rel_edge_intensity('Control', 1, 31)

df = pd.DataFrame()

df['Tubule_intensity_std'] = pd.Series(np.concatenate((atl_std, cl_std, rt_std, ct_std)))
df['Group'] = pd.Series(
    np.concatenate((['ATL'] * len(atl_std), ['Climp'] * len(cl_std), ['RTN'] * len(rt_std), ['Control'] * len(ct_std))))

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
