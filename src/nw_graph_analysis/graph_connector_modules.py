import numpy as np
import math
import networkx as nx
from collections import OrderedDict
from skimage.graph import route_through_array
from sklearn.neighbors import NearestNeighbors
from er_graph_extraction import get_skel, get_graph_from_skel


def get_connected_graph(skel_img_path):
    """
    Connect the graph based on missing near-node connections
    """
    graph = get_graph_from_skel(skel_img_path)


def get_nbrs(nodes_array):
    """
    Get the nearest neighbors per node
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
    # sourcery skip: assign-if-exp, dict-comprehension
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