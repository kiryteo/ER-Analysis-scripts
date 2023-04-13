"""
Modules to connect the nodes close to each other
disconnected in the skeleton due to data (extremely low signal)
and method (skeletonization) limitations.
"""


import numpy as np
import math
import networkx as nx
from collections import OrderedDict
from skimage.graph import route_through_array
from sklearn.neighbors import NearestNeighbors


def get_nbrs(nodes_array):
    """
    Get the nearest neighbors per node
    @param nodes_array: np array with nodes are [x, y] lists
    """

    # Create a NearestNeighbors object and fit the data
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(nodes_array)

    # Get the distances and indices of the nearest neighbors
    distances, indices = nbrs.kneighbors(nodes_array)

    # Return the indices of the nearest neighbors for each element
    return distances[:, 1], indices[:, 1]


def nearest_node(distances, nbrs):
    # Create a dictionary that maps node indices to their nearest neighbor and the corresponding distance
    node_dict = {i: (nbrs[i], distances[i]) for i in range(len(nbrs))}

    # Create a dictionary that maps each node to its nearest neighbor and the corresponding distance
    nn_dict = {}
    for node, (nbr, dist) in node_dict.items():
        # If the neighbor is not already in the nearest neighbor dictionary, or the distance is shorter than the previous one, update the dictionary
        if nbr not in nn_dict or nn_dict[nbr][1] > dist:
            nn_dict[nbr] = (node, dist)

    # Return the dictionaries for nearest neighbors and node dictionary
    return node_dict, nn_dict


# Find the closest neighbor of a given node in a graph
def get_closest_from_nbr(graph, node):
    # Get the list of neighbors and their distances to the node
    neighbors = list(graph.neighbors(node))
    distances = [graph.edges[node, neighbor, 0]['weight'] for neighbor in neighbors]

    # Find the index of the closest neighbor in the list of neighbors
    # closest_neighbor_index = distances.index(min(distances))
    closest_neighbor_index = np.argmin(distances)

    # Return the closest neighbor and its distance to the node
    return neighbors[closest_neighbor_index], min(distances)


# Get the coordinates of the path between a given node and its closest neighbor in a graph
def get_path_coords(er_input, cost_arr, g_nodes_array, node, fin_dict):
    # Get the starting and ending coordinates of the path
    start_coord = tuple(g_nodes_array[node][:2])
    end_coord = tuple(g_nodes_array[fin_dict[node][0]][:2])

    # Find the path coordinates using the `route_through_array()` function
    path_coords, _ = route_through_array(cost_arr, start=start_coord, end=end_coord, fully_connected=True)

    # locations without signal in the path
    zero_signal_coords = sum(er_input[loc] == 0 for loc in path_coords)

    signal_coords = len(path_coords) - zero_signal_coords

    if zero_signal_coords > signal_coords:
        return None

    # Return the path coordinates as a NumPy array if the path is short enough, otherwise return None
    return np.array(path_coords) if len(path_coords) < 20 else None


def connect_low_degree_nodes(temp_graph, node, fin_dict, path_coords):
    # add edge between the close nodes and get length of edge (distance)
    temp_graph.add_edge(node, fin_dict[node][0])
    # total_distance = np.sum(np.linalg.norm(np.diff(path_coords, axis=0), axis=1))

    total_distance = sum(
        math.sqrt((path_coords[i + 1][0] - path_coords[i][0]) ** 2 + (path_coords[i + 1][1] - path_coords[i][1]) ** 2)
        for i in range(len(path_coords) - 1))

    return {(node, fin_dict[node][0], 0): {'pts': path_coords, 'weight': total_distance}}


def remove_edge_if_exists(temp_graph, node, neighbor):
    if temp_graph.has_edge(node, neighbor) or temp_graph.has_edge(neighbor, node):
        temp_graph.remove_edge(node, neighbor)
        temp_graph.remove_nodes_from((node, neighbor))


def connect_nodes(er_input, temp_graph, n1, n2, fin_dict, cost_arr, g_nodes_array):
    # If the two nodes are not already connected
    if fin_dict[n1][0] != n2:

        # Calculate the shortest path between the two nodes
        path_coords = get_path_coords(er_input, cost_arr, g_nodes_array, n1, fin_dict)

        # If a path exists and there is no existing edge between the nodes, connect them
        if path_coords is not None and not temp_graph.has_edge(n1, fin_dict[n1][0]):
            edge_data = connect_low_degree_nodes(temp_graph, n1, fin_dict, path_coords)

            # Update the edge attributes of the graph with the new connection
            nx.set_edge_attributes(temp_graph, edge_data)


def get_updated_degree_nodes(temp_graph):
    # Initialize empty lists for nodes with degree 1, degree 2, and higher degrees
    deg_one_nodes, deg_two_nodes, high_deg_nodes = [], [], []

    # Loop through each node in the graph and its degree
    for node, degree in temp_graph.degree:
        # Check if the node is a point of interest ('o' in the node attributes)
        if 'o' in temp_graph.nodes[node]:
            # Get the point coordinates
            point = list(temp_graph.nodes[node]['o'])

            # Classify the node based on its degree
            if degree == 1:
                deg_one_nodes.append(point)
            elif degree == 2:
                deg_two_nodes.append(point)
            else:
                high_deg_nodes.append(point)

    # Convert the lists to NumPy arrays and return them
    return np.array(deg_one_nodes), np.array(deg_two_nodes), np.array(high_deg_nodes)


def create_new_path(path_nbr1, path_nbr2):
    # Create an ordered dictionary from the union of path_nbr1 and path_nbr2
    union_dict = OrderedDict.fromkeys(map(tuple, path_nbr1 + path_nbr2))

    # Convert the keys of the ordered dictionary to a list of lists and return it
    return list(map(list, union_dict.keys()))


def get_new_edge_data(temp_graph, nbr_a, nbr_b, path_a, path_b):
    # Convert the two paths to a numpy array of coordinates
    path_coords = np.array(create_new_path(path_a, path_b))

    # Calculate the total distance between the endpoints of the two paths
    total_distance = sum(
        math.sqrt((path_coords[i + 1][0] - path_coords[i][0]) ** 2 + (path_coords[i + 1][1] - path_coords[i][1]) ** 2)
        for i in range(len(path_coords) - 1))

    # Add the new edge to the temporary graph
    temp_graph.add_edge(nbr_a, nbr_b)

    # Return a dictionary of edge data containing the path coordinates and total distance
    return {(nbr_a, nbr_b, 0): {'pts': path_coords, 'weight': total_distance}}


def high_deg_connections(temp_graph, node, nbr1, nbr2):
    # Get the path coordinates
    path_nbr1 = [[int(x) for x in a] for a in temp_graph[node][nbr1][0]['pts']]
    path_nbr2 = [[int(x) for x in a] for a in temp_graph[node][nbr2][0]['pts']]

    # Check if the paths share a common starting point
    if path_nbr1[0] == path_nbr2[0]:
        # Reverse path_nbr2 if necessary
        path_nbr2 = path_nbr2[::-1]
        # Create new edge data and add the edge to the temporary graph
        edge_data = get_new_edge_data(temp_graph, nbr2, nbr1, path_nbr2, path_nbr1)

    # Check if the ending point of path_nbr1 matches the starting point of path_nbr2
    elif path_nbr1[0] == path_nbr2[-1]:
        # Create new edge data and add the edge to the temporary graph
        edge_data = get_new_edge_data(temp_graph, nbr2, nbr1, path_nbr2, path_nbr1)

    # Check if the ending point of path_nbr2 matches the starting point of path_nbr1
    elif path_nbr1[-1] == path_nbr2[0]:
        # Create new edge data and add the edge to the temporary graph
        edge_data = get_new_edge_data(temp_graph, nbr1, nbr2, path_nbr1, path_nbr2)

    # Otherwise, reverse path_nbr2 and create new edge data
    else:
        path_nbr2 = path_nbr2[::-1]
        edge_data = get_new_edge_data(temp_graph, nbr1, nbr2, path_nbr1, path_nbr2)

    # Remove the node from the temporary graph and set edge attributes
    temp_graph.remove_node(node)
    nx.set_edge_attributes(temp_graph, edge_data)


def process_node(temp_graph, node):
    # Get the neighbors of the node
    nbrs = list(temp_graph.neighbors(node))

    # If the node has less or more than 2 neighbors, return
    if len(nbrs) != 2:
        return

    # Get the neighbors of the neighbors
    nbr1, nbr2 = nbrs
    deg1, deg2 = temp_graph.degree(nbr1), temp_graph.degree(nbr2)

    # If both neighbors have at least one neighbor themselves, connect them directly and remove the node
    if deg1 >= 1 and deg2 >= 1:
        high_deg_connections(temp_graph, node, nbr1, nbr2)


def get_updated_neighbor_dict(graph):
    # sourcery skip: assign-if-exp, dict-comprehension
    # Get the nodes of the graph as a list of points
    graph_nodes = np.array([graph.nodes[i]['o'] for i in graph.nodes])
    graph_nodes_list = graph_nodes.tolist()

    # Dictionary containing the closest neighbor of each node, and the distance to that neighbor
    closest_neighbor_dict = {}
    for node in graph.nodes():
        neighbor_id, dist = get_closest_from_nbr(graph, node)
        closest_neighbor_dict[node] = (neighbor_id, dist)

    # Get the distances and neighbors for all points
    distances, neighbors_all = get_nbrs(graph_nodes)

    # Get the nearest neighbor and its id for all points
    neighbor_dict, neighbor_id_dict = nearest_node(distances, neighbors_all)

    # for k, v in closest_neighbor_dict.items():
    #     if k in neighbor_id_dict:
    #         fin_dict[k] = neighbor_id_dict[k]

    updated_neighbor_dict = {}

    # Compare the closest neighbor to the nearest neighbor, and choose the closest one
    for node, neighbor in closest_neighbor_dict.items():
        if neighbor[1] < neighbor_dict[node][1]:
            updated_neighbor_dict[node] = closest_neighbor_dict[node]
        else:
            updated_neighbor_dict[node] = neighbor_dict[node]

    return updated_neighbor_dict, graph_nodes_list
