import sknw
import networkx as nx
import numpy as np
import imageio


def skel_to_graph(skel):
    """

    @param skel: skeleton input image
    @return: set of nodes, list of degree per node
    """
    g = sknw.build_sknw(skel, iso=False)
    G = nx.Graph()

    node_set = g.nodes()

    G.add_nodes_from(node_set)
    G.add_edges_from(g.edges)

    degree_list = G.degree
    return node_set, degree_list


def get_junctions(skel_img):
    """

    @param skel_img: Input skel image (ndarray, binary)
    @return: list of Nodes with degree > 2
    """

    skel = imageio.imread(skel_img)

    # Build graph from the skeleton
    node_set, degree_list = skel_to_graph(skel)
    node_coords = np.array([node_set[node]['o'] for node in node_set])

    return [node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2]


