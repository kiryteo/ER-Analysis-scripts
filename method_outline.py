import sknw
import networkx as nx
import imageio
import matplotlib.pyplot as plt
import numpy as np
import itertools
from skimage import draw

# skel = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/skel/C12/C12_decon_t000_ch00_skel.png')

# skel = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/er_mean_proc/climp12_er_mean_proc_enhance_skel.png')


skel = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png')

# img = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure2/A1_t0.png')
#
# skel = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure2/A1_decon_t000_ch00_skel.png')
#
# skimg = np.stack((skel, skel, skel), axis=2)

# print(skimg.shape)
# exit()

def skel_to_graph(skel):
    g = sknw.build_sknw(skel, iso=False)
    G = nx.Graph()

    node_set = g.nodes()
    edge_set = g.edges()

    G.add_nodes_from(node_set)
    G.add_edges_from(edge_set)

    degree_list = G.degree
    return g, node_set, edge_set, degree_list

graph, node_set, edge_set, d = skel_to_graph(skel)
# print(node_set)
# print(edge_set)
# exit()

# for (s, e) in graph.edges():
#     pse = graph[s][e]['pts']
#     # print(pse)
#     # print(pse[0][0], pse[0][1], pse[-1][0], pse[-1][1])
#     # exit()
#
#     rl, cl = draw.line(pse[0][0], pse[0][1], pse[-1][0], pse[-1][1])
#     for a, b in zip(rl, cl):
#         skimg[a, b] = [255, 255, 0]

# for each in edge_set:
#     print(edge_set[each])
# exit()

# edge_coords = np.array([edge_set[edge]['pts'] for edge in edge_set])
#
# edg = list(itertools.chain.from_iterable(edge_coords))
# edg = np.array(edg)
# exit()

node_coords = np.array([node_set[node]['o'] for node in node_set])
newps = [node_coords[j] for j, val in enumerate(d) if val[1] > 2]
# ed_coords = np.array([node_set[node]['pts'] for node in node_set])
# print(node_coords[2])
# print(ed_coords[2])
# exit()

nps = [[each[0], each[1]] for each in newps]
nps = np.array(nps)

# def get_junction_image(nps):
#     """
#
#     @param newps: List of nodes
#     @return: Image with nodes -> 1, else 0
#     """
#     brpts_img = np.zeros((128, 128))
#     for each in nps:
#         brpts_img[each[0], each[1]] = 255.
#     return brpts_img
#
# brimg = get_junction_image(nps)
# imageio.imsave('ATL1_t0_junctions_refined.png', brimg)
#
# exit()

# s = np.where(skel>0)
# skel[s] = 0
plt.imshow(skel, cmap='gray')
# plt.imshow(imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_junc_projection.png'), cmap='gray')
# plt.imshow(img, cmap='gray')
# plt.imshow(skimg)
# plt.plot(edg[:, 1], edg[:, 0], '.', markerfacecolor='red', markeredgecolor='red', mew=0.2)
# plt.plot(node_coords[:, 1], node_coords[:, 0], 's', markerfacecolor='blue', markeredgecolor='blue', mew=0.1, markersize=3)
plt.plot(nps[:, 1], nps[:, 0], 'o', markerfacecolor='magenta', markeredgecolor='magenta', mew=0.75, markersize=4)
plt.axis('off')
# plt.savefig('Climp_12_CC_and_proj_junctions', bbox_inches='tight', pad_inches=0, dpi=700)
# plt.close()

plt.show()



















