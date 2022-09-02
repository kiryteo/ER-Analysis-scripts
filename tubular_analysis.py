import imageio
import skimage
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
# from skan import skeleton_to_csgraph
from networkx.drawing.nx_agraph import graphviz_layout
import sknw
import networkx as nx
import numpy as np

mean_proj_img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A1_mean.png')
# plt.imshow(mean_proj_img)

# bin_img = mean_proj_img > 0
thresh = threshold_otsu(mean_proj_img)
bin_img = mean_proj_img > thresh

skel = pcv.morphology.skeletonize(mask=bin_img)
branchpts = pcv.morphology.find_branch_pts(skel_img=skel)

# print(len(np.where(branchpts==255)[0]))

dil_brpts = pcv.dilate(gray_img=branchpts, ksize=3, i=1)

tubules = skel - dil_brpts

lab_tubules = skimage.measure.label(tubules)

# a = skeleton_to_csgraph(skel)
g = sknw.build_sknw(skel, iso=False)
G = nx.Graph()
# inters = degrees > 2

G.add_nodes_from(g.nodes)
G.add_edges_from(g.edges)

nodes = g.nodes()
ps = np.array([nodes[i]['o'] for i in nodes])
# print(ps[:,1])
# print(ps[:,0])
# print(len(ps))

# nx.draw_networkx(G, pos=pos, with_labels=True, node_size=10)
# nx.draw(G)
degree_list = G.degree

# print(ps)

# get all the nodes with degree greater than 2
newps = []
for i, val in enumerate(degree_list):
    if val[1] > 2:
        newps.append(ps[i])

# print(newps[0])
# print(len(newps))

# print(len(degree_list))
#
# print(degree_list)

# pos = graphviz_layout(G)
#
# print(pos)

# plt.show()
# print(pos)

# exit()

# plt.imshow(tubules)
# plt.imshow(lab_tubules)
# plt.show()

# plt.imshow(skel)
# plt.imshow(tubules)

# fig = plt.figure()
# plt.axis('off')
# r, c = 1, 4
#
# fig.add_subplot(r, c, 1)
# plt.imshow(mean_proj_img)
#
# fig.add_subplot(r, c, 2)
# plt.imshow(skel)
#
# fig.add_subplot(r, c, 3)
# plt.imshow(tubules)
#
# fig.add_subplot(r, c, 4)
# plt.imshow(lab_tubules)
#
# plt.show()

# plt.imshow(bin_img)
# plt.show()