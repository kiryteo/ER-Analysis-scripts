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


def tub_analysis():
    mean_proj_img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A1_mean.png')
# plt.imshow(mean_proj_img)

# bin_img = mean_proj_img > 0
    thresh = threshold_otsu(mean_proj_img)
    bin_img = mean_proj_img > thresh

    skel = pcv.morphology.skeletonize(mask=bin_img)
    # branchpts = pcv.morphology.find_branch_pts(skel_img=skel)

    g = sknw.build_sknw(skel, iso=False)
    G = nx.Graph()

    G.add_nodes_from(g.nodes)
    G.add_edges_from(g.edges)

    nodes = g.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])

    # nx.draw_networkx(G, pos=pos, with_labels=True, node_size=10)
    degree_list = G.degree

    # get all the nodes with degree greater than 2
    newps = []
    for i, val in enumerate(degree_list):
        if val[1] > 2:
            newps.append(ps[i])

    brpts_img = np.zeros((128, 128))
    # brpts_img[newps] = 1.
    for each in newps:
        brpts_img[each[0], each[1]] = 255.

    dil_brpts = pcv.dilate(gray_img=brpts_img, ksize=3, i=1)
    tubules = skel - dil_brpts

    tubules[np.where(tubules<0)] = 0

    lab_tubules = skimage.measure.label(tubules)

    return lab_tubules

    # op = 0.5 * skel + 0.5 * brpts_img
    # plt.imshow(op)
    # plt.show()

    # print(len(np.where(branchpts==255)[0]))


lab_tubules = tub_analysis()

lt = []
for each in range(1, len(np.unique(lab_tubules))):
    tub_len = len(np.where(lab_tubules==each)[0])
    if tub_len > 5:
        lt.append(np.where(lab_tubules==each))

# print(lt)

intensity_vals = []
for i in range(100):
    l = []
    img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0%s_ch00.tif'%f'{i:02d}')
    img = (img - img.min()) / (img.max() - img.min())
    for each in lt:
        l.append(img[each])
    intensity_vals.append(l)

# print(intensity_vals)
# exit()
# print(len(intensity_vals))

newlt = []

for tub_list in range(len(intensity_vals[0])):
    for frame in range(100):
        newlt.append(intensity_vals[frame][tub_list])


mean_list = []
for i in range(0, len(newlt), 100):
    mean_list.append(np.mean(newlt[i:i+100], axis=0))

# print(mean_list[5])
for i, val in enumerate(mean_list):
    plt.plot(mean_list[i])
    plt.show()


exit()


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

g = sknw.build_sknw(skel, iso=False)
G = nx.Graph()

G.add_nodes_from(g.nodes)
G.add_edges_from(g.edges)

nodes = g.nodes()
ps = np.array([nodes[i]['o'] for i in nodes])

# nx.draw_networkx(G, pos=pos, with_labels=True, node_size=10)
degree_list = G.degree

# print(ps)

# get all the nodes with degree greater than 2
newps = []
for i, val in enumerate(degree_list):
    if val[1] > 2:
        newps.append(ps[i])

# print(newps)

nps = []
for each in newps:
    nps.append([each[0], each[1]])

nps = np.array(nps)

# print(nps)
plt.imshow(mean_proj_img, cmap='gray')

# for (s,e) in g.edges():
#     ps = g[s][e]['pts']
#     plt.plot(ps[:,1], ps[:,0], 'green')

plt.plot(ps[:, 1], ps[:, 0], 'y.')
plt.plot(nps[:, 1], nps[:, 0], 'r.')

# plt.show()

plt.suptitle('Junction detection')
plt.title('Yellow: all branching areas, Red: Only degree 3 or more areas')
plt.savefig('Refined_junction_detection', bbox_inches='tight')
plt.close()
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