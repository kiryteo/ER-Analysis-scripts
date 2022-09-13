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
import cv2




# from skan import draw
#
# tub = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/A1_mean_skel_tubules.png')
# for i in range(100):
#     fig, ax = plt.subplots()
#     img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0%s_ch00.png'%f'{i:02d}')
#     op = draw.overlay_skeleton_2d(img, tub, dilate=0, axes=ax)
#     plt.savefig('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/A1_decon_t0%s_ch00_er_overlay.png'%f'{i:02d}', bbox_inches='tight', pad_inches=0)
#     plt.close()
# # # cv2.imwrite('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/A1_decon_t000_ch00_brpts_overlay.png', op)
# #
#
# exit()


def tub_analysis():

    mean_proj_img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A1_mean.png')

    # perform thresholding + binarization + skel
    thresh = threshold_otsu(mean_proj_img)
    bin_img = mean_proj_img > thresh

    skel = pcv.morphology.skeletonize(mask=bin_img)

    # Build graph from the skeleton
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

    # Obtain the junctions
    brpts_img = np.zeros((128, 128))
    # brpts_img[newps] = 1.
    for each in newps:
        brpts_img[each[0], each[1]] = 255.

    # Dilate junctions and remove them to get individual tubules
    dil_brpts = pcv.dilate(gray_img=brpts_img, ksize=3, i=1)
    tubules = skel - dil_brpts

    tubules[np.where(tubules<0)] = 0

    # label tubules
    lab_tubules = skimage.measure.label(tubules)

    return lab_tubules

    # op = 0.5 * skel + 0.5 * brpts_img
    # plt.imshow(op)
    # plt.show()

    # print(len(np.where(branchpts==255)[0]))


lab_tubules = tub_analysis()


# print(np.where(lab_tubules==102))
# print(np.where(dil_lab_tubules==102))
# plt.imshow(lab_tubules)
# plt.show()


def get_tubule_XY(lab_tubules):
    lt = {}
    # lt = []
    for each in range(1, len(np.unique(lab_tubules))):
        tub_len = len(np.where(lab_tubules==each)[0])
        if tub_len > 5:
            # lt.append(np.where(lab_tubules==each))
            lt[each] = np.where(lab_tubules==each)

    coords = {}
    for num, pts in lt.items():
        coords[num] = []
        for (x, y) in zip(pts[0], pts[1]):
            coords[num].append((x,y))
    return coords



# lt = []
# lt.append(np.where(lab_tubules==102))
#
#
# coords = {}
# for num, pts in lt.items():
#     coords[num] = []
#     for (x, y) in zip(pts[0], pts[1]):
#         coords[num].append((x,y))
#
# print(coords)


coords = get_tubule_XY(lab_tubules)


ndt = {}
for tubule_num, coord_list in coords.items():
    # print(coord_list)
    ndt[tubule_num] = []
    for i in range(100):
        img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0%s_ch00.tif'%f'{i:02d}')
        img = (img - img.min()) / (img.max() - img.min())
        l = []
        for (x, y) in coord_list:
            if (x > 0 and x < 127) and (y > 0 and y < 127):
                mean_val = (img[x,y] + img[x+1,y] + img[x-1,y] + img[x,y-1] + img[x,y+1]) / 5
            else:
                mean_val = img[x,y]
            l.append(mean_val)
        ndt[tubule_num].append(l)

# print(len(ndt[1]))
# print(len(ndt[1][4]))

mean_list = []

for k, v in ndt.items():
    mean_list.append(np.mean(v, axis=0))




# intensity_vals = []
# for i in range(100):
#     l = []
#     img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0%s_ch00.tif'%f'{i:02d}')
#     img = (img - img.min()) / (img.max() - img.min())
#     # for num, points in coords.items():
#     #     for (x, y) in points:
#
#     for each in lt:
#         l.append(img[each])
#     intensity_vals.append(l)
#

# newlt = []
#
# # for tub_list in range(len(intensity_vals[0])):
# #     for frame in range(100):
# #         newlt.append(intensity_vals[frame][tub_list])
#
#
# mean_list = []
# for i in range(0, len(newlt), 100):
#     mean_list.append(np.mean(newlt[i:i+100], axis=0))
#
# print(mean_list[5])
# exit()

for i, val in enumerate(mean_list):
    # plt.plot(mean_list[i])
    # plt.title('Intensity profile for highlighted tubule', fontsize=24)
    # plt.xlabel('Distance along tubule (pixels)', fontsize=20)
    # plt.ylabel('Average normalized intensity over time (100 frames)', fontsize=20)
    # plt.show()
    print(np.var(mean_list[i]))


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

plt.axis('off')
plt.suptitle('Junction detection')
plt.title('Yellow: all branching areas, Red: Only degree 3 or more areas')
plt.savefig('Refined_junction_detection', bbox_inches='tight')
plt.close()
