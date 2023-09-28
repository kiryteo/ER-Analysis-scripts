import cv2
import sknw
from skimage.morphology import skeletonize
import networkx as nx
import json
import matplotlib.pyplot as plt
import collections
from skimage import io
import numpy as np

import imageio

# plt.switch_backend("agg")

import imageio

import sknw
import numpy as np


img = imageio.imread('comparison_figures/climp1_er_mean.png')
# binimg = imageio.imread('/localhome/asa420/Downloads/ERnet-v2-main/images/climp1_er_mean_proc_enhance.png')

gt_skel = imageio.imread('/localhome/asa420/MIAL/data/confocal-data/Climp/climp1_proc_skel.png')

skel = imageio.imread('/localhome/asa420/MIAL/data/confocal-data/Climp/climp1_er_mean_proc_enhance_skel.png')

analyzer_input = imageio.imread('comparison_figures/processed data/images/ER_er_ip.png')

analyzer_skel = imageio.imread('comparison_figures/processed data/images/ER_skele_bin.png')



plt.imshow(img, cmap='gray')

graph = sknw.build_sknw(gt_skel, multi=False, iso=False)

for (s,e) in graph.edges():
    ps = graph[s][e]['pts']
    plt.plot(ps[:,1], ps[:,0], 'green')

# draw node by o
nodes = graph.nodes()
ps = np.array([nodes[i]['o'] for i in nodes])

plt.plot(ps[:,1], ps[:,0], 'r.', markersize=3)

plt.axis('off')

# plt.show()
plt.savefig('climp1_er_mean_out_GT_graph.png', bbox_inches='tight', pad_inches=0, dpi=300)

plt.close()

exit()




# analyzer_enh = imageio.imread('comparison_figures/processed data/images/ER_enh.png')

# print(analyzer_enh.shape)

# exit()


# print(analyzer_input.shape)

# if analyzer_skel[:,:,0].all() == analyzer_skel[:,:,1].all() == analyzer_skel[:,:,2].all() == analyzer_skel[:,:,3].all():
#     print('yes')

# exit()

# fig, ax = plt.subplots(1, 4, figsize=(15, 15))

# ax[0].imshow(analyzer_skel[:,:,0], cmap='gray')
# ax[1].imshow(analyzer_skel[:,:,1], cmap='gray')
# ax[2].imshow(analyzer_skel[:,:,2], cmap='gray')
# ax[3].imshow(analyzer_skel[:,:,3], cmap='gray')

# er = analyzer_input[:,:,1]
# er = er[32:er.shape[0]-26, 32:er.shape[0]-26]

op = analyzer_skel[:,:,0]


# plt.imshow(er, cmap='gray')
# plt.show()

# exit()

op[np.where(op > 0)] = 1

skel = skeletonize(op).astype(np.uint16)

# graph = sknw.build_sknw(gt_skel, multi=False, iso=False)
graph = sknw.build_sknw(skel, multi=False, iso=False)

# draw image
import matplotlib.pyplot as plt
plt.imshow(er, cmap='gray')
# plt.imshow(binimg, cmap='gray')

# draw edges by pts
for (s,e) in graph.edges():
    ps = graph[s][e]['pts']
    plt.plot(ps[:,1], ps[:,0], 'green')

# draw node by o
nodes = graph.nodes()
ps = np.array([nodes[i]['o'] for i in nodes])


plt.plot(ps[:,1], ps[:,0], 'r.', markersize=2.5)

plt.axis('off')

plt.savefig('climp1_analyzer_graph.png', bbox_inches='tight', pad_inches=0, dpi=300)

plt.close()

# plt.show()

exit()


def remove_isolated_pixels(image):
    connectivity = 8

    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for label in range(num_stats):
        if stats[label, cv2.CC_STAT_AREA] < 50:
            new_image[labels == label] = 0

    return new_image


def binariseImage(I):
    if len(I.shape) > 2:
        ind = I[:, :, 0] > 250
    else:
        ind = I > 250
    Ibin = np.zeros((I.shape[0], I.shape[1])).astype("uint8")
    Ibin[ind] = 255
    Ibin = remove_isolated_pixels(Ibin)
    return Ibin


def getGraph():
    # opimg = img
    # plt.imshow(img, cmap="gray")
    # img = binariseImage(img) / 255
    img = imageio.imread('comparison_figures/climp1_er_mean_out_ERNet.png') / 255
    ske = skeletonize(img).astype(np.uint16)
    # ske = img.astype('uint16')

    # build graph from skeleton
    graph = sknw.build_sknw(ske)

    # draw image
    # plt.figure(figsize=(15, 15))
    plt.imshow(img, cmap="gray")

    # draw edges by pts
    for s, e in graph.edges():
        ps = graph[s][e]["pts"]
        plt.plot([ps[0, 1], ps[-1, 1]], [ps[0, 0], ps[-1, 0]], "green")

    # draw node by o
    nodes = graph.nodes()
    ps = np.array([nodes[i]["o"] for i in nodes])
    plt.plot(ps[:, 1], ps[:, 0], "r.", markersize=2)

    plt.axis("off")

    plt.savefig('climp1_er_mean_out_ERNet_graph.png', bbox_inches='tight', pad_inches=0, dpi=300)

    plt.close()

    # plt.savefig("%s_fig_graph.jpg" % basename, bbox_inches="tight", pad_inches=0, dpi=300)
    # plt.close()
    # open("%s_edges.dat" % basename, "w").write(
    #     str(graph.edges()).replace("(", "[").replace(")", "]")
    # )
    # open("%s_nodes.dat" % basename, "w").write(
    #     str(graph.nodes()).replace("(", "[").replace(")", "]")
    # )

    # edges = np.array(graph.edges())

    # return edges, nodes





# getGraph()


# exit()

# img = imageio.imread('comparison_figures/processed data/images/ER_skele.png')

# data = img[:,:,0]

# data = data[32:data.shape[0]-26, 32:data.shape[0]-26]

# print(data.shape)

# # exit()

# plt.axis('off')
# plt.imshow(data, cmap='gray')
# # plt.show()

# plt.savefig('comparison_figures/processed data/images/ER_skele_bin.png', bbox_inches='tight', pad_inches=0, dpi=300)

# plt.close()

# exit()



