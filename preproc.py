import imageio
import skimage
from skimage.filters import threshold_local, threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
import copy
from skimage.color import rgb2gray
import cv2
from skimage import measure
from skimage.morphology import area_closing
import scipy.ndimage
import sknw
import networkx as nx
from plantcv import plantcv as pcv


# img = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_CC_Contours_grayscale.png')
# print(img.shape)
# exit()

# img = imageio.imread('mv50_gray.png')
# ot = threshold_local(img, 75)
# img = img > ot
# sk = pcv.morphology.skeletonize(mask=img)
# cv2.imwrite('mv50_gray_bin_local.png', sk)
# exit()

# img = imageio.imread('/localhome/asa420/Downloads/AnalyzER_Simon_Fraser_University/some_GTs/GFP-HDEL_mv_50_Airyscan Processing_t1.jpg')
# # print(img.shape)
# img_gr = rgb2gray(img)
# cv2.imwrite('mv50_gray.png', img_gr*255)
# exit()

img = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_t0_ch0.png')

aop = skimage.morphology.area_opening(img, area_threshold=2)
erod = skimage.morphology.erosion(aop)
aop = skimage.morphology.area_opening(erod, area_threshold=2)
cl = skimage.morphology.area_closing(aop, area_threshold=32)
aop = skimage.morphology.area_opening(cl, area_threshold=2)

cv2.imwrite('morph-ops_c12_t0.png', aop)

loc = threshold_local(aop, 3)
loc = threshold_local(loc, 3)

cv2.imwrite('thresh_ops_c12_t0.png', loc)

# cv2.imwrite('AnalyzER-mv50.png', loc*255)

exit()


skel = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/er_mean_proc/climp12_er_mean_proc_enhance_skel.png')


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

node_coords = np.array([node_set[node]['o'] for node in node_set])
newps = []
for j, val in enumerate(d):
    if val[1] > 2:
        newps.append(node_coords[j])

nps = []
for each in newps:
    nps.append([each[0], each[1]])

nps = np.array(nps)

er_mean = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/er_mean/climp12_er_mean.png')
er_proj = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/er_mean_proc/climp12_er_mean_proc.png')


# er_proj[59, 52] = 258
#
# ot = threshold_otsu(er_proj)
# e = np.where(er_proj>ot)
#
# er_proj[e] = ot


# exit()

# er_proj[e] = er_proj[e] - 5
# ref_junc = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/Climp_12_proj_junctions.png')

# plt.imshow(er_proj)
# plt.imshow(contours)
# plt.imshow(ref_junc)
# plt.show()
# exit()


img = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_junc_projection.png')

op = area_closing(img, area_threshold=3)

contours = measure.find_contours(op)



fig, ax = plt.subplots()
# s = np.where(img>0)
# img[s] = 0
ax.imshow(er_mean, cmap=plt.cm.gray)

for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], 'cyan', linewidth=1)

ax.plot(nps[:, 1], nps[:, 0], 'o', markerfacecolor='magenta', markeredgecolor='magenta', mew=0.75, markersize=3)

# ax.imshow(ref_junc)

plt.axis('off')
# plt.show()
plt.savefig('C12_er_proj_Contours_ref_junc.png', bbox_inches='tight', pad_inches=0, dpi=700)
# plt.close()
# ax.axis('image')
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()
exit()

# er_mean = np.zeros((128,128))
# for i in range(100):
#     img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/junctions/C12/C12_decon_t0%s_ch00_junc.png'%(f'{i:02d}'))
#     # img = (img - img.min())/(img.max() - img.min())
#     er_mean += img
# cv2.imwrite('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/junctions/C12_junc_projection.png', (er_mean)*255)

# exit()

img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/files/C12_decon_t099_ch00.tif')
op = (img - img.min()) / (img.max() - img.min())
op = op * 255
cv2.imwrite('C12_t99_ch0.png', op)
exit()
#
# img = imageio.imread('/localhome/asa420/Downloads/Oligos_PLP1_abcam_MBP_DAPI_40x_z0_ch00.tif')
# op = (img - img.min()) / (img.max() - img.min())
# op = op * 255
# cv2.imwrite('/localhome/asa420/Downloads/Oligos_PLP1_abcam_MBP_DAPI_40x_z0_ch00_std.png', op)
# exit()

def preprocess_samples():
    img = imageio.imread('/localhome/asa420/Downloads/O4+ Day 81 on IBIDI_1_std.png')
    # std_img = ((img) / (img.max() - img.min())) * 255
    # plt.imshow(img)
    # plt.show()
    # exit()
    # img = rgb2gray(img)
    aop = skimage.morphology.area_opening(img, area_threshold=2)
    # plt.imshow(aop)
    # plt.show()
    # exit()

    # erod = skimage.morphology.erosion(aop)
    # nerod = copy.deepcopy(erod)
    # # plt.imshow(erod)
    # # plt.show()
    # nerod[np.where(erod)]= img[np.where(erod)]
    #
    # nimg = img - nerod
    #
    # plt.imshow(nimg)
    # plt.show()
    #
    # exit()
    # aop = skimage.morphology.area_opening(nimg, area_threshold=2)

    # plt.imshow(aop)
    # plt.show()
    #
    # exit()

    cl = skimage.morphology.area_closing(aop, area_threshold=32)
    # plt.imshow(cl)
    # plt.show()
    #
    # exit()
    aop = skimage.morphology.area_opening(cl, area_threshold=2)
    # plt.imshow(aop)
    # plt.show()

    # print(aop.shape)
    aop = rgb2gray(aop)
    # plt.imshow(aop)
    # plt.show()
    #
    # exit()
    loc = threshold_local(aop, 3)
    # plt.imshow(loc)
    # plt.show()
    # exit()

    loc = threshold_local(loc, 3)
    # plt.axis('off')
    # plt.imshow(loc, cmap='gray')
    # plt.show()
    # exit()

    imageio.imsave('/localhome/asa420/Downloads/O4+ Day 81 on IBIDI_1_std_proc.png', loc)
    # plt.savefig('/localhome/asa420/Downloads/Oligos_PLP1_abcam_MBP_DAPI_40x_z0_ch00_std_proc.png', bbox_inches='tight', pad_inches=0, dpi=400)
    # cv2.imwrite('/localhome/asa420/Downloads/Oligos_PLP1_abcam_MBP_DAPI_40x_z0_ch00_std_proc.png', loc*255)

# preprocess_samples()
#
# exit()

# img = imageio.imread('/localhome/asa420/Downloads/Oligos_PLP1_abcam_MBP_DAPI_40x_z0_Composite.png')
# print(img.min())
# print(img.max())

# img = imageio.imread('/localhome/asa420/Downloads/Oligos_preproc_enhance.png')
# aop = skimage.morphology.area_opening(img, area_threshold=8)

from skimage.filters.rank import median
from skimage.morphology import disk

# aop = scipy.ndimage.median_filter(img, size=4)

# fig, axes = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
# ax = axes.ravel()

# ax[0].imshow(skimage.morphology.area_opening(img, area_threshold=12))
# ax[1].imshow(median(img, disk(2)))

# aop = median(img, disk(1))
# aop = threshold_local(aop, 3)

# cv2.imwrite('/localhome/asa420/Downloads/Oligos_preproc_enh_proc.png', aop)

# plt.imshow(aop)
# plt.show()

from plantcv import plantcv as pcv

nimg = imageio.imread('/localhome/asa420/Downloads/Oligos_PLP1abcam_MBP_DAPI_40x_z0_ch00_std_proc_enhance.png')
sk = pcv.morphology.skeletonize(nimg)

# op = median(sk, disk(1))
# op = skimage.morphology.area_opening(sk, area_threshold=1)
# op = skimage.morphology.area_opening(op, area_threshold=1)
# op = skimage.morphology.area_closing(op, area_threshold=12)

cv2.imwrite('/localhome/asa420/Downloads/Oligos_PLP1abcam_MBP_DAPI_40x_z0_ch00_std_proc_enhance_skel.png', sk)
# plt.imshow(op)
# plt.show()

