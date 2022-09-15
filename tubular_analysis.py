import imageio
import skimage
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
# from skan import skeleton_to_csgraph
from networkx.drawing.nx_agraph import graphviz_layout
import seaborn as sns
import sknw
import networkx as nx
import numpy as np
import cv2
# from scipy import stats
from KDEpy import FFTKDE
from skimage import exposure

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

# for i in range(100):
#     l = []
#     path = '/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0%s_ch00.tif'%f'{i:02d}'
#     img = imageio.imread(path)
#     l.extend(img.flatten())
#
#
# plt.hist(l)
# plt.show()
# exit()

def tub_analysis(mean_img):

    mean_proj_img = imageio.imread(mean_img)

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

#
# lab_tubules = tub_analysis()


# print(np.where(lab_tubules==102))
# print(np.where(dil_lab_tubules==102))
# plt.imshow(lab_tubules)
# plt.show()

def junction_flow(mean_img):
    mean_proj_img = imageio.imread(mean_img)

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
    # dil_brpts = pcv.dilate(gray_img=brpts_img, ksize=3, i=1)

    # return newps, dil_brpts
    return newps

newps = junction_flow('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A1_mean.png')

# print(newps)

def junction_analysis(group):
    l_mean = []
    l_std = []

    # junc_vals stores the
    junc_vals = []
    for i in range(100):
        l = []
        if group == 'ATL':
            path = '/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0%s_ch00.tif'%f'{i:02d}'
        elif group == 'Climp':
            path = '/localhome/asa420/MIAL/data/confocal_movies/Climp/files/C1_decon_t0%s_ch00.tif'%f'{i:02d}'
        elif group == 'Control':
            path = '/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_1_decon_t0%s.tif'%f'{i:02d}'
        else:
            path = '/localhome/asa420/MIAL/data/confocal_movies/RTN/files/R1_decon_t0%s_ch00.tif'%f'{i:02d}'
        img = imageio.imread(path)

        # p2, p98 = np.percentile(img, (2, 98))
        # img = exposure.rescale_intensity(img, in_range=(p2, p98))
        # img = (img - img.min()) / (img.max() - img.min())
        img = exposure.equalize_hist(img)
        # print(img.max())
        # print(img.min())

        for each in newps:
            x, y = each[0], each[1]
            mean_val = (img[x+1, y] + img[x-1, y] + img[x, y] + img[x, y-1] + img[x, y+1]) / 5
            l.append(mean_val)
        junc_vals.append(l)

    # for jun in junc_vals:
    #     l_mean.append(np.mean(jun))
    #     l_std.append(np.std(jun))

    # return junc_vals, l_mean, l_std
    return junc_vals


atl_junc_vals = junction_analysis('ATL')
climp_junc_vals = junction_analysis('Climp')
ctrl_junc_vals = junction_analysis('Control')
rtn_junc_vals = junction_analysis('RTN')

kde_atl = FFTKDE(bw='silverman', kernel='triweight')
xatl, yatl = FFTKDE(bw='silverman', kernel='triweight').fit(atl_junc_vals)(2**11)
yatl[xatl<=0.001] = 0
yatl = yatl * 2

xcl, ycl = FFTKDE(bw='silverman', kernel='triweight').fit(climp_junc_vals)(2**11)
ycl[xcl<=0.001] = 0
ycl = ycl * 2

xct, yct = FFTKDE(bw='silverman', kernel='triweight').fit(ctrl_junc_vals)(2**11)
yct[xct<=0.001] = 0
yct = yct * 2

xrt, yrt = FFTKDE(bw='silverman', kernel='triweight').fit(rtn_junc_vals)(2**11)
yrt[xrt<=0.001] = 0
yrt = yrt * 2

plt.plot(xatl, yatl, label='ATL')
plt.plot(xcl, ycl, label='Climp')
plt.plot(xct, yct, label='Control')
plt.plot(xrt, yrt, label='RTN')



# sns.distplot(atl_junc_vals[0], label='ATL')
# sns.distplot(climp_junc_vals[0], label='Climp')
# sns.distplot(ctrl_junc_vals[0], label='Control')
# sns.distplot(rtn_junc_vals[0], label='RTN')
plt.legend()

# sns.distplot(junc_vals[0])
plt.show()
# plt.hist(l_mean)
# plt.hist(l_std)
# plt.show()



exit()



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


# coords = get_tubule_XY(lab_tubules)


def get_ndt(coords):
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
    return ndt

def get_ndt_mean(ndt):
    mean_list = []

    for k, v in ndt.items():
        mean_list.append(list(np.mean(v, axis=0)))

    return mean_list


def variance_analysis():
    ATL_vals = []
    for series_num in range(1, 15):
        mean_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A%s_mean.png'%f'{series_num}'
        lab_tubules = tub_analysis(mean_img)
        coords = get_tubule_XY(lab_tubules)
        ndt = get_ndt(coords)
        mean_list = get_ndt_mean(ndt)
        var_list = []
        for each in mean_list:
            var_val = np.var(each)
            var_list.append(var_val)
        ATL_vals.extend(var_list)

    Climp_vals = []
    for series_num in range(1, 15):
        mean_img = '/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/Climp_mean_proj/C%s_mean.png'%f'{series_num}'
        lab_tubules = tub_analysis(mean_img)
        coords = get_tubule_XY(lab_tubules)
        ndt = get_ndt(coords)
        mean_list = get_ndt_mean(ndt)
        var_list = []
        for each in mean_list:
            var_val = np.var(each)
            var_list.append(var_val)
        Climp_vals.extend(var_list)

    Ctrl_vals = []
    for series_num in range(1, 15):
        mean_img = '/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/Ctrl_mean_proj/Ct%s_mean.png'%f'{series_num}'
        lab_tubules = tub_analysis(mean_img)
        coords = get_tubule_XY(lab_tubules)
        ndt = get_ndt(coords)
        mean_list = get_ndt_mean(ndt)
        var_list = []
        for each in mean_list:
            var_val = np.var(each)
            var_list.append(var_val)
        Ctrl_vals.extend(var_list)

    RTN_vals = []
    for series_num in range(1, 15):
        mean_img = '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/RTN_mean_proj/R%s_mean.png'%f'{series_num}'
        lab_tubules = tub_analysis(mean_img)
        coords = get_tubule_XY(lab_tubules)
        ndt = get_ndt(coords)
        mean_list = get_ndt_mean(ndt)
        var_list = []
        for each in mean_list:
            var_val = np.var(each)
            var_list.append(var_val)
        RTN_vals.extend(var_list)

    sns.distplot(Climp_vals, label='Climp')
    sns.distplot(Ctrl_vals, label='Control')
    sns.distplot(ATL_vals, label='ATL')
    sns.distplot(RTN_vals, label='RTN')
    plt.legend()
    plt.show()

# variance_analysis()
# exit()

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

var_list = []
for i, val in enumerate(mean_list):
    # plt.plot(mean_list[i])
    # plt.title('Intensity profile for highlighted tubule', fontsize=24)
    # plt.xlabel('Distance along tubule (pixels)', fontsize=20)
    # plt.ylabel('Average normalized intensity over time (100 frames)', fontsize=20)
    # plt.show()
    var_list.append(np.var(mean_list[i]))


# plt.hist(var_list)
import seaborn as sns
sns.boxplot(var_list)
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

plt.axis('off')
plt.suptitle('Junction detection')
plt.title('Yellow: all branching areas, Red: Only degree 3 or more areas')
plt.savefig('Refined_junction_detection', bbox_inches='tight')
plt.close()
