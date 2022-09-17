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
from skimage import filters


def intensity_variation():
    l = []
    for i in range(100):
        img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0%s_ch00.tif'%f'{i:02d}')
        # img = exposure.equalize_hist(img)
        # l.append(img.max())
        # l.append(img.mean())
        l.append(np.median(img))
    plt.plot(l)
    plt.xlabel('Frame number')
    plt.ylabel('Median intensity value')#Max intensity value')
    plt.title('Median intensity per frame - confocal ATL series 1')
    plt.show()


# intensity_variation()
# exit()


def junction_location_plotter():
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
    # plt.imshow(mean_proj_img, cmap='gray')

    # for (s,e) in g.edges():
    #     ps = g[s][e]['pts']
    #     plt.plot(ps[:,1], ps[:,0], 'green')

    for i in range(100):
        img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t0%s_ch00_skel.png'%f'{i:02d}')
        plt.imshow(img, cmap='gray')
        # plt.plot(ps[:, 1], ps[:, 0], 'y.')
        plt.plot(nps[:, 1], nps[:, 0], 'r.')

        plt.show()




def get_unique_components(path):
    img = imageio.imread(path)
    cc = skimage.measure.label(img)
    return len(np.unique(cc)) - 1

def count_connected_components():
    l1 = []
    for num in range(1, 31):
        for i in range(100):
            path = '/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/skel/Ct%s/Ct%s_decon_t0%s_ch00_skel.png'%(f'{num}', f'{num}', f'{i:02d}')
            num_components = get_unique_components(path)
            l1.append(num_components)

            # cc = skimage.measure.label(img)
            # cnt = 0
            # for each in np.unique(cc):
            #     if len(np.where(cc==each)) == 1:
            #         continue
            #     else:
            #         cnt += 1
            # # l4.append(len(np.unique(cc)) - 1)
            # l1.append(cnt)

    l2 = []
    for num in range(1, 27):
        for i in range(100):
            path = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A%s/A%s_decon_t0%s_ch00_skel.png'%(f'{num}', f'{num}', f'{i:02d}')
            num_components = get_unique_components(path)
            l2.append(num_components)

    l3 = []
    for num in range(1, 31):
        for i in range(100):
            path = '/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/skel/C%s/C%s_decon_t0%s_ch00_skel.png'%(f'{num}', f'{num}', f'{i:02d}')
            num_components = get_unique_components(path)
            l3.append(num_components)

    l4 = []
    for num in range(1, 30):
        for i in range(100):
            path = '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/skel/R%s/R%s_decon_t0%s_ch00_skel.png'%(f'{num}', f'{num}', f'{i:02d}')

            num_components = get_unique_components(path)
            l4.append(num_components)

    sns.distplot(l1, label='Control')
    sns.distplot(l2, label='ATL')
    sns.distplot(l3, label='Climp')
    sns.distplot(l4, label='RTN')
    plt.legend()
    plt.title('Number of connected components in skeletons for all groups')
    plt.xlabel('Number of connected components per sample')
    plt.show()


# count_connected_components()
# exit()

def preproc_rolling_ball():
    img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/preproc/A1/A1_decon_t000_ch00_proc.png')
    op = skimage.restoration.rolling_ball(img, radius=1)
    # op = skimage.morphology.erosion(op)
    # op = skimage.morphology.erosion(op)
    # op = skimage.morphology.erosion(op)
    # op = skimage.morphology.erosion(op)

    edges = filters.roberts(op)
    edges = skimage.morphology.area_closing(edges, area_threshold=2)
    # edges = filters.sobel(op)
    # edges = (edges > 0)
    thr = threshold_otsu(edges)
    edges = edges > thr

    skel = skimage.morphology.skeletonize(edges)

    fig, ax = plt.subplots()
    r, c = 1, 3

    fig.add_subplot(r,c,1)
    plt.imshow(img)

    fig.add_subplot(r,c,2)
    plt.imshow(edges)

    fig.add_subplot(r,c,3)
    plt.imshow(skel)

    plt.show()


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


# newps = junction_flow('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A1_mean.png')

# print(newps)

def get_junc_vals(path, newps):
    l = []
    img = imageio.imread(path)
    img = (img - img.min()) / (img.max() - img.min())
    # img = exposure.equalize_hist(img)
    for each in newps:
        x, y = each[0], each[1]
        mean_val = (img[x + 1, y] + img[x - 1, y] + img[x, y] + img[x, y - 1] + img[x, y + 1] + img[x+1, y+1] + img[x-1, y-1] + img[x-1, y+1] + img[x+1, y-1]) / 9
        # mean_val = (img[x + 1, y] + img[x - 1, y] + img[x, y] + img[x, y - 1] + img[x, y + 1]) / 5
        l.append(mean_val)
        # if mean_val == 0:
        #     continue
        # else:
        #     l.append(mean_val)
    return l


def junction_analysis(group, channel):
    """

    @param group: Protein group to analyze
    @param channel: ERmoxGFP=0, mCherry=1
    @return: mean intensity values for junction region
    """

    # l_mean = []
    # l_std = []

    prefix = '/localhome/asa420/MIAL/data/confocal_movies/'
    # junc_vals stores the

    if group == 'ATL':
        junc_vals = []
        for series_num in range(1, 27):
            newps = junction_flow(prefix + 'ATL/new_op_jul/ATL_mean_proj/A%s_mean.png' % f'{series_num}')
            for i in range(100):
                path = prefix + 'ATL/files/A%s_decon_t0%s_ch0%s.tif' % (f'{series_num}', f'{i:02d}', f'{channel}')

                l = get_junc_vals(path, newps)
                junc_vals.append(l)

    elif group == 'Climp':
        junc_vals = []
        for series_num in range(1, 31):
            newps = junction_flow(prefix + 'Climp/new_op_jul/Climp_mean_proj/C%s_mean.png' % f'{series_num}')
            for i in range(100):
                path = prefix + 'Climp/files/C%s_decon_t0%s_ch0%s.tif' % (f'{series_num}', f'{i:02d}', f'{channel}')

                l = get_junc_vals(path, newps)
                junc_vals.append(l)

    elif group == 'Control':
        junc_vals = []
        for series_num in range(1, 31):
            newps = junction_flow(prefix + 'Control/new_op_jul/Ctrl_mean_proj/Ct%s_mean.png' % f'{series_num}')
            for i in range(100):
                path = prefix + 'Control/files/img_%s_decon_t0%s.tif' % (f'{series_num}', f'{i:02d}')

                l = get_junc_vals(path, newps)
                junc_vals.append(l)

    else:
        junc_vals = []
        for series_num in range(1, 30):
            newps = junction_flow(prefix + 'RTN/new_op_jul/RTN_mean_proj/R%s_mean.png' % f'{series_num}')
            for i in range(100):
                path = prefix + 'RTN/files/R%s_decon_t0%s_ch0%s.tif' % (f'{series_num}', f'{i:02d}', f'{channel}')

                l = get_junc_vals(path, newps)
                junc_vals.append(l)


        # # p2, p98 = np.percentile(img, (2, 98))
        # # img = exposure.rescale_intensity(img, in_range=(p2, p98))
        # img = (img - img.min()) / (img.max() - img.min())
        # # img = exposure.equalize_hist(img)

        # for each in newps:
        #     x, y = each[0], each[1]
        #     mean_val = (img[x + 1, y] + img[x - 1, y] + img[x, y] + img[x, y - 1] + img[x, y + 1]) / 5
        #     l.append(mean_val)
        # junc_vals.append(l)

    # for jun in junc_vals:
    #     l_mean.append(np.mean(jun))
    #     l_std.append(np.std(jun))

    # return junc_vals, l_mean, l_std
    return junc_vals




# atl_junc_vals3 = junction_analysis(3,'ATL')
# atl_junc_vals4 = junction_analysis(4,'ATL')
# atl_junc_vals5 = junction_analysis(5,'ATL')
# atl_junc_vals6 = junction_analysis(6,'ATL')
# atl_junc_vals7 = junction_analysis(7,'ATL')
# atl_junc_vals8 = junction_analysis(8,'ATL')

# print(atl_junc_vals)
import pandas as pd


def get_junc_variance(group_junc):
    variance_val_list = []
    for i in range(0, len(group_junc), 100):
        l1 = group_junc[i:i+100]
        l1 = np.array(l1).T.tolist()

        for each in l1:
            var_val = np.var(each)
            variance_val_list.append(var_val)

    return variance_val_list


def plot_junc_analysis():
    atl_egfp_junc_vals = junction_analysis('ATL', 0)
    atl_mch_junc_vals = junction_analysis('ATL', 1)
    # print(len(atl_junc_vals))
    # print(len(atl_junc_vals[100]))
    # print(len(atl_junc_vals[0][0]))

    climp_egfp_junc_vals = junction_analysis('Climp', 0)
    climp_mch_junc_vals = junction_analysis('Climp', 1)

    # ctrl_junc_vals = junction_analysis('Control')

    rtn_egfp_junc_vals = junction_analysis('RTN', 0)
    rtn_mch_junc_vals = junction_analysis('RTN', 1)

    # ctrl_temporal = np.array(ctrl_junc_vals).T.tolist()


    # temporal_vals = list(map(list, zip(*atl_junc_vals)))

    # print(len(temporal_vals))
    # print(len(atl_junc_vals[0]))


    # t1 = []
    # for i in range(0, len(atl_egfp_junc_vals), 100):
    #     l1 = atl_egfp_junc_vals[i:i+100]
    #     l1 = np.array(l1).T.tolist()
    #
    #     for each in l1:
    #         var_val = np.var(each)
    #         t1.append(var_val)

    t1mch = []
    for i in range(0, len(atl_egfp_junc_vals), 100):
        l1 = atl_egfp_junc_vals[i:i+100]
        l1 = np.array(l1).T.tolist()

        for each in l1:
            var_val = np.var(each)
            t1mch.append(var_val)

    t2 = []
    for i in range(0, len(climp_junc_vals), 100):
        l1 = climp_junc_vals[i:i+100]
        l1 = np.array(l1).T.tolist()

        for each in l1:
            var_val = np.var(each)
            t2.append(var_val)

    t3 = []
    for i in range(0, len(ctrl_junc_vals), 100):
        l1 = ctrl_junc_vals[i:i+100]
        l1 = np.array(l1).T.tolist()

        for each in l1:
            var_val = np.var(each)
            t3.append(var_val)

    t4 = []
    for i in range(0, len(rtn_junc_vals), 100):
        l1 = rtn_junc_vals[i:i+100]
        l1 = np.array(l1).T.tolist()

        for each in l1:
            var_val = np.var(each)
            t4.append(var_val)

    df = pd.DataFrame()
    tseries = pd.Series(np.concatenate((t1,t2,t3,t4)))
    ls = pd.Series(np.concatenate((['ATL']*len(t1), ['Climp']*len(t2), ['Control']*len(t3), ['RTN']*len(t4))))

    df['Junction Intensity variance'] = tseries
    df['Group'] = ls

    # sns.distplot(t1, label='ATL')
    # sns.distplot(t2, label='Climp')
    # sns.distplot(t3, label='Control')
    # sns.distplot(t4, label='RTN')
    sns.boxplot(data=df, x='Junction Intensity variance', y='Group')
    plt.legend()
    plt.title('Junction intensity flow variance over time for all movies across groups')
    plt.show()
    # print(len(l1[0]))

    # for ser_num in range(0, len(atl_junc_vals), 100):
    #     val_list =

    # print(atl_junc_vals.shape)
    # temporal_vals = np.transpose(atl_junc_vals)
    #
    # print(len(temporal_vals[0]))
    # print(len(atl_junc_vals[0]))
    # print(len(temporal_vals[0]))
    # plt.plot(temporal_vals[0])
    # plt.hist(temporal_vals[0])
    # plt.show()
    # ctrl_temporal = list(map(list, zip(*ctrl_junc_vals)))

    # atl1 = temporal_vals[0][:100]
    ctl_var_list = []
    # for i in range(len()
    # print(len(ctrl_temporal))
    # ctl1 = ctrl_temporal[0][:100]

    # ft = np.fft.fft(atl1)
    # print(ft)

    # plt.plot(ctl1)
    # sns.boxplot(ctl1)
    # plt.show()

    # print(len(atl_junc_vals))
    # print(len(atl_junc_vals[1]))
    # df = pd.DataFrame()
    #
    # l = []
    # for each in atl_junc_vals:
    #     l.append(each)
    # for each in climp_junc_vals:
    #     l.append(each)
    # for each in ctrl_junc_vals:
    #     l.append(each)
    # for each in rtn_junc_vals:
    #     l.append(each)
    #
    # l2 = ['ATL'] * len(atl_junc_vals)
    # l3 = ['Climp'] * len(climp_junc_vals)
    # l4 = ['Control'] * len(ctrl_junc_vals)
    # l5 = ['RTN'] * len(rtn_junc_vals)
    #
    # groups = np.concatenate((l2,l3,l4,l5))
    # s2 = pd.Series(groups)
    #
    # s = pd.Series(l)
    # df['Junc intensity variation'] = s
    # df['groups'] = s2
    #
    # df['Junc intensity variation'] = pd.Series()
    # df['Junc intensity variation'].append(atl_junc_vals)
    # df['Junc intensity variation'].append(climp_junc_vals)
    # df['Junc intensity variation'].append(ctrl_junc_vals)
    # df['Junc intensity variation'].append(rtn_junc_vals)

    # xatl, yatl = FFTKDE(bw='silverman', kernel='triweight').fit(atl_junc_vals)(2 ** 11)
    # yatl[xatl <= 0.001] = 0
    # yatl = yatl * 2

    # xatl4, yatl4 = FFTKDE(bw='silverman', kernel='triweight').fit(atl_junc_vals4)(2**11)
    # yatl4[xatl4<=0.001] = 0
    # yatl4 = yatl4 * 2
    #
    # xatl5, yatl5 = FFTKDE(bw='silverman', kernel='triweight').fit(atl_junc_vals5)(2**11)
    # yatl5[xatl5<=0.001] = 0
    # yatl5 = yatl5 * 2
    #
    #
    # xatl6, yatl6 = FFTKDE(bw='silverman', kernel='triweight').fit(atl_junc_vals6)(2**11)
    # yatl6[xatl6<=0.001] = 0
    # yatl6 = yatl6 * 2
    #
    # xatl7, yatl7 = FFTKDE(bw='silverman', kernel='triweight').fit(atl_junc_vals7)(2**11)
    # yatl7[xatl7<=0.001] = 0
    # yatl7 = yatl7 * 2
    #
    #
    # xatl8, yatl8 = FFTKDE(bw='silverman', kernel='triweight').fit(atl_junc_vals8)(2**11)
    # yatl8[xatl8<=0.001] = 0
    # yatl8 = yatl8 * 2


    # xcl, ycl = FFTKDE(bw='silverman', kernel='triweight').fit(climp_junc_vals)(2 ** 11)
    # ycl[xcl <= 0.001] = 0
    # ycl = ycl * 2
    #
    # xct, yct = FFTKDE(bw='silverman', kernel='triweight').fit(ctrl_junc_vals)(2 ** 11)
    # yct[xct <= 0.001] = 0
    # yct = yct * 2
    #
    # xrt, yrt = FFTKDE(bw='silverman', kernel='triweight').fit(rtn_junc_vals)(2 ** 11)
    # yrt[xrt <= 0.001] = 0
    # yrt = yrt * 2

    # plt.plot(xatl3, yatl3, label='ATL3')
    # plt.plot(xatl4, yatl4, label='ATL4')
    # plt.plot(xatl5, yatl5, label='ATL5')
    # plt.plot(xatl6, yatl6, label='ATL6')
    # plt.plot(xatl7, yatl7, label='ATL7')
    # plt.plot(xatl8, yatl8, label='ATL8')
    # plt.plot(xatl, yatl, label='ATL')
    # plt.plot(xcl, ycl, label='Climp')
    # plt.plot(xct, yct, label='Control')
    # plt.plot(xrt, yrt, label='RTN')
    # sns.stripplot(data=df, x='Junc intensity variation', y='groups', hue='groups')
    # sns.swarmplot(data=climp_junc_vals, x='Climp')
    # sns.swarmplot(data=ctrl_junc_vals, x='Control')
    # sns.swarmplot(data=rtn_junc_vals, x='RTN')
    #
    # sns.distplot(atl_junc_vals, label='ATL')
    # sns.distplot(climp_junc_vals, label='Climp')
    # sns.distplot(ctrl_junc_vals, label='Control')
    # sns.distplot(rtn_junc_vals, label='RTN')
    # plt.title('Intensity values (standardized 0-1) in junction regions across groups - kernel 3x3 square')
    # plt.xlabel('Intensity value')
    # plt.legend()
    # #
    # # # sns.distplot(junc_vals[0])
    # plt.show()


plot_junc_analysis()
exit()

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

    tubules[np.where(tubules < 0)] = 0

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



def get_tubule_XY(lab_tubules):
    lt = {}
    # lt = []
    for each in range(1, len(np.unique(lab_tubules))):
        tub_len = len(np.where(lab_tubules == each)[0])
        if tub_len > 5:
            # lt.append(np.where(lab_tubules==each))
            lt[each] = np.where(lab_tubules == each)

    coords = {}
    for num, pts in lt.items():
        coords[num] = []
        for (x, y) in zip(pts[0], pts[1]):
            coords[num].append((x, y))
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
            img = imageio.imread(
                '/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0%s_ch00.tif' % f'{i:02d}')
            img = (img - img.min()) / (img.max() - img.min())
            l = []
            for (x, y) in coord_list:
                if (x > 0 and x < 127) and (y > 0 and y < 127):
                    mean_val = (img[x, y] + img[x + 1, y] + img[x - 1, y] + img[x, y - 1] + img[x, y + 1]) / 5
                else:
                    mean_val = img[x, y]
                if mean_val == 0:
                    continue
                else:
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
    for series_num in range(1, 2):
        mean_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A%s_mean.png' % f'{series_num}'
        lab_tubules = tub_analysis(mean_img)
        coords = get_tubule_XY(lab_tubules)
        ndt = get_ndt(coords)
        mean_list = get_ndt_mean(ndt)
        var_list = []
        for each in mean_list:
            var_val = np.var(each)
            if var_val == 0:
                continue
            else:
                var_list.append(var_val)
        ATL_vals.extend(var_list)

    Climp_vals = []
    for series_num in range(1, 2):
        mean_img = '/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/Climp_mean_proj/C%s_mean.png' % f'{series_num}'
        lab_tubules = tub_analysis(mean_img)
        coords = get_tubule_XY(lab_tubules)
        ndt = get_ndt(coords)
        mean_list = get_ndt_mean(ndt)
        var_list = []
        for each in mean_list:
            var_val = np.var(each)
            if var_val == 0:
                continue
            else:
                var_list.append(var_val)
        Climp_vals.extend(var_list)

    Ctrl_vals = []
    for series_num in range(1, 2):
        mean_img = '/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/Ctrl_mean_proj/Ct%s_mean.png' % f'{series_num}'
        lab_tubules = tub_analysis(mean_img)
        coords = get_tubule_XY(lab_tubules)
        ndt = get_ndt(coords)
        mean_list = get_ndt_mean(ndt)
        var_list = []
        for each in mean_list:
            var_val = np.var(each)
            if var_val == 0:
                continue
            else:
                var_list.append(var_val)
        Ctrl_vals.extend(var_list)

    RTN_vals = []
    for series_num in range(1, 2):
        mean_img = '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/RTN_mean_proj/R%s_mean.png' % f'{series_num}'
        lab_tubules = tub_analysis(mean_img)
        coords = get_tubule_XY(lab_tubules)
        ndt = get_ndt(coords)
        mean_list = get_ndt_mean(ndt)
        var_list = []
        for each in mean_list:
            var_val = np.var(each)
            if var_val == 0:
                continue
            else:
                var_list.append(var_val)
        RTN_vals.extend(var_list)

    # sns.distplot(Climp_vals, label='Climp')
    # sns.distplot(Ctrl_vals, label='Control')
    # sns.distplot(ATL_vals, label='ATL')
    # sns.distplot(RTN_vals, label='RTN')

    import pandas as pd
    df = pd.DataFrame()
    l = np.concatenate((Climp_vals, Ctrl_vals, ATL_vals, RTN_vals))
    df['variance_values'] = pd.Series(l)
    a = ['ATL'] * len(ATL_vals)
    cl = ['Climp'] * len(Climp_vals)
    ct = ['Control'] * len(Ctrl_vals)
    r = ['RTN'] * len(RTN_vals)
    l2 = np.concatenate((cl,ct,a,r))
    df['group'] = pd.Series(l2)

    sns.violinplot(data=df, x='variance_values', y='group')
    # plt.legend()
    plt.show()


variance_analysis()
exit()

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



# plt.axis('off')
# plt.suptitle('Junction detection')
# plt.title('Yellow: all branching areas, Red: Only degree 3 or more areas')
# plt.savefig('Refined_junction_detection', bbox_inches='tight')
# plt.close()
