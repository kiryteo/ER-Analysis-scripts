import imageio
import skimage
from skimage.filters import threshold_otsu, threshold_local
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import seaborn as sns
import os
from sklearn.feature_selection import f_oneway
import sknw
import networkx as nx
import numpy as np
from skimage import metrics
import pandas as pd
import scipy
import pickle
from scipy import spatial
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage import measure

from junction_analysis_modules import JunctionAnalysis as JA
import graph_connector_modules as gcm

max_val = 999
confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'
junc_analysis = JA(confocal_data_path)



def get_std_img(path):
    """
    Get standardised image
    @param path: path to image
    @return: standardised image
    """
    img = imageio.imread(path)
    if img.max() == img.min():
        return img
    img_max = img.max()
    img_min = img.min()
    return (img - img_min) / (img_max - img_min)


# patch1_vals -> list of intensity values per patch

def get_junctions_per_cc_id(label_ids, per_frame_junctions, labelled_img, region):
    """
    Get junctions per CC id
    @param label_ids: dict of CC ids and corresponding junctions
    @param per_frame_junctions: list of junctions per frame
    @param labelled_img: labelled image
    @param region: 'iso' or 'non-iso'
    @return: dict of CC ids and junctions
    """
    # get CC ids with only one junction -> isolated junctions
    if region == 'iso':
        index_list = [idx for idx, junctions in label_ids.items() if idx != 0 and len(junctions) == 1]
    # get CC ids with more than one junction -> fuzzy junctions
    else:
        index_list = [idx for idx, junctions in label_ids.items() if idx != 0 and len(junctions) > 1]

    # get junctions per CC id
    label_id_junctions = {}
    for junction in per_frame_junctions:
        idx = labelled_img[tuple(junction)]
        if idx in index_list:
            label_id_junctions.setdefault(idx, set()).add(tuple(junction))

    return label_id_junctions


# ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions('ATL', 1)
# label_ids, assigned_components = junc_analysis.get_ref_junc_per_CC_id(ref_junctions, labelled_img)
# label_id_junctions = get_junctions_per_cc_id(label_ids, per_frame_junctions, labelled_img, 'fuz')


# get pixel data per CC id
def get_CC_patch_data(junction, group, num):
    """
    Get pixel data per CC id

    junction: junction
    group: 'ATL', 'Climp', 'Control', 'RTN'
    num: number of series

    returns list of lists of pixel values
    """
    
    group_prefixes = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}

    pixel_data_egfp = []
    pixel_data_mch = []

    for i in range(100): # Get data for all 100 images
        file_name_egfp = f'{group_prefixes[group]}{num}_decon_t0{i:02d}_ch00_std.png'
        file_path_egfp = os.path.join(confocal_data_path, group, 'new_op_jul', 'std_egfp', file_name_egfp)
        input_egfp = imageio.imread(file_path_egfp)
        pixel_data_egfp.append(input_egfp[junction])

        if group != 'Control': # Control group does not have MCH data
            file_name_mch = f'{group_prefixes[group]}{num}_decon_t0{i:02d}_ch01_std.png'
            file_path_mch = os.path.join(confocal_data_path, group, 'new_op_jul', 'std_mch', file_name_mch)
            input_mch = imageio.imread(file_path_mch)
            pixel_data_mch.append(input_mch[junction])

    return pixel_data_egfp, pixel_data_mch


def get_sequence_CC_pixel_data(label_id_junctions, group, num):
    """
    Get pixel data per CC id
    @param label_id_junctions: dict of CC ids and junctions
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param num: number of series
    @return: list of lists of lists of lists of pixel values
    """

    #Get the patch data for all junctions in the image
    sequence_data_egfp = []
    sequence_data_mch = []

    for idx, junctions in label_id_junctions.items():
        junctions = list(junctions)
        for junction in junctions:
            pixel_data_egfp, pixel_data_mch = get_CC_patch_data(junction, group, num)
            sequence_data_egfp.append(pixel_data_egfp)
            if pixel_data_mch:
                sequence_data_mch.append(pixel_data_mch)
    return sequence_data_egfp, sequence_data_mch


def get_per_CC_pixel_data(group, num_series, region):
    """
    Get pixel data per CC id
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param num_series: number of series
    @param channel: 'egfp' or 'mch'
    @return: list of lists of lists of lists of pixel values
    """

    group_data_egfp = []
    group_data_mch = []

    for num in range(1, num_series + 1):
        ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions(group, num)
        label_ids, assigned_components = junc_analysis.get_ref_junc_per_CC_id(ref_junctions, labelled_img)
        label_id_junctions = get_junctions_per_cc_id(label_ids, per_frame_junctions, labelled_img, region)

        sequence_data_egfp, sequence_data_mch = get_sequence_CC_pixel_data(label_id_junctions, group, num)
        group_data_egfp.append(sequence_data_egfp)
        if sequence_data_mch:
            group_data_mch.append(sequence_data_mch)
    return group_data_egfp, group_data_mch


def get_mean_std_per_CC_pixel_data(data, measure):
    """
    Get mean and std of pixel data per CC id
    @param data: list of lists of lists of lists of pixel values
    @return: list of means and list of stds
    """
    if data:
        measure_data = []
        for seq_num in data:
            for junction_data in seq_num:
                if measure == 'mean':
                    measure_data.append(np.mean(np.array(junction_data, dtype=np.float)/255.))
                elif measure == 'std':
                    measure_data.append(np.std(np.array(junction_data, dtype=np.float)/255.))
        return measure_data


def create_per_CC_pixel_data_pickles(group, num_series, region):
    """
    Create pickles of pixel data per CC id
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param num_series: number of sequences per group
    @param region: 'isolated' or 'fuzzy'
    """
    egfp_data, mch_data = get_per_CC_pixel_data(group, num_series, region)
    
    pickle.dump(egfp_data, open(f'{group}_egfp_{region}_data.pkl', 'wb'))
    if mch_data:
        pickle.dump(mch_data, open(f'{group}_mch_{region}_data.pkl', 'wb'))


# create_per_CC_pixel_data_pickles('ATL', 26, 'iso')
# create_per_CC_pixel_data_pickles('Climp', 31, 'iso')
# create_per_CC_pixel_data_pickles('RTN', 29, 'iso')
# create_per_CC_pixel_data_pickles('Control', 31, 'iso')
# create_per_CC_pixel_data_pickles('ATL', 26, 'fuz')
# create_per_CC_pixel_data_pickles('Climp', 31, 'fuz')
# create_per_CC_pixel_data_pickles('RTN', 29, 'fuz')
# create_per_CC_pixel_data_pickles('Control', 31, 'fuz')

# exit()



def get_region_areas(label_id_junctions):
    """
    Get areas of regions
    @param label_id_junctions: dict of CC ids and junctions
    @return: list of areas
    """
    areas = []
    for id, juncs in label_id_junctions.items():
        if len(juncs) < 500:
            areas.append(len(juncs))

    return areas


def get_cc_ids(labelled_img, region):
    """
    Get CC ids for the specified region
    @param labelled_img: labelled image
    @param region: isolated or fuzzy
    @return: list of CC ids
    """

    # Create a dictionary to store per component data
    cc_data = {cc_id: [] for cc_id in np.unique(labelled_img)}

    # Populate the dictionary with locations for the specified region
    for loc in region:
        loc_x, loc_y = loc[0], loc[1]
        cc_id = labelled_img[loc_x, loc_y]
        cc_data[cc_id].append(loc)

    # Extract CC ids for the specified region
    cc_ids = [cc_id for cc_id, data in cc_data.items() if cc_id > 0 and len(data) > 0]

    return cc_ids


def get_region_cc(group, series_num, region):
    """
    Get CC ids for the specified region
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param series_num: series number
    @param region: isolated or fuzzy
    @return: list of CC ids
    """
    ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions(group, series_num)

    # dict with ids as key and (x, y) as value
    label_ids, unassigned_cc_dict = junc_analysis.separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)

    # iso, fuz, unk: list of lists with x, y
    iso, fuz, unk = junc_analysis.get_junction_areas(label_ids, unassigned_cc_dict)

    if region == 'iso':
        return get_cc_ids(labelled_img, iso), labelled_img
    else:
        return get_cc_ids(labelled_img, fuz), labelled_img


def get_region_areas_per_group(group, num_series, region):
    """
    Get areas of regions per group
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param num_series: number of series
    @param region: 'iso' or 'non-iso'
    @return: list of lists of areas
    """
    area_data = []
    for num in range(1, num_series + 1):
        ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions(group, num)
        label_ids, unassigned_cc_dict = junc_analysis.separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)
        label_id_junctions = get_junctions_per_cc_id(label_ids, per_frame_junctions, labelled_img, region)
        area_vals = get_region_areas(label_id_junctions)
        area_data.append(area_vals)

    return area_data


def cc_signal(group, channel, region):
    """
    Calculate deposit
    @param num_series: number of series
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param region: 'iso' or 'non-iso'
    @return: list of lists of deposits
    """

    # groups = {'ATL': 26, 'Climp': 31, 'Control': 31, 'RTN': 29}
    groups = {'ATL': 26, 'Climp': 31, 'RTN': 29}
    channel_idx = 0 if channel == 'egfp' else 1

    group_data = []
    for series_num in range(1, groups[group] + 1):
        region_cc, labelled_img = get_region_cc(group, series_num, region)

        # CC coords per series
        region_cc_coords = {each: np.where(labelled_img == each) for each in region_cc}

        # Data: num_cc * 100
        series_values = []
        for i in range(100):
            # if group == 'Control':
            #     path = f'{confocal_data_path}/Control/files/img_{series_num}_decon_t0{i:02d}.tif'
            # else:
            #     path = f'{confocal_data_path}/{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch0{channel_idx}.tif'
            path = f'{confocal_data_path}/{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch0{channel_idx}.tif'
            img = get_std_img(path)
            region_means = [np.mean(img[coords]) for coords in region_cc_coords.values()]
            series_values.append(region_means)

        series_values = np.array(series_values)

        group_data.append(series_values.T)

    return group_data




# egfp_seq_data, mch_seq_data, region_cc_coords = calc_deposit(1, 'ATL', 'iso')
# exit()

# calc_deposit_net_norm
def cc_signal_net_norm(num_series, group, region):
    """
    Calculate deposit
    @param num_series: number of series
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param region: 'iso' or 'fuz'
    @return: list of lists of deposits
    """

    # Get CC ids for the specified region
    region_cc, labelled_img = get_region_cc(group, num, region)
    region_cc_coords = {each: np.where(labelled_img == each) for each in region_cc}

    egfp_seq_data = []
    mch_seq_data = []

    # mnmx_egfp = []
    # mnmx_mch = []

    if group == 'Control':
        path_skel = f'{confocal_data_path}/{group}/new_op_jul/skel_max_proj/Ct{num}_max.png'
    else:
        path_skel = f'{confocal_data_path}/{group}/new_op_jul/skel_max_proj/{group[0]}{num}_max.png'

    for i in range(100):
        if group == 'Control':
            path_EGFP = f'{confocal_data_path}/Control/files/img_{num}_decon_t0{i:02d}.tif'
            path_mch = None
        else:
            path_EGFP = f'{confocal_data_path}/{group}/files/{group[0]}{num}_decon_t0{i:02d}_ch00.tif'
            path_mch = f'{confocal_data_path}/{group}/files/{group[0]}{num}_decon_t0{i:02d}_ch01.tif'
            # path_skel = f'{confocal_data_path}/{group}/new_op_jul/skel/{group[0]}{num}/{group[0]}{num}_decon_t0{i:02d}_ch00_skel.png'

        # Read images and get extract data
        img_egfp = imageio.imread(path_EGFP)
        skel = imageio.imread(path_skel)
        data_egfp = img_egfp[np.where(skel)]
        egfp_norm = (img_egfp - min(data_egfp)) / (max(data_egfp) - min(data_egfp))
        egfp_frame_data = [np.mean(egfp_norm[v]) for v in region_cc_coords.values()]

        egfp_seq_data.append(egfp_frame_data)

        if path_mch is not None:
            img_mch = imageio.imread(path_mch)

            data_mch = img_mch[np.where(skel)]

            mch_norm = (img_mch - min(data_mch)) / (max(data_mch) - min(data_mch))

            mch_frame_data = [np.mean(mch_norm[v]) for v in region_cc_coords.values()]

            mch_seq_data.append(mch_frame_data)

    egfp_seq_data = np.array(egfp_seq_data)

    mch_seq_data = np.array(mch_seq_data)

    return egfp_seq_data.T, mch_seq_data.T, region_cc_coords


def cc_signal_cc_norm(num_series, group, region):
    """
    Calculate deposit
    @param num_series: number of series
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param region: 'iso' or 'non-iso'
    @return: list of lists of deposits
    """


    region_cc, labelled_img = get_region_cc(group, num, region)

    region_cc_coords = {each: np.where(labelled_img == each) for each in region_cc}

    egfp_seq_data = []
    mch_seq_data = []

    for i in range(100):
        if group == 'Control':
            path_EGFP = f'{confocal_data_path}/Control/files/img_{num}_decon_t0{i:02d}.tif'
            path_mch = None
        else:
            path_EGFP = f'{confocal_data_path}/{group}/files/{group[0]}{num}_decon_t0{i:02d}_ch00.tif'
            path_mch = f'{confocal_data_path}/{group}/files/{group[0]}{num}_decon_t0{i:02d}_ch01.tif'

        img_egfp = imageio.imread(path_EGFP)

        # img_egfp = get_std_img(path_EGFP)
        egfp_frame_data = [np.mean(img_egfp[v]) for v in region_cc_coords.values()]

        egfp_seq_data.append(egfp_frame_data)
        # print(region_cc_coords.values())
        # exit()
        # frame_data = []

        # egfp_frame_data = [np.mean(img_egfp[v]) for v in region_cc_coords.values()]
        # for v in region_cc_coords.values():
        #     std_junc = (img_egfp[v] - min(img_egfp[v])) / (max(img_egfp[v]) - min(img_egfp[v]))
        # frame_data.append(img_egfp[v])
        # print(std_junc)
        # exit()
        # frame_data.append(np.mean(std_junc))

        # print(frame_data)
        # exit()

        # frame_data = (frame_data - min(frame_data)) / (max(frame_data) - min(frame_data))

        # print(frame_data)
        # exit()

        # egfp_seq_data.append(egfp_frame_data)
        # egfp_seq_data.append(frame_data)

        if path_mch is not None:
            img_mch = imageio.imread(path_mch)

            mch_frame_data = [np.mean(img_mch[v]) for v in region_cc_coords.values()]

            mch_seq_data.append(mch_frame_data)

    egfp_seq_data = np.array(egfp_seq_data)

    mch_seq_data = np.array(mch_seq_data)

    op_egfp = []
    op_mch = []

    for each in egfp_seq_data.T:
        each = (each - each.min()) / (each.max() - each.min())
        op_egfp.append(each)

    for each in mch_seq_data.T:
        each = (each - each.min()) / (each.max() - each.min())
        op_mch.append(each)

    op_egfp = np.array(op_egfp)
    op_mch = np.array(op_mch)

    # return egfp_seq_data.T, mch_seq_data.T, region_cc_coords
    return op_egfp, op_mch, region_cc_coords


def calc_egfp_deposit(group, channel, num_series, region):
    """
    Calculate EGFP deposit
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param channel: 'EGFP' or 'mCherry'
    @param num_series: number of series
    @param region: 'iso' or 'non-iso'
    @return: list of lists of deposits
    """

    channel_idx = 1 if channel == 'mCherry' else 0
    series_data = []

    for series_num in range(1, num_series + 1):
        region_cc, labelled_img = get_region_cc(group, series_num, region)
        region_cc_coords = {each: np.where(labelled_img == each) for each in region_cc}

        series_values = []
        for i in range(100):
            if group == 'Control':
                path = f'{confocal_data_path}/Control/files/img_{series_num}_decon_t0{i:02d}.tif'
            else:
                path = f'{confocal_data_path}/{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch0{channel_idx}.tif'
            img = get_std_img(path)
            region_means = [np.mean(img[coords]) for coords in region_cc_coords.values()]
            series_values.extend(region_means)

        series_data.extend(np.array(series_values))

    return series_data


def cc_area_measure(group, region, rstart, rend):
    """
    Calculate the area of each connected component
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param region: 'isolated' or 'fuzzy'
    @param rstart: start series number
    @param rend: end series number
    @return: list of areas
    """
    cc_area_list = []

    for series_num in range(rstart, rend + 1):
        region_cc, labelled_img = get_region_cc(group, series_num, region)
        regions = regionprops(labelled_img)

        if len(regions) == 0:
            print("No connected components in series %d" % series_num)
            continue

        cc_areas = [regions[each - 1]['Area'] for each in region_cc]
        cc_area_list.extend(cc_areas)

    # with open(f'cc_area_{group}_{region}.pkl', 'wb') as f:
    #     pickle.dump(cc_area_list, f)

    return cc_area_list


def stat_analysis(cc_area_atl, cc_area_climp, cc_area_rtn, cc_area_ctrl):
    """
    Perform statistical analysis
    @param cc_area_atl: list of areas of ATL
    @param cc_area_climp: list of areas of Climp
    @param cc_area_rtn: list of areas of RTN
    @param cc_area_ctrl: list of areas of Control
    @return: None
    """
    f_stat, p_val = f_oneway(cc_area_atl, cc_area_climp, cc_area_rtn, cc_area_ctrl)

    mc = MultiComparison(pd.concat([cc_area_atl, cc_area_climp, cc_area_rtn, cc_area_ctrl]), pd.Series(
        ["ATL"] * len(cc_area_atl) + ["Climp"] * len(cc_area_climp) + ["RTN"] * len(cc_area_rtn) + ["Control"] * len(
            cc_area_ctrl)))
    result = mc.tukeyhsd()

    stat, p_val = kruskal(cc_area_atl, cc_area_climp, cc_area_rtn, cc_area_ctrl)

    series_pairs = [(cc_area_atl, cc_area_climp), (cc_area_atl, cc_area_rtn), (cc_area_atl, cc_area_ctrl),
                    (cc_area_climp, cc_area_rtn), (cc_area_climp, cc_area_ctrl), (cc_area_rtn, cc_area_ctrl)]

    for pair in series_pairs:
        u_stat, p_val = mannwhitneyu(pair[0], pair[1], alternative='two-sided')
        print("Mann-Whitney U test between", pair[0].name, "and", pair[1].name)
        print("U-statistic:", u_stat)
        print("p-value:", p_val)


def get_junc_patches(newps, img):
    """
    Get patches around junctions
    @param newps: list of junction coordinates
    @param img: image
    @return: list of patches
    """
    img_patches = []
    for x, y in newps:
        if 1 < x < 126 and 1 < y < 126:
            patch = img[x - 1:x + 2, y - 1:y + 2].flatten()
            img_patches.append(patch)
    # for num, coordinate in enumerate(newps):
    #     x, y = newps[num]
    #     if x > 1 and x < 126 and y > 1 and y < 126:
    #         coord_vals = [img[x - 1, y], img[x + 1, y], img[x, y], img[x, y - 1], img[x, y + 1], img[x - 1, y - 1],
    #                       img[x - 1, y + 1], img[x + 1, y - 1],
    #                       img[x + 1, y + 1]]  # , img[x+2, y], img[x-2, y], img[x, y+2], img[x, y-2]]
    #         img_patches.append(coord_vals)
    return img_patches


def junc_patch_mean(newps, img):
    """
    Get mean of patches around junctions
    @param newps: list of junction coordinates
    @param img: image
    @return: list of patch means
    """
    junc_patches = []
    dt = {}
    for num, coordinate in enumerate(newps):
        x, y = newps[num]
        if x > 1 and x < 126 and y > 1 and y < 126:
            coord_vals = [img[x - 1, y], img[x + 1, y], img[x, y], img[x, y - 1], img[x, y + 1], img[x - 1, y - 1],
                          img[x - 1, y + 1], img[x + 1, y - 1],
                          img[x + 1, y + 1]]  # , img[x+2, y], img[x-2, y], img[x, y+2], img[x, y-2]]
            junc_mean = np.mean(coord_vals)
            # if junc_mean > thr:
            junc_patches.append(junc_mean)
            dt[(x, y)] = junc_mean
    return junc_patches, dt


def per_patch_variation(group, channel):
    """
    Get variation in patch means
    @param group: group name
    @param channel: channel number
    @return: None
    """

    for num_series in range(1, 2):
        er_img = ''
        mean_img = confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

        graph = junc_analysis.skel_to_graph(mean_img)
        newps = junc_analysis.get_ref_junctions(graph)

        sl = []

        for frame in range(100):
            if group == 'Control':
                path = confocal_data_path + f'{group}/files/img_{num_series}_decon_t0{frame:02d}.tif'

            else:
                path = confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t0{frame:02d}_ch0{channel}.tif'

            img = get_std_img(path)

            # im1_patch_vals = get_junc_patches(newps, img)
            im1_patch_vals = junc_patch_mean(newps, img)
            sl.append(im1_patch_vals)

        slt = np.array(sl)
        print(slt.shape)
        print(len(sl))
        print(len(sl[0]))
        exit()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        X = slt
        Y = slt.T


def patch_variation_viz():
    """
    Visualize variation in patch means
    @return: coord_dt: dictionary of patch means
    """

    er_img = ''
    mean_img = f'{confocal_data_path}ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'

    graph = junc_analysis.skel_to_graph(mean_img)
    newps = junc_analysis.get_ref_junctions(graph)
    sl = []
    dtl = []

    for i in range(100):
        path = f'{confocal_data_path}ATL/files/A1_decon_t0{i:02d}_ch00.tif'
        img = get_std_img(path)

        im1_patch_vals, dt = junc_patch_mean(newps, img)
        sl.append(im1_patch_vals)
        dtl.append(dt)

    coord_dt = {}
    for each in dtl:
        for k, v in each.items():
            if k not in coord_dt.keys():
                coord_dt[k] = []
            coord_dt[k].append(v)
    return coord_dt


def junc_area_locator(group, num_series):
    """

    @param group: group to be analyzed
    @param num_series: sequence number
    @return: dt, dictionary with matched junctions per reference junction - nearest neighbour approach
    """
    # mean_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'
    group_pref = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}

    er_img = ''
    mean_img = confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

    # Get junction coordinates from projection frame
    # newps = junc_analysis.get_junctions(er_img, mean_img)
    graph = junc_analysis.skel_to_graph(mean_img)
    newps = junc_analysis.get_ref_junctions(graph)

    nps = [[each[0], each[1]] for each in newps]
    nps = np.array(nps)
    # print(nps.shape)

    # sort the array for nearest neighbour matching per frame
    # nps_sorted is the mean projection (reference) junction list
    nps_sorted = sorted(nps, key=lambda t: t[0])

    dt = {}
    for frame in range(100):

        er_img = ''
        sk_img = confocal_data_path + f'{group}/new_op_jul/skel/{group_pref[group]}{num_series}/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_skel.png'

        sk_newps = junc_analysis.get_junctions(er_img, sk_img)

        sk_nps = [[each[0], each[1]] for each in sk_newps]
        sk_nnode_coords = np.array(sk_nps)

        # sk_nps_sorted is the per frame junction list
        sk_nps_sorted = sorted(sk_nps, key=lambda t: t[0])

        # matching of junction candidates
        for elem in nps_sorted:
            for sk_elem in sk_nps_sorted:
                dst = np.sqrt((sk_elem[0] - elem[0]) ** 2 + (sk_elem[1] - elem[1]) ** 2)
                if dst < 3:
                    if (elem[0], elem[1]) not in dt.keys():
                        dt[(elem[0], elem[1])] = []
                    dt[(elem[0], elem[1])].append([sk_elem, frame, dst])
    return dt


def refine_junc_dt(dt, min_presence=50):
    return {k: v for k, v in dt.items() if len(v) > min_presence}










def get_label_id(regions, iso, junc_id):
    for j in range(len(regions)):
        region_coords = regions[j].coords
        for each in region_coords:
            a, b = each[0], each[1]
            if a == iso[junc_id][0] and b == iso[junc_id][1]:
                return j


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure





def per_frame_num_junctions(labelled_img, iso):
    iso_cc = get_cc_ids(labelled_img, iso)

    # get lists to store the count of junctions within CC per frame
    iso_junc_num = []
    fuz_junc_num = []

    for i in range(100):
        temp_iso = []
        temp_fuz = []
        junc_frame = imageio.imread(
            confocal_data_path + f'ATL/new_op_jul/junctions/A1/A1_decon_t0{i:02d}_ch00_junc.png')

        # get the junc locations
        locations = np.where(junc_frame > 0)

        for locx, locy in zip(locations[0], locations[1]):
            cc_id = labelled_img[locx, locy]
            if cc_id in iso_cc:
                temp_iso.append(cc_id)
            else:
                temp_fuz.append(cc_id)
        iso_junc_num.append(temp_iso)
        fuz_junc_num.append(temp_fuz)

    return iso_junc_num, fuz_junc_num


def calc_egfp_junction_intensity_nbrhood(group, series_num, channel):
    ch = 1 if channel == 'mCherry' else 0
    sl = []
    for series_num in range(1, series_num + 1):
        nps, skdata, labelled_img = label_junctions(group, series_num)
        label_vals, unassigned_cc_dict = separate_junc_cc(nps, skdata, labelled_img)
        iso, fuz, unk = get_junction_areas(label_vals, unassigned_cc_dict)

        ln = []
        for i in range(100):
            frame_data = []
            if group == 'Control':
                path = confocal_data_path + f'Control/files/img_{series_num}_decon_t0{i:02d}.tif'

            else:
                path = confocal_data_path + f'{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch0{ch}.tif'

            img = get_std_img(path)
            for each in iso:
                x, y = each[0], each[1]
                mean_val = (img[x - 1, y - 1] + img[x - 1, y] + img[x - 1, y + 1] + img[x, y - 1] + img[x, y] + img[
                    x, y + 1] + img[x + 1, y - 1] + img[x + 1, y] + img[x + 1, y + 1]) / 9
                frame_data.append(mean_val)
            ln.extend(frame_data)
        ln = np.array(ln)

        sl.extend(ln)

    return sl





# exit()


def calc_egfp_deposit(group, channel, series_num, region):
    ch = 1 if channel == 'mCherry' else 0
    sl = []
    for series_num in range(1, series_num + 1):

        nps, skdata, labelled_img = label_junctions(group, series_num)
        label_vals, unassigned_cc_dict = separate_junc_cc(nps, skdata, labelled_img)
        iso, fuz, unk = get_junction_areas(label_vals, unassigned_cc_dict)
        # iso_cc = get_cc_ids(labelled_img, iso)
        if region == 'iso':
            region_cc = get_cc_ids(labelled_img, iso)
        else:
            region_cc = get_cc_ids(labelled_img, fuz)

        # dict to store the coords for each cc id
        # iso_cc_coords = {}
        # for each in iso_cc:
        #     iso_cc_coords[each] = np.where(labelled_img==each)

        region_cc_coords = {each: np.where(labelled_img == each) for each in region_cc}
        # for each cc, we obtain the index of dispersion per frame

        ln = []
        for i in range(100):
            if group == 'Control':
                path = confocal_data_path + f'Control/files/img_{series_num}_decon_t0{i:02d}.tif'

            else:
                path = confocal_data_path + f'{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch0{ch}.tif'

            # path = confocal_data_path + '%s/files/%s_decon_t0%s_ch0%s.tif'%(f'{group}', f'{group[0]}{series_num}', f'{i:02d}', f'{ch}')
            img = get_std_img(path)
            frame_data = [np.mean(img[v]) for v in region_cc_coords.values()]
            # ln.append(np.std(frame_data) / np.mean(frame_data))
            # ln.append(frame_data)
            ln.extend(frame_data)
        ln = np.array(ln)
        # m = np.std(ln, axis=0)

        # sl.extend(m)
        sl.extend(ln)
        # print(len(sl))
        # print(len(sl[0]))
        # print(sl[0])
        # exit()
    return sl






def per_movie_num_junctions(group, num_series):
    grp_iso = []
    grp_fuz = []

    for i in range(1, num_series + 1):
        nps, skdata, labelled_img = label_junctions(group, i)
        label_vals, unassigned_cc_dict = separate_junc_cc(nps, skdata, labelled_img)
        iso, fuz, unk = get_junction_areas(label_vals, unassigned_cc_dict)
        # grp_iso.append(len(iso))
        # grp_fuz.append(len(fuz))
        # grp_iso.append(len(iso_area))
        # grp_fuz.append(len(fuz_area))

    return grp_iso, grp_fuz





def per_movie_junctions_area(group, num_series):
    grp_iso = []
    grp_fuz = []

    for i in range(1, num_series + 1):
        nps, skdata, labelled_img = label_junctions(group, i)
        label_vals, unassigned_cc_dict = separate_junc_cc(nps, skdata, labelled_img)
        iso, fuz, unk = get_junction_areas(label_vals, unassigned_cc_dict)
        # grp_iso.append(len(iso))
        # grp_fuz.append(len(fuz))
        # grp_iso.extend(iso_area)
        # grp_fuz.extend(fuz_area)

    return grp_iso, grp_fuz







def per_cc_analysis():
    nps, skdata, labelled_img = label_junctions('ATL', 1)

    print(np.unique(labelled_img))
    cc_dict = {num: [] for num in range(1, len(np.unique(labelled_img)))}
    exit()
    l = []
    for i in range(100):
        junc_frame = imageio.imread(
            confocal_data_path + f'ATL/new_op_jul/junctions/A1/A1_decon_t0{i:02d}_ch00_junc.png')

        junctions = np.where(junc_frame > 0)
        for x, y in zip(junctions[0], junctions[1]):
            cc_id = labelled_img[x, y]
            cc_dict[cc_id] = ...

        # exit()








def junction_var_median():
    var_vals = [each[0] for each in n3]
    med = np.median(var_vals)
    mn = np.mean(var_vals)

    for i, each in enumerate(var_vals):
        if each < mn:
            # print(n1[i])
            print(len(dt[(n1[i][0], n1[i][1])]))




def get_mean_patch_intensity(er_input, a, b):
    return (er_input[a, b] + er_input[a - 1, b - 1] + er_input[a + 1, b + 1] + er_input[a, b + 1] + er_input[a, b - 1] +
            er_input[a + 1, b - 1] + er_input[a - 1, b + 1] + er_input[a + 1, b] + er_input[a - 1, b]) / 9


def get_junc_patch(er_input, a, b):
    patch = np.zeros((3, 3))
    pa = patch.flatten()
    vals = [er_input[a - 1, b - 1], er_input[a, b - 1], er_input[a + 1, b - 1], er_input[a - 1, b], er_input[a, b],
            er_input[a + 1, b], er_input[a - 1, b + 1], er_input[a, b + 1], er_input[a + 1, b + 1]]
    for i, each in enumerate(vals):
        pa[i] = each
    patch = np.resize(pa, (3, 3))
    return patch


def get_junc_lists(group, num_series):
    junc_to_analyze = junc_area_locator(group, num_series)
    newdt = {}
    for k, v in junc_to_analyze.items():
        v_new = []

        j = 0
        for i in range(100):
            if j < len(v):
                data = v[j]
                if data[1] == i:
                    v_new.append(data)
                    j = j + 1
                else:
                    v_new.append(max_val)
            else:
                v_new.append(max_val)

        newdt[k] = v_new
    return newdt


def junc_intensity_variation(group, num_series):
    newdt = get_junc_lists(group, num_series)

    if group == 'Control':
        ref_input = imageio.imread(confocal_data_path + f'Control/files/img_{num_series}_decon_t000.tif')

    else:
        ref_input = imageio.imread((confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t000_ch00.tif'))

    ref_input = (ref_input - ref_input.min()) / (ref_input.max() - ref_input.min())

    mean_val_dt = {}
    for k, v in newdt.items():
        mean_val_dt[k] = []
        a, b = k[0], k[1]
        m_val = get_mean_patch_intensity(ref_input, a, b)

        if v[0] != max_val:
            a, b = v[0][0][0], v[0][0][1]
            m_val = get_mean_patch_intensity(ref_input, a, b)
        mean_val_dt[k].append(m_val)
        for i in range(1, 100):
            if group == 'Control':
                er_input = imageio.imread((confocal_data_path + f'Control/files/img_{num_series}_decon_t0{i:02d}.tif'))

            else:
                er_input = imageio.imread(
                    (confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t0{i:02d}_ch00.tif'))

            er_input = (er_input - er_input.min()) / (er_input.max() - er_input.min())
            if v[i] is not max_val:
                a, b = v[i][0][0], v[i][0][1]
                m_val = get_mean_patch_intensity(er_input, a, b)
            else:
                # print(a, b)
                # m_val = get_mean_patch_intensity(er_input, a, b)
                m_val = get_mean_patch_intensity(er_input, k[0], k[1])
            mean_val_dt[k].append(m_val)
    return mean_val_dt


def junc_intensity_variation_runner():
    mean_val_dt = junc_intensity_variation('RTN', 1)

    for k1, v1 in mean_val_dt.items():
        # plt.title('Junction ref location: %s'%f'{k1}')
        # plt.plot(v1, linestyle='--', marker='o')
        # grd = np.gradient(v1)
        # plt.plot(grd, linestyle='--', marker='o')
        # plt.show()
        f = np.abs(np.fft.fft(v1))
        k = np.fft.fftfreq(len(v1))
        plt.plot(k)
        plt.show()
        break




def junction_location_plotter(group, num_series):
    # mean_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'

    group_pref = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}
    mean_img = confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

    graph = junc_analysis.skel_to_graph(mean_img)
    newps = junc_analysis.get_ref_junctions(graph)

    nps = [[each[0], each[1]] for each in newps]
    nps = np.array(nps)

    # print(nps[46, 1], nps[46, 0])
    #
    # exit()
    # plt.imshow(mean_proj_img, cmap='gray')

    # for (s,e) in g.edges():
    #     ps = g[s][e]['pts']
    #     plt.plot(ps[:,1], ps[:,0], 'green')
    # for dr in range(len(nps)):
    #     os.makedirs('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/A1_junc_viz/junc%s'%f'{dr+1}')

    for frame in range(100):

        # ER Input image
        if group == 'Control':
            path = confocal_data_path + f'{group}/files/img_{num_series}_decon_t0{frame:02d}.tif'

        else:
            path = confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t0{frame:02d}_ch00.tif'

        img = imageio.imread(path)
        img = (img - img.min()) / (img.max() - img.min())

        # ER Skel image
        # sk_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t0%s_ch00_skel.png'%f'{i:02d}'
        sk_img = confocal_data_path + f'{group}/new_op_jul/skel/{group_pref[group]}{num_series}/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_skel.png'

        sk_newps = get_junctions(sk_img)

        sk_nps = [[each[0], each[1]] for each in sk_newps]
        sk_nps = np.array(sk_nps)

        # img = img * 255.
        # # plt.plot(ps[:, 1], ps[:, 0], 'y.')
        # for j in range(47, 48):
        # plt.figure(figsize=(128/77, 128/77))
        plt.imshow(img, cmap='gray')
        # plt.imshow(imageio.imread(img), cmap='gray')
        plt.axis('off')
        # plt.title('t=%s'%f'{i}')
        plt.plot(nps[:, 1], nps[:, 0], 'r.')
        plt.plot(sk_nps[:, 1], sk_nps[:, 0], 'b.')
        # y = nps[0, 1]
        # x = nps[0, 0]
        # cv2.rectangle(img, (x-1, y-1), (x+1, y+1), (0, 0, 255), 2)

        plt.savefig((
                                confocal_data_path + f'{group}/new_op_jul/{group.lower()}_junc_viz/{group_pref[group]}{num_series}_t{frame:02d}.png'),
                    bbox_inches='tight', pad_inches=0)

        plt.close()


# junction_location_plotter('RTN', 1)
# exit()



def per_patch_pixel_fourier(group, channel):
    grp_dict = {}
    nd = {}
    sldt = []

    for num_series in range(1, 2):
        mean_img = confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

        graph = junc_analysis.skel_to_graph(mean_img)
        newps = junc_analysis.get_ref_junctions(graph)

        sl = []
        for frame in range(100):
            if group == 'Control':
                path = confocal_data_path + f'{group}/files/img_{num_series}_decon_t0{frame:02d}.tif'

            else:
                path = confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t0{frame:02d}_ch0{channel}.tif'

            img = imageio.imread(path)
            img = (img - img.min()) / (img.max() - img.min())

            # thr = threshold_otsu(img)

            # im1_patch_vals = get_junc_patches(newps, img)
            im1_patch_vals = junc_patch_mean(newps, img)
            # print(max(im1_patch_vals))
            # print(min(im1_patch_vals))
            sl.append(im1_patch_vals)
            # sldt.append(dt)

        # for each in sl:
        #     print(len(each))

        # slt shape: (9, num_patches, 100)
        slt = np.array(sl).T
        ht = slt.flatten()

        lval = np.quantile(ht, 0.05)
        hval = np.quantile(ht, 0.95)

        print(lval)
        print(hval)

        plt.hist(ht)
        plt.show()

        exit()

        #        sldt -> dict with key per junction and its mean patch value
        # sldt_ar = np.array(sldt).T
        # klist = sldt_ar[0].keys()

        # for k in klist:

        # for fr in range(len(slt)):
        #     plt.plot(slt[fr])
        #     plt.title('')
        #     plt.show()

        exit()

        # slt shape -> (93, 100)

    #     grp_dict[num_series] = slt
    #
    # return grp_dict


# atl_egfp = per_patch_pixel_fourier('ATL', 0)
# per_patch_pixel_fourier('ATL', 0)

# exit()


# atl_mc = per_patch_pixel_fourier('ATL', 1)
# climp_egfp = per_patch_pixel_fourier('Climp', 0)
# climp_mc = per_patch_pixel_fourier('Climp', 1)
# rtn_egfp = per_patch_pixel_fourier('RTN', 0)
# rtn_mc = per_patch_pixel_fourier('RTN', 1)
#

def inter_channel_correlation(c1, c2):
    data = []
    for (v1, v2) in zip(c1.values(), c2.values()):
        data.extend(np.corrcoef(mv1, mv2)[0, 1] for mv1, mv2 in zip(v1, v2))
    return data




# Obtain variance per pixel in the patch
# over 100 frames so we get 9 values per patch and then
# get the index of dispersion per patch

# ser_list = []
# for junc_num, junc in enumerate(slt):
#     ser_list.append(np.var(junc) / np.mean(junc))

# print(len(ser_list))
# for tf in range(len(sl)):
#     # a = slt.T[patch_num]
#     a = slt[tf]
#
#     # print(a.shape)
#     # exit()
#     ser_list.append(np.var(a) / np.mean(a))
# sns.distplot(a, hist=False)
# plt.plot(a)

# ser_list.append(np.abs(np.fft.fft(a)))
# plt.xlabel()

# label = 'movie num: ' + str(num_series)
# plt.plot(ser_list, label=label)
# # grp_list.extend(ser_list)
# plt.ylabel('Index of dispersion values', fontsize=16)
# plt.xlabel('Timestep', fontsize=16)
# plt.title('Index of dispersion for all patches (junctions) in a movie over time - RTN group', fontsize=20)
# plt.legend()
# plt.show()
# print(grp_list)
# return grp_list





# atl_list = per_patch_pixel_fourier('ATL', 0)
# climp_list = per_patch_pixel_fourier('Climp', 0)
# ctrl_list = per_patch_pixel_fourier('Control', 0)
# rtn_list = per_patch_pixel_fourier('RTN', 0)

# # print(atl_list)
# import pandas as pd

# df = pd.DataFrame()

# df['patch_intensity_vals'] = pd.Series(atl_list + climp_list + ctrl_list + rtn_list)
# df['group'] = pd.Series()

# exit()






# ATL_list = per_patch_pixel_fourier('ATL', 0)
# Climp_list = per_patch_pixel_fourier('Climp', 0)
# Ctrl_list = per_patch_pixel_fourier('Control', 0)
# RTN_list = per_patch_pixel_fourier('RTN', 0)
#
# sns.distplot(ATL_list, hist=False, label='ATL')
# sns.distplot(Climp_list, hist=False, label='Climp')
# sns.distplot(Ctrl_list, hist=False, label='Control')
# sns.distplot(RTN_list, hist=False, label='RTN')
#
# plt.legend()
# plt.xlabel('Variance')
# plt.title('')
# plt.show()

# exit()

def per_patch_pixel_variance(group):
    grp_list = []
    for num_series in range(1, 25):
        mean_img = confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

        graph = junc_analysis.skel_to_graph(mean_img)
        newps = junc_analysis.get_ref_junctions(graph)

        sl = []

        for frame in range(100):
            if group == 'Control':
                path = confocal_data_path + f'{group}/files/img_{num_series}_decon_t0{frame:02d}.tif'

            else:
                path = confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t0{frame:02d}_ch00.tif'

            img = imageio.imread(path)
            img = (img - img.min()) / (img.max() - img.min())

            im1_patch_vals = get_junc_patches(newps, img)
            sl.append(im1_patch_vals)

        # slt shape: (9, num_patches, 100)
        slt = np.array(sl)

        for patch_num in range(len(sl[0])):
            a = slt[:, patch_num, :]
        vl = [np.var(each) for each in a.T]
        ser_list = [np.var(vl) / np.mean(vl)]
        grp_list.extend(ser_list)

    return grp_list


# atl_list = per_patch_pixel_variance('ATL')
# climp_list = per_patch_pixel_variance('Climp')
# ctrl_list = per_patch_pixel_variance('Control')
# rtn_list = per_patch_pixel_variance('RTN')

# print(atl_list)
# print(len(atl_list[0]))
# sns.boxplot(atl_list)
# sns.boxplot(climp_list)
# sns.boxplot(ctrl_list)
# sns.boxplot(rtn_list)
# plt.show()


def per_patch_if_corr():
    grp_list = []
    for num_series in range(1, 25):
        mean_img = confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

        graph = junc_analysis.skel_to_graph(mean_img)
        newps = junc_analysis.get_ref_junctions(graph)

        sl = []

        for frame in range(100):
            if group == 'Control':
                path = confocal_data_path + f'{group}/files/img_{num_series}_decon_t0{frame:02d}.tif'

            else:
                path = confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t0{frame:02d}_ch00.tif'

            img = imageio.imread(path)
            img = (img - img.min()) / (img.max() - img.min())

            im1_patch_vals = get_junc_patches(newps, img)
            sl.append(im1_patch_vals)

            # slt shape: (9, num_patches, 100)
            slt = np.array(sl)

            # Obtain variance per pixel in the patch
            # over 100 frames so we get 9 values per patch and then
            # get the index of dispersion per patch

            ser_list = []
            for patch_num in range(len(sl[0])):
                a = slt[:, patch_num, :]
                vl = [np.var(each) for each in a.T]
                ser_list.append(np.var(vl) / np.mean(vl))

            grp_list.extend(ser_list)

    return grp_list


# op = np.abs(np.fft.fft(a))

# print(op)
# plt.plot(op)
# plt.show()

# print(a)
# print(im1_patch_vals)
# im1_patch_vals[1] = np.reshape(im1_patch_vals[1], (3,3))
# # print(im1_patch_vals[0])
# print(len(im1_patch_vals))
# print(len(im1_patch_vals[0]))
#
# op = np.abs(np.fft.fft2(im1_patch_vals[1]))
# # plt.imshow(op)
# plt.plot(op)
# plt.show()


def get_group_dif(group, metric):
    grp_list = []
    global met_val
    if group == 'Control':
        for series_num in range(1, 25):
            newps = get_junctions(confocal_data_path + f'Control/new_op_jul/Ctrl_mean_proj/Ct{series_num}_mean.png')

            ser_list = []
            for i in range(99):
                i1 = imageio.imread(
                    f'/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_{series_num}_decon_t0{i:02d}.tif')

                i2 = imageio.imread(
                    f'/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_{series_num}_decon_t0{i + 1:02d}.tif')

                i1 = (i1 - i1.min()) / (i1.max() - i1.min())
                i2 = (i2 - i2.min()) / (i2.max() - i2.min())

                img1_patch_vals = get_junc_patches(newps, i1)
                img2_patch_vals = get_junc_patches(newps, i2)

                l = []
                for i in range(len(img1_patch_vals)):
                    if metric == 'ssim':
                        met_val = metrics.structural_similarity(np.array(img1_patch_vals[i]),
                                                                np.array(img2_patch_vals[i]))
                    elif metric == 'cos_sim':
                        met_val = 1 - spatial.distance.cosine((img1_patch_vals[i]), (img2_patch_vals[i]))
                    elif metric == 'norm_cc':
                        i1 = np.array(img1_patch_vals[i])
                        i2 = np.array(img1_patch_vals[i])
                        met_val = scipy.signal.correlate(i1, i2)
                    elif metric == 'nmi':
                        met_val = mutual_information_2d(img1_patch_vals[i], img2_patch_vals[i])
                    elif metric == 'IF_corr':
                        valnum = np.corrcoef(img1_patch_vals[i], img2_patch_vals[i])
                        met_val = valnum[0, 1]

                    l.append(met_val)
                ser_list.extend(l)
            grp_list.extend(ser_list)
    else:
        for series_num in range(1, 25):
            newps = get_junctions(
                confocal_data_path + f'{group}/new_op_jul/{group}_mean_proj/{group[0]}{series_num}_mean.png')

            ser_list = []
            for i in range(98):
                i1 = imageio.imread(
                    f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch00.tif')

                i2 = imageio.imread(
                    f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/{group[0]}{series_num}_decon_t0{i + 1:02d}_ch00.tif')

                i1 = (i1 - i1.min()) / (i1.max() - i1.min())
                i2 = (i2 - i2.min()) / (i2.max() - i2.min())

                img1_patch_vals = get_junc_patches(newps, i1)
                img2_patch_vals = get_junc_patches(newps, i2)

                l = []
                for i in range(len(img1_patch_vals)):
                    if metric == 'ssim':
                        met_val = metrics.structural_similarity(np.array(img1_patch_vals[i]),
                                                                np.array(img2_patch_vals[i]))
                    elif metric == 'cos_sim':
                        met_val = 1 - spatial.distance.cosine((img1_patch_vals[i]), (img2_patch_vals[i]))
                    elif metric == 'norm_cc':
                        i1 = np.array(img1_patch_vals[i])
                        i2 = np.array(img1_patch_vals[i])
                        met_val = scipy.signal.correlate(i1, i2)
                    elif metric == 'nmi':
                        met_val = mutual_information_2d(img1_patch_vals[i], img2_patch_vals[i])
                    elif metric == 'IF_corr':
                        valnum = np.corrcoef(img1_patch_vals[i], img2_patch_vals[i])
                        met_val = valnum[0, 1]

                    l.append(met_val)
                ser_list.extend(l)
            grp_list.extend(ser_list)

    return grp_list
