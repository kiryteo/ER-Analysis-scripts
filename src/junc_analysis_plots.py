import numpy as np
from skimage.measure import label, regionprops
import imageio
import sknw
import networkx as nx
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'


def get_std_img(path):
    img = imageio.imread(path)
    return (img - img.min()) / (img.max() - img.min())


def skel_to_graph(skel):
    g = sknw.build_sknw(skel, iso=False)
    G = nx.Graph()

    node_set = g.nodes()

    G.add_nodes_from(node_set)
    G.add_edges_from(g.edges)

    degree_list = G.degree
    return node_set, degree_list


def get_junction_types(nps, lab):
    """

    @param nps: (ndarray) reference junctions
    @param lab: (ndarray) connected components for junctions
    @return: label_vals (dict) provides corresponding reference junctions per cc_id, assigned_components (list) provides cc with at least 1 reference junction
    """
    label_vals = {}

    assigned_components = []

    for each in nps:
        if lab[each[0], each[1]] != 0:
            if lab[each[0], each[1]] not in label_vals.keys():
                label_vals[(lab[each[0], each[1]])] = []
            label_vals[(lab[each[0], each[1]])].append([each[0], each[1]])
            assigned_components.append(lab[each[0], each[1]])
    return label_vals, assigned_components


def get_uncertain_junctions(lab, skdata, num_components, assigned_components):
    unassigned_components = [x for x in num_components if x not in assigned_components]

    unassigned_cc_dict = {}

    for each in skdata:
        cc_label = lab[each[0], each[1]]
        if cc_label != 0 and cc_label in unassigned_components:
            if cc_label not in unassigned_cc_dict.keys():
                unassigned_cc_dict[(lab[each[0], each[1]])] = []
            unassigned_cc_dict[(lab[each[0], each[1]])].append([each[0], each[1]])
    return unassigned_cc_dict


def separate_junc_cc(nps, skdata, labelled_img):
    regions = regionprops(labelled_img)

    # cc_list = []
    # for idx in range(1, labelled_img.max()):
    #     lab_i = props[idx].label

    # cc_area_dict = {}
    # for idx, props in enumerate(regions):
    #     cc_area_dict[idx] = props.area
    # cc_area_dict[idx] = [props.area, props.axis_major_length]

    num_components = np.unique(labelled_img)
    # print(num_components)

    label_vals, assigned_components = get_junction_types(nps, labelled_img)
    # print(label_vals)

    unassigned_cc_dict = get_uncertain_junctions(labelled_img, skdata, num_components, assigned_components)

    # return label_vals, cc_area_dict, unassigned_cc_dict
    return label_vals, unassigned_cc_dict


def get_junctions(mean_img):
    """

    @param mean_img: Input mean projection skel image (ndarray, binary)
    @return: list of Nodes with degree > 2
    """

    # if mean_img is obtained via skel mean projection
    # step, then use otsu else directly read the image

    # mean_proj_img = imageio.imread(mean_img)
    #
    # # perform thresholding + binarization + skel
    # thresh = threshold_otsu(mean_proj_img)
    # bin_img = mean_proj_img > thresh

    # skel = pcv.morphology.skeletonize(mask=mean_proj_img)

    skel = imageio.imread(mean_img)

    # Build graph from the skeleton
    node_set, degree_list = skel_to_graph(skel)
    node_coords = np.array([node_set[node]['o'] for node in node_set])

    return [node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2]


def get_all_junc(group, num_series):
    """

    @param group: group to be analyzed
    @param num_series: sequence number
    @return: nps (list) - provides all junctions with degree > 2 from the mean projection proc skeleton, skdata (list) - provides all junctions per skel frame
    """
    # mean_img = 'confocal_data_pathATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'

    group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}

    mean_img = f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

    # Get junction coordinates from projection frame
    newps = get_junctions(mean_img)

    nps = [[each[0], each[1]] for each in newps]
    skdata = []
    for frame in range(100):

        sk_img = f'{confocal_data_path}{group}/new_op_jul/skel/{group_pref[group]}{num_series}/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_skel.png'



        sk_newps = get_junctions(sk_img)

        sk_nps = [[each[0], each[1]] for each in sk_newps]
        # sk_nps = np.array(sk_nps)
        skdata.extend(sk_nps)

    return nps, skdata


def label_junctions(group, series_num):

    # fig, ax = plt.subplots()
    nps, skdata = get_all_junc(group, series_num)

    nps = np.array(nps)
    skdata = np.array(skdata)

    spread_img = np.zeros((128, 128))
    for each in skdata:
        spread_img[each[0], each[1]] = 255.


    # fig.add_subplot(1,2,1)
    # plt.imshow(spread_img)
    #
    labelled_img = label(spread_img, connectivity=2)
    # imageio.imsave('Climp12_junc_labelled.png', labelled_img)
    # fig.add_subplot(1,2,2)
    # plt.axis('off')
    # plt.imshow(labelled_img, cmap='gray')
    # plt.savefig('Climp12_junc_labelled.png', bbox_inches='tight', pad_inches=0, dpi=700)
    # plt.close()
    # plt.show()
    # exit()

    return nps, skdata, labelled_img


def get_junction_areas(label_vals, unassigned_cc_dict):
    isolated_junc = []
    isolated_junc_area = []
    fuzzy_junc = []
    fuzzy_junc_area = []
    for k, v in label_vals.items():
        if k != 0:
            if len(v) == 1:
                isolated_junc.append(v[0])
                # try:
                #     isolated_junc_area.append(cc_area_dict[k])
                # except:
                #     pass
            else:
                fuzzy_junc.append(v)
                # try:
                #     fuzzy_junc_area.append(cc_area_dict[k])
                # except:
                #     pass

    unknown_junc = [v for k, v in unassigned_cc_dict.items()]
    iso = np.array(isolated_junc)

    fuz = list(itertools.chain.from_iterable(fuzzy_junc))
    fuz = np.array(fuz)

    unk = list(itertools.chain.from_iterable(unknown_junc))
    unk = np.array(unk)

    # iso_area = list(itertools.chain.from_iterable(isolated_junc_area))
    # iso_area = np.array(isolated_junc_area)

    # fuz_area = list(itertools.chain.from_iterable(fuzzy_junc_area))
    # fuz_area = np.array(fuzzy_junc_area)


    return iso, fuz, unk#, iso_area, fuz_area


def get_cc_ids(labelled_img, region):
    """

    @param labelled_img: Input with all CC areas
    @param region: (list) iso or fuz
    @return: isolated or fuzzy region CC ids list
    """

    num_cc = np.unique(labelled_img)
    # dict to store per component data
    dt = {each: [] for each in num_cc}
    for loc in region:
        locx, locy = loc[0], loc[1]
        cc_id = labelled_img[locx, locy]
        dt[cc_id] = loc

    dt_vals = dt.values()

    return [i for i, num in enumerate(dt_vals) if i > 0 and len(num) != 0]


def calc_egfp_deposit(group, channel, series_num, region):

    ch = 1 if channel=='mCherry' else 0
    sl = []
    for series_num in range(1, series_num+1):

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

        region_cc_coords = {each: np.where(labelled_img==each) for each in region_cc}
        # for each cc, we obtain the index of dispersion per frame

        ln = []
        for i in range(100):
            if group == 'Control':
                path = f'{confocal_data_path}Control/files/img_{series_num}_decon_t0{i:02d}.tif'


            else:
                path = f'{confocal_data_path}{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch0{ch}.tif'


            # path = confocal_data_path + '%s/files/%s_decon_t0%s_ch0%s.tif'%(f'{group}', f'{group[0]}{series_num}', f'{i:02d}', f'{ch}')
            img = get_std_img(path)
            frame_data = [np.mean(img[v]) for v in region_cc_coords.values()]
            # ln.append(np.std(frame_data) / np.mean(frame_data))
            # ln.append(frame_data)
            ln.extend(frame_data)
        ln = np.array(ln)

        sl.extend(ln)
    return sl


def junction_cc_mean_distplot(channel):

    atl = calc_egfp_deposit('ATL', channel, 26, 'iso')
    climp = calc_egfp_deposit('Climp', channel, 31, 'fuz')
    ctrl = calc_egfp_deposit('Control', channel, 31, 'fuz')
    rtn = calc_egfp_deposit('RTN', channel, 29, 'fuz')

    sns.distplot(atl, label='ATL', hist=False)
    sns.distplot(climp, label='Climp', hist=False)
    sns.distplot(ctrl, label='Control', hist=False)
    sns.distplot(rtn, label='RTN', hist=False)

    plt.title(f'{channel} Mean Intensity per fuzzy area junction CC patch', fontsize=18)
    plt.xlabel(f'{channel} intensity mean values', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    # plt.legend(fontsize=14) # for distplot only

    plt.savefig(f'{channel}_fuzzy_CCs_mean_intensity_per_patch', bbox_inches='tight', pad_inches=0.2)
    plt.close()
    # plt.show()


# junction_cc_mean_distplot('ERmoxGFP')
# exit()


def junction_cc_mean_boxplot(channel, region):

    region_name = 'isolated' if region == 'iso' else 'fuzzy'
    atl = calc_egfp_deposit('ATL', channel, 26, region)

    climp = calc_egfp_deposit('Climp', channel, 31, region)
    ctrl = calc_egfp_deposit('Control', channel, 31, region)
    rtn = calc_egfp_deposit('RTN', channel, 29, region)

    df = pd.DataFrame()
    df['Values'] = pd.Series(np.concatenate((atl, climp, ctrl, rtn)))
    df['ids'] = pd.Series(np.concatenate((np.arange(1, len(atl)+1), np.arange(1, len(climp)+1), np.arange(1, len(ctrl)+1), np.arange(1, len(rtn)+1))))
    df['Group'] = pd.Series(np.concatenate((['ATL'] * len(atl), ['Climp'] * len(climp), ['Control'] * len(ctrl), ['RTN'] * len(rtn))))
    #
    ax = sns.violinplot(data=df, y='Group', x='Values')
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=13)
    # sns.scatterplot(data=df, x='ids', y='Values', hue='Group', style='Group')

    # plt.suptitle(f'{channel} deposit in isolated region junction CCs across conditions', fontsize=22)
    # plt.title('Variance of junction CC mean per patch over 100 frames', fontsize=14)
    plt.title(f'{channel} Mean Intensity per {region_name} area junction CC patch', fontsize=18)
    plt.xlabel(f'{channel} intensity mean values', fontsize=14)
    plt.ylabel('ER-Shaping proteins', fontsize=14)
    # plt.legend(fontsize=14) # for distplot only

    plt.savefig(f'{channel}_{region_name}_CCs_mean_intensity_per_patch_violinplot', bbox_inches='tight', pad_inches=0.2)
    plt.close()
    # plt.show()

# junction_cc_mean_boxplot('ERmoxGFP', 'fuz')
# junction_cc_mean_boxplot('mCherry', 'fuz')

import cv2

def per_ref_junc_deposit():
    egfp_img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean/atl1_er_mean.png')

    # print(egfp_img.max())

    mch_stack = np.zeros((128, 128))
    for i in range(100):
        mch_img = f'/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0{i:02d}_ch01.tif'
        mch_img = get_std_img(mch_img) * 255
        mch_stack += mch_img
    mch_op = mch_stack / 100

    cv2.imwrite('ATL1_mch_mean.png', mch_op)
    # exit()

    # print(mch_op.max())
    # exit()

    ref_skel = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'
    junctions = get_junctions(ref_skel)

    # print(junctions)
    egfp_data = []
    mch_data = []
    for each in junctions:
        x, y = each[0], each[1]
        mean_egfp_val = (egfp_img[x-1, y-1] + egfp_img[x-1, y] + egfp_img[x-1, y+1] + egfp_img[x, y-1] + egfp_img[x, y] + egfp_img[x, y+1] + egfp_img[x+1, y-1] + egfp_img[x+1, y] + egfp_img[x+1, y+1]) / 9
        egfp_data.append(mean_egfp_val)
        mean_mch_val = (mch_op[x-1, y-1] + mch_op[x-1, y] + mch_op[x-1, y+1] + mch_op[x, y-1] + mch_op[x, y] + mch_op[x, y+1] + mch_op[x+1, y-1] + mch_op[x+1, y] + mch_op[x+1, y+1]) / 9
        mch_data.append(mean_mch_val)

    # print(max(egfp_data))
    # print(max(mch_data))
    # exit()

    # plt.plot(egfp_data, label='EGFP')
    # plt.plot(mch_data, label='mCherry')
    df = pd.DataFrame()
    # df['egfp'] = pd.Series(egfp_data)
    # df['mcherry'] = pd.Series(mch_data)
    df['intensity'] = pd.Series(np.concatenate((egfp_data, mch_data)))
    df['junction_num'] = pd.Series(np.concatenate((np.arange( len(egfp_data)+1), np.arange(len(mch_data)+1))))
    df['channel'] = pd.Series(np.concatenate((['EGFP'] * len(egfp_data), ['mCherry'] * len(mch_data))))

    # print(df['channel'])
    # print(df['junctions'])
    plt.title('ATL series 1 junction intensity values (mean per 3x3 patch)', fontsize=14)

    sns.scatterplot(data=df, x='junction_num', y='intensity', hue='channel')

    plt.show()


per_ref_junc_deposit()
exit()


def junction_cc_mean_distplot_per_group(group, series_num):

    egfp_data = calc_egfp_deposit(group, 'ERmoxGRP', series_num, 'fuz')
    mch_data = calc_egfp_deposit(group, 'mCherry', series_num, 'fuz')
    return egfp_data if group == 'Control' else (egfp_data, mch_data)
    # climp = calc_egfp_deposit('Climp', channel, 31, 'fuz')
    # ctrl = calc_egfp_deposit('Control', channel, 31, 'fuz')
    # rtn = calc_egfp_deposit('RTN', channel, 29, 'fuz')


atl_egfp, atl_mch = junction_cc_mean_distplot_per_group('ATL', 1)
# climp_egfp, climp_mch = junction_cc_mean_distplot_per_group('Climp', 31)
# rtn_egfp, rtn_mch = junction_cc_mean_distplot_per_group('RTN', 29)
# ctrl_egfp = junction_cc_mean_distplot_per_group('Control', 31)


# plt.scatter(atl_egfp, label='ATL_ERmoxGFP')
# plt.scatter(atl_mch, label='ATL_mCherry')

df = pd.DataFrame()
df['Values'] = pd.Series(np.concatenate((atl_egfp, atl_mch)))
df['channel'] = pd.Series(np.concatenate((np.arange(1, len(atl_egfp)+1), np.arange(1, len(atl_mch)+1))))

sns.scatterplot(data=df, x='Values', y='channel', hue='Values')

# plt.legend()

plt.show()

exit()

# sns.distplot(atl_egfp, label='ATL_ERmoxGFP', hist=False)
# sns.distplot(atl_mch, label='ATL_mCherry', hist=False)
# sns.distplot(climp_egfp, label='Climp_ERmoxGFP', hist=False)
# sns.distplot(climp_mch, label='Climp_mCherry', hist=False)
# sns.distplot(rtn_egfp, label='RTN_ERmoxGFP', hist=False)
# sns.distplot(rtn_mch, label='RTN_mCherry', hist=False)
# sns.distplot(ctrl_egfp, label='Control_ERmoxGFP', hist=False)

# sns.distplot(climp, label='Climp', hist=False)
# sns.distplot(ctrl, label='Control', hist=False)
# sns.distplot(rtn, label='RTN', hist=False)

plt.title('Mean Intensity per fuzzy junction area CC patch for all conditions', fontsize=18)
plt.xlabel('Intensity mean values', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.legend(fontsize=12) # for distplot only

plt.savefig('All_conditions_fuzzy_CCs_mean_intensity_per_patch_across_channels', bbox_inches='tight', pad_inches=0.2)
plt.close()
# plt.show()

# junction_cc_mean_distplot_per_group('RTN', 29)