import numpy as np
from skimage.measure import label, regionprops
import imageio
import sknw
import networkx as nx
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from numpy.polynomial.polynomial import polyfit
import pandas as pd
import scipy
import os
from scipy.stats import pearsonr
import statannot
from statsmodels.stats.multicomp import MultiComparison
from scipy.stats import kruskal, mannwhitneyu
import pickle as pkl

from structure_extraction import node_connector, get_updated_degree_nodes
from junction_analysis_modules import JunctionAnalysis as JA

confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies'

GROUP_PREF = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}
VALID_GROUPS = ['ATL', 'Climp', 'RTN', 'Control']
VALID_CONNECTIONS = ['iso-iso', 'iso-fuz', 'fuz-fuz']
VALID_CHANNELS = ['egfp', 'mch']
VALID_MEASURES = ['tubules', 'tub-mean']

junc_analysis = JA(confocal_data_path)


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


def get_junctions(er_input_path, mean_img):
    """

    @param mean_img: Input mean projection skel image (ndarray, binary)
    @return: list of Nodes with degree > 2
    """

    conn_graph = node_connector(er_input_path, mean_img)

    node_set, degree_list = conn_graph.nodes, conn_graph.degree

    # skel = imageio.imread(mean_img)
    #
    # sk_graph = sknw.build_sknw(skel, multi=True, iso=False)
    #
    # # Build graph from the skeleton
    # # node_set, degree_list = skel_to_graph(skel)
    #
    # node_set, degree_list = sk_graph.nodes, sk_graph.degree

    node_coords = np.array([node_set[node]['o'] for node in node_set])

    return [node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2]


def get_all_junc(group, num_series):
    """

    @param group: group to be analyzed
    @param num_series: sequence number
    @return: nps (list) - provides all junctions with degree > 2 from the mean projection proc skeleton, skdata (list) - provides all junctions per skel frame
    """

    group_pref = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}

    er_mean_img_name = f'{group.lower()}{num_series}_er_mean.png'
    er_mean_img = os.path.join(confocal_data_path, group, 'new_op_jul', 'er_mean', er_mean_img_name)

    skel_mean_img_name = f'{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'
    skel_mean_img = os.path.join(confocal_data_path, group, 'new_op_jul', 'er_mean_proc', skel_mean_img_name)

    # conn_graph = node_connector(er_mean_img, skel_mean_img)
    # #
    # node_set, degree_list = conn_graph.nodes, conn_graph.degree
    #
    # node_coords = np.array([node_set[node]['o'] for node in node_set])
    #
    # newps = [node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2]
    ref_junctions = get_junctions(er_mean_img, skel_mean_img)

    ref_junctions = [[each[0], each[1]] for each in ref_junctions]
    per_frame_junctions = []

    ### Get junction coordinates from projection frame
    # newps = get_junctions(er_mean_img, skel_mean_img)
    # newps = get_junctions(skel_mean_img)

    for frame in range(100):
        er_img = f'{confocal_data_path}/{group}/new_op_jul/std_egfp/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_std.png'
        sk_img = f'{confocal_data_path}/{group}/new_op_jul/skel/{group_pref[group]}{num_series}/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_skel.png'

        junctions = get_junctions(er_img, sk_img)
        # sk_newps = get_junctions(sk_img)

        junctions = [[each[0], each[1]] for each in junctions]
        # sk_nps = np.array(sk_nps)
        per_frame_junctions.extend(junctions)

    return ref_junctions, per_frame_junctions


def label_junctions(group, series_num):
    """
    Creates the connected component for junctions
    """
    ref_junctions, per_frame_junctions = get_all_junc(group, series_num)

    ref_junctions = np.array(ref_junctions)
    per_frame_junctions = np.array(per_frame_junctions)

    spread_img = np.zeros((128, 128))
    for each in per_frame_junctions:
        spread_img[each[0], each[1]] = 255.

    labelled_img = label(spread_img, connectivity=2)
    return ref_junctions, per_frame_junctions, labelled_img


def get_cc_ids(labelled_img, region):
    # sourcery skip: inline-immediately-returned-variable
    """
    Returns a list of isolated or fuzzy region CC ids.

    Args:
    - labelled_img: numpy.ndarray, input with all CC areas
    - region: list of tuples, iso or fuz

    Returns:
    - list of ints, CC ids for the specified region
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



def get_ref_junc_per_CC_id(reference_junctions, connected_components):
    """
    Return the reference junctions per connected component and the list of connected components with at least 1 reference
    junction.

    :param reference_junctions: (ndarray) the reference junctions
    :param connected_components: (ndarray) the connected components for the junctions
    :return: label_values (dict) provides corresponding reference junctions per cc_id,
             assigned_components (list) provides cc with at least 1 reference junction
    """
    label_ids = {}
    assigned_components = []

    for junction in reference_junctions:
        if connected_components[junction[0], junction[1]] != 0:
            cc_id = connected_components[junction[0], junction[1]]
            if cc_id not in label_ids:
                label_ids[cc_id] = []
            label_ids[cc_id].append([junction[0], junction[1]])
            assigned_components.append(cc_id)

    return label_ids, assigned_components


def get_uncertain_junctions(labelled_img, per_frame_junctions, num_components, assigned_components):
    unassigned_components = [x for x in num_components if x not in assigned_components]
    unassigned_cc_dict = {}
    for each in per_frame_junctions:
        cc_label = labelled_img[each[0], each[1]]
        if cc_label != 0 and cc_label in unassigned_components:
            if cc_label not in unassigned_cc_dict.keys():
                unassigned_cc_dict[(labelled_img[each[0], each[1]])] = []
            unassigned_cc_dict[(labelled_img[each[0], each[1]])].append([each[0], each[1]])
    return unassigned_cc_dict


def separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img):
    num_components = np.unique(labelled_img)
    label_ids, assigned_components = get_ref_junc_per_CC_id(ref_junctions, labelled_img)
    unassigned_cc_dict = get_uncertain_junctions(labelled_img, per_frame_junctions, num_components, assigned_components)
    return label_ids, unassigned_cc_dict


def get_junction_areas(label_ids, unassigned_cc_dict):
    isolated_junctions = []
    fuzzy_junctions = []
    unknown_junctions = []

    for cc_id, junctions in label_ids.items():
        if cc_id != 0:
            if len(junctions) == 1:
                isolated_junctions.append(junctions[0])
            else:
                fuzzy_junctions.append(junctions)

    for junctions in unassigned_cc_dict.values():
        unknown_junctions.extend(junctions)

    isolated_junctions = np.array(isolated_junctions)

    fuzzy_junctions = list(itertools.chain.from_iterable(fuzzy_junctions))
    fuzzy_junctions = np.array(fuzzy_junctions)

    unknown_junctions = list(itertools.chain.from_iterable(unknown_junctions))
    unknown_junctions = np.array(unknown_junctions)

    return isolated_junctions, fuzzy_junctions, unknown_junctions


def get_junctions_per_cc_id(label_ids, per_frame_junctions, labelled_img, region):

    if region == 'iso':
        ids = [id for id, junctions in label_ids.items() if id!=0 and len(junctions) == 1]
    else:
        ids = [id for id, junctions in label_ids.items() if id != 0 and len(junctions) > 1]

    label_id_junctions = {}
    for junction in per_frame_junctions:
        id = labelled_img[tuple(junction)]
        if id in ids:
            label_id_junctions.setdefault(id, set()).add(tuple(junction))

    return label_id_junctions


def get_per_CC_pixel_data(group, num_series, channel):
    """
    Variation for each pixel in a CC over 100 frames
    """

    group_prefixes = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}
    dir_name = 'std_egfp' if channel == 'egfp' else 'std_mch'
    ch = 0 if channel == 'egfp' else 1
    group_data = []

    for num in range(1, num_series+1):
        ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions(group, num)
        label_ids, assigned_components = get_ref_junc_per_CC_id(ref_junctions, labelled_img)
        label_id_junctions = get_junctions_per_cc_id(label_ids, per_frame_junctions, labelled_img, 'iso')

        sequence_data = []
        for id, junctions in label_id_junctions.items():
            junctions = list(junctions)
            for junc in junctions:
                pixel_data = []
                for i in range(100):
                    file_name = f'{group_prefixes[group]}{num}_decon_t0{i:02d}_ch{ch:02d}_std.png'
                    file_path = os.path.join(confocal_data_path, group, 'new_op_jul', dir_name, file_name)

                    input = imageio.imread(file_path)
                    pixel_data.append(input[junc])
                sequence_data.append(pixel_data)
        group_data.append(sequence_data)
    return group_data


atl = get_per_CC_pixel_data('ATL', 2, 'egfp')
print(atl[0])
exit()



def get_region_areas(label_id_junctions):
    areas = []
    for id, juncs in label_id_junctions.items():
        if len(juncs) < 500:
            areas.append(len(juncs))

    return areas


def get_region_areas_per_group(group, num_series, region):
    area_data = []
    for num in range(1, num_series + 1):
        ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions(group, num)
        label_ids, unassigned_cc_dict = separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)
        label_id_junctions = get_junctions_per_cc_id(label_ids, per_frame_junctions, labelled_img, region)
        # print(label_id_junctions)
        # exit()
        area_vals = get_region_areas(label_id_junctions)
        area_data.append(area_vals)

    return area_data


def plot_region_areas(channel):
    atl = get_region_areas_per_group('ATL', 26, 'fuz')
    a1, a2, a3 = atl[:10], atl[10:20], atl[20:]
    climp = get_region_areas_per_group('Climp', 31, 'fuz')
    c1, c2, c3 = climp[:10], climp[10:20], climp[20:]
    rtn = get_region_areas_per_group('RTN', 29, 'fuz')
    r1, r2, r3 = rtn[:10], rtn[10:20], rtn[20:]

    if channel == 'egfp':
        ctrl = get_region_areas_per_group('Control', 31, 'fuz')
        ct1, ct2, ct3 = ctrl[:10], ctrl[10:20], ctrl[20:]

    df = pd.DataFrame()

    # df['data_tubule_mean'] = pd.Series(np.concatenate((atl, climp, rtn)))
    df['data_tubule_mean'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3, ct1, ct2, ct3)))

    df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1),
                                                ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2),
                                                ['R3'] * len(r3), ['R1'] * len(ct1), ['R2'] * len(ct2),
                                                ['R3'] * len(ct3))))

    df['Group'] = pd.Series(np.concatenate((
        ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1), ['Climp'] * len(c2),
        ['Climp'] * len(c3), ['RTN'] * len(r1), ['RTN'] * len(r2), ['RTN'] * len(r3), ['Control'] * len(ct1), ['Control'] * len(ct2), ['Control'] * len(ct3))))

    # ax = sns.boxenplot(data=df, x='Group', y='data_tubule_mean')
    ax = sns.boxenplot(data=df, x='Replicate', y='data_tubule_mean', hue='Group', dodge=True)  # , yscale='log')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

    # egfp
    replicates = ['R1', 'R2', 'R3']

    if channel == 'egfp':
        groups = ['ATL', 'Climp', 'RTN', 'Control']
    else:
        groups = ['ATL', 'Climp', 'RTN']
    box_pairs = [((x, y), (x, z)) for i, x in enumerate(replicates) for j, y in enumerate(groups) for z in groups[j + 1 :]]

    # box_pairs = [(('R1', 'ATL'), ('R1', 'Climp')), (('R1', 'ATL'), ('R1', 'RTN')), (('R1', 'Climp'), ('R1', 'RTN')), (('R1', 'ATL'), ('R1', 'Control')), (('R1', 'Climp'), ('R1', 'Control')), (('R1', 'RTN'), ('R1', 'Control')), (('R2', 'ATL'), ('R2', 'Climp')), (('R2', 'ATL'), ('R2', 'RTN')), (('R2', 'Climp'), ('R2', 'RTN')), (('R2', 'ATL'), ('R2', 'Control')), (('R2', 'Climp'), ('R2', 'Control')), (('R2', 'RTN'), ('R2', 'Control')), (('R3', 'ATL'), ('R3', 'Climp')), (('R3', 'ATL'), ('R3', 'RTN')), (('R3', 'Climp'), ('R3', 'RTN')), (('R3', 'ATL'), ('R3', 'Control')), (('R3', 'Climp'), ('R3', 'Control')), (('R3', 'RTN'), ('R3', 'Control'))]

    statannot.add_stat_annotation(ax, x='Replicate', y='data_tubule_mean', hue='Group', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    # statannot.add_stat_annotation(ax, x='Group', y='data_tubule_mean', data=df, box_pairs=box_pairs,
    #                               test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    # plt.title('Cross-correlation between ERmoxGFP and mCherry over sequence for tubule intensity mean in all tubules',
    #           fontsize=18)
    plt.title('CC area for Fuzzy CCs across conditions', fontsize=20)
    plt.grid(True)
    plt.xlabel('Replicate', fontsize=20)
    plt.ylabel('Fuzzy CCs area', fontsize=18)

    plt.show()


def ref_junc_data_per_group(group, num_series, channel):
    group_data = []
    # grp_data_egfp, grp_data_mch = [], []

    dirname, ch = ('std_egfp', 0) if channel == 'egfp' else ('std_mch', 1)
    group_pref = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}

    for num in range(1, num_series+1):
        ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions(group, num)
        label_ids, unassigned_cc_dict = separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)
        iso, fuz, unk = get_junction_areas(label_ids, unassigned_cc_dict)

        # store per movie list of per ref junc list with 100 values
        data = []
        for each in iso:
            # store 100 values per ref junc
            junc_signal_vals = []
            # junc_sig_egfp = []
            # junc_sig_mch = []
            for i in range(100):
                input = imageio.imread(f'{confocal_data_path}/{group}/new_op_jul/{dirname}/{group_pref[group]}{num}_decon_t0{i:02d}_ch{ch:02d}_std.png')
                junc_signal_vals.append(input[each[0], each[1]])
            data.append(junc_signal_vals)

        # store list of per movie data list
        group_data.append(data)

    return group_data


# def create_junc_data_pickles():
#     atl_e = ref_junc_data_per_group('ATL', 26, 'egfp')
#     atl_m = ref_junc_data_per_group('ATL', 26, 'mcherry')
#     climp_e = ref_junc_data_per_group('Climp', 31, 'egfp')
#     climp_m = ref_junc_data_per_group('Climp', 31, 'mcherry')
#     rtn_e = ref_junc_data_per_group('RTN', 29, 'egfp')
#     rtn_m = ref_junc_data_per_group('RTN', 29, 'mcherry')
#     ctrl_e = ref_junc_data_per_group('Control', 31, 'egfp')
#
#     # with open('atl_ref_junc_egfp.pkl', 'wb') as fl:
#     #     pkl.dump(group_data, fl)


def ref_junc_variation(data, measure):
    measure_vals = []
    if measure == 'mean':
        for each in data:
            # l = []
            for junc in each:
                junc = np.array(junc) / 255
                # l.append(np.mean(junc))
                # measure_vals.append(l)
                measure_vals.append(np.mean(junc))
    elif measure == 'std':
        for each in data:
            # l = []
            for junc in each:
                junc = np.array(junc) / 255
                # l.append(np.std(junc))
            # measure_vals.append(l)
                measure_vals.append(np.std(junc))
    return measure_vals


def write_pickle(group, channel, data):
    with open(f'{group.lower()}_ref_junc_{channel}.pkl', 'wb') as fl:
        pkl.dump(data, fl)


def plot_ref_junc_variation():

    atl_data = pkl.load(open('atl_ref_junc_mch.pkl', 'rb'))
    # atl_data = ref_junc_data_per_group('ATL', 26, 'mch')
    # write_pickle('ATL', 'mch', atl_data)
    # a1, a2, a3 = atl_data[:10], atl_data[10:20], atl_data[20:]
    # a1, a2, a3 = ref_junc_variation(a1, 'std'), ref_junc_variation(a2, 'std'), ref_junc_variation(a3, 'std')

    atl = ref_junc_variation(atl_data, 'mean')

    # climp_data = ref_junc_data_per_group('Climp', 31, 'mch')
    # write_pickle('Climp', 'mch', climp_data)
    climp_data = pkl.load(open('climp_ref_junc_mch.pkl', 'rb'))
    # c1, c2, c3 = climp_data[:10], climp_data[10:20], climp_data[20:]
    # c1, c2, c3 = ref_junc_variation(c1, 'std'), ref_junc_variation(c2, 'std'), ref_junc_variation(c3, 'std')
    climp = ref_junc_variation(climp_data, 'mean')

    # rtn_data = ref_junc_data_per_group('RTN', 29, 'mch')
    # write_pickle('RTN', 'mch', rtn_data)
    rtn_data = pkl.load(open('rtn_ref_junc_mch.pkl', 'rb'))
    # r1, r2, r3 = rtn_data[:10], rtn_data[10:20], rtn_data[20:]
    # r1, r2, r3 = ref_junc_variation(r1, 'std'), ref_junc_variation(r2, 'std'), ref_junc_variation(r3, 'std')
    rtn = ref_junc_variation(rtn_data, 'mean')

    # ctrl_data = get_region_areas_per_group('Control', 31, 'mch')
    # write_pickle('Control', 'mch', ctrl_data)

    # ctrl_data = pkl.load(open('control_ref_junc_egfp.pkl', 'rb'))
    # ct1, ct2, ct3 = ctrl_data[:10], ctrl_data[10:20], ctrl_data[20:]
    # ct1, ct2, ct3 = ref_junc_variation(ct1, 'mean'), ref_junc_variation(ct2, 'std'), ref_junc_variation(ct3, 'std')
    # ctrl = ref_junc_variation(ctrl_data, 'mean')

    # print(ct1)
    # exit()
    df = pd.DataFrame()

    df['data_tubule_mean'] = pd.Series(np.concatenate((atl, climp, rtn)))

    # df['data_tubule_mean'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3, ct1, ct2, ct3)))
    # df['data_tubule_mean'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3)))

    # df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1),
    #                                             ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2),
    #                                             ['R3'] * len(r3), ['R1'] * len(ct1), ['R2'] * len(ct2),
    #                                             ['R3'] * len(ct3))))

    # df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1),
    #                                                                                         ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2),
    #                                                                                         ['R3'] * len(r3))))

    df['Group'] = pd.Series(np.concatenate((['ATL']*len(atl), ['Climp']*len(climp), ['RTN']*len(rtn))))

    # df['Group'] = pd.Series(np.concatenate((
    #     ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1), ['Climp'] * len(c2),
    #     ['Climp'] * len(c3), ['RTN'] * len(r1), ['RTN'] * len(r2), ['RTN'] * len(r3), ['Control'] * len(ct1), ['Control'] * len(ct2), ['Control'] * len(ct3))))
    #

    # df['Group'] = pd.Series(np.concatenate((
    #     ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1), ['Climp'] * len(c2),
    #     ['Climp'] * len(c3), ['RTN'] * len(r1), ['RTN'] * len(r2), ['RTN'] * len(r3))))

    ax = sns.boxenplot(data=df, x='Group', y='data_tubule_mean')
    # ax = sns.boxenplot(data=df, x='Replicate', y='data_tubule_mean', hue='Group', dodge=True)  # , yscale='log')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

    box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('Climp', 'RTN')]

    # egfp
    # box_pairs = [(('R1', 'ATL'), ('R1', 'Climp')), (('R1', 'ATL'), ('R1', 'RTN')), (('R1', 'Climp'), ('R1', 'RTN')), (('R1', 'ATL'), ('R1', 'Control')), (('R1', 'Climp'), ('R1', 'Control')), (('R1', 'RTN'), ('R1', 'Control')), (('R2', 'ATL'), ('R2', 'Climp')), (('R2', 'ATL'), ('R2', 'RTN')), (('R2', 'Climp'), ('R2', 'RTN')), (('R2', 'ATL'), ('R2', 'Control')), (('R2', 'Climp'), ('R2', 'Control')), (('R2', 'RTN'), ('R2', 'Control')), (('R3', 'ATL'), ('R3', 'Climp')), (('R3', 'ATL'), ('R3', 'RTN')), (('R3', 'Climp'), ('R3', 'RTN')), (('R3', 'ATL'), ('R3', 'Control')), (('R3', 'Climp'), ('R3', 'Control')), (('R3', 'RTN'), ('R3', 'Control'))]

    # box_pairs = [(('R1', 'ATL'), ('R1', 'Climp')), (('R1', 'ATL'), ('R1', 'RTN')), (('R1', 'Climp'), ('R1', 'RTN')), (('R2', 'ATL'), ('R2', 'Climp')), (('R2', 'ATL'), ('R2', 'RTN')), (('R2', 'Climp'), ('R2', 'RTN')), (('R3', 'ATL'), ('R3', 'Climp')), (('R3', 'ATL'), ('R3', 'RTN')), (('R3', 'Climp'), ('R3', 'RTN'))]

    # statannot.add_stat_annotation(ax, x='Replicate', y='data_tubule_mean', hue='Group', data=df, box_pairs=box_pairs,
    #                               test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    statannot.add_stat_annotation(ax, x='Group', y='data_tubule_mean', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    # plt.title('Cross-correlation between ERmoxGFP and mCherry over sequence for tubule intensity mean in all tubules',
    #           fontsize=18)
    # plt.title('CC area for Fuzzy CCs across conditions', fontsize=20)
    plt.title('Reference junction mean over sequence for isolated CCs in mCherry channel', fontsize=20)
    plt.grid(True)
    # plt.xlabel('Replicate', fontsize=20)
    plt.xlabel('Group', fontsize=20)
    plt.ylabel('Mean over sequence per reference junction', fontsize=18)

    plt.show()

# atl_e = ref_junc_data_per_group('ATL', 10, 'egfp')
# atl_mean = ref_junc_variation(atl_e, 'mean')
# climp_e = ref_junc_data_per_group('Climp', 10, 'egfp')
# climp_mean = ref_junc_variation(climp_e, 'mean')
# rtn_e = ref_junc_data_per_group('RTN', 10, 'egfp')
# rtn_mean = ref_junc_variation(rtn_e, 'mean')
# ctrl_e = ref_junc_data_per_group('Control', 10, 'egfp')
# ctrl_mean = ref_junc_variation(ctrl_e, 'mean')
#
# # print(atl_mean)
# sns.distplot(atl_mean, hist=False, label='atl')
# sns.distplot(climp_mean, hist=False, label='climp')
# sns.distplot(rtn_mean, hist=False, label='rtn')
# sns.distplot(ctrl_mean, hist=False, label='ctrl')
# plt.legend()
# plt.show()
#
# exit()


plot_ref_junc_variation()


exit()



def get_region_cc(group, series_num, region):
    ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions(group, series_num)

    # dict with ids as key and (x, y) as value
    label_ids, unassigned_cc_dict = separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)

    # iso, fuz, unk: list of lists with x, y
    iso, fuz, unk = get_junction_areas(label_ids, unassigned_cc_dict)

    if region == 'iso':
        return get_cc_ids(labelled_img, iso), labelled_img
    else:
        return get_cc_ids(labelled_img, fuz), labelled_img

    # dict to store the coords for each cc id
    # iso_cc_coords = {}
    # for each in iso_cc:
    #     iso_cc_coords[each] = np.where(labelled_img==each)


##################################################################

def calc_egfp_deposit(group, channel, num_series, region):
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


def junction_cc_mean_boxplot(channel, region):
    region_name = 'isolated' if region == 'iso' else 'fuzzy'
    atl = calc_egfp_deposit('ATL', channel, 26, region)

    climp = calc_egfp_deposit('Climp', channel, 31, region)
    ctrl = calc_egfp_deposit('Control', channel, 31, region)
    rtn = calc_egfp_deposit('RTN', channel, 29, region)

    df = pd.DataFrame()
    df['Values'] = pd.Series(np.concatenate((atl, climp, ctrl, rtn)))
    df['ids'] = pd.Series(np.concatenate((np.arange(1, len(atl) + 1), np.arange(1, len(climp) + 1),
                                          np.arange(1, len(ctrl) + 1), np.arange(1, len(rtn) + 1))))
    df['Group'] = pd.Series(
        np.concatenate((['ATL'] * len(atl), ['Climp'] * len(climp), ['Control'] * len(ctrl), ['RTN'] * len(rtn))))
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


def cc_area_measure(group, region, rstart, rend):
    cc_area_list = []

    for series_num in range(rstart, rend + 1):
        region_cc, labelled_img = get_region_cc(group, series_num, region)

        regions = regionprops(labelled_img)
        l = [regions[each - 1]['Area'] for each in region_cc]
        cc_area_list.extend(l)

    return cc_area_list


def stat_analysis(cc_area_atl, cc_area_climp, cc_area_rtn, cc_area_ctrl):
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


def plot_cc_area(a1, a2, a3, c1, c2, c3, r1, r2, r3, ct1, ct2, ct3, region):
    df = pd.DataFrame()
    # df['CC_area'] = pd.Series(np.concatenate((cc_area_atl, cc_area_climp, cc_area_rtn, cc_area_ctrl)))
    # df['Group'] = pd.Series(np.concatenate((['ATL'] * len(cc_area_atl), ['Climp'] * len(cc_area_climp),
    #                                         ['RTN'] * len(cc_area_rtn), ['Control'] * len(cc_area_ctrl))))

    df['CC_area'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3, ct1, ct2, ct3)))
    df['Group'] = pd.Series(np.concatenate((
        ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1),
        ['Climp'] * len(c2), ['Climp'] * len(c3), ['RTN'] * len(r1),
        ['RTN'] * len(r2), ['RTN'] * len(r3), ['Control'] * len(r3),
        ['Control'] * len(r3), ['Control'] * len(r3))))

    df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1),
                                                ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2),
                                                ['R3'] * len(r3), ['R1'] * len(ct1), ['R2'] * len(ct2),
                                                ['R3'] * len(ct3))))

    # ax = sns.swarmplot(data=df, x='Group', y='CC_area', hue='Replicate', dodge=True)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax = sns.boxenplot(data=df, x='Replicate', y='CC_area', hue='Group', dodge=True)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    # sns.boxplot(data=df, x='Group', y='CC_area', hue='replicate', color='white', dodge=True)

    plt.yscale('log')

    channel = 'egfp'

    replicates = ['R1', 'R2', 'R3']
    groups = ['ATL', 'Climp', 'RTN', 'Control']

    if channel == 'egfp':
        box_pairs = [(tuple([x, y]), tuple([x, z])) for i, x in enumerate(replicates) for j, y in enumerate(groups) for z in groups[j+1:]]
    else:
        groups = ['ATL', 'Climp', 'RTN']
        box_pairs = [(tuple([x, y]), tuple([x, z])) for i, x in enumerate(replicates) for j, y in enumerate(groups) for z in groups[j+1:]]

    statannot.add_stat_annotation(ax, x='Replicate', y='CC_area', hue='Group', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    region_name = 'Isolated' if region == 'iso' else 'Fuzzy'

    plt.suptitle(f'{region_name} CC area across conditions', fontsize=20)
    plt.title('CC area denotes the total movement of each junction', fontsize=18)
    plt.grid(True)
    plt.xlabel('Replicate', fontsize=18)
    plt.ylabel('CC_area (movement of junctions), log scale', fontsize=18)
    plt.show()


def plot_cc_area_all(a1, c1, r1, ct1, region):
    df = pd.DataFrame()
    # df['CC_area'] = pd.Series(np.concatenate((cc_area_atl, cc_area_climp, cc_area_rtn, cc_area_ctrl)))
    # df['Group'] = pd.Series(np.concatenate((['ATL'] * len(cc_area_atl), ['Climp'] * len(cc_area_climp),
    #                                         ['RTN'] * len(cc_area_rtn), ['Control'] * len(cc_area_ctrl))))

    df['CC_area'] = pd.Series(np.concatenate((a1, c1, r1, ct1)))
    df['Group'] = pd.Series(np.concatenate((
        ['ATL'] * len(a1), ['Climp'] * len(c1), ['RTN'] * len(r1), ['Control'] * len(ct1))))

    # df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1),
    #                                             ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2),
    #                                             ['R3'] * len(r3), ['R1'] * len(ct1), ['R2'] * len(ct2),
    #                                             ['R3'] * len(ct3))))

    # ax = sns.swarmplot(data=df, x='Group', y='CC_area', hue='Replicate', dodge=True)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax = sns.boxenplot(data=df, x='Group', y='CC_area')
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    # sns.boxplot(data=df, x='Group', y='CC_area', hue='replicate', color='white', dodge=True)

    plt.yscale('log')

    box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('ATL', 'Control'), ('Climp', 'RTN'), ('Climp', 'Control'),
                 ('RTN', 'Control')]

    statannot.add_stat_annotation(ax, x='Group', y='CC_area', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    region_name = 'Isolated' if region == 'iso' else 'Fuzzy'

    plt.suptitle(f'{region_name} CC area across conditions', fontsize=20)
    plt.title('CC area denotes the total movement of each junction', fontsize=18)
    plt.grid(True)
    plt.xlabel('Group', fontsize=18)
    plt.ylabel('CC_area (movement of junctions), log scale', fontsize=18)
    plt.show()


# a1 = pd.Series(cc_area_measure('ATL', 'fuz', 1, 26))
# c1 = pd.Series(cc_area_measure('Climp', 'fuz', 1, 31))
# r1 = pd.Series(cc_area_measure('RTN', 'fuz', 1, 29))
# ct1 = pd.Series(cc_area_measure('Control', 'fuz', 1, 31))
#
# plot_cc_area_all(a1, c1, r1, ct1, 'fuz')
# exit()


def get_data_cc_area(region):
    # at1 = (pd.Series(cc_area_measure('ATL', region, 1, 10)))

    # sns.displot(at3)
    # plt.xlabel('CC Area', fontsize=15)
    # plt.title('ATL-Replicate3 fuzzy CC distribution', fontsize=20)
    # plt.show()

    at1 = (pd.Series(cc_area_measure('ATL', region, 1, 10)))
    cl1 = (pd.Series(cc_area_measure('Climp', region, 1, 10)))
    rt1 = (pd.Series(cc_area_measure('RTN', region, 1, 10)))
    ctrl1 = (pd.Series(cc_area_measure('Control', region, 1, 10)))
    at2 = (pd.Series(cc_area_measure('ATL', region, 11, 20)))
    cl2 = (pd.Series(cc_area_measure('Climp', region, 11, 20)))
    rt2 = (pd.Series(cc_area_measure('RTN', region, 11, 20)))
    ctrl2 = (pd.Series(cc_area_measure('Control', region, 11, 20)))
    at3 = (pd.Series(cc_area_measure('ATL', region, 21, 26)))
    cl3 = (pd.Series(cc_area_measure('Climp', region, 21, 31)))
    rt3 = (pd.Series(cc_area_measure('RTN', region, 21, 29)))
    ctrl3 = (pd.Series(cc_area_measure('Control', region, 21, 31)))

    plot_cc_area(at1, at2, at3, cl1, cl2, cl3, rt1, rt2, rt3, ctrl1, ctrl2, ctrl3, region)


# get_data_cc_area('fuz')
# exit()


def calc_deposit(num_series, group, region):
    for num in range(num_series, num_series + 1):

        region_cc, labelled_img = get_region_cc(group, num, region)

        region_cc_coords = {each: np.where(labelled_img == each) for each in region_cc}

        ln_egfp = []
        ln_mch = []

        for i in range(30):
            if group == 'Control':
                path_EGFP = f'{confocal_data_path}/Control/files/img_{num}_decon_t0{i:02d}.tif'
                path_mch = None
            else:
                path_EGFP = f'{confocal_data_path}/{group}/files/{group[0]}{num}_decon_t0{i:02d}_ch00.tif'
                path_mch = f'{confocal_data_path}/{group}/files/{group[0]}{num}_decon_t0{i:02d}_ch01.tif'

            img_egfp = get_std_img(path_EGFP)
            frame_data_egfp = [np.mean(img_egfp[v]) for v in region_cc_coords.values()]

            ln_egfp.append(frame_data_egfp)

            if path_mch is not None:
                img_mch = get_std_img(path_mch)
                frame_data_mch = [np.mean(img_mch[v]) for v in region_cc_coords.values()]
                # j57_data.append([img_mch[v] for v in region_cc_coords.values()])
                ln_mch.append(frame_data_mch)

        ln_egfp = np.array(ln_egfp)

        ln_mch = np.array(ln_mch)
        # print(j57_data[:5])
        # exit()

    return ln_egfp.T, ln_mch.T, region_cc_coords


# ln_egfp, ln_mch, region_cc_coords = calc_deposit(1, 'ATL', 'iso')
# exit()


def calc_deposit_net_norm(num_series, group, region):
    global ln_egfp, region_cc_coords, op_egfp, op_mch
    global ln_mch

    for num in range(num_series, num_series + 1):
        region_cc, labelled_img = get_region_cc(group, num, region)

        region_cc_coords = {each: np.where(labelled_img == each) for each in region_cc}

        ln_egfp = []
        ln_mch = []

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

            img_egfp = imageio.imread(path_EGFP)
            skel = imageio.imread(path_skel)

            data_egfp = img_egfp[np.where(skel)]

            egfp_norm = (img_egfp - min(data_egfp)) / (max(data_egfp) - min(data_egfp))

            frame_data_egfp = [np.mean(egfp_norm[v]) for v in region_cc_coords.values()]

            ln_egfp.append(frame_data_egfp)

            if path_mch is not None:
                img_mch = imageio.imread(path_mch)

                data_mch = img_mch[np.where(skel)]

                # minval = min(data_mch)
                # maxval = max(data_mch)
                #
                # mnmx_mch.append((minval, maxval))
                mch_norm = (img_mch - min(data_mch)) / (max(data_mch) - min(data_mch))

                frame_data_mch = [np.mean(mch_norm[v]) for v in region_cc_coords.values()]

                ln_mch.append(frame_data_mch)

        ln_egfp = np.array(ln_egfp)

        ln_mch = np.array(ln_mch)

        # print(ln_egfp.shape)

    ######################################
    # op_egfp = []
    # op_mch = []
    #
    # for i, each in enumerate(ln_egfp):
    #     each = (each - mnmx_egfp[i][0]) / (mnmx_egfp[i][1] - mnmx_egfp[i][0])
    #     op_egfp.append(each)
    #
    # for i, each in enumerate(ln_mch):
    #     each = (each - mnmx_mch[i][0]) / (mnmx_mch[i][1] - mnmx_mch[i][0])
    #     op_mch.append(each)
    ######################################

    # for each in ln_egfp.T:
    #     each = (each - each.min())/(each.max() - each.min())
    #     op_egfp.append(each)

    # for each in ln_mch.T:
    #     each = (each - each.min())/(each.max() - each.min())
    #     op_mch.append(each)

    ######################################
    # op_egfp = np.array(op_egfp).T
    # op_mch = np.array(op_mch).T
    ######################################

    return ln_egfp.T, ln_mch.T, region_cc_coords


def calc_deposit_cc_norm(num_series, group, region):
    # ch = 1 if channel=='mCherry' else 0
    # global ln
    global ln_egfp, region_cc_coords, op_egfp, op_mch
    global ln_mch
    # sl = []
    for num in range(num_series, num_series + 1):
        region_cc, labelled_img = get_region_cc(group, num, region)

        region_cc_coords = {each: np.where(labelled_img == each) for each in region_cc}

        ln_egfp = []
        ln_mch = []

        for i in range(100):
            if group == 'Control':
                path_EGFP = f'{confocal_data_path}/Control/files/img_{num}_decon_t0{i:02d}.tif'
                path_mch = None
            else:
                path_EGFP = f'{confocal_data_path}/{group}/files/{group[0]}{num}_decon_t0{i:02d}_ch00.tif'
                path_mch = f'{confocal_data_path}/{group}/files/{group[0]}{num}_decon_t0{i:02d}_ch01.tif'

            img_egfp = imageio.imread(path_EGFP)

            # img_egfp = get_std_img(path_EGFP)
            frame_data_egfp = [np.mean(img_egfp[v]) for v in region_cc_coords.values()]

            ln_egfp.append(frame_data_egfp)
            # print(region_cc_coords.values())
            # exit()
            # frame_data = []

            # frame_data_egfp = [np.mean(img_egfp[v]) for v in region_cc_coords.values()]
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

            # ln_egfp.append(frame_data_egfp)
            # ln_egfp.append(frame_data)

            if path_mch is not None:
                # img_mch = get_std_img(path_mch)
                img_mch = imageio.imread(path_mch)

                # frame_data = []

                frame_data_mch = [np.mean(img_mch[v]) for v in region_cc_coords.values()]
                # j57_data.append([img_mch[v] for v in region_cc_coords.values()])

                # for v in region_cc_coords.values():
                #     std_junc = (img_mch[v] - min(img_mch[v])) / (max(img_mch[v]) - min(img_mch[v]))
                #     frame_data.append(np.mean(std_junc))

                ln_mch.append(frame_data_mch)

        ln_egfp = np.array(ln_egfp)

        ln_mch = np.array(ln_mch)

        op_egfp = []
        op_mch = []

        for each in ln_egfp.T:
            each = (each - each.min()) / (each.max() - each.min())
            op_egfp.append(each)

        for each in ln_mch.T:
            each = (each - each.min()) / (each.max() - each.min())
            op_mch.append(each)

        op_egfp = np.array(op_egfp)
        op_mch = np.array(op_mch)
        # print(op_egfp.shape)
        # print(ln_egfp.T.shape)
        # exit()

    # return ln_egfp.T, ln_mch.T, region_cc_coords
    return op_egfp, op_mch, region_cc_coords


def create_tubule_junc_plot(er_path, skeleton_path):
    er_img = imageio.imread(er_path)
    plt.imshow(er_img, cmap='gray')

    graph = node_connector(er_path, skeleton_path)

    exclude_edges = [(node1, node2) for node1, node2 in graph.edges() if
                     graph.degree(node1) == 1 or graph.degree(node2) == 1]

    for (start_node, end_node) in graph.edges():
        if graph[start_node][end_node][0]:
            ps = graph[start_node][end_node][0]['pts']
            if (start_node, end_node) not in exclude_edges:
                plt.plot(ps[:, 1], ps[:, 0], 'red')
            else:
                plt.plot(ps[:, 1], ps[:, 0], 'green')

    deg_one_nodes, deg_two_nodes, high_deg_nodes = get_updated_degree_nodes(graph)

    if len(deg_one_nodes) != 0:
        plt.plot(deg_one_nodes[:, 1], deg_one_nodes[:, 0], 'o', markerfacecolor='yellow', markeredgecolor='yellow',
                 mew=0.5, markersize=3)

    if len(high_deg_nodes) != 0:
        plt.plot(high_deg_nodes[:, 1], high_deg_nodes[:, 0], 'o', markerfacecolor='blue', markeredgecolor='blue',
                 mew=0.5, markersize=3)

    plt.show()


def get_intersection(a, b):
    # get common elements between 2 ndarrays (list of nodes)
    return np.array([x for x in a if np.any(np.all(x == b, axis=1))])


def get_edges(conn_graph, iso_ids, fuz_ids, connection):
    # Find the edges between iso-iso, iso-fuz, fuz-fuz
    if connection == 'iso-iso':
        if len(iso_ids) > 0:
            return [(u, v) for (u, v) in conn_graph.edges() if (u in iso_ids and v in iso_ids)]

    elif connection == 'iso-fuz':
        if len(iso_ids) > 0 and len(fuz_ids) > 0:
            return [(u, v) for (u, v) in conn_graph.edges() if
                    ((u in iso_ids and v in fuz_ids) or (u in fuz_ids and v in iso_ids))]

    elif connection == 'fuz-fuz':
        if len(fuz_ids) > 0:
            return [(u, v) for (u, v) in conn_graph.edges() if (u in fuz_ids and v in fuz_ids)]

    else:
        return list(conn_graph.edges())


def get_tubule_data(group, series_num, connection):
    group_pref = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}

    # sourcery skip: inline-immediately-returned-variable
    er_input_path = f'{confocal_data_path}/{group}/new_op_jul/er_mean/{group.lower()}{series_num}_er_mean.png'
    skel_path = f'{confocal_data_path}/{group}/new_op_jul/er_mean_proc/{group.lower()}{series_num}_er_mean_proc_enhance_skel.png'

    conn_graph = node_connector(er_input_path, skel_path)

    deg_one_nodes, deg_two_nodes, high_deg_nodes = get_updated_degree_nodes(conn_graph)

    # graph = junc_analysis.skel_to_graph(mean_img)
    ref_junctions = junc_analysis.get_junctions(conn_graph)

    ref_junctions = [[each[0], each[1]] for each in ref_junctions]

    per_frame_junctions = []
    for frame in range(100):
        er_path = f'{confocal_data_path}/{group}/new_op_jul/std_egfp/{group_pref[group]}{series_num}_decon_t0{frame:02d}_ch00_std.png'
        skeleton_path = f'{confocal_data_path}/{group}/new_op_jul/skel/{group_pref[group]}{series_num}/{group_pref[group]}{series_num}_decon_t0{frame:02d}_ch00_skel.png'

        graph = node_connector(er_path, skeleton_path)

        # create_tubule_junc_plot(er_path, skeleton_path)

        junctions = junc_analysis.get_junctions(graph)

        junc_array = [[junc[0], junc[1]] for junc in junctions]
        per_frame_junctions.extend(junc_array)

    ref_junctions = np.array(ref_junctions)
    per_frame_junctions = np.array(per_frame_junctions)

    spread_img = np.zeros((128, 128))
    for each in per_frame_junctions:
        spread_img[each[0], each[1]] = 255.

    labelled_img = label(spread_img, connectivity=2)

    # label_vals: dict with ids as key and (x, y) as value
    label_vals, unassigned_cc_dict = junc_analysis.separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)

    # iso, fuz, unk: list of lists with x, y
    iso, fuz, unk = junc_analysis.get_junction_areas(label_vals, unassigned_cc_dict)

    # Find the iso and fuz from high_deg_nodes
    intersection_iso = get_intersection(iso, high_deg_nodes)
    intersection_fuz = get_intersection(fuz, high_deg_nodes)

    # Find the iso/ fuz nodes in conn_graph

    if len(intersection_iso) > 0:
        iso_ids = [k for k in conn_graph.nodes if (
                conn_graph.nodes[k]['o'][0] in intersection_iso[:, 0] and conn_graph.nodes[k]['o'][
            1] in intersection_iso[:, 1])]
    else:
        iso_ids = []
    if len(intersection_fuz) > 0:
        fuz_ids = [k for k in conn_graph.nodes if (
                conn_graph.nodes[k]['o'][0] in intersection_fuz[:, 0] and conn_graph.nodes[k]['o'][
            1] in intersection_fuz[:, 1])]
    else:
        fuz_ids = []

    edges = get_edges(conn_graph, iso_ids, fuz_ids, connection)

    return edges, conn_graph


# edges, conn_graph = get_tubule_data('ATL', 1, 'iso-iso')
# edge_pts = [conn_graph[u][v][0]['pts'] for (u, v) in edges]
#
# def get_er_input(num, group, channel, series_num, ch_id):
#     er_input_path = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/std_{channel}/A1_decon_t0{num:02d}_ch0{ch_id}_std.png'
#     er = imageio.imread(er_input_path)
#     # er = (er - er.min()) / (er.max() - er.min())
#     return er / 255
#
# egfp = get_er_input(0, 'ATL', 'egfp', 1, 0)
# mch = get_er_input(0, 'ATL', 'mch', 1, 1)
#
# # egfp = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/std_egfp/A1_decon_t000_ch00_std.png') / 255
# # mch = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/std_mch/A1_decon_t000_ch01_std.png') / 255
#
# a = edge_pts[0]
# print(egfp[a[:, 0], a[:, 1]])
# print(mch[a[:, 0], a[:, 1]])
# exit()


def tubule_sequence_data(group, series_num, connection, channel, measure):
    """
    Returns sequence data for tubule images from a given group, series, connection, and channel.

    Args:
    - group (str): Name of the group, must be one of ['ATL', 'Climp', 'RTN', 'Control'].
    - series_num (int): Series number.
    - connection (str): Type of connection, must be one of ['iso-iso', 'iso-fuz', 'fuz-fuz'].
    - channel (str): Channel name, must be one of ['egfp', 'mch'].

    Returns:
    - seq_data (list): A list of lists of pixel values for each edge in image sequence.
    """

    assert group in VALID_GROUPS, f"Invalid group name: {group}"
    assert connection in VALID_CONNECTIONS, f"Invalid connection type: {connection}"
    assert channel in VALID_CHANNELS, f"Invalid channel name: {channel}"
    assert measure in VALID_MEASURES, f"Invalid measure name: {measure}"

    group_pref = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}

    def get_er_input(num, group, channel, series_num, ch_id):
        er_input_path = f'{confocal_data_path}/{group}/new_op_jul/std_{channel}/{group_pref[group]}{series_num}_decon_t0{num:02d}_ch0{ch_id}_std.png'
        er = imageio.imread(er_input_path)
        # er = (er - er.min()) / (er.max() - er.min())
        return er / 255

    edges, conn_graph = get_tubule_data(group, series_num, connection)

    if not edges:
        return None

    # get coordinates for edges
    edge_pts = [conn_graph[u][v][0]['pts'] for (u, v) in edges]

    seq_data = []

    # ch_id = 0 if channel == 'egfp' else 1
    if channel == 'egfp':
        ch_id = 0
    else:
        ch_id = 1

    for each in edge_pts:
        edge_data = []
        for i in range(100):
            er = get_er_input(i, group, channel, series_num, ch_id)
            edge_data.append(er[each[:, 0], each[:, 1]])
            # if measure == 'tubules':
            #     edge_data.append(er[each[:, 0], each[:, 1]])
            # elif measure == 'tub-mean':
            #     edge_data.append(np.mean(er[each[:, 0], each[:, 1]]))
        seq_data.append(edge_data)

    return seq_data


def create_tub_data_pickles(group, total_series, connection, channel, measure):
    group_data = []
    for i in range(1, total_series + 1):
        seq_data = np.array(tubule_sequence_data(group, i, connection, channel, measure))
        group_data.append(seq_data)

    with open(f'{group.lower()}_{connection}_{measure}_{channel}.pkl', 'wb') as fl:
        pkl.dump(group_data, fl)


# groups = {'ATL': 26, 'Climp': 31, 'Control': 31, 'RTN': 29}
# connections = ['iso-iso', 'iso-fuz', 'fuz-fuz']
# channels = ['egfp', 'mch']
#
# create_tub_data_pickles('ATL', 26, 'iso-iso', 'egfp', 'tubules')
# create_tub_data_pickles('ATL', 26, 'iso-iso', 'mch', 'tubules')
# create_tub_data_pickles('ATL', 26, 'iso-fuz', 'egfp', 'tubules')
# create_tub_data_pickles('ATL', 26, 'iso-fuz', 'mch', 'tubules')
# create_tub_data_pickles('ATL', 26, 'fuz-fuz', 'egfp', 'tubules')
# create_tub_data_pickles('ATL', 26, 'fuz-fuz', 'mch', 'tubules')
#
# create_tub_data_pickles('Climp', 31, 'iso-iso', 'egfp', 'tubules')
# create_tub_data_pickles('Climp', 31, 'iso-iso', 'mch', 'tubules')
# create_tub_data_pickles('Climp', 31, 'iso-fuz', 'egfp', 'tubules')
# create_tub_data_pickles('Climp', 31, 'iso-fuz', 'mch', 'tubules')
# create_tub_data_pickles('Climp', 31, 'fuz-fuz', 'egfp', 'tubules')
# create_tub_data_pickles('Climp', 31, 'fuz-fuz', 'mch', 'tubules')
#
# create_tub_data_pickles('RTN', 29, 'iso-iso', 'egfp', 'tubules')
# create_tub_data_pickles('RTN', 29, 'iso-iso', 'mch', 'tubules')
# create_tub_data_pickles('RTN', 29, 'iso-fuz', 'egfp', 'tubules')
# create_tub_data_pickles('RTN', 29, 'iso-fuz', 'mch', 'tubules')
# create_tub_data_pickles('RTN', 29, 'fuz-fuz', 'egfp', 'tubules')
# create_tub_data_pickles('RTN', 29, 'fuz-fuz', 'mch', 'tubules')
#
# create_tub_data_pickles('Control', 31, 'iso-iso', 'egfp', 'tubules')
# create_tub_data_pickles('Control', 31, 'iso-fuz', 'egfp', 'tubules')
# create_tub_data_pickles('Control', 31, 'fuz-fuz', 'egfp', 'tubules')
#
# exit()


def create_pickles(groups: dict, connections: list, channels: list, measure: list) -> None:
    """
    Creates pickles of tubule data for the given groups, connections, channels, and measure.

    Args:
        groups (dict): A dictionary of group names and the number of series in each group.
        connections (list): A list of connection types.
        channels (list): A list of channel names.
        measure (str): The name of the measure to be pickled.

    Returns:
        None
    """

    for group, num_series in groups.items():
        for connection in connections:
            if group != 'Control':
                for channel in channels:
                    create_tub_data_pickles(group, num_series, connection, channel, measure)
            else:
                create_tub_data_pickles(group, num_series, connection, 'egfp', measure)





def pickle_creation_runner():
    groups = {'ATL': 26, 'Climp': 31, 'Control': 31, 'RTN': 29}
    # connections = ['iso-iso', 'iso-fuz', 'fuz-fuz']
    # channels = ['egfp', 'mch']
    # measure = ['tubule']
    # measure = VALID_MEASURES
    create_pickles(groups, VALID_CONNECTIONS, VALID_CHANNELS, VALID_MEASURES)


def get_group_data(group):
    a1e = get_pickle_data(group, 'iso-iso', 'egfp')
    a2e = get_pickle_data(group, 'iso-fuz', 'egfp')
    a3e = get_pickle_data(group, 'fuz-fuz', 'egfp')
    a1m = get_pickle_data(group, 'iso-iso', 'mch')
    a2m = get_pickle_data(group, 'iso-fuz', 'mch')
    a3m = get_pickle_data(group, 'fuz-fuz', 'mch')

    eg = a1e + a2e + a3e
    mc = a1m + a2m + a3m

    egr1 = eg[:10]
    mcr1 = mc[:10]
    egr2 = eg[10:20]
    mcr2 = mc[10:20]
    egr3 = eg[20:]
    mcr3 = mc[20:]

    corr_data_r1 = []
    corr_data_r2 = []
    corr_data_r3 = []

    for tub_eg, tub_mch in zip(egr1, mcr1):
        corr_data_r1.extend(np.corrcoef(i, j)[0][1] for i, j in zip(tub_eg, tub_mch))

    for tub_eg, tub_mch in zip(egr2, mcr2):
        corr_data_r2.extend(np.corrcoef(i, j)[0][1] for i, j in zip(tub_eg, tub_mch))

    for tub_eg, tub_mch in zip(egr3, mcr3):
        corr_data_r3.extend(np.corrcoef(i, j)[0][1] for i, j in zip(tub_eg, tub_mch))

    return corr_data_r1, corr_data_r2, corr_data_r3


a1, a2, a3 = get_group_data('ATL')
c1, c2, c3 = get_group_data('Climp')
r1, r2, r3 = get_group_data('RTN')
# control = get_group_data('Control')


df = pd.DataFrame()

# df['data_tubule_mean'] = pd.Series(np.concatenate((atl, climp, rtn)))
df['data_tubule_mean'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3)))

df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1),
                                            ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2),
                                            ['R3'] * len(r3))))

df['Group'] = pd.Series(np.concatenate((
    ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1), ['Climp'] * len(c2),
    ['Climp'] * len(c3), ['RTN'] * len(r1), ['RTN'] * len(r2), ['RTN'] * len(r3))))

# ax = sns.boxenplot(data=df, x='Group', y='data_tubule_mean')
ax = sns.boxenplot(data=df, x='Replicate', y='data_tubule_mean', hue='Group', dodge=True)  # , yscale='log')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

# egfp
channel = 'egfp'
replicates = ['R1', 'R2', 'R3']

if channel == 'egfp':
    groups = ['ATL', 'Climp', 'RTN', 'Control']
else:
    groups = ['ATL', 'Climp', 'RTN']

box_pairs = [((x, y), (x, z)) for i, x in enumerate(replicates) for j, y in enumerate(groups) for z in groups[j + 1 :]]

# box_pairs = [(('R1', 'ATL'), ('R1', 'Climp')), (('R1', 'ATL'), ('R1', 'RTN')), (('R1', 'Climp'), ('R1', 'RTN')),
#              (('R2', 'ATL'), ('R2', 'Climp')), (('R2', 'ATL'), ('R2', 'RTN')), (('R2', 'Climp'), ('R2', 'RTN')),
#              (('R3', 'ATL'), ('R3', 'Climp')), (('R3', 'ATL'), ('R3', 'RTN')), (('R3', 'Climp'), ('R3', 'RTN'))]

statannot.add_stat_annotation(ax, x='Replicate', y='data_tubule_mean', hue='Group', data=df, box_pairs=box_pairs,
                              test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

# statannot.add_stat_annotation(ax, x='Group', y='data_tubule_mean', data=df, box_pairs=box_pairs,
#                               test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

plt.title('Cross-correlation between ERmoxGFP and mCherry over sequence for tubule intensity mean in all tubules',
          fontsize=18)
plt.grid(True)
plt.xlabel('Replicate', fontsize=20)
plt.ylabel('Cross-correlation value', fontsize=18)

plt.show()

exit()


def plot_seq_mean_tubule_mean(group, channel, connection):
    filename = f'{group.lower()}_{connection}_{channel}.pkl'
    with open(filename, 'rb') as f:
        data = pkl.load(f)

    d1 = []

    for each in data:
        try:
            d1.extend(np.std(i) for i in each if len(i) > 0)
        except:
            pass
    return d1


# a1, a2, a3 = plot_seq_mean_tubule_mean('ATL', 'mch', 'iso-iso')
# c1, c2, c3 = plot_seq_mean_tubule_mean('Climp', 'mch', 'iso-iso')
# # ct1, ct2, ct3 = plot_seq_mean_tubule_mean('Control', 'mch', 'fuz-fuz')
# r1, r2, r3 = plot_seq_mean_tubule_mean('RTN', 'mch', 'iso-iso')


a1 = plot_seq_mean_tubule_mean('ATL', 'egfp', 'fuz-fuz')
c1 = plot_seq_mean_tubule_mean('Climp', 'egfp', 'fuz-fuz')
# ct1, ct2, ct3 = plot_seq_mean_tubule_mean('Control', 'mch', 'fuz-fuz')
ct1 = plot_seq_mean_tubule_mean('Control', 'egfp', 'fuz-fuz')
r1 = plot_seq_mean_tubule_mean('RTN', 'egfp', 'fuz-fuz')

# a1, a2, a3 = get_channel_corr('ATL', 'fuz-fuz')
# c1, c2, c3 = get_channel_corr('Climp', 'fuz-fuz')
# r1, r2, r3 = get_channel_corr('RTN', 'fuz-fuz')

### for complete group
# a1 = get_channel_corr('ATL', 'iso-iso')
# c1 = get_channel_corr('Climp', 'iso-iso')
# r1 = get_channel_corr('RTN', 'iso-iso')


df = pd.DataFrame()
# df['data_tubule_mean'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, ct1, ct2, ct3, r1, r2, r3)))
# df['data_tubule_mean'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3)))

df['data_tubule_mean'] = pd.Series(np.concatenate((a1, c1, ct1, r1)))

# df['Group'] = pd.Series(np.concatenate((
#     ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1), ['Climp'] * len(c2), ['Climp'] * len(c3), ['Control'] * len(ct1), ['Control'] * len(ct2), ['Control'] * len(ct3), ['RTN'] * len(r1), ['RTN'] * len(r2), ['RTN'] * len(r3))))


df['Group'] = pd.Series(np.concatenate((
    ['ATL'] * len(a1), ['Climp'] * len(c1), ['Control'] * len(ct1), ['RTN'] * len(r1))))

# df['Group'] = pd.Series(np.concatenate((
#     ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1), ['Climp'] * len(c2), ['Climp'] * len(c3), ['RTN'] * len(r1), ['RTN'] * len(r2), ['RTN'] * len(r3))))
#
# df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1), ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(ct1), ['R2'] * len(ct2), ['R3'] * len(ct3), ['R1'] * len(r1), ['R2'] * len(r2), ['R3'] * len(r3))))

# df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1), ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2), ['R3'] * len(r3))))

ax = sns.boxenplot(data=df, x='Group', y='data_tubule_mean')
# ax = sns.boxenplot(data=df, x='Replicate', y='data_tubule_mean', hue='Group', dodge=True)  # , yscale='log')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

# box_pairs = [(('R1', 'ATL'), ('R1', 'Climp')), (('R1', 'ATL'), ('R1', 'Control')), (('R1', 'ATL'), ('R1', 'RTN')), (('R1', 'Climp'), ('R1', 'RTN')), (('R1', 'Climp'), ('R1', 'Control')), (('R1', 'Control'), ('R1', 'RTN')), (('R2', 'ATL'), ('R2', 'Climp')), (('R2', 'ATL'), ('R2', 'Control')), (('R2', 'ATL'), ('R2', 'RTN')), (('R2', 'Climp'), ('R2', 'RTN')), (('R2', 'Climp'), ('R2', 'Control')), (('R2', 'Control'), ('R2', 'RTN')), (('R3', 'ATL'), ('R3', 'Climp')), (('R3', 'ATL'), ('R3', 'Control')), (('R3', 'ATL'), ('R3', 'RTN')), (('R3', 'Climp'), ('R3', 'RTN')), (('R3', 'Climp'), ('R3', 'Control')), (('R3', 'Control'), ('R3', 'RTN'))]

# box_pairs = [(('R1', 'ATL'), ('R1', 'Climp')), (('R1', 'ATL'), ('R1', 'RTN')), (('R1', 'Climp'), ('R1', 'RTN')), (('R2', 'ATL'), ('R2', 'Climp')), (('R2', 'ATL'), ('R2', 'RTN')), (('R2', 'Climp'), ('R2', 'RTN')), (('R3', 'ATL'), ('R3', 'Climp')), (('R3', 'ATL'), ('R3', 'RTN')), (('R3', 'Climp'), ('R3', 'RTN'))]

# mcherry
# box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('Climp', 'RTN')]

# egfp
groups = ['ATL', 'Climp', 'RTN', 'Control']
box_pairs = [(x, y) for i, x in enumerate(groups) for j, y in enumerate(groups) if i < j and y not in groups[:i] + groups[i+1:j]]

# box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('ATL', 'Control'), ('Climp', 'Control'), ('Climp', 'RTN'),
#              ('Control', 'RTN')]

# statannot.add_stat_annotation(ax, x='Replicate', y='data_tubule_mean', hue='Group', data=df, box_pairs=box_pairs,
#                               test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

statannot.add_stat_annotation(ax, x='Group', y='data_tubule_mean', data=df, box_pairs=box_pairs,
                              test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

# plt.suptitle('Isolated CC area across conditions', fontsize=20)
# plt.title('Standard Deviation per sequence for junction CC mean intensity (isolated junctions)', fontsize=18)
# measure_name = 'Standard deviation' if measure == 'std' else 'Mean'
# region_name = 'isolated' if region == 'iso' else 'fuzzy'
plt.title('Standard deviation of sequence for tubule intensity mean (ERmoxGFP) in fuz-fuz edges', fontsize=18)

# plt.title('Cross-correlation between ERmoxGFP and mCherry over sequence for tubule intensity mean in iso-iso edges', fontsize=18)
plt.grid(True)
plt.xlabel('Group', fontsize=20)
plt.ylabel('Standard deviation value', fontsize=18)

plt.show()
# figm = plt.get_current_fig_manager()
# figm.full_screen_toggle()
# fig.savefig('Seq_std_tub_mean_fuz_fuz.png', bbox_inches='tight', pad_inches=0.2, dpi=300)
# plt.close()
# plt.show()
# fig = plt.gcf()
# fig.savefig('dfdfd')

exit()

mean_vals_atl = []
mean_vals_climp = []
mean_vals_rtn = []
mean_vals_ctrl = []
for i in range(1, 11):
    seq_data_atl = np.array(tubule_sequence_analysis('ATL', i, 'iso-iso'))
    seq_data_climp = np.array(tubule_sequence_analysis('Climp', i, 'iso-iso'))
    seq_data_rtn = np.array(tubule_sequence_analysis('RTN', i, 'iso-iso'))
    seq_data_ctrl = np.array(tubule_sequence_analysis('Control', i, 'iso-iso'))

    mean_vals_atl.extend(np.mean(each) for each in seq_data_atl)
    mean_vals_climp.extend(np.mean(each) for each in seq_data_climp)
    mean_vals_rtn.extend(np.mean(each) for each in seq_data_rtn)
    mean_vals_ctrl.extend(np.mean(each) for each in seq_data_ctrl)

# mean_vals = []
# std_vals = []
# for each in seq_data_atl:
#     mean_vals.append(np.mean(each))
#     std_vals.append(np.std(each))

sns.distplot(mean_vals_atl, hist=False, label='ATL')
sns.distplot(mean_vals_climp, hist=False, label='Climp')
sns.distplot(mean_vals_rtn, hist=False, label='RTN')
sns.distplot(mean_vals_ctrl, hist=False, label='Control')
plt.legend()
plt.title('Mean over sequence for tubule intensity mean value')
# plt.hist(mean_vals)
plt.show()

exit()


def tubule_intensity_analysis(group, series_num, connection, measure):
    edges, conn_graph = get_tubule_data(group, series_num, connection)

    if edges:
        er_input_path = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/er_mean/{group.lower()}{series_num}_er_mean.png'
        #
        er = imageio.imread(er_input_path)

        er = (er - er.min()) / (er.max() - er.min())

        edge_pts = [conn_graph[u][v][0]['pts'] for (u, v) in edges]
        vals = []
        if measure == 'mean':
            vals.extend(np.mean(er[each[:, 0], each[:, 1]]) for each in edge_pts)
        else:
            vals.extend(np.std(er[each]) for each in edge_pts)

        return vals


# for i in range(1, 11):
#     tubule_intensity_analysis('Control', i, 'iso-iso')
# exit()


atl = []
climp = []
rtn = []
ctrl = []

for i in range(1, 11):
    vals = tubule_intensity_analysis('ATL', i, 'iso-iso', 'mean')
    atl.extend(vals)
    vals = tubule_intensity_analysis('Climp', i, 'iso-iso', 'mean')
    climp.extend(vals)
    vals = tubule_intensity_analysis('RTN', i, 'iso-iso', 'mean')
    rtn.extend(vals)
    vals = tubule_intensity_analysis('Control', i, 'iso-iso', 'mean')
    ctrl.extend(vals)

df = pd.DataFrame()

df['Tubule mean '] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3)))

sns.distplot(atl, hist=False, label='ATL')
sns.distplot(climp, hist=False, label='Climp')
sns.distplot(rtn, hist=False, label='RTN')
sns.distplot(ctrl, hist=False, label='Control')
plt.show()
exit()


def tubule_length_analysis(conn_graph, edges):
    # sourcery skip: inline-immediately-returned-variable
    # Find the length of each edge in iso-iso
    connection_length = [conn_graph[u][v][0]['weight'] for (u, v) in edges]

    return connection_length


def get_group_len_data(group, connection):
    l1 = []
    for i in range(1, 11):
        edges, conn_graph = get_tubule_data(group, i, connection)
        ln = tubule_length_analysis(conn_graph, edges)
        l1.extend(ln)
    return l1


atl1 = get_group_len_data('ATL', 'iso-iso')
atl2 = get_group_len_data('ATL', 'iso-fuz')
atl3 = get_group_len_data('ATL', 'fuz-fuz')

climp1 = get_group_len_data('Climp', 'iso-iso')
climp2 = get_group_len_data('Climp', 'iso-fuz')
climp3 = get_group_len_data('Climp', 'fuz-fuz')
# ctrl = get_group_len_data('Control', 'iso-iso')
rtn1 = get_group_len_data('RTN', 'iso-iso')
rtn2 = get_group_len_data('RTN', 'iso-fuz')
rtn3 = get_group_len_data('RTN', 'fuz-fuz')


# sns.distplot(atl1+atl2+atl3, hist=False, label='ATL')
# sns.distplot(climp1+climp2+climp3, hist=False, label='Climp')
# # sns.distplot(ctrl, hist=False, label='Control')
# sns.distplot(rtn1+rtn2+rtn3, hist=False, label='RTN')
# plt.show()
# exit()


def get_above_mean_across_groups():
    atl_list_eg = []
    atl_list_mch = []

    climp_list_eg = []
    climp_list_mch = []

    rtn_list_eg = []
    rtn_list_mch = []

    ctrl_list_eg = []

    # for i in range(1, 27):
    #     ln_egfp, ln_mch = calc_deposit(i, 'ATL', 'iso')
    #
    #     cnt_eg_list = []
    #     cnt_mch_list = []
    #     for each_eg, each_mch in zip(ln_egfp, ln_mch):
    #         eg_mean = np.mean(each_eg)
    #         mch_mean = np.mean(each_mch)
    #         cnt_eg = sum(t > eg_mean for t in each_eg)
    #         cnt_eg_list.append(cnt_eg/100)
    #         cnt_mch = sum(t > mch_mean for t in each_mch)
    #         cnt_mch_list.append(cnt_mch/100)
    #
    #     # print(cnt_list)
    #     atl_list_eg.extend(cnt_eg_list)
    #     atl_list_mch.extend(cnt_mch_list)

    for i in range(1, 27):
        ln_egfp = calc_deposit(i, 'ATL', 'iso')

        cnt_eg_list = []
        for each_eg in ln_egfp:
            eg_mean = np.mean(each_eg)
            cnt_eg = sum(t > eg_mean for t in each_eg)
            cnt_eg_list.append(cnt_eg / 100)

        atl_list_eg.extend(cnt_eg_list)

    for i in range(1, 32):
        ln_egfp = calc_deposit(i, 'Climp', 'iso')

        cnt_eg_list = []
        for each_eg in ln_egfp:
            eg_mean = np.mean(each_eg)
            cnt_eg = sum(t > eg_mean for t in each_eg)
            cnt_eg_list.append(cnt_eg / 100)

        climp_list_eg.extend(cnt_eg_list)

    # for i in range(1, 32):
    #     ln_egfp = calc_deposit(i, 'Control', 'iso')
    #
    #     cnt_eg_list = []
    #     for each_eg in ln_egfp:
    #         eg_mean = np.mean(each_eg)
    #         cnt_eg = sum(t > eg_mean for t in each_eg)
    #         cnt_eg_list.append(cnt_eg/100)
    #
    #     ctrl_list_eg.extend(cnt_eg_list)

    for i in range(1, 30):
        ln_egfp = calc_deposit(i, 'RTN', 'iso')

        # print(ln_egfp)
        cnt_eg_list = []
        for each_eg in ln_egfp:
            eg_mean = np.mean(each_eg)
            # print(eg_mean)
            cnt_eg = sum(t > eg_mean for t in each_eg)
            # print(cnt_eg)
            cnt_eg_list.append(cnt_eg / 100)

        rtn_list_eg.extend(cnt_eg_list)

    # print(above_mean_list)
    # print(len(above_mean_list))
    plt.title('Isolated CCs in mCherry across movies with intensity above mean value for all conditions', fontsize=18)
    plt.xlabel('Fraction of timeframes with intensity above mean', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    sns.distplot(atl_list_eg, label='ATL', hist=False)
    sns.distplot(climp_list_eg, label='Climp', hist=False)
    # sns.distplot(ctrl_list_eg, label='Control', hist=False)
    sns.distplot(rtn_list_eg, label='RTN', hist=False)
    plt.legend(fontsize=13)
    plt.show()


def junc_bar_plots(ser_num, group, junc_num):
    ln_egfp, ln_mch, region_cc_coords = calc_deposit(ser_num, group, 'iso')

    df = pd.DataFrame({'ERmoxGFP': ln_egfp[junc_num], 'mCherry': ln_mch[junc_num]}, index=np.arange(0, 10))

    df.plot.bar(rot=45)

    plt.ylabel('Mean intensity value', fontsize=14)
    plt.xlabel('Timeframe', fontsize=14)
    #    xr = np.arange(0, len(ma_egfp_5))
    #    plt.xticks(xr, rotation=45)
    # plt.title(f'{group} series 1, isolated CC {junc_num+1} mean intensity variation for both channels', fontsize=18)
    #    plt.title(f'{group} Series{ser_num} isolated junc{junc_num} - CC mean intensity variation (mov. avg 3)', fontsize=18)
    plt.title(f'{group} Series{ser_num} isolated junc{junc_num} - CC mean intensity variation', fontsize=18)
    plt.legend()
    #    plt.legend(loc='upper right')
    # plt.savefig(f'ATL1_j{ser_num}_ma5', bbox_inches='tight', pad_inches=1)
    # plt.close()
    plt.show()


def junc_line_charts_init(ser_num, group, junc_num):
    global junc_id
    ln_egfp, ln_mch, region_cc_coords = calc_deposit(ser_num, group, 'iso')

    ids = list(region_cc_coords.keys())

    # print(ids)
    # exit()

    for i, val in enumerate(ids):
        if junc_num == val:
            junc_id = i
            break

    plt.plot(ln_egfp[junc_id], label='EGFP')
    plt.plot(ln_mch[junc_id], label='mCherry')

    plt.ylabel('Mean intensity value', fontsize=14)
    plt.xlabel('Timeframe', fontsize=14)

    plt.title(f'{group} Series{ser_num} isolated junc{junc_num} - CC mean intensity variation', fontsize=18)
    plt.legend()
    plt.show()


def junc_line_charts(ser_num, group, junc_num):
    global junc_id
    ln_egfp, ln_mch, region_cc_coords = calc_deposit(ser_num, group, 'iso')

    print(ln_egfp.shape)
    print(ln_egfp[0])
    exit()

    ln_egfp_mod = []

    for junc_data in ln_egfp:
        # print(junc_data)
        # exit()
        data = []
        for each in junc_data:
            data.extend(each)
        # print(np.amax(junc_data))

        # print(max(data))
        # print(min(data))
        # print(data)

        # print(junc_data - min(data))
        # print(max(data) - min(data))
        #
        # exit()
        std_data = (junc_data - min(data)) / (max(data) - min(data))
        # std_data = np.array(std_data)
        # print(std_data)
        # print(np.mean(std_data.T))
        # print(junc_data)
        l = []
        for e in std_data:
            l.append(np.mean(e))
            ln_egfp_mod.append(l)

    print(np.array(ln_egfp_mod).shape)
    exit()

    # print(ln_egfp.shape)
    # print(ln_egfp[0][:])
    # print(ln_egfp[0].T)
    # exit()

    # df = pd.DataFrame({'Timeframe': np.arange(0, 100), 'Mean Intensity value': ln_egfp[junc_num]})

    # df.plot.bar(x='Timeframe', y='Mean Intensity value', rot=45)

    ##################
    #    df = pd.DataFrame({'ERmoxGFP': ln_egfp[junc_num], 'mCherry': ln_mch[junc_num]}, index = np.arange(0, 10))

    #    df.plot.bar(rot=45)
    ##################

    #    plt.show()

    # cumsum_vec_mch = np.cumsum(np.insert(ln_mch[junc_num], 0, 0))
    # w = mva
    # ma_vec_mch = (cumsum_vec_mch[w:] - cumsum_vec_mch[:-w]) / w

    # cumsum_vec_eg = np.cumsum(np.insert(ln_egfp[junc_num], 0, 0))
    # w = mva
    # ma_vec_eg = (cumsum_vec_eg[w:] - cumsum_vec_eg[:-w]) / w

    #    ma_egfp_3 = np.convolve(ln_egfp[junc_num], np.ones(3), 'valid') / 3

    # ma_mch_3 = np.convolve(ln_mch[junc_num], np.ones(3), 'valid') / 3

    # ma_egfp_5 = np.convolve(ln_egfp[junc_num], np.ones(5), 'valid') / 5

    #    ma_mch_3 = np.convolve(ln_mch[junc_num], np.ones(3), 'valid') / 3

    #    ma_mch_5 = np.convolve(ln_mch[junc_num], np.ones(5), 'valid') / 5

    # ma_egfp_7 = np.convolve(ln_egfp[junc_num], np.ones(7), 'valid') / 7

    # ma_mch_7 = np.convolve(ln_mch[junc_num], np.ones(7), 'valid') / 7

    # ma_egfp_9 = np.convolve(ln_egfp[junc_num], np.ones(9), 'valid') / 9

    # ma_mch_9 = np.convolve(ln_mch[junc_num], np.ones(9), 'valid') / 9

    #    print(ln_egfp[:5])

    #    exit()

    # print(ln_egfp[junc_num])
    # print(ln_mch[junc_num])

    #############################################

    ids = list(region_cc_coords.keys())

    # print(ids)
    # exit()

    for i, val in enumerate(ids):
        if junc_num == val:
            junc_id = i
            break

    #############################################

    # idx = label_vals[junc_num]
    # print(idx)

    # for i, val in enumerate(iso):
    #     if idx[0][0] == val[0] and idx[0][1] == val[1]:
    #         junc_id = i
    #         break

    # print(junc_id)
    # print(iso)
    # print(region_cc_coords)
    # exit()

    #############################################

    # print(ln_mch[junc_id])
    # print(ln_mch[junc_id+1])

    std_egfp = (ln_egfp[junc_id] - ln_egfp[junc_id].min()) / (ln_egfp[junc_id].max() - ln_egfp[junc_id].min())
    std_mch = (ln_mch[junc_id] - ln_mch[junc_id].min()) / (ln_mch[junc_id].max() - ln_mch[junc_id].min())

    plt.plot(std_egfp, label='EGFP')
    plt.plot(std_mch, label='mCherry')

    # plt.plot(ln_egfp[junc_id], label='EGFP')
    # plt.plot(ln_mch[junc_id], label='mCherry')

    # plt.plot(ln_egfp[junc_id+1], label='CC mean intensity (EGFP)')
    # plt.plot(ln_mch[junc_id+1], label='CC mean intensity (mCherry)')

    #    plt.plot(ln_egfp[junc_num], label='EGFP')
    #    plt.bar(ln_egfp[junc_num], label='EGFP')

    #    plt.plot(ma_egfp_3, label='EGFP')
    # plt.plot(ma_egfp_5, label='EGFP')
    # plt.plot(ma_mch_9, label='mCherry')
    # plt.plot(ma_egfp_9, label='EGFP')
    # plt.plot(ma_mch_9, label='mCherry')
    # plt.plot(ma_egfp_5, label='EGFP, mva=5')
    #    plt.plot(ln_mch[junc_num], label='mCherry')
    #    plt.bar(ln_mch[junc_num], label='mCherry')

    #    plt.plot(ma_mch_3, label='mCherry')
    #    plt.plot(ma_mch_5, label='mCherry')
    # plt.plot(ma_egfp_7, label='EGFP, mva=7')
    # plt.plot(ma_mch_7, label='mCherry, mva=7')
    # plt.plot(ma_egfp_9, label='EGFP, mva=9')
    # plt.plot(ma_mch_9, label='mCherry, mva=9')

    plt.ylabel('Mean intensity value', fontsize=14)
    plt.xlabel('Timeframe', fontsize=14)
    #    xr = np.arange(0, len(ma_egfp_5))
    #    plt.xticks(xr, rotation=45)

    # xr = np.arange(30, 60)
    # print(xr)
    # plt.xticks(range(30, 60))
    # exit()
    # plt.xticks(ln_egfp[junc_id], labels=xr)
    # ax.set_xticklabels(xr)

    # plt.title(f'{group} series 1, isolated CC {junc_num+1} mean intensity variation for both channels', fontsize=18)
    #    plt.title(f'{group} Series{ser_num} isolated junc{junc_num} - CC mean intensity variation (mov. avg 3)', fontsize=18)
    plt.title(f'{group} Series{ser_num} isolated junc{junc_num} - CC mean intensity variation', fontsize=18)
    plt.legend()
    #    plt.legend(loc='upper right')
    # plt.savefig(f'ATL1_j{ser_num}_ma5', bbox_inches='tight', pad_inches=1)
    # plt.close()
    plt.show()


# def junc_line_mean_std(group, num_movies, region):
def junc_line_mean_std(group, repl_start, repl_end, region, measure):
    egfp_list = []
    mch_list = []

    for ser_num in range(repl_start, repl_end + 1):
        global junc_id

        ln_egfp, ln_mch, region_cc_coords = calc_deposit_net_norm(ser_num, group, region)

        ids = list(region_cc_coords.keys())

        # for i, val in enumerate(ids):
        #     if junc_num == val:
        #         junc_id = i
        #         break
        l_eg = []
        l_mc = []

        for idx, val in enumerate(ids):
            mx_egfp = max(ln_egfp[idx])
            mn_egfp = min(ln_egfp[idx])
            std_egfp = (ln_egfp[idx] - mn_egfp) / (mx_egfp - mn_egfp)
            if measure == 'mean':
                l_eg.append(np.mean(std_egfp))
            else:
                l_eg.append(np.std(std_egfp))
            egfp_list.extend(l_eg)
            if group != 'Control':
                mx_mch = max(ln_mch[idx])
                mn_mch = min(ln_mch[idx])
                std_mch = (ln_mch[idx] - mn_mch) / (mx_mch - mn_mch)
                if measure == 'mean':
                    l_mc.append(np.mean(std_mch))
                else:
                    l_mc.append(np.std(std_mch))
                mch_list.extend(l_mc)
            else:
                mch_list = None

    # return egfp_list, mch_list
    return mch_list


def correlation_analysis(group, repl_start, repl_end, region):
    corr_list = []

    for ser_num in range(repl_start, repl_end + 1):
        global junc_id
        # ln_egfp, ln_mch, region_cc_coords = calc_deposit_cc_norm(ser_num, group, 'iso')

        ln_egfp, ln_mch, region_cc_coords = calc_deposit_net_norm(ser_num, group, region)

        ids = list(region_cc_coords.keys())
        # print(ids)
        # exit()
        l = []

        for idx, val in enumerate(ids):
            mx_egfp = max(ln_egfp[idx])
            mn_egfp = min(ln_egfp[idx])

            mx_mch = max(ln_mch[idx])
            mn_mch = min(ln_mch[idx])

            std_egfp = (ln_egfp[idx] - mn_egfp) / (mx_egfp - mn_egfp)
            std_mch = (ln_mch[idx] - mn_mch) / (mx_mch - mn_mch)

            p = pearsonr(std_egfp, std_mch)
            l.append(p[0])
            corr_list.extend(l)

    return corr_list


def corr_plots(region):
    a1 = correlation_analysis('ATL', 1, 10, region)
    a2 = correlation_analysis('ATL', 11, 20, region)
    a3 = correlation_analysis('ATL', 21, 26, region)
    c1 = correlation_analysis('Climp', 1, 10, region)
    c2 = correlation_analysis('Climp', 11, 20, region)
    c3 = correlation_analysis('Climp', 21, 31, region)
    r1 = correlation_analysis('RTN', 1, 10, region)
    r2 = correlation_analysis('RTN', 11, 20, region)
    r3 = correlation_analysis('RTN', 21, 29, region)

    df = pd.DataFrame()

    df['Channel correlation'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3)))

    df['Group'] = pd.Series(np.concatenate((
        ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1),
        ['Climp'] * len(c2), ['Climp'] * len(c3), ['RTN'] * len(r1),
        ['RTN'] * len(r2), ['RTN'] * len(r3))))
    df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1),
                                                ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2),
                                                ['R3'] * len(r3))))

    ax = sns.boxplot(data=df, x='Replicate', y='Channel correlation', hue='Group', dodge=True)  # , yscale='log')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

    box_pairs = [(('R1', 'ATL'), ('R1', 'Climp')), (('R1', 'ATL'), ('R1', 'RTN')), (('R1', 'Climp'), ('R1', 'RTN')),
                 (('R2', 'ATL'), ('R2', 'Climp')), (('R2', 'ATL'), ('R2', 'RTN')), (('R2', 'Climp'), ('R2', 'RTN')),
                 (('R3', 'ATL'), ('R3', 'Climp')), (('R3', 'ATL'), ('R3', 'RTN')), (('R3', 'Climp'), ('R3', 'RTN'))]
    statannot.add_stat_annotation(ax, x='Replicate', y='Channel correlation', hue='Group', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    # plt.suptitle('Isolated CC area across conditions', fontsize=20)
    # plt.title('Standard Deviation per sequence for junction CC mean intensity (isolated junctions)', fontsize=18)
    # measure_name = 'Standard deviation' if measure == 'std' else 'Mean'
    region_name = 'isolated' if region == 'iso' else 'fuzzy'
    plt.title(f'Cross correlation between EGFP and mCherry channels in {region_name} CC mean intensity sequences',
              fontsize=18)
    plt.grid(True)
    plt.xlabel('Replicate', fontsize=18)
    plt.ylabel('Pearson correlation coefficient value', fontsize=18)
    plt.show()

    # plt.title(
    #     f'{group} cross correlation between EGFP annd mCherry {reg} CC mean intensity sequences across replicates',
    #     fontsize=18)
    # plt.xlabel('Pearson correlation coefficient value')
    # plt.legend()
    # plt.show()


# corr_plots('fuz')
# exit()


def total_data_variation_plots(group, region, measure):
    # atl_egfp, atl_mch = junc_line_mean_std('ATL', 1, 10, 'fuz')
    # climp_egfp, climp_mch = junc_line_mean_std('Climp', 1, 10, 'fuz')
    # rtn_egfp, rtn_mch = junc_line_mean_std('RTN', 1, 10, 'fuz')
    # # ctrl_egfp, ctrl_mch = junc_line_mean_std('Control', 1, 10, 'iso')
    #
    # # sns.distplot(atl_egfp, hist=False, label='ATL_egfp')
    # sns.distplot(atl_mch, hist=False, label='ATL_mch')
    # # sns.distplot(climp_egfp, hist=False, label='Climp_egfp')
    # sns.distplot(climp_mch, hist=False, label='Climp_mch')
    # # sns.distplot(rtn_egfp, hist=False, label='RTN_egfp')
    # sns.distplot(rtn_mch, hist=False, label='RTN_mch')
    # # sns.distplot(ctrl_egfp, hist=False, label='Control_egfp')
    # plt.title('Standard deviation of Junction CC mean intensity for fuzzy junction CCs (mCherry) - Replicate 1')
    # plt.xlabel('Standard deviation over 100 frames per CC intensity mean value')
    # plt.legend()
    # plt.show()

    if group == 'ATL':
        r3end = 26
    elif group == 'Climp' or group == 'Control':
        r3end = 31
    else:
        r3end = 29

    egfp_r1, mch_r1 = junc_line_mean_std(group, 1, 10, region, measure)
    egfp_r2, mch_r2 = junc_line_mean_std(group, 11, 20, region, measure)
    egfp_r3, mch_r3 = junc_line_mean_std(group, 21, r3end, region, measure)

    reg = 'fuzzy' if region == 'fuz' else 'isolated'

    sns.distplot(egfp_r1, hist=False, label=f'{group}_r1')
    sns.distplot(egfp_r2, hist=False, label=f'{group}_r2')
    sns.distplot(egfp_r3, hist=False, label=f'{group}_r3')
    plt.title(
        f'{group} - Standard deviation of Junction CC mean intensity for {reg} junction CCs (egfp) across replicates',
        fontsize=16)
    plt.xlabel('Standard deviation over 100 frames per CC intensity mean value')
    plt.legend()
    plt.show()
    # climp_egfp, climp_mch = junc_line_mean_std('Climp', 1, 10, 'fuz')
    # rtn_egfp, rtn_mch = junc_line_mean_std('RTN', 1, 10, 'fuz')


# total_data_variation_plots('Control', 'iso', 'std')
# exit()

def full_data_variation_plots(region, measure, channel):
    a_r1_egfp = junc_line_mean_std('ATL', 1, 2, region, measure)
    a_r2_egfp = junc_line_mean_std('ATL', 11, 12, region, measure)
    a_r3_egfp = junc_line_mean_std('ATL', 21, 22, region, measure)
    c_r1_egfp = junc_line_mean_std('Climp', 1, 2, region, measure)
    c_r2_egfp = junc_line_mean_std('Climp', 11, 12, region, measure)
    c_r3_egfp = junc_line_mean_std('Climp', 21, 22, region, measure)
    r_r1_egfp = junc_line_mean_std('RTN', 1, 2, region, measure)
    r_r2_egfp = junc_line_mean_std('RTN', 11, 12, region, measure)
    r_r3_egfp = junc_line_mean_std('RTN', 21, 22, region, measure)
    # ct_r1_egfp = junc_line_mean_std('Control', 1, 10, region, measure)
    # ct_r2_egfp = junc_line_mean_std('Control', 11, 20, region, measure)
    # ct_r3_egfp = junc_line_mean_std('Control', 21, 31, region, measure)

    df = pd.DataFrame()

    df['data_junc_CC_mean'] = pd.Series(np.concatenate((
        a_r1_egfp, a_r2_egfp, a_r3_egfp, c_r1_egfp, c_r2_egfp, c_r3_egfp,
        r_r1_egfp, r_r2_egfp,
        r_r3_egfp)))  # , ct_r1_egfp, ct_r2_egfp, ct_r3_egfp)))
    df['Group'] = pd.Series(np.concatenate((
        ['ATL'] * len(a_r1_egfp), ['ATL'] * len(a_r2_egfp), ['ATL'] * len(a_r3_egfp),
        ['Climp'] * len(c_r1_egfp), ['Climp'] * len(c_r2_egfp),
        ['Climp'] * len(c_r3_egfp), ['RTN'] * len(r_r1_egfp),
        ['RTN'] * len(r_r2_egfp), ['RTN'] * len(
            r_r3_egfp))))  # , ['Control']*len(ct_r1_egfp), ['Control']*len(ct_r2_egfp), ['Control']*len(ct_r3_egfp))))
    df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a_r1_egfp), ['R2'] * len(a_r2_egfp),
                                                ['R3'] * len(a_r3_egfp), ['R1'] * len(c_r1_egfp),
                                                ['R2'] * len(c_r2_egfp), ['R3'] * len(c_r3_egfp),
                                                ['R1'] * len(r_r1_egfp), ['R2'] * len(r_r2_egfp), ['R3'] * len(
        r_r3_egfp))))  # ,['R1']*len(ct_r1_egfp),['R2']*len(ct_r2_egfp),['R3']*len(ct_r3_egfp))))
    ax = sns.boxplot(data=df, x='Replicate', y='data_junc_CC_mean', hue='Group', dodge=True)  # , yscale='log')

    # sns.swarmplot(data=df, x='Replicate', y='data_junc_CC_mean', hue='Group', dodge=True, linewidth=0)

    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    # sns.boxplot(data=df, x='Group', y='CC_area', hue='replicate', color='white', dodge=True)
    # hue_order = ['ATL', 'Climp', 'RTN', 'Control']
    # box_pairs = [(('R1', 'ATL'), ('R1', 'Climp')), (('R1', 'ATL'), ('R1', 'RTN')), (('R1', 'ATL'), ('R1', 'Control')), (('R1', 'Climp'), ('R1', 'RTN')), (('R1', 'Climp'), ('R1', 'Control')), (('R1', 'RTN'), ('R1', 'Control')), (('R2', 'ATL'), ('R2', 'Climp')), (('R2', 'ATL'), ('R2', 'RTN')), (('R2', 'ATL'), ('R2', 'Control')), (('R2', 'Climp'), ('R2', 'RTN')), (('R2', 'Climp'), ('R2', 'Control')), (('R2', 'RTN'), ('R2', 'Control')), (('R3', 'ATL'), ('R3', 'Climp')), (('R3', 'ATL'), ('R3', 'RTN')), (('R3', 'ATL'), ('R3', 'Control')), (('R3', 'Climp'), ('R3', 'RTN')), (('R3', 'Climp'), ('R3', 'Control')), (('R3', 'RTN'), ('R3', 'Control'))]

    replicates = ['R1', 'R2', 'R3']
    groups = ['ATL', 'Climp', 'RTN', 'Control']

    box_pairs = [((x, y), (x, z)) for i, x in enumerate(replicates) for j, y in enumerate(groups) for z in groups[j + 1 :]]


    # box_pairs = [(('R1', 'ATL'), ('R1', 'Climp')), (('R1', 'ATL'), ('R1', 'RTN')), (('R1', 'Climp'), ('R1', 'RTN')),
    #              (('R2', 'ATL'), ('R2', 'Climp')), (('R2', 'ATL'), ('R2', 'RTN')), (('R2', 'Climp'), ('R2', 'RTN')),
    #              (('R3', 'ATL'), ('R3', 'Climp')), (('R3', 'ATL'), ('R3', 'RTN')), (('R3', 'Climp'), ('R3', 'RTN'))]
    statannot.add_stat_annotation(ax, x='Replicate', y='data_junc_CC_mean', hue='Group', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    # plt.suptitle('Isolated CC area across conditions', fontsize=20)
    # plt.title('Standard Deviation per sequence for junction CC mean intensity (isolated junctions)', fontsize=18)
    measure_name = 'Standard deviation' if measure == 'std' else 'Mean'
    region_name = 'isolated' if region == 'iso' else 'fuzzy'
    plt.title(
        f'{measure_name} per sequence for junction CC mean intensity ({region_name} junctions) - {channel} channel',
        fontsize=18)
    plt.grid(True)
    plt.xlabel('Replicate', fontsize=18)
    plt.ylabel(f'{measure_name} value per sequence', fontsize=18)
    plt.show()


# full_data_variation_plots('iso', 'std', 'mCherry')
# total_data_variation_plots('Control', 'iso', 'std')
# exit()


def junc_line_charts_norm(ser_num, group, junc_num):
    egfp_list = []
    mch_list = []
    global junc_id
    # ln_egfp, ln_mch, region_cc_coords = calc_deposit_cc_norm(ser_num, group, 'iso')

    ln_egfp, ln_mch, region_cc_coords = calc_deposit_net_norm(ser_num, group, 'fuz')

    # print(region_cc_coords)
    # exit()
    # print(ln_egfp)
    # print(ln_mch)
    # exit()

    ids = list(region_cc_coords.keys())
    # print(ids)
    # exit()

    for i, val in enumerate(ids):
        if junc_num == val:
            junc_id = i
            break

    mx_egfp = max(ln_egfp[junc_id])
    mn_egfp = min(ln_egfp[junc_id])

    mx_mch = max(ln_mch[junc_id])
    mn_mch = min(ln_mch[junc_id])

    std_egfp = (ln_egfp[junc_id] - mn_egfp) / (mx_egfp - mn_egfp)
    std_mch = (ln_mch[junc_id] - mn_mch) / (mx_mch - mn_mch)

    print(pearsonr(std_egfp, std_mch))
    exit()

    # plt.plot(ln_egfp[junc_id], label='EGFP')
    # plt.plot(ln_mch[junc_id], label='mCherry')

    # plt.plot(std_egfp, label='EGFP')
    # plt.plot(std_mch, label='mCherry')

    plt.ylabel('Mean intensity value', fontsize=14)
    plt.xlabel('Timeframe', fontsize=14)
    #    xr = np.arange(0, len(ma_egfp_5))
    #    plt.xticks(xr, rotation=45)

    # xr = np.arange(30, 60)
    # print(xr)
    # plt.xticks(range(30, 60))
    # exit()
    # plt.xticks(ln_egfp[junc_id], labels=xr)
    # ax.set_xticklabels(xr)

    # plt.title(f'{group} series 1, isolated CC {junc_num+1} mean intensity variation for both channels', fontsize=18)
    #    plt.title(f'{group} Series{ser_num} isolated junc{junc_num} - CC mean intensity variation (mov. avg 3)', fontsize=18)

    # plt.title(f'{group} Series{ser_num} isolated junc{junc_num} - CC mean intensity variation', fontsize=18)
    # plt.legend()

    plt.title(f'{group} Series{ser_num} fuzzy CC{junc_num} - mean intensity variation', fontsize=18)
    plt.legend()

    #    plt.legend(loc='upper right')
    # plt.savefig(f'ATL1_j{ser_num}_ma5', bbox_inches='tight', pad_inches=1)
    # plt.close()
    plt.show()


# junc_line_charts_init(1, 'ATL', 2)
junc_line_charts_norm(13, 'RTN', 106)
exit()


def get_lincharts(group, junc_num):
    ln_egfp, ln_mch, region_cc_coords = calc_deposit(1, group, 'iso')

    # print(list(region_cc_coords.values())[junc_num])
    # print(region_cc_coords.values())
    # exit()

    # mov_avg = np.convolve(ln_mch[5], np.ones(5), 'valid') / 5
    eg_mean = np.mean(ln_egfp[junc_num])
    mch_mean = np.mean(ln_mch[junc_num])
    scaled_ln_mch = ln_mch[junc_num] * (eg_mean / mch_mean)

    cumsum_vec_mch = np.cumsum(np.insert(scaled_ln_mch, 0, 0))
    w = 5
    ma_vec_mch = (cumsum_vec_mch[w:] - cumsum_vec_mch[:-w]) / w

    cumsum_vec_eg = np.cumsum(np.insert(ln_egfp[junc_num], 0, 0))
    w = 5
    ma_vec_eg = (cumsum_vec_eg[w:] - cumsum_vec_eg[:-w]) / w

    bin_list_eg = []
    bin_list_mch = []
    # for each_eg in ln_egfp:

    bin_ma_mch = []
    bin_ma_egfp = []

    for each in ma_vec_eg:
        if each > eg_mean:
            bin_ma_egfp.append(1)
        else:
            bin_ma_egfp.append(0)

    for each in ma_vec_mch:
        if each > mch_mean:
            bin_ma_mch.append(1)
        else:
            bin_ma_mch.append(0)

    for t in ln_egfp[junc_num]:
        if t > eg_mean:
            bin_list_eg.append(1)
        else:
            bin_list_eg.append(0)

    for t in scaled_ln_mch:
        if t > mch_mean:
            bin_list_mch.append(1)
        else:
            bin_list_mch.append(0)

    egfp_mean_lt = [eg_mean] * 100

    # import scipy.signal
    #
    # corr = scipy.signal.correlate(ma_vec_eg, ma_vec_mch)
    #
    # lags = scipy.signal.correlation_lags(len(ma_vec_mch), len(ma_vec_eg))
    #
    # corr /= np.max(corr)
    #
    # plt.plot(lags, corr)
    # plt.show()
    # corr = (len(ma_vec_eg) - len(ma_vec_mch) + 1) * [0]

    # Go through lag components one-by-one
    # for l in range(len(corr)):
    #     corr[l] = sum([ma_vec_eg[i+l] * ma_vec_mch[i] for i in range(len(ma_vec_mch))])

    # print(corr)

    # # Remove padded correlations
    #     print(ma_vec_eg)
    #     cr_corr = corr[(len(ma_vec_eg)-len(ma_vec_mch)-1):len(corr)-((len(ma_vec_eg)-len(ma_vec_mch)-1))]

    # print(cr_corr)
    # plt.plot(cr_corr)

    # plt.plot(bin_list_eg, label='EGFP_bin')
    # plt.plot(bin_list_mch, label='mCherry_bin')
    # plt.plot(egfp_mean_lt, label='Mean value')

    plt.plot(ln_egfp[junc_num], label='CC mean intensity (EGFP)')
    plt.plot(scaled_ln_mch, label='CC mean intensity (mCherry)')
    # plt.plot(ma_vec_mch, label='mov_avg_mCherry')
    # plt.plot(ma_vec_eg, label='mov_avg_egfp')
    plt.plot(bin_ma_mch, label='binarized mov_avg mCherry')
    plt.plot(bin_ma_egfp, label='binarized mov_avg EGFP')
    plt.plot(egfp_mean_lt, label='Mean value')

    plt.ylabel('Mean intensity value', fontsize=14)
    plt.xlabel('Timeframe', fontsize=14)
    plt.title(f'{group} series 1, isolated CC {junc_num + 1} mean intensity variation for both channels', fontsize=18)
    plt.legend()
    plt.show()
    exit()
    #
    #     # print(eg_mean)
    #     # cnt_eg = sum(t > eg_mean for t in each_eg)
    #     # print(cnt_eg)
    #     # cnt_eg_list.append(cnt_eg / 100)
    #
    #
    plt.plot(ln_egfp[5], label='CC mean intensity (EGFP)')
    plt.plot(scaled_ln_mch, label='CC mean intensity (mCherry)')
    plt.plot(ma_vec_mch, label='mov_avg_mCherry')
    plt.plot(ma_vec_eg, label='mov_avg_egfp')
    plt.xlabel()
    plt.legend()
    plt.show()
    #
    exit()

    mch_mean = np.mean(ln_mch[5])

    egfp_mean = np.mean(ln_egfp[5])

    mch_mean_lt = [mch_mean] * 100
    egfp_mean_lt = [egfp_mean] * 100

    plt.plot(mch_mean_lt, label='mCherry_mean')
    plt.plot(egfp_mean_lt, label='EGFP_mean')
    plt.plot(ln_mch[5], label='mCherry')
    plt.plot(ln_egfp[5], label='EGFP')

    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Mean Intensity value', fontsize=15)
    plt.title('ATL series 1, isolated CC 6 mean intensity over time', fontsize=18)

    plt.legend()
    plt.show()


get_lincharts('RTN', 20)
exit()

# op = np.logical_and(ln_egfp, ln_mch)
# print(op)

# print(ln.shape)


exit()

from scipy.stats import pearsonr


def ref_junc_deposit():
    egfp_img_path = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean/atl1_er_mean.png'

    egfp_img = get_std_img(egfp_img_path)


def per_ref_junc_deposit(group, region, series_num):
    # egfp_img_path = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean/atl1_er_mean.png'

    # egfp_img = get_std_img(egfp_img_path)
    # print(egfp_img.max())

    # mch_stack = np.zeros((128, 128))
    # for i in range(100):
    #     mch_img = f'/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0{i:02d}_ch01.tif'
    #     mch_img = get_std_img(mch_img)
    #     mch_stack += mch_img
    # mch_op = mch_stack / 100

    # region_name = 'isolated' if region == 'iso' else 'fuzzy'
    atl_mch_iso = calc_egfp_deposit(group, 'mCherry', series_num, region)
    atl_egfp_iso = calc_egfp_deposit(group, 'ERmoxGFP', series_num, region)

    corr = pearsonr(atl_egfp_iso, atl_mch_iso)
    # print(corr)
    # print(len(atl_egfp_iso))

    df = pd.DataFrame()

    df['ERmoxGFP'] = pd.Series(atl_egfp_iso)
    df['mCherry'] = pd.Series(atl_mch_iso)

    fig = plt.gcf()
    fig.set_size_inches(15, 15, forward=True)
    fig.set_dpi(150)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    # plt.figure(figsize=(10,10), dpi=200)
    # sns.scatterplot(data=df, x='mCherry', y='ERmoxGFP')
    b, m = polyfit(df['mCherry'], df['ERmoxGFP'], 1)
    num_pts = len(df['mCherry'])
    plt.plot(df['mCherry'], df['ERmoxGFP'], '.', alpha=(1 / num_pts) * 8000)
    plt.plot(df['mCherry'], b + m * df['mCherry'], '-')
    plt.title('%s - %s junctions, corr=%.2f' % (group, region, corr[0]), fontsize=20)
    plt.xlabel('mCherry', fontsize=18)
    plt.ylabel('ERmoxGFP', fontsize=18)

    plt.show()
    # plt.savefig(f'{group}_{region}_junc_intensity', bbox_inches='tight', pad_inches=0.2)
    # plt.close()

    # exit()

    # climp = calc_egfp_deposit('Climp', channel, 31, region)
    # ctrl = calc_egfp_deposit('Control', channel, 31, region)
    # rtn = calc_egfp_deposit('RTN', channel, 29, region)

    # cv2.imwrite('ATL1_mch_mean.png', mch_op)
    # exit()

    # print(mch_op.max())
    # exit()

    # ref_skel = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'
    # junctions = get_junctions(ref_skel)
    #
    # # print(junctions)
    # egfp_data = []
    # mch_data = []
    # for each in junctions:
    #     x, y = each[0], each[1]
    #     mean_egfp_val = (egfp_img[x-1, y-1] + egfp_img[x-1, y] + egfp_img[x-1, y+1] + egfp_img[x, y-1] + egfp_img[x, y] + egfp_img[x, y+1] + egfp_img[x+1, y-1] + egfp_img[x+1, y] + egfp_img[x+1, y+1]) / 9
    #     egfp_data.append(mean_egfp_val)
    #     mean_mch_val = (mch_op[x-1, y-1] + mch_op[x-1, y] + mch_op[x-1, y+1] + mch_op[x, y-1] + mch_op[x, y] + mch_op[x, y+1] + mch_op[x+1, y-1] + mch_op[x+1, y] + mch_op[x+1, y+1]) / 9
    #     mch_data.append(mean_mch_val)

    # print(max(egfp_data))
    # print(max(mch_data))
    # exit()

    # plt.plot(egfp_data, label='EGFP')
    # plt.plot(mch_data, label='mCherry')
    # df = pd.DataFrame()
    # df['egfp'] = pd.Series(egfp_data)
    # df['mcherry'] = pd.Series(mch_data)
    # df['intensity'] = pd.Series(np.concatenate((egfp_data, mch_data)))
    # df['junction_num'] = pd.Series(np.concatenate((np.arange( len(egfp_data)+1), np.arange(len(mch_data)+1))))
    # df['channel'] = pd.Series(np.concatenate((['EGFP'] * len(egfp_data), ['mCherry'] * len(mch_data))))

    # df['egfp'] = pd.Series(egfp_data)
    # df['mcherry'] = pd.Series(mch_data)
    #
    # # print(df['channel'])
    # # print(df['junctions'])
    # plt.title('ATL series 1 junction intensity values (mean per 3x3 patch)', fontsize=14)
    #
    # sns.lmplot(data=df, x='egfp', y='mcherry', lowess=True)
    #
    # plt.show()


per_ref_junc_deposit('ATL', 'iso', 26)
# per_ref_junc_deposit('ATL', 'fuz', 26)
# per_ref_junc_deposit('Climp', 'iso', 31)
# per_ref_junc_deposit('Climp', 'fuz', 31)
# per_ref_junc_deposit('RTN', 'iso', 29)
# per_ref_junc_deposit('RTN', 'fuz', 29)
# per_ref_junc_deposit('Climp', 'iso', 31)
# per_ref_junc_deposit('iso')
# per_ref_junc_deposit('iso')
# per_ref_junc_deposit('iso')
# per_ref_junc_deposit('iso')
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
df['channel'] = pd.Series(np.concatenate((np.arange(1, len(atl_egfp) + 1), np.arange(1, len(atl_mch) + 1))))

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
plt.legend(fontsize=12)  # for distplot only

plt.savefig('All_conditions_fuzzy_CCs_mean_intensity_per_patch_across_channels', bbox_inches='tight', pad_inches=0.2)
plt.close()
# plt.show()

# junction_cc_mean_distplot_per_group('RTN', 29)
