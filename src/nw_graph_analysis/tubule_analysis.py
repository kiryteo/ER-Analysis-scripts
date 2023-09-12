import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cv2
import os
import sys
import pickle as pkl
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import scipy.ndimage as ndimage
from skimage.measure import label

from structure_extraction import *
from junction_analysis_modules import JunctionAnalysis as JA


confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'
junc_analysis = JA(confocal_data_path)


def get_edges(conn_graph, iso_ids, fuz_ids, connection):
    """
    @param conn_graph: graph object
    @param iso_ids: list of iso ids
    @param fuz_ids: list of fuz ids
    @param connection: type of connection
    """
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


def get_intersection(a, b):
    """
    get common elements between 2 ndarrays (list of nodes)
    @param a: list of lists
    @param b: list of lists
    """
    return np.array([x for x in a if np.any(np.all(x == b, axis=1))])


def get_tubule_data(group, series_num, connection):
    """
    
    @param group: group name
    @param series_num: series number
    @param connection: type of connection
    """
    group_pref = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}

    # sourcery skip: inline-immediately-returned-variable
    # mean_er_path = f'{confocal_data_path}/{group}/new_op_jul/er_mean/{group.lower()}{series_num}_er_mean.png'
    mean_skel_path = f'{confocal_data_path}/{group}/new_op_jul/er_mean_proc/{group.lower()}{series_num}_proc_skel.png'

    # ref_junctions = junc_analysis.get_junctions(er_input_path, skel_path)

    conn_graph = junc_analysis.skel_to_graph(mean_skel_path)
    ref_junctions = junc_analysis.get_ref_junctions(conn_graph)

    deg_one_nodes, deg_two_nodes, high_deg_nodes = get_updated_degree_nodes(conn_graph)

    # graph = junc_analysis.skel_to_graph(mean_img)
    # ref_junctions = junc_analysis.get_junctions(conn_graph)

    ref_junctions = [[junc[0], junc[1]] for junc in ref_junctions]

    per_frame_junctions = []
    for frame in range(100):
        er_path = f'{confocal_data_path}/{group}/new_op_jul/std_egfp/{group_pref[group]}{series_num}_decon_t0{frame:02d}_ch00_std.png'
        skeleton_path = f'{confocal_data_path}/{group}/new_op_jul/skel/{group_pref[group]}{series_num}/{group_pref[group]}{series_num}_decon_t0{frame:02d}_ch00_skel.png'

        # graph = node_connector(er_path, skeleton_path)

        # # create_tubule_junc_plot(er_path, skeleton_path)

        junctions, graph = junc_analysis.get_junctions(er_path, skeleton_path)

        junc_array = [[junc[0], junc[1]] for junc in junctions]
        per_frame_junctions.extend(junc_array)

    ref_junctions = np.array(ref_junctions)
    per_frame_junctions = np.array(per_frame_junctions)

    spread_img = np.zeros((128, 128))
    for junc in per_frame_junctions:
        spread_img[junc[0], junc[1]] = 255.

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


import imageio

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

    # assert group in VALID_GROUPS, f"Invalid group name: {group}"
    # assert connection in VALID_CONNECTIONS, f"Invalid connection type: {connection}"
    # assert channel in VALID_CHANNELS, f"Invalid channel name: {channel}"
    # assert measure in VALID_MEASURES, f"Invalid measure name: {measure}"

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

    ch_id = 0 if channel == 'egfp' else 1

    for pt in edge_pts:
        edge_data = []
        for i in range(100):
            er = get_er_input(i, group, channel, series_num, ch_id)
            edge_data.append(er[pt[:, 0], pt[:, 1]])
            # if measure == 'tubules':
            #     edge_data.append(er[pt[:, 0], pt[:, 1]])
            # elif measure == 'tub-mean':
            #     edge_data.append(np.mean(er[pt[:, 0], pt[:, 1]]))
        seq_data.append(edge_data)

    return seq_data

# seq_data = tubule_sequence_data('ATL', 1, 'iso-iso', 'egfp', 'mean')
# exit()


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
            vals.extend(np.mean(er[pt[:, 0], pt[:, 1]]) for pt in edge_pts)
        else:
            vals.extend(np.std(er[pt]) for pt in edge_pts)

        return vals


def tubule_length_analysis(conn_graph, edges):
    # Find the length of each edge in iso-iso
    connection_length = [
        conn_graph[u][v][0]['weight'] for (u, v) in edges
    ]

    return connection_length


def get_group_len_data(group, connection):
    """Get tubule lengths for all tubules in a given group.

    :param group: The group to analyze.
    :param connection: The connection to the database.
    :returns: A list of tubule lengths.
    """
    tubule_lengths = []
    for tubule_number in range(1, 11):
        edges, conn_graph = get_tubule_data(group, tubule_number, connection)
        ln = tubule_length_analysis(conn_graph, edges)
        tubule_lengths.extend(ln)
    return tubule_lengths


def correlation_analysis(group, repl_start, repl_end, region):
    corr_list = []

    for ser_num in range(repl_start, repl_end + 1):
        global junc_id
        # ln_egfp, ln_mch, region_cc_coords = calc_deposit_cc_norm(ser_num, group, 'iso')

        ln_egfp, ln_mch, region_cc_coords = calc_deposit_net_norm(ser_num, group, region)

        ids = list(region_cc_coords.keys())
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


def plot_seq_mean_tubule_mean(group, channel, connection):
    filename = f'{group.lower()}_{connection}_{channel}.pkl'
    with open(filename, 'rb') as f:
        data = pkl.load(f)

    d1 = []

    for d in data:
        try:
            d1.extend(np.std(i) for i in d if len(i) > 0)
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