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
import statannot

import imageio

from junction_analysis_modules import JunctionAnalysisModules as JAM


home = os.path.expanduser('~')
confocal_data_path = '/MIAL/data/confocal-data/'
junc_analysis = JAM(confocal_data_path)


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

    mean_skel_path = f'{home}{confocal_data_path}/vess_enh_unet/{group.lower()}/gt_skel/{group.lower()}{series_num}_proc_skel.png'

    conn_graph = junc_analysis.skel_to_graph(mean_skel_path)

    ref_junctions = junc_analysis.get_junctions(mean_skel_path)

    ref_junctions = [[junc[0], junc[1]] for junc in ref_junctions]

    per_frame_junctions = []
    for frame in range(100):

        skeleton_path = f'{home}{confocal_data_path}/vess_enh_unet/{group.lower()}/skel/{group_pref[group]}{series_num}_decon_t0{frame:02d}_ch00_skel.png'

        junctions = junc_analysis.get_junctions(skeleton_path)

        junc_array = [[junc[0], junc[1]] for junc in junctions]
        per_frame_junctions.extend(junc_array)

    ref_junctions = np.array(ref_junctions)
    per_frame_junctions = np.array(per_frame_junctions)

    spread_img = np.zeros((128, 128))
    for junc in per_frame_junctions:
        spread_img[junc[0], junc[1]] = 255.

    labelled_img = label(spread_img, connectivity=2)
    label_vals = junc_analysis.separate_junc_cc(ref_junctions, labelled_img)

    # iso, fuz, unk: list of lists with x, y
    iso, fuz = junc_analysis.get_junction_areas(label_vals)

    edges = get_edges(conn_graph, iso, fuz, connection)

    return edges, conn_graph


def get_tubule_len_data(group, series_num, connection):
    data = []
    for seq in range(1, series_num + 1):
        edges, conn_graph = get_tubule_data(group, seq, connection)

        seq_data = [len(conn_graph[edge[0]][edge[1]][0]['pts']) for edge in edges]
        data.append(seq_data)
    return data

import pickle as pkl

def save_tubule_len_data(group, length, condition, filename):
    data = get_tubule_len_data(group, length, condition)
    with open(filename, 'wb') as f:
        pkl.dump(data, f)

# Usage

def run_tubule_analysis():
    save_tubule_len_data('ATL', 26, 'iso-iso', 'atl_iso-iso_tubule_len.pkl')
    save_tubule_len_data('Climp', 31, 'iso-iso', 'climp_iso-iso_tubule_len.pkl')
    save_tubule_len_data('Control', 31, 'iso-iso', 'control_iso-iso_tubule_len.pkl')
    save_tubule_len_data('RTN', 29, 'iso-iso', 'rtn_iso-iso_tubule_len.pkl')


def load_data(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)

def create_dataframe(control_data, rtn_data, climp_data, atl_data):
    df = pd.DataFrame()
    df['data_tubule_len'] = pd.Series(np.concatenate((control_data, rtn_data, climp_data, atl_data)))
    df['Group'] = pd.Series(np.concatenate((['Control'] * len(control_data), ['Reticulon'] * len(rtn_data), ['Climp'] * len(climp_data), ['Atlastin'] * len(atl_data))))
    return df

def plot_tubule_length(df, output_filename):
    ax = sns.boxplot(data=df, x='Group', y='data_tubule_len', showfliers=False, width=0.9)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

    yt = ax.get_yticks()
    ymax = max(yt)
    yt = [f'{y:.2f}' for y in yt]

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, rotation=90)
    ax.set_yticklabels(yt, fontsize=13)
    ax.set_ylim(0, ymax + 0.2)

    box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('ATL', 'Control'), ('Climp', 'Control'), ('Climp', 'RTN'), ('Control', 'RTN')]

    # statannot.add_stat_annotation(ax, x='Group', y='data_tubule_len', data=df, box_pairs=box_pairs,
    #                               test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    # plt.title('Tubule length in iso-iso edges', fontsize=18)
    ax.grid(axis='y')
    # plt.xlabel('Group', fontsize=20)
    plt.ylabel('Tubule length', fontsize=15)

    plt.gcf().set_size_inches(2.5, 8)
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

def run_analysis():
    atl_data = load_data('atl_iso-iso_tubule_len.pkl')
    climp_data = load_data('climp_iso-iso_tubule_len.pkl')
    control_data = load_data('control_iso-iso_tubule_len.pkl')
    rtn_data = load_data('rtn_iso-iso_tubule_len.pkl')

    df = create_dataframe(control_data, rtn_data, climp_data, atl_data)
    plot_tubule_length(df, 'tubule_len_iso-iso.png')


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

        seq_data.append(edge_data)

    return seq_data


def tubule_intensity_analysis(group, series_num, connection, measure):
    edges, conn_graph = get_tubule_data(group, series_num, connection)

    if edges:
        er_input_path = f'{home}/MIAL/data/confocal_movies/{group}/new_op_jul/er_mean/{group.lower()}{series_num}_er_mean.png'
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



def plot_seq_mean_tubule_mean(group, channel, connection):
    """
    Calculate the standard deviation of tubule means for a given group, channel, and connection.

    Args:
        group (str): The group name.
        channel (str): The channel name.
        connection (str): The connection type.

    Returns:
        list: A list of standard deviations for each sequence.
    """
    filename = f'{group.lower()}_{connection}_{channel}.pkl'
    
    try:
        with open(filename, 'rb') as f:
            data = pkl.load(f)
    except (FileNotFoundError, pkl.UnpicklingError) as e:
        print(f"Error loading file {filename}: {e}")
        return []

    std_devs = []

    for sequence in data:
        try:
            std_devs.extend(np.std(item) for item in sequence if len(item) > 0)
        except TypeError as e:
            print(f"Error processing sequence {sequence}: {e}")
            continue

    return std_devs


def plotter(data):
    df = pd.DataFrame()

    df['data_tubule_mean'] = pd.Series(np.concatenate((data[0], data[1], data[2], data[3])))

    df['Group'] = pd.Series(np.concatenate((
        ['ATL'] * len(a1), ['Climp'] * len(c1), ['Control'] * len(ct1), ['RTN'] * len(r1))))

    ax = sns.boxenplot(data=df, x='Group', y='data_tubule_mean')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

    # egfp
    groups = ['ATL', 'Climp', 'RTN', 'Control']
    box_pairs = [(x, y) for i, x in enumerate(groups) for j, y in enumerate(groups) if i < j and y not in groups[:i] + groups[i+1:j]]

    statannot.add_stat_annotation(ax, x='Group', y='data_tubule_mean', data=df, box_pairs=box_pairs,
                                test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    plt.title('Standard deviation of sequence for tubule intensity mean (ERmoxGFP) in fuz-fuz edges', fontsize=18)

    plt.grid(True)
    plt.xlabel('Group', fontsize=20)
    plt.ylabel('Standard deviation value', fontsize=18)

    plt.show()


connections = ['iso-iso', 'iso-fuz', 'fuz-fuz']
channels = ['egfp', 'mch']
groups = ['ATL', 'Climp', 'RTN', 'Control']

a1 = plot_seq_mean_tubule_mean('ATL', 'egfp', 'fuz-fuz')
c1 = plot_seq_mean_tubule_mean('Climp', 'egfp', 'fuz-fuz')
ct1 = plot_seq_mean_tubule_mean('Control', 'egfp', 'fuz-fuz')
r1 = plot_seq_mean_tubule_mean('RTN', 'egfp', 'fuz-fuz')

plotter([a1, c1, ct1, r1])