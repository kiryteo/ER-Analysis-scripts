import pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import statannot
import itertools
import matplotlib.pyplot as plt

import os

home = os.path.expanduser('~')
confocal_data_path = '/MIAL/data/confocal_movies'

from tubule_data_plotting import *
from tubule_data_loading import *
from tubule_data_processing import *


def get_group_box_pairs(channel):
    groups = (
        ['ATL', 'Climp', 'RTN']
        if channel == 'mch'
        else ['ATL', 'Climp', 'RTN', 'Control']
    )
    return list(itertools.combinations(groups, 2))

def get_box_pairs(channel):
    """
    Get all pairs of groups across regions for a given channel.

    Args:
        channel (str): The channel name ('mch' or other).

    Returns:
        list: A list of tuples representing pairs of groups across regions.
    """
    # Get group names
    if channel == 'mch':
        groups = ['ATL', 'Climp', 'RTN']
    else:
        groups = ['ATL', 'Climp', 'RTN', 'Control']

    # Get region names
    regions = ['R1', 'R2', 'R3']

    # Get all pairs of groups across regions
    box_pairs = []
    for region in regions:
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                pair = ((region, group1), (region, group2))
                box_pairs.append(pair)

    return box_pairs

def get_correlation_data_per_replicate(data_egfp, data_mch):
    """
    Get the correlation data for each replicate.

    Args:
        data_egfp (list): Data from eGFP.
        data_mch (list): Data from mCherry.

    Returns:
        tuple: Correlation data for each region (R1, R2, R3).
    """
    def calculate_correlation(data1, data2):
        return [np.corrcoef(tub1, tub2)[0][1] for tub1, tub2 in zip(data1, data2)]

    correlation_data_r1 = calculate_correlation(data_egfp[:10], data_mch[:10])
    correlation_data_r2 = calculate_correlation(data_egfp[10:20], data_mch[10:20])
    correlation_data_r3 = calculate_correlation(data_egfp[20:], data_mch[20:])

    return correlation_data_r1, correlation_data_r2, correlation_data_r3

def get_channel_corr(group, connection):
    """
    Get correlation data for a given group and connection.

    Args:
        group (str): Group name.
        connection (str): Connection type.

    Returns:
        list: Correlation data per replicate.
    """
    data_egfp = load_pickle_data(f'{group.lower()}_{connection}.pkl')
    data_mch = load_pickle_data(f'{group.lower()}_{connection}_mch.pkl')

    if data_egfp is not None and data_mch is not None and connection != 'None':
        return get_correlation_data_per_replicate(data_egfp, data_mch)
    return None

def get_per_pixel_variation_over_sequence(group, connection, channel, variation):
    """
    Get the per-pixel variation over sequence for a given group, connection, channel, and variation.

    Args:
        group (str): Group name.
        connection (str): Connection type.
        channel (str): Channel name.
        variation (str): Variation type ('mean' or 'std').

    Returns:
        list: Per-pixel variation over sequence.
    """
    filepath = f'{home}/ER-Analysis-scripts/src/nw_graph_analysis/pickles/{group.lower()}_{connection}_tubules_{channel}.pkl'
    data = load_pickle_data(filepath)
    if data is None:
        return []

    data = filter_data(data)
    variation_vals = []

    for series in data:
        for tubule in series:
            transposed_list = list(map(list, zip(*tubule)))
            for pixel_seq_vals in transposed_list:
                if variation == 'mean':
                    variation_vals.append(np.mean(pixel_seq_vals))
                else:
                    variation_vals.append(np.std(pixel_seq_vals))
    return variation_vals

def get_per_tubule_variation_over_sequence(group, connection, channel, variation):
    """
    Get the per-tubule variation over sequence for a given group, connection, channel, and variation.

    Args:
        group (str): Group name.
        connection (str): Connection type.
        channel (str): Channel name.
        variation (str): Variation type ('mean' or 'std').

    Returns:
        list: Per-tubule variation over sequence.
    """
    filepath = f'{home}/ER-Analysis-scripts/src/nw_graph_analysis/pickles/{group.lower()}_{connection}_tubules_{channel}.pkl'
    data = load_pickle_data(filepath)
    if data is None:
        return []

    data = filter_data(data)
    variation_vals = []

    for series in data:
        for tubule in series:
            frame_data = [np.mean(frame) for frame in tubule]
            if variation == 'mean':
                variation_vals.append(np.mean(frame_data))
            else:
                variation_vals.append(np.std(frame_data))
    return variation_vals

def get_per_tubule_correlation_over_sequence(group, connection, channel):
    """
    Get the per-tubule correlation over sequence for a given group, connection, and channel.

    Args:
        group (str): Group name.
        connection (str): Connection type.
        channel (str): Channel name.

    Returns:
        list: Per-tubule correlation over sequence.
    """
    filepath = f'{home}/ER-Analysis-scripts/src/nw_graph_analysis/pickles/{group.lower()}_{connection}_tubules_{channel}.pkl'
    data = load_pickle_data(filepath)
    if data is None:
        return []

    data = filter_data(data)
    correlation_vals = []

    for series in data:
        for tubule in series:
            frame_data = [np.mean(frame) for frame in tubule]
            correlation_vals.append(frame_data)
    return correlation_vals

def load_data(group, connection, channel=None, variation=None):
    if channel and variation:
        return get_per_tubule_variation_over_sequence(group, connection, channel, variation)
    return get_per_tubule_correlation_over_sequence(group, connection, channel)

def create_dataframe(groups, connection, channel=None, variation=None, data_type='correlation'):
    data = []
    labels = []
    for group in groups:
        group_data = load_data(group, connection, channel, variation)
        data.extend(group_data)
        labels.extend([group] * len(group_data))
    
    column_name = 'Per-tubule-correlation' if data_type == 'correlation' else 'Tubule-mean'
    return pd.DataFrame({column_name: data, 'Group': labels})

def plot_per_tubule_correlation(connection):
    groups = ['ATL', 'Climp', 'RTN']
    df = create_dataframe(groups, connection)
    colors = sns.color_palette(n_colors=4)
    palette = {'Reticulon': colors[1], 'Climp': colors[2], 'Atlastin': colors[3]}
    plot_data(df, 'Group', 'Per-tubule-correlation', palette, (-0.29, 0.75), (15, 15), 'Group', 'Cross-correlation over sequence', 'Per-tubule correlation over sequence', f'Seq_Correlation_Tubule_mean_{connection}_v3.png')

def plot_per_tubule_variation_over_sequence(connection, channel, variation):
    groups = ['Control', 'ATL', 'Climp', 'RTN'] if channel == 'egfp' else ['ATL', 'Climp', 'RTN']
    df = create_dataframe(groups, connection, channel, variation, data_type='variation')
    plot_data(df, 'Group', 'Tubule-mean', None, (0, 0.13), (13, 13), 'Group', f'{variation} over sequence', f'{variation} over sequence', f'Seq_{variation}_Tubule_mean_{connection}_{channel}_v4.png')

def get_per_pixel_correlation_over_sequence(group, connection):
    egfp_data = pkl.load(open(f'{home}/ER-Analysis-scripts/src/nw_graph_analysis/{group.lower()}_{connection}_tubules_egfp.pkl', 'rb'))
    mch_data = pkl.load(open(f'{home}/ER-Analysis-scripts/src/nw_graph_analysis/{group.lower()}_{connection}_tubules_mch.pkl', 'rb'))

    egfp_data = filter_data(egfp_data)
    mch_data = filter_data(mch_data)

    correlation_vals = []
    for egfp, mch in zip(egfp_data, mch_data):
        for egfp_tubule, mch_tubule in zip(egfp, mch):
            for egfp_pixel_seq_vals, mch_pixel_seq_vals in zip(zip(*egfp_tubule), zip(*mch_tubule)):
                correlation_vals.append(np.corrcoef(egfp_pixel_seq_vals, mch_pixel_seq_vals)[0][1])
    
    return correlation_vals

def load_data(group, connection, suffix):
    file_path = f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/{group.lower()}_{connection}_tubules_{suffix}.pkl'
    return pkl.load(open(file_path, 'rb'))

def per_pixel_correlation(egfp_data, mch_data):
    egfp = filter_data(egfp_data)
    mch = filter_data(mch_data)
    corr_vals = []

    for s1, s2 in zip(egfp, mch):
        for t1, t2 in zip(s1, s2):
            transposed_egfp = list(map(list, zip(*t1)))
            transposed_mch = list(map(list, zip(*t2)))
            corr_vals.extend(np.corrcoef(e1, e2)[0, 1] for e1, e2 in zip(transposed_egfp, transposed_mch))

    return corr_vals

def get_group_correlation_data(group, connection):
    egfp = load_data(group, connection, 'egfp')
    mch = load_data(group, connection, 'mch')
    return egfp, mch

def get_replicate_data(egfp, mch):
    replicates = [(egfp[i:i+10], mch[i:i+10]) for i in range(0, len(egfp), 10)]
    return [per_pixel_correlation(e, m) for e, m in replicates]

def plot_data(df, x, y, hue=None, box_pairs=None, title='', xlabel='', ylabel=''):
    ax = sns.boxenplot(data=df, x=x, y=y, hue=hue, dodge=True if hue else False)
    if box_pairs:
        add_stat_annotation(ax, x=x, y=y, hue=hue, data=df, box_pairs=box_pairs, test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    plt.grid(True)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=20)
    plt.show()

def plot_per_pixel_correlation(connection, plottype):
    groups = ['ATL', 'Climp', 'RTN']
    data = {group: get_group_correlation_data(group, connection) for group in groups}
    df = pd.DataFrame()

    if plottype == 'all':
        per_pixel_data = np.concatenate([per_pixel_correlation(*data[group]) for group in groups])
        group_labels = np.concatenate([[group] * len(per_pixel_correlation(*data[group])) for group in groups])
        df['Per-pixel-variation'] = pd.Series(per_pixel_data)
        df['Group'] = pd.Series(group_labels)
        box_pairs = get_group_box_pairs('mch')
        plot_data(df, 'Group', 'Per-pixel-variation', box_pairs=box_pairs, title=f'Per pixel cross-correlation over 100 frames between ERmoxGFP and mCherry in each {connection} connection tubule', xlabel='Group', ylabel='Cross-correlation values')
    else:
        replicate_data = {group: get_replicate_data(*data[group]) for group in groups}
        per_pixel_data = np.concatenate([np.concatenate(replicate_data[group]) for group in groups])
        group_labels = np.concatenate([[group] * len(np.concatenate(replicate_data[group])) for group in groups])
        replicate_labels = np.concatenate([['R1', 'R2', 'R3'] * len(replicate_data[group][0]) for group in groups])
        df['Per-pixel-variation'] = pd.Series(per_pixel_data)
        df['Group'] = pd.Series(group_labels)
        df['Replicate'] = pd.Series(replicate_labels)
        box_pairs = get_box_pairs('mch')
        plot_data(df, 'Replicate', 'Per-pixel-variation', hue='Group', box_pairs=box_pairs, title=f'Per pixel cross-correlation over 100 frames between ERmoxGFP and mCherry in each {connection} connection tubule', xlabel='Replicates', ylabel='Cross-correlation values')

def plot_per_pixel_variation(connection, channel, variation, plottype):
    groups = ['ATL', 'Climp', 'RTN']
    if channel == 'egfp':
        groups.append('Control')
    data = {group: get_per_pixel_variation_over_sequence(group, connection, channel, variation) for group in groups}
    df = pd.DataFrame()

    if plottype == 'all':
        per_pixel_data = np.concatenate([data[group] for group in groups])
        group_labels = np.concatenate([[group] * len(data[group]) for group in groups])
        df['Per-pixel-variation'] = pd.Series(per_pixel_data)
        df['Group'] = pd.Series(group_labels)
        box_pairs = get_group_box_pairs(channel)
        plot_data(df, 'Group', 'Per-pixel-variation', box_pairs=box_pairs, title=f'Per-pixel {variation} over sequence for {connection} tubules in {channel}', xlabel='Group', ylabel=f'Per pixel {variation} over 100 frames in each tubule')
    else:
        replicate_data = {group: [data[group][i:i+10] for i in range(0, len(data[group]), 10)] for group in groups}
        per_pixel_data = np.concatenate([np.concatenate(replicate_data[group]) for group in groups])
        group_labels = np.concatenate([[group] * len(np.concatenate(replicate_data[group])) for group in groups])
        replicate_labels = np.concatenate([['R1', 'R2', 'R3'] * len(replicate_data[group][0]) for group in groups])
        df['Per-pixel-variation'] = pd.Series(per_pixel_data)
        df['Group'] = pd.Series(group_labels)
        df['Replicate'] = pd.Series(replicate_labels)
        box_pairs = get_box_pairs(channel)
        plot_data(df, 'Replicate', 'Per-pixel-variation', hue='Group', box_pairs=box_pairs, title=f'Per-pixel {variation} over sequence for {connection} tubules in {channel}', xlabel='Replicates', ylabel=f'Per pixel {variation} over 100 frames in each tubule')


def get_egfp_plots(connection, channel, variation):
    atl = get_pickle_data('ATL', connection, 'tub_mean', channel)
    climp = get_pickle_data('Climp', connection, 'tub_mean', channel)
    rtn = get_pickle_data('RTN', connection, 'tub_mean', channel)

    ar1, ar2, ar3 = get_variation(atl, variation)
    cr1, cr2, cr3 = get_variation(climp, variation)
    rr1, rr2, rr3 = get_variation(rtn, variation)

    data = {
        'tub-mean': np.concatenate((ar1, ar2, ar3, cr1, cr2, cr3, rr1, rr2, rr3)),
        'Group': np.concatenate((
            ['ATL'] * len(ar1) + ['ATL'] * len(ar2) + ['ATL'] * len(ar3) +
            ['Climp'] * len(cr1) + ['Climp'] * len(cr2) + ['Climp'] * len(cr3) +
            ['RTN'] * len(rr1) + ['RTN'] * len(rr2) + ['RTN'] * len(rr3))),
        'Replicate': np.concatenate((
            ['R1'] * len(ar1) + ['R2'] * len(ar2) + ['R3'] * len(ar3) +
            ['R1'] * len(cr1) + ['R2'] * len(cr2) + ['R3'] * len(cr3) +
            ['R1'] * len(rr1) + ['R2'] * len(rr2) + ['R3'] * len(rr3)))
    }

    if channel == 'egfp':
        ctrl = get_pickle_data('Control', connection, 'tub_mean', channel)
        ctr1, ctr2, ctr3 = get_variation(ctrl, variation)
        data['tub-mean'] = np.concatenate((data['tub-mean'], ctr1, ctr2, ctr3))
        data['Group'] = np.concatenate((data['Group'],
            ['Control'] * len(ctr1) + ['Control'] * len(ctr2) + ['Control'] * len(ctr3)))
        data['Replicate'] = np.concatenate((data['Replicate'],
            ['R1'] * len(ctr1) + ['R2'] * len(ctr2) + ['R3'] * len(ctr3)))

    df = pd.DataFrame(data)

    ax = sns.boxenplot(data=df, x='Replicate', y='tub-mean', hue='Group', dodge=True)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

    box_pairs = get_box_pairs(channel)
    statannot.add_stat_annotation(ax, x='Replicate', y='tub-mean', hue='Group', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    ch_name = 'ERmoxGFP' if channel == 'egfp' else 'mCherry'
    plt.title(f'{variation.capitalize()} of sequence for tubule intensity mean ({ch_name}) in {connection} edges', fontsize=18)
    plt.grid(True)
    plt.xlabel('Replicate', fontsize=18)
    plt.ylabel(f'Tubular {variation}', fontsize=18)
    plt.show()

def get_length_per_tubule_variation(data, measure):
    data = filter_data(data)
    if measure == 'var':
        return [np.var([len(tubule_seq[0]) for tubule_seq in series if len(tubule_seq[0]) > 3]) for series in data]
    else:
        return [np.mean([len(tubule_seq[0]) for tubule_seq in series if len(tubule_seq[0]) > 3]) for series in data]

def load_and_get_variation(group, connection, channel, measure):
    conn = load_annot_tub_pickles(group, connection, channel)
    return get_length_per_tubule_variation(conn, measure)

def plot_tubule_length_distribution(connection, channel, measure):
    groups = ['Control', 'RTN', 'Climp', 'ATL']
    variations = [load_and_get_variation(group, connection, channel, measure) for group in groups]

    df = pd.DataFrame({
        'tub-length': np.concatenate(variations),
        'Group': np.concatenate([[group] * len(var) for group, var in zip(groups, variations)])
    })

    ax = sns.boxplot(data=df, x='Group', y='tub-length', showfliers=False, width=0.9)
    ax.set_xlim(-1, 4.0)
    ax.set_ylim(0, 81)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, rotation=90)
    ax.set_yticklabels([f'{y}' for y in ax.get_yticks()], fontsize=13)

    measure_name = 'variance' if measure == 'var' else 'mean (pixels)'
    conn_name = {
        'iso-iso': 'isolated-isolated',
        'iso-fuz': 'isolated-overlapping'
    }.get(connection, 'overlapping-overlapping')

    plt.grid(True)
    plt.xlabel('Group', fontsize=15)
    plt.ylabel(f'Tubule length {measure_name}', fontsize=15)
    plt.gcf().set_size_inches(2.2, 6)
    plt.savefig(f'Seq_{measure}_tub_length_{connection}_v2.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def load_and_get_lengths(group, connection, channel):
    conn = load_annot_tub_pickles(group, connection, channel)
    return get_length_per_tubule(conn)

def plot_tubule_length_distribution_all():
    groups = ['ATL', 'Climp', 'RTN', 'Control']
    lengths = [load_and_get_lengths(group, 'iso-iso', 'egfp') for group in groups]

    df = pd.DataFrame({
        'tub-length': np.concatenate(lengths),
        'Group': np.concatenate([[group] * len(length) for group, length in zip(groups, lengths)])
    })

    ax = sns.boxenplot(data=df, x='Group', y='tub-length', showfliers=False, linewidth=2)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    ax.set_yticklabels([f'{y:.2f}' for y in ax.get_yticks()], fontsize=18)

    plt.title('Tubule length in all tubules across groups', fontsize=24)
    plt.grid(True)
    plt.xlabel('Group', fontsize=24)
    plt.ylabel('Tubule length (pixels)', fontsize=24)
    plt.show()


def get_length_per_tubule(data):
    data = filter_data(data)
    return [len(tubule[0]) for series in data for tubule in series if len(tubule[0]) > 3]

def get_edge_length(group, connection, measure, channel):
    tubule_data = get_pickle_data(group, connection, measure, channel)
    r1_length = get_length_per_tubule(tubule_data[:10])
    r2_length = get_length_per_tubule(tubule_data[10:20])
    r3_length = get_length_per_tubule(tubule_data[20:])
    return r1_length, r2_length, r3_length


def get_combined_edge_length(group, conditions, structure, marker):
    combined_lengths = [0, 0, 0]
    for condition in conditions:
        lengths = get_edge_length(group, condition, structure, marker)
        combined_lengths = [sum(x) for x in zip(combined_lengths, lengths)]
    return combined_lengths

def run_edge_length_analysis():
    conditions = ['iso-iso', 'iso-fuz', 'fuz-fuz']
    structure = 'tubules'
    marker = 'egfp'
    groups = ['ATL', 'Climp', 'RTN', 'Control']
    return [get_combined_edge_length(group, conditions, structure, marker) for group in groups]

def prepare_dataframe(lengths, groups, plottype):
    if plottype == 'group':
        combined_lengths = [sum(length) for length in zip(*lengths)]
        df = pd.DataFrame({
            'tub-mean': np.concatenate(combined_lengths),
            'Group': np.concatenate([[group] * len(length) for group, length in zip(groups, combined_lengths)])
        })
    else:
        df = pd.DataFrame({
            'tub-mean': np.concatenate([length for sublist in lengths for length in sublist]),
            'Group': np.concatenate([[group] * len(length) for group, sublist in zip(groups, lengths) for length in sublist]),
            'Replicate': np.concatenate([['R1'] * len(lengths[0][0]), ['R2'] * len(lengths[0][1]), ['R3'] * len(lengths[0][2]),
                                          ['R1'] * len(lengths[1][0]), ['R2'] * len(lengths[1][1]), ['R3'] * len(lengths[1][2]),
                                          ['R1'] * len(lengths[2][0]), ['R2'] * len(lengths[2][1]), ['R3'] * len(lengths[2][2]),
                                          ['R1'] * len(lengths[3][0]), ['R2'] * len(lengths[3][1]), ['R3'] * len(lengths[3][2])])
        })
    return df

def plot_data(df, connection, plottype):
    if plottype == 'group':
        ax = sns.boxenplot(data=df, x='Group', y='tub-mean')
        box_pairs = get_group_box_pairs('egfp')
        statannot.add_stat_annotation(ax, x='Group', y='tub-mean', data=df, box_pairs=box_pairs,
                                      test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')
        plt.xlabel('Group', fontsize=18)
    else:
        ax = sns.boxenplot(data=df, x='Replicate', y='tub-mean', hue='Group', dodge=True)
        box_pairs = get_box_pairs('egfp')
        statannot.add_stat_annotation(ax, x='Replicate', y='tub-mean', hue='Group', data=df, box_pairs=box_pairs,
                                      test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')
        plt.xlabel('Replicates', fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    plt.title(f'Tubule length in {connection} connections for all sequences', fontsize=18)
    plt.grid(True)
    plt.ylabel('Tubule length (pixels)', fontsize=18)
    plt.show()

def compare_groups(connection, plottype):
    groups = ['ATL', 'Climp', 'RTN', 'Control']
    lengths = run_edge_length_analysis()
    df = prepare_dataframe(lengths, groups, plottype)
    plot_data(df, connection, plottype)