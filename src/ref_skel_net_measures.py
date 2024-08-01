import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sknw
import imageio
import seaborn as sns
import pandas as pd
import statannot
from junction_analysis_modules import JunctionAnalysis as JA

confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'
junc_analysis = JA(confocal_data_path)

seq_per_group = {'ATL': 26, 'RTN': 29, 'Climp': 30, 'Control': 30}

def read_image(group, seq_num):
    """Read the image for a given group and sequence number."""
    return imageio.imread(f'{confocal_data_path}ref_skels/ref_skels_{group.lower()}/{group.lower()}{seq_num}_proc_skel.png')

def build_graph_from_image(img):
    """Convert an image to a graph using sknw."""
    return sknw.build_sknw(img, multi=False)

def get_graph_data(group, measure):
    """
    Get graph data for a specific group and measure.

    Args:
        group (str): The group name.
        measure (function): The measure function to apply to the graph.

    Returns:
        list: A list of measure values for each sequence.
    """
    data = []
    num_seq = seq_per_group[group]

    for i in range(1, num_seq + 1):
        img = read_image(group, i)
        g = build_graph_from_image(img)
        dac = measure(g)
        data.append(dac)

    return data

def get_graph_measure_data(group):
    """
    Get graph measure data for a specific group.

    Args:
        group (str): The group name.

    Returns:
        list: A list of density values for each sequence.
    """
    data = []
    num_seq = seq_per_group[group]

    for i in range(1, num_seq + 1):
        img = read_image(group, i)
        G = build_graph_from_image(img)
        density_data = nx.density(G)
        data.append(density_data)

    return data

def get_graph_data_nodes(data):
    """
    Extract node values from graph data.

    Args:
        data (list): List of graph data.

    Returns:
        list: List of node values.
    """
    nodes_vals = []
    for seq in data:
        nodes_vals.extend(v for k, v in seq.items())
    return nodes_vals

def get_iso_fuz_nodes_data(group, measure, region):
    """
    Get isolated or fuzzy nodes data for a specific group, measure, and region.

    Args:
        group (str): The group name.
        measure (function): The measure function to apply to the graph.
        region (str): The region type ('iso' or 'fuz').

    Returns:
        list: A list of measure values for the specified region.
    """
    data = []
    num_seq = seq_per_group[group]

    for ser_num in range(1, num_seq + 1):
        nps, skdata, labelled_img = junc_analysis.label_junctions(group, ser_num)
        label_vals, unassigned_cc_dict = junc_analysis.separate_junc_cc(nps, skdata, labelled_img)
        iso, fuz, unk = junc_analysis.get_junction_areas(label_vals, unassigned_cc_dict)

        reg_data = [[x[0], x[1]] for x in (iso if region == 'iso' else fuz)]
        img = read_image(group, ser_num)
        G = build_graph_from_image(img)
        nodes = G.nodes()
        ps = [nodes[i]['o'] for i in nodes]
        pse = [[x[0], x[1]] for x in ps]
        d = measure(G)
        ids = [pse.index(x) for x in reg_data if x in pse]
        data.extend([d[i] for i in ids])

    return data

def plot_iso_fuz_graph_measures(measure, region):
    """
    Plot isolated or fuzzy graph measures.

    Args:
        measure (function): The measure function to apply to the graph.
        region (str): The region type ('iso' or 'fuz').
    """
    atl = get_iso_fuz_nodes_data('ATL', measure, region)
    rtn = get_iso_fuz_nodes_data('RTN', measure, region)
    climp = get_iso_fuz_nodes_data('Climp', measure, region)
    control = get_iso_fuz_nodes_data('Control', measure, region)
    measure_name = measure.__name__
    df = pd.DataFrame()
    df[measure_name] = pd.Series(np.concatenate((control, rtn, climp, atl)))
    df['Group'] = pd.Series(np.concatenate((['Control']*len(control), ['Reticulon']*len(rtn), ['Climp']*len(climp), ['Atlastin']*len(atl))))
    ax = sns.boxplot(x='Group', y=measure_name, data=df, showfliers=False)
    box_pairs = [('Atlastin', 'Climp'), ('Atlastin', 'Reticulon'), ('Atlastin', 'Control'), ('Climp', 'Reticulon'), ('Climp', 'Control'), ('Reticulon', 'Control')]
    statannot.add_stat_annotation(ax, x='Group', y=measure_name, data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='star', loc='inside', verbose=2, fontsize=15)
    ax.set_xlim(-1, 4.0)
    yt = ax.get_yticks()
    yt = [f'{y:.2f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16, rotation=90)
    ax.grid(axis='y')
    plt.ylabel(measure_name, fontsize=18)
    plt.xlabel('Group', fontsize=18)
    plt.gcf().set_size_inches(2.2, 7)
    plt.savefig(f'{measure_name}_{region}.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def get_iso_fuz_nodes_ids(group):
    """
    Get isolated or fuzzy nodes IDs for a specific group.

    Args:
        group (str): The group name.
    """
    num_seq = seq_per_group[group]

    for ser_num in range(1, num_seq + 1):
        np, skdata, labelled_img = junc_analysis.label_junctions(group, ser_num)
        label_vals, unassigned_cc_dict = junc_analysis.separate_junc_cc(np, skdata, labelled_img)
        iso, fuz, unk = junc_analysis.get_junction_areas(label_vals, unassigned_cc_dict)
        img = read_image(group, ser_num)

def plot_measures(measure):
    """
    Plot measures for all groups.

    Args:
        measure (function): The measure function to apply to the graph.
    """
    climp_data = get_graph_data('Climp', measure)
    rtn_data = get_graph_data('RTN', measure)
    control_data = get_graph_data('Control', measure)
    atl_data = get_graph_data('ATL', measure)
    measure_name = measure.__name__
    df = pd.DataFrame()
    df[measure_name] = pd.Series(np.concatenate((atl_data, rtn_data, climp_data, control_data)))
    df['Group'] = pd.Series(np.concatenate((['Control']*len(control_data), ['Reticulon']*len(rtn_data), ['Climp']*len(climp_data), ['Atlastin']*len(atl_data))))
    ax = sns.boxplot(x='Group', y=measure_name, data=df, showfliers=False)
    box_pairs = [('Atlastin', 'Climp'), ('Atlastin', 'Reticulon'), ('Atlastin', 'Control'), ('Climp', 'Reticulon'), ('Climp', 'Control'), ('Reticulon', 'Control')]
    statannot.add_stat_annotation(ax, x='Group', y=measure_name, data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='star', loc='inside', verbose=2, fontsize=15)
    ax.set_xlim(-1, 4.0)
    yt = ax.get_yticks()
    yt = [f'{y:.2f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, rotation=90)
    ax.grid(axis='y')
    plt.xlabel(measure_name, fontsize=14)
    plt.ylabel('Data distribution', fontsize=14)
    plt.gcf().set_size_inches(2.2, 7)
    plt.savefig(f'{measure_name}_box.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()