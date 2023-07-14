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


seq_per_group = {'ATL': 26, 'RTN': 29, 'Climp':30, 'Control':30}

def get_graph_data(group, measure):
    data = []

    num_seq = seq_per_group[group]

    for i in range(1, num_seq+1):
        img = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ref_skels/ref_skels_{group.lower()}/{group.lower()}{i}_proc_skel.png')

        # convert to graph
        g = sknw.build_sknw(img, multi=False)
        dac = measure(g)
        # dac = nx.degree_assortativity_coefficient(g)
        data.append(dac)

    return data


def get_graph_measure_data(group):
    data = []

    num_seq = seq_per_group[group]

    for i in range(1, num_seq+1):
        img = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ref_skels/ref_skels_{group.lower()}/{group.lower()}{i}_proc_skel.png')

        G = sknw.build_sknw(img, multi=False)

        # degree_data = [d for n, d in G.degree()]

        # avg degree
        # data.append(np.mean(degree_data))

        # std degree
        # data.append(np.std(degree_data))

        # median degree
        # data.append(np.median(degree_data))

        density_data = nx.density(G)
        data.append(density_data)

    return data


def get_graph_data_nodes(data):
    nodes_vals = []
    for seq in data:
        nodes_vals.extend(v for k, v in seq.items())
    return nodes_vals


def get_iso_fuz_nodes_data(group, measure, region):

    data = []
    num_seq = seq_per_group[group]

    for ser_num in range(1, num_seq+1):
        nps, skdata, labelled_img = junc_analysis.label_junctions(group, ser_num)

        label_vals, unassigned_cc_dict = junc_analysis.separate_junc_cc(nps, skdata, labelled_img)
        iso, fuz, unk = junc_analysis.get_junction_areas(label_vals, unassigned_cc_dict)

        # print(iso)

        if region == 'iso':
            reg_data = [[x[0], x[1]] for x in iso]
        else:
            reg_data = [[x[0], x[1]] for x in fuz]

        img = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ref_skels/ref_skels_{group.lower()}/{group.lower()}{ser_num}_proc_skel.png')

        G = sknw.build_sknw(img, multi=False)

        nodes = G.nodes()
        # ps = np.array([nodes[i]['o'] for i in nodes])
        ps = [nodes[i]['o'] for i in nodes]

        pse = [[x[0], x[1]] for x in ps]

        d = measure(G)
        ids = [pse.index(x) for x in reg_data if x in pse]

        data.extend([d[i] for i in ids])

        return data


# atl = get_iso_fuz_nodes_data('ATL', nx.clustering, 'iso')
# print(atl)
# exit()

# plot_measures(nx.closeness_centrality) # nodes dict
# plot_measures(nx.degree_centrality) # nodes dict
# plot_measures(nx.betweenness_centrality) # nodes dict

# plot_measures(nx.average_degree_connectivity)

def plot_iso_fuz_graph_measures(measure, region):
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
    # plt.show()

    # exit()
    plt.gcf().set_size_inches(2.2, 7)

    # plt.title(f'{measure_name} distribution', fontsize=18)

    plt.savefig(
        f'{measure_name}_{region}.png', bbox_inches='tight', pad_inches=0.1
    )
    plt.close()

# sns.distplot(atl, hist=False, rug=True, label='ATL')
# sns.distplot(rtn, hist=False, rug=True, label='RTN')
# sns.distplot(climp, hist=False, rug=True, label='Climp')
# sns.distplot(control, hist=False, rug=True, label='Control')

# plt.xlabel('Closeness centrality')
# plt.ylabel('Data distribution')
# plt.title('Closeness centrality distribution')
# plt.legend()
# plt.show()
plot_iso_fuz_graph_measures(nx.closeness_centrality, 'iso')
plot_iso_fuz_graph_measures(nx.closeness_centrality, 'fuz')

plot_iso_fuz_graph_measures(nx.degree_centrality, 'iso')
plot_iso_fuz_graph_measures(nx.degree_centrality, 'fuz')

plot_iso_fuz_graph_measures(nx.betweenness_centrality, 'iso')
plot_iso_fuz_graph_measures(nx.betweenness_centrality, 'fuz')

# add percolation_centrality


exit()


def get_iso_fuz_nodes_ids(group):

    num_seq = seq_per_group[group]

    for ser_num in range(1, num_seq+1):
        np, skdata, labelled_img = junc_analysis.label_junctions(group, ser_num)

        label_vals, unassigned_cc_dict = junc_analysis.separate_junc_cc(np, skdata, labelled_img)

        iso, fuz, unk = junc_analysis.get_junction_areas(label_vals, unassigned_cc_dict)



        img = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ref_skels/ref_skels_{group.lower()}/{group.lower()}{ser_num}_proc_skel.png')




# get_iso_fuz_nodes_data('ATL')
# exit()


def plot_measures(measure):
    climp_data = get_graph_data('Climp', measure)
    rtn_data = get_graph_data('RTN', measure)
    control_data = get_graph_data('Control', measure)
    atl_data = get_graph_data('ATL', measure)

    # climp_data = get_graph_measure_data('Climp')
    # rtn_data = get_graph_measure_data('RTN')
    # control_data = get_graph_measure_data('Control')
    # atl_data = get_graph_measure_data('ATL')

    # print(climp_data)
    # exit()

    # atl_data = get_graph_data_nodes(atl_data)
    # rtn_data = get_graph_data_nodes(rtn_data)
    # climp_data = get_graph_data_nodes(climp_data)
    # control_data = get_graph_data_nodes(control_data)

    # measure_name = 'Graph density'
    measure_name = measure.__name__

    df = pd.DataFrame()
    df[measure_name] = pd.Series(np.concatenate((atl_data, rtn_data, climp_data, control_data)))

    df['Group'] = pd.Series(np.concatenate((['Control']*len(control_data), ['Reticulon']*len(rtn_data), ['Climp']*len(climp_data), ['Atlastin']*len(atl_data))))

    ax = sns.boxplot(x='Group', y=measure_name, data=df, showfliers=False)

    box_pairs = [('Atlastin', 'Climp'), ('Atlastin', 'Reticulon'), ('Atlastin', 'Control'), ('Climp', 'Reticulon'), ('Climp', 'Control'), ('Reticulon', 'Control')]
    # box_pairs = [('Atlastin', 'Climp'), ('Atlastin', 'Reticulon'), ('Climp', 'Reticulon')]
    # # box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('Climp', 'RTN')]

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

    # plt.title(f'{measure_name} distribution', fontsize=18)

    plt.savefig(f'{measure_name}_box.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # plt.legend()
    # plt.show()

# plot_measures()
# exit()

# plot_measures(nx.degree_assortativity_coefficient)
# plot_measures(nx.average_clustering) # float
# plot_measures(nx.degree_pearson_correlation_coefficient)
# plot_measures(nx.local_efficiency) # float
# plot_measures(nx.global_efficiency) # float

plot_measures(nx.closeness_centrality) # nodes dict
# plot_measures(nx.degree_centrality) # nodes dict
# plot_measures(nx.betweenness_centrality) # nodes dict

# plot_measures(nx.average_degree_connectivity) # nodes dict
 
# plot_measures(nx.eccentricity)  # nodes dict



### no differences for the following measures

# plot_measures(nx.average_degree_connectivity) # nodes dict
# plot_measures(nx.average_neighbor_degree) # nodes dict
# plot_measures(nx.laplacian_centrality) # nodes dict
# plot_measures(nx.clustering) # nodes dict


# plot_measures(nx.constraint)    # nodes dict
# plot_measures(nx.effective_size)    # nodes dict
# plot_measures(nx.closeness_vitality)   # nodes dict


# no application yet

# plot_measures(nx.voterank) # list of voterank scores
# plot_measures(nx.dominating_set) # dominating set

# plot_measures(nx.flow_hierarchy)   # float needs digraph


exit()

#plot_measures(nx.eigenvector_centrality)
#plot_measures(nx.closeness_centrality)
#plot_measures(nx.information_centrality)
plot_measures(nx.group_betweenness_centrality)
plot_measures(nx.group_closeness_centrality)
plot_measures(nx.group_degree_centrality)
plot_measures(nx.load_centrality)
plot_measures(nx.global_reaching_centrality)


#plot_measures(nx.prominent_group)


#plot_measures(nx.subgraph_centrality) # nodes dict
#plot_measures(nx.harmonic_centrality) # nodes dict
