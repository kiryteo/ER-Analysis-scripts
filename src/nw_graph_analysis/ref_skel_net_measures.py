import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sknw
import imageio
import seaborn as sns

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


def get_graph_data_nodes(data):
    nodes_vals = []
    for seq in data:
        nodes_vals.extend(v for k, v in seq.items())
    return nodes_vals


def plot_measures(measure):
    climp = get_graph_data('Climp', measure)
    rtn = get_graph_data('RTN', measure)
    control = get_graph_data('Control', measure)
    atl = get_graph_data('ATL', measure)

    atl_data = get_graph_data_nodes(atl)
    rtn_data = get_graph_data_nodes(rtn)
    climp_data = get_graph_data_nodes(climp)
    control_data = get_graph_data_nodes(control)

    measure_name = measure.__name__

    df = pd.DataFrame()
    df[measure_name] = pd.Series(np.concatenate((atl_data, rtn_data, climp_data, control_data)))

    df['Group'] = pd.Series(np.concatenate(['Atlastin']*len(atl_data), ['Reticulon']*len(rtn_data), ['Climp']*len(climp_data), ['Control']*len(control_data)))

    sns.boxplot(x='Group', y=measure_name, data=df, showfliers=False)

    box_pairs = [('Atlastin', 'Climp'), ('Atlastin', 'Reticulon'), ('Atlastin', 'Control'), ('Climp', 'Reticulon'), ('Climp', 'Control'), ('Reticulon', 'Control')]
    # box_pairs = [('Atlastin', 'Climp'), ('Atlastin', 'Reticulon'), ('Climp', 'Reticulon')]
    # # box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('Climp', 'RTN')]

    # statannot.add_stat_annotation(ax, x='Group', y='CC_mean', data=df, box_pairs=box_pairs,
                                #   test='Mann-Whitney', text_format='star', loc='inside', verbose=2, fontsize=15)

    

    plt.xlabel(measure_name, fontsize=14)
    plt.ylabel('Density', fontsize=14)

    plt.title(f'{measure_name} distribution', fontsize=18)

    plt.legend()
    plt.show()


# plot_measures(nx.degree_assortativity_coefficient)
# plot_measures(nx.average_clustering) # float
# plot_measures(nx.degree_pearson_correlation_coefficient)
# plot_measures(nx.local_efficiency) # float
# plot_measures(nx.global_efficiency) # float

# plot_measures(nx.closeness_centrality) # nodes dict
# plot_measures(nx.degree_centrality) # nodes dict
# plot_measures(nx.betweenness_centrality) # nodes dict

plot_measures(nx.average_degree_connectivity) # nodes dict
plot_measures(nx.eccentricity)  # nodes dict



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
