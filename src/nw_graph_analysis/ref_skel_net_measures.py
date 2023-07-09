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


def plot_measures(measure):
    climp = get_graph_data('Climp', measure)
    rtn = get_graph_data('RTN', measure)
    control = get_graph_data('Control', measure)
    atl = get_graph_data('ATL', measure)

    sns.distplot(atl, hist=False, rug=True, label='ATL')
    sns.distplot(rtn, hist=False, rug=True, label='RTN')
    sns.distplot(climp, hist=False, rug=True, label='Climp')
    sns.distplot(control, hist=False, rug=True, label='Control')

    measure_name = measure.__name__

    plt.xlabel(measure_name)
    plt.ylabel('Density')

    plt.title(f'{measure_name} distribution')

    plt.legend()
    plt.show()


plot_measures(nx.degree_assortativity_coefficient)
plot_measures(nx.average_degree_connectivity)
plot_measures(nx.degree_pearson_correlation_coefficient)
plot_measures(nx.average_neighbor_degree)



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
