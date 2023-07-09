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
plot_measures(nx.average_clustering)
plot_measures(nx.average_node_connectivity)
plot_measures(nx.average_shortest_path_length)
plot_measures(nx.diameter)
plot_measures(nx.eccentricity)
plot_measures(nx.radius)
plot_measures(nx.periphery)
plot_measures(nx.center)
plot_measures(nx.density)
plot_measures(nx.transitivity)
plot_measures(nx.average_neighbor_degree)
plot_measures(nx.edge_connectivity)
plot_measures(nx.node_connectivity)
plot_measures(nx.edge_disjoint_paths)
plot_measures(nx.node_disjoint_paths)
plot_measures(nx.number_connected_components)
plot_measures(nx.node_boundary)
plot_measures(nx.node_clique_number)
plot_measures(nx.node_disjoint_paths)



