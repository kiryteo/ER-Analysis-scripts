import sknw
import imageio
import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_graph_data_pickles(group):
    graph_data = []
    group_len = {'ATL': 26, 'Climp': 31, 'RTN': 29, 'Control':31}
    group_prefix = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}
    for ser in range(1, group_len[group]+1):
        ser_data = []
        for frame in range(100):
            skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/{group_prefix[group]}{ser}/{group_prefix[group]}{ser}_decon_t0{frame:02d}_ch00_skel.png')
            graph = sknw.build_sknw(skel, multi=False)

            ser_data.append(graph)
        graph_data.append(ser_data)

    with open(f'{group.lower()}_graph_data.pkl', 'wb') as f:
        pickle.dump(graph_data, f)


def compute_average_degree_connectivity(graph):
    return nx.average_degree_connectivity(graph)


def compute_average_neighbor_degree(graph):
    return nx.average_neighbor_degree(graph)


def compute_average_clustering(graph):
    return nx.average_clustering(graph)


def compute_average_node_connectivity(graph):
    return nx.average_node_connectivity(graph)


def compute_closeness_centrality(graph):
    return nx.closeness_centrality(graph)


def compute_degree_centrality(graph):
    return nx.degree_centrality(graph)


def compute_betweenness_centrality(graph):
    return nx.betweenness_centrality(graph)


def compute_number_of_cliques(graph):
    return sum(1 for _ in nx.find_cliques(graph))


def compute_clique_number(graph):
    return max(len(c) for c in nx.find_cliques(graph))


def compute_transitivity(graph):
    return nx.transitivity(graph)


def compute_num_connected_components(graph):
    return nx.number_connected_components(graph)


def compute_diameter(graph):
    return nx.diameter(graph)


def compute_radius(graph):
    return nx.radius(graph)


def compute_local_efficiency(graph):
    return nx.local_efficiency(graph)


def compute_global_efficiency(graph):
    return nx.global_efficiency(graph)


# def compute_graph_edit_distance(graph1, graph2):
#     return nx.graph_edit_distance(graph1, graph2)


def compute_s_metric(graph):
    return nx.s_metric(graph)


def temporal_graph_analyzer(group, feature_func):
    # get temporal graphs per series per group

    with open(f'{group.lower()}_graph_data.pkl', 'rb') as f:
        graph_data = pickle.load(f)

    # # get temporal graph properties per series per group
    feature_data = []

    graph_data = np.array(graph_data)
    # print(graph_data.shape)
    for ser in range(graph_data.shape[0]):
        ser_data = []
        for frame in range(100):
            dac = feature_func(graph_data[ser, frame])
            ser_data.append(dac)
        feature_data.append(ser_data)

    # mean_data = np.mean(feature_data, axis=0)
    # std_data = np.std(feature_data, axis=0)
    # print(mean_data.shape)
    # print(mean_data)

    return np.array(feature_data).flatten()
    # return mean_data, std_data

# temporal_graph_analyzer('ATL')
# exit()

def get_graph_edits(group):
    with open(f'{group.lower()}_graph_data.pkl', 'rb') as f:
        graph_data = pickle.load(f)

    graph_data = np.array(graph_data)
    graph_edits = []
    for ser in range(graph_data.shape[0]):
        ser_data = []
        for frame in range(1, 100):
            graph_edit = compute_graph_edit_distance(graph_data[ser, frame-1], graph_data[ser, frame])
            ser_data.append(graph_edit)
        graph_edits.append(ser_data)

    return np.array(graph_edits).flatten()


# def plot_features_across_groups(feature_func):
def plot_features_across_groups():
    # atl_mean_data, atl_std_data = temporal_graph_analyzer('ATL')

    # feature = feature_func.__name__[8:]
    # atl_data = temporal_graph_analyzer('ATL', feature_func)
    # ids_atl = []
    # for i in range(1, 101):
    #     ids_atl.extend([i]*26)
    
    # climp_data = temporal_graph_analyzer('Climp', feature_func)
    # ids_climp = []
    # for i in range(1, 101):
    #     ids_climp.extend([i]*31)

    # rtn_data = temporal_graph_analyzer('RTN', feature_func)
    # ids_rtn = []
    # for i in range(1, 101):
    #     ids_rtn.extend([i]*29)

    # control_data = temporal_graph_analyzer('Control', feature_func)
    # ids_control = []
    # for i in range(1, 101):
    #     ids_control.extend([i]*31)

    # atl_mean = np.mean(atl_data)

    # plt.plot(atl_mean_data, label='ATL')
    # plt.fill_between(range(100), atl_mean_data-atl_std_data, atl_mean_data+atl_std_data, alpha=0.5)

    atl_data = get_graph_edits('ATL')
    ids_atl = []
    for i in range(1, 100):
        ids_atl.extend([i]*26)

    climp_data = get_graph_edits('Climp')
    ids_climp = []
    for i in range(1, 100):
        ids_climp.extend([i]*31)

    rtn_data = get_graph_edits('RTN')
    ids_rtn = []
    for i in range(1, 100):
        ids_rtn.extend([i]*29)

    control_data = get_graph_edits('Control')
    ids_control = []
    for i in range(1, 100):
        ids_control.extend([i]*31)


    sns.regplot(x=ids_control, y=control_data, lowess=True, scatter=False, label='Control')
    sns.regplot(x=ids_rtn, y=rtn_data, lowess=True, scatter=False, label='RTN')
    sns.regplot(x=ids_climp, y=climp_data, lowess=True, scatter=False, label='Climp')
    sns.regplot(x=ids_atl, y=atl_data, lowess=True, scatter=False, label='ATL')
    
    plt.xlabel('Time (frames)', fontsize=16)
    # plt.ylabel(f'{feature}', fontsize=16)
    # plt.title(f'Temporal {feature} across groups', fontsize=18)
    plt.ylabel('Graph edit distance', fontsize=16)
    plt.title('Graph edit distance across groups', fontsize=18)
    plt.legend()
    plt.show()

plot_features_across_groups()
exit()

# plot_features_across_groups(compute_avg_degree_connectivity)
# plot_features_across_groups(compute_avg_neighbor_degree)
# plot_features_across_groups(compute_average_clustering)
 
# plot_features_across_groups(compute_avg_node_connectivity)
# plot_features_across_groups(compute_closeness_centrality)
# plot_features_across_groups(compute_degree_centrality)
# plot_features_across_groups(compute_betweenness_centrality)

# transitivity
# node_connectivity



# plot_features_across_groups('average_neighbor_degree')
# plot_features_across_groups('average_clustering')
# plot_features_across_groups('average_node_connectivity')
# plot_features_across_groups('closeness_centrality')
# plot_features_across_groups('degree_centrality')
# plot_features_across_groups('betweenness_centrality')
