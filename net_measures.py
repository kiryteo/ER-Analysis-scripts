import networkx as nx
import skimage
import imageio
import sknw
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def create_graphs():
    for i in range(100):
        ske = imageio.imread(
            '/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A9/A9_decon_t0%s_ch00_skel.png' % f'{i:02d}')
        grph = sknw.build_sknw(ske)
        G = nx.Graph()

        G.add_edges_from(grph.edges)
        options = {
            'node_size': 20,
        }
        nx.draw(G, **options)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        # plt.show()
        plt.title('ATL_S9 - frame number: %s' % f'{i:02d}')
        plt.savefig(
            '/localhome/asa420/MIAL/data/confocal_movies/ATL/ATL_S9_graphs/A9_decon_t0%s_ch00_graph.png' % f'{i:02d}',
            bbox_inches='tight', pad_inches=0)
        plt.close()


# max_degree_list = []


def get_nx_graph(group, series_num):
    """
    @param group: Condition for analysis.
    @param series_num: sequences number to analyze
    @return: Dict with graphs for all the samples from the sequence.
    """
    graph_dict = {}

    pref = '/localhome/asa420/MIAL/data/confocal_movies/'
    if group == 'ATL':
        prefix = pref + 'ATL/new_op_jul/skel/'
        series_init = 'A'
    elif group == 'Climp':
        prefix = pref + 'Climp/new_op_jul/skel/'
        series_init = 'C'
    elif group == 'Control':
        prefix = pref + 'Control/new_op_jul/skel/'
        series_init = 'Ct'
    else:
        prefix = pref + 'RTN/new_op_jul/skel/'
        series_init = 'R'

    for i in range(100):
        ske = imageio.imread(prefix + '%s/%s_decon_t0%s_ch00_skel.png' % (
        f'{series_init}{series_num}', f'{series_init}{series_num}', f'{i:02d}'))
        grph = sknw.build_sknw(ske)
        G = nx.Graph()

        G.add_nodes_from(grph.nodes)
        G.add_edges_from(grph.edges)
        graph_dict[i] = G

    return graph_dict


def get_features_nodes(group, series_num):
    dt = get_nx_graph(group, series_num)
    nodes_list = []
    for idx, graph in dt.items():
        nodes_list.append(len(graph.nodes))
    return nodes_list


class GetGraphFeatures:
    def __init__(self):
        pass

    @staticmethod
    def get_features_edges(group, series_num):
        dt = get_nx_graph(group, series_num)
        edges_list = []
        for idx, graph in dt.items():
            edges_list.append(len(graph.edges))
        return edges_list

    @staticmethod
    def get_features_avg_degree(group, series_num):
        dt = get_nx_graph(group, series_num)
        avg_deg_list = []
        for idx, graph in dt.items():
            deg = graph.degree()
            deg_sum = 0
            for (a, b) in deg:
                deg_sum += b
            avg_deg_list.append(deg_sum)
        return avg_deg_list

    @staticmethod
    def get_features_conn_components(group, series_num):
        dt = get_nx_graph(group, series_num)
        conn_comp_list = []
        for idx, graph in dt.items():
            conn_comp_list.append(nx.number_connected_components(graph))
        return conn_comp_list

    @staticmethod
    def get_features_betn_centrality(group, series_num):
        dt = get_nx_graph(group, series_num)
        bet_cen_list = []
        for idx, graph in dt.items():
            # print(nx.betweenness_centrality(graph).values())
            bet_cen_list.append(
                sum(nx.betweenness_centrality(graph).values()) / len(nx.betweenness_centrality(graph)))
        return bet_cen_list


def plot_feature_graphs():
    Graph_features = GetGraphFeatures()
    ATL_total = []
    for i in range(1, 27):
        # ATL_list = []
        group = 'ATL'

        avg_deg_list = np.array(Graph_features.get_features_avg_degree(group, i))
        #avg_deg_list = (avg_deg_list - min(avg_deg_list)) / (max(avg_deg_list) - min(avg_deg_list))

        ATL_total.append(avg_deg_list)

        # bet_cet_list = []
        # bet_cet_list = np.array(Graph_features.get_features_betn_centrality(group, i))
        # bet_cet_list = (bet_cet_list - min(bet_cet_list)) / (max(bet_cet_list) - min(bet_cet_list))
        # plt.title('%s_Series_%s_betweenness_centrality_measure' % (f'{group}', f'{i}'))
        # plt.title('%s_Series_%s_avg_degree_measure' % (f'{group}', f'{i}'))
        # plt.ylabel('Avg degree')
        # plt.xlabel('Frame number')
        # plt.plot(avg_deg_list)
        # # plt.show()
        # plt.savefig('%s_Series_%s_avg_degree' % (f'{group}', f'{i}'), bbox_inches='tight')
        # plt.close()


plot_feature_graphs()
exit()


def get_avg_degree_unweighted():
    graph_dict = get_nx_graph()
    avg_degree_list = []
    for idx, graph in graph_dict.items():
        avg_degree = nx.average_degree_connectivity(graph)
        avg_degree_list.append(avg_degree)

    return avg_degree_list


def get_avg_clustering():
    graph_dict = get_nx_graph()
    avg_cluster_list = []
    # for idx, graph in graph_dict.items():

    for i in range(100):
        avg_cl = nx.average_clustering()


# for i in range(100):
#     avg_deg_list = []
#     ske = imageio.imread(
#         '/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A9/A9_decon_t0%s_ch00_skel.png' % f'{i:02d}')
#     grph = sknw.build_sknw(ske)
#     G = nx.Graph()
#
#     # edges = grph.edges()
#     G.add_nodes_from(grph.nodes)
#     G.add_edges_from(grph.edges)
#     # options = {
#     #     'node_size': 20,
#     # }
#
#     # max_degree_list.append(max(set(d for n, d in G.degree())))
#     # betn_dict = nx.betweenness_centrality(G)
#     # print(bct.degree.degrees_dir(G))
#
#     # cl_dict = nx.closeness_centrality(G)
#
#     # print(list(nx.connected_components(G)))
#     # print(len(list(nx.connected_components(G))))
#
#     avg_deg_list.append(nx.average_degree_connectivity(G))
# print(avg_deg_list)
