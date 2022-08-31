import networkx as nx
import skimage
import imageio
import sknw
import matplotlib.pyplot as plt
import bct
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
        ske = imageio.imread(prefix + '%s/%s_decon_t0%s_ch00_skel.png'%(f'{series_init}{series_num}', f'{series_init}{series_num}', f'{i:02d}'))
        grph = sknw.build_sknw(ske)
        G = nx.Graph()

        G.add_nodes_from(grph.nodes)
        G.add_edges_from(grph.edges)
        graph_dict[i] = G

    return graph_dict


class GetGraphFeatures:
    def __init__(self):
        self.nodes_list = []
        self.edges_list = []
        self.avg_deg_list = []
        self.conn_comp_list = []
        self.bet_cen_list = []

    def get_features_nodes(self, group, series_num):
        dt = get_nx_graph(group, series_num)
        for idx, graph in dt.items():
            self.nodes_list.append(len(graph.nodes))
        return self.nodes_list

    def get_features_edges(self, group, series_num):
        dt = get_nx_graph(group, series_num)
        for idx, graph in dt.items():
            self.edges_list.append(len(graph.edges))
        return self.edges_list

    def get_features_avg_degree(self, group, series_num):
        dt = get_nx_graph(group, series_num)
        for idx, graph in dt.items():
            deg = graph.degree()
            deg_sum = 0
            for (a, b) in deg:
                deg_sum += b
            self.avg_deg_list.append(deg_sum)
        return self.avg_deg_list

    def get_features_conn_components(self, group, series_num):
        dt = get_nx_graph(group, series_num)
        for idx, graph in dt.items():
            self.conn_comp_list.append(nx.number_connected_components(graph))
        return self.conn_comp_list

    def get_features_betn_centrality(self, group, series_num):
        dt = get_nx_graph(group, series_num)
        for idx, graph in dt.items():
            # print(nx.betweenness_centrality(graph).values())
            self.bet_cen_list.append(
                sum(nx.betweenness_centrality(graph).values()) / len(nx.betweenness_centrality(graph)))
        return self.bet_cen_list


def plot_feature_graphs():
    Graph_features = GetGraphFeatures()
    for i in range(1, 27):
        # ATL_list = []
        group = 'ATL'
        bet_cet_list = Graph_features.get_features_betn_centrality(group, i)
        plt.title('%s_Series_%s_betweenness_centrality_measure'%(f'{group}', f'{i}'))
        plt.ylabel('centrality')
        plt.xlabel('Frame number')
        plt.plot(bet_cet_list)
        # plt.show()
        plt.savefig('%s_Series_%s_btn_centrality'%(f'{group}', f'{i}'), bbox_inches='tight')
        plt.close()


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

# l = []
# for k, v in betn_dict.items():
#     l.append(v)
#
# print(cl_dict)

# plt.plot(l)
# plt.show()

# print(max_degree_list)
# plt.plot(max_degree_list)
# plt.show()


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
