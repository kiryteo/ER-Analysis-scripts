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
            avg_deg_list.append(deg_sum/ len(deg))
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



def combined_graph_feature_plot():

    fig = plt.figure()
    plt.axis('off')
    r,c = 4, 1

    atl = imageio.imread('/localhome/asa420/MIAL/graph_features/ATL_avg_degree_boxplot.png')
    climp = imageio.imread('/localhome/asa420/MIAL/graph_features/Climp_avg_degree_boxplot.png')
    control = imageio.imread('/localhome/asa420/MIAL/graph_features/Control_avg_degree_boxplot.png')
    rtn = imageio.imread('/localhome/asa420/MIAL/graph_features/RTN_avg_degree_boxplot.png')

    fig.add_subplot(r,c,1)
    plt.imshow(atl)

    fig.add_subplot(r,c,2)
    plt.imshow(climp)

    fig.add_subplot(r,c,3)
    plt.imshow(control)

    fig.add_subplot(r,c,4)
    plt.imshow(rtn)

    plt.show()

# combined_graph_feature_plot()
# exit()


def get_feature(group):
    group_total_features = []
    for i in range(1, 27):
        avg_deg_list = np.array(Graph_features.get_features_avg_degree(group, i))
        group_total_features.append(avg_deg_list)

    timestep_features = []
    for i in range(100):
        l = []
        for each in group_total_features:
            l.append(each[i])
        # lt.append(np.mean(l))
        timestep_features.append(l)


def plot_feature_graphs():
    Graph_features = GetGraphFeatures()

    # ATL_total = []
    # for i in range(1, 27):
    #     group = 'ATL'
    #
    #     avg_deg_list = np.array(Graph_features.get_features_avg_degree(group, i))
    #
    #     ATL_total.append(avg_deg_list)
    # #
    # lt_atl = []
    # lt_std = []
    # for i in range(100):
    #     l = []
    #     for each in ATL_total:
    #         l.append(each[i])
    #     # lt.append(np.mean(l))
    #     # lt_std.append(np.std(l))
    #     lt_atl.append(l)



    Climp_total = []
    for i in range(1, 32):
        group = 'Climp'

        avg_deg_list = np.array(Graph_features.get_features_avg_degree(group, i))
        #avg_deg_list = (avg_deg_list - min(avg_deg_list)) / (max(avg_deg_list) - min(avg_deg_list))

        # print(avg_deg_list)


        Climp_total.append(avg_deg_list)
    #
    lt_climp = []
    lt_std_climp = []
    for i in range(100):
        l = []
        for each in Climp_total:
            l.append(each[i])
        # lt_climp.append(np.mean(l))
        # lt_std_climp.append(np.std(l))
        lt_climp.append(l)


    # Ctrl_total = []
    # for i in range(1, 32):
    #     group = 'Control'
    #
    #     avg_deg_list = np.array(Graph_features.get_features_avg_degree(group, i))
    #     #avg_deg_list = (avg_deg_list - min(avg_deg_list)) / (max(avg_deg_list) - min(avg_deg_list))
    #
    #     # print(avg_deg_list)
    #
    #
    #     Ctrl_total.append(avg_deg_list)
    # #
    # lt_ctrl = []
    # for i in range(100):
    #     l = []
    #     for each in Ctrl_total:
    #         l.append(each[i])
    #     # lt_climp.append(np.mean(l))
    #     # lt_std_climp.append(np.std(l))
    #     lt_ctrl.append(l)



    # RTN_total = []
    # for i in range(1, 30):
    #     group = 'RTN'
    #
    #     avg_deg_list = np.array(Graph_features.get_features_avg_degree(group, i))
    #     #avg_deg_list = (avg_deg_list - min(avg_deg_list)) / (max(avg_deg_list) - min(avg_deg_list))
    #
    #     # print(avg_deg_list)
    #
    #
    #     RTN_total.append(avg_deg_list)
    # #
    # lt_rtn = []
    # for i in range(100):
    #     l = []
    #     for each in RTN_total:
    #         l.append(each[i])
    #     # lt_climp.append(np.mean(l))
    #     # lt_std_climp.append(np.std(l))
    #     lt_rtn.append(l)


    dct = {}

    # ind = np.arange(1, 101)

    for idx, tstep in enumerate(lt_climp):
        dct[idx+1] = tstep

    # print(dct)

    # dct = {"ATL": lt, "Climp": lt_climp}
    #
    fig, ax = plt.subplots()
    ax.boxplot(dct.values())
    # ax.set_xticks(ind)
    ax.set_xticklabels(dct.keys())
    plt.ylim([1.25, 2.6])
    plt.title('Climp average degree variation for all movies across t=1 to t=100')
    plt.xlabel('Timestamp')
    plt.ylabel('average degree')

    # sns.boxplot(lt)
    # sns.boxplot(lt_climp)
    # plt.plot(lt_climp, label='mean Climp')
    # plt.plot(lt_std_climp, label='std Climp')
    # plt.legend()
    plt.savefig('Climp average degree variation', bbox_inches='tight', pad_inches=1)

    # plt.show()
    plt.close()

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
