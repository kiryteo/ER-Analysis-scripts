import networkx as nx
import skimage
import imageio
import sknw
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def create_graphs():
    for i in range(100):
        ske = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A9/A9_decon_t0{i:02d}_ch00_skel.png')

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
        plt.title(f'ATL_S9 - frame number: {i:02d}')
        plt.savefig(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/ATL_S9_graphs/A9_decon_t0{i:02d}_ch00_graph.png', bbox_inches='tight', pad_inches=0)

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
        prefix = f'{pref}ATL/new_op_jul/skel/'
        series_init = 'A'
    elif group == 'Climp':
        prefix = f'{pref}Climp/new_op_jul/skel/'
        series_init = 'C'
    elif group == 'Control':
        prefix = f'{pref}Control/new_op_jul/skel/'
        series_init = 'Ct'
    else:
        prefix = f'{pref}RTN/new_op_jul/skel/'
        series_init = 'R'

    for i in range(100):
        ske = imageio.imread((prefix + f'{series_init}{series_num}/{series_init}{series_num}_decon_t0{i:02d}_ch00_skel.png'))

        grph = sknw.build_sknw(ske)
        G = nx.Graph()

        G.add_nodes_from(grph.nodes)
        G.add_edges_from(grph.edges)
        graph_dict[i] = G

    return graph_dict




class GetGraphFeatures:
    def __init__(self):
        pass

    @staticmethod
    def get_features_nodes(group, series_num):
        """

        @param group: group to analyze
        @param series_num: sequence number
        @return: list: number of nodes in the graph (frame)
        """
        dt = get_nx_graph(group, series_num)
        return [len(graph.nodes) for idx, graph in dt.items()]

    @staticmethod
    def get_features_edges(group, series_num):
        dt = get_nx_graph(group, series_num)
        return [len(graph.edges) for idx, graph in dt.items()]

    @staticmethod
    def get_features_avg_degree(group, series_num):
        dt = get_nx_graph(group, series_num)
        avg_deg_list = []
        for idx, graph in dt.items():
            deg = graph.degree()
            deg_sum = sum(b for a, b in deg)
            avg_deg_list.append(deg_sum / len(deg))
        return avg_deg_list

    @staticmethod
    def get_features_conn_components(group, series_num):
        dt = get_nx_graph(group, series_num)
        return [nx.number_connected_components(graph) for idx, graph in dt.items()]

    @staticmethod
    def get_features_betn_centrality(group, series_num):
        dt = get_nx_graph(group, series_num)
        return [sum(nx.betweenness_centrality(graph).values()) / len(nx.betweenness_centrality(graph)) for idx, graph in dt.items()]


def combined_graph_feature_plot():
    fig = plt.figure()
    plt.axis('off')
    r, c = 4, 1

    atl = imageio.imread('/localhome/asa420/MIAL/graph_features/ATL_avg_degree_boxplot.png')
    climp = imageio.imread('/localhome/asa420/MIAL/graph_features/Climp_avg_degree_boxplot.png')
    control = imageio.imread('/localhome/asa420/MIAL/graph_features/Control_avg_degree_boxplot.png')
    rtn = imageio.imread('/localhome/asa420/MIAL/graph_features/RTN_avg_degree_boxplot.png')

    fig.add_subplot(r, c, 1)
    plt.imshow(atl)

    fig.add_subplot(r, c, 2)
    plt.imshow(climp)

    fig.add_subplot(r, c, 3)
    plt.imshow(control)

    fig.add_subplot(r, c, 4)
    plt.imshow(rtn)

    plt.show()


# combined_graph_feature_plot()
# exit()




def get_feature(group, feature):
    """

    @param group: string: group to analyze
    @param feature: string: feature to analyze
    @return: list (n, n): group_total_features
    """

    Graph_features = GetGraphFeatures()
    group_total_features = []
    if group == 'ATL':
        num_series = 26
    elif group in ['Climp', 'Control']:
        num_series = 31
    else:
        num_series = 29

    if feature == 'avg_degree':
        for i in range(1, num_series + 1):
            feature_list = np.array(Graph_features.get_features_avg_degree(group, i))
            group_total_features.append(feature_list)

    elif feature == 'conn_components':
        for i in range(1, num_series + 1):
            feature_list = np.array(Graph_features.get_features_conn_components(group, i))
            group_total_features.append(feature_list)

    elif feature == 'nodes':
        for i in range(1, num_series + 1):
            feature_list = np.array(Graph_features.get_features_nodes(group, i))
            group_total_features.append(feature_list)

    elif feature == 'edges':
        for i in range(1, num_series + 1):
            feature_list = np.array(Graph_features.get_features_avg_degree(group, i))
            group_total_features.append(feature_list)

    elif feature == 'betn_centrality':
        for i in range(1, num_series + 1):
            feature_list = np.array(Graph_features.get_features_avg_degree(group, i))
            group_total_features.append(feature_list)


    else:
        print("Select the correct graph feature.")
        exit()

    return group_total_features

atl_nodes = get_feature('ATL', 'nodes')
print(len(atl_nodes))
print(len(atl_nodes[0]))

exit()

def get_timestep_features(group_total_features):
    timestep_features = []
    for i in range(100):
        l = [each[i] for each in group_total_features]
        # lt.append(np.mean(l))
        timestep_features.append(l)

    return timestep_features


def plot_feature_graphs():

    atl_feature = get_feature('ATL', 'betn_centrality')
    ATL_timestep_features = get_timestep_features(atl_feature)

    climp_feature = get_feature('Climp', 'betn_centrality')
    Climp_timestep_features = get_timestep_features(climp_feature)

    control_feature = get_feature('Control', 'betn_centrality')
    Control_timestep_features = get_timestep_features(control_feature)

    rtn_feature = get_feature('RTN', 'betn_centrality')
    RTN_timestep_features = get_timestep_features(rtn_feature)

    # df = pd.DataFrame()
    # df['atl'] = ATL_timestep_features
    # df['climp'] = Climp_timestep_features
    # df['control'] = Control_timestep_features
    # df['rtn'] = RTN_timestep_features



    boxplot_dict_atl = {idx + 1: tstep for idx, tstep in enumerate(ATL_timestep_features)}


    # print(boxplot_dict_atl)
    # exit()

    boxplot_dict_climp = {idx + 1: tstep for idx, tstep in enumerate(Climp_timestep_features)}


    boxplot_dict_ctrl = {idx + 1: tstep for idx, tstep in enumerate(Control_timestep_features)}


    boxplot_dict_rtn = {idx + 1: tstep for idx, tstep in enumerate(RTN_timestep_features)}


    #
    # df['atl'] = boxplot_dict_atl.values()
    # df['climp'] = boxplot_dict_climp.values()
    # df['control'] = boxplot_dict_ctrl.values()
    # df['rtn'] = boxplot_dict_rtn.values()
    #
    # sns.boxplot(data=df, palette='flare')
    # plt.show()
    # dct = {"ATL": lt, "Climp": lt_climp}

    #
    fig, ax = plt.subplots()
    c = 'red'
    d = 'green'
    e = 'blue'
    f = 'yellow'
    bp1 = ax.boxplot(boxplot_dict_atl.values(), patch_artist=True, boxprops=dict(facecolor=c, color=c),
               capprops=dict(color=c),
               whiskerprops=dict(color=c),
               flierprops=dict(color=c, markeredgecolor=c),
               medianprops=dict(color=c))
    bp2 = ax.boxplot(boxplot_dict_climp.values(), patch_artist=True, boxprops=dict(facecolor=d, color=d),
               capprops=dict(color=d),
               whiskerprops=dict(color=d),
               flierprops=dict(color=d, markeredgecolor=d),
               medianprops=dict(color=d))
    bp3 = ax.boxplot(boxplot_dict_ctrl.values(), patch_artist=True, boxprops=dict(facecolor=e, color=e),
               capprops=dict(color=e),
               whiskerprops=dict(color=e),
               flierprops=dict(color=e, markeredgecolor=e),
               medianprops=dict(color=e))
    bp4 = ax.boxplot(boxplot_dict_rtn.values(), patch_artist=True, boxprops=dict(facecolor=f, color=f),
               capprops=dict(color=f),
               whiskerprops=dict(color=f),
               flierprops=dict(color=f, markeredgecolor=f),
               medianprops=dict(color=f))
    # ax.set_xticks(ind)
    # ax.set_xticklabels(np.arange(1, 101), rotation=45)
    # ax.set_xticklabels(boxplot_dict_atl.keys())
    # plt.ylim([1.25, 2.6]) # avg degree
    # plt.ylim([25, 360]) #num nodes
    # plt.ylim([0, 120]) # conn components
    plt.title('Betweenness centrality variation for all movies across t=1 to t=100')
    plt.xlabel('Timestamp')
    plt.ylabel('Betweenness centrality')
    # plt.legend()
    ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], bp4["boxes"][0]], ['ATL', 'Climp', 'Control', 'RTN'], loc='upper right')
    #
    # # sns.boxplot(lt)
    # # sns.boxplot(lt_climp)
    # # plt.plot(lt_climp, label='mean Climp')
    # # plt.plot(lt_std_climp, label='std Climp')
    # # plt.legend()
    # # plt.savefig('Climp average degree variation', bbox_inches='tight', pad_inches=1)
    #
    plt.show()
    # plt.close()

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

    # Climp_total = []
    # for i in range(1, 32):
    #     group = 'Climp'
    #
    #     avg_deg_list = np.array(Graph_features.get_features_avg_degree(group, i))
    #     #avg_deg_list = (avg_deg_list - min(avg_deg_list)) / (max(avg_deg_list) - min(avg_deg_list))
    #
    #     # print(avg_deg_list)
    #
    #
    #     Climp_total.append(avg_deg_list)
    # #
    # lt_climp = []
    # lt_std_climp = []
    # for i in range(100):
    #     l = []
    #     for each in Climp_total:
    #         l.append(each[i])
    #     # lt_climp.append(np.mean(l))
    #     # lt_std_climp.append(np.std(l))
    #     lt_climp.append(l)

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

    # dct = {}
    #
    # # ind = np.arange(1, 101)
    #
    # for idx, tstep in enumerate(lt_climp):
    #     dct[idx + 1] = tstep
    #
    # # print(dct)
    #
    # # dct = {"ATL": lt, "Climp": lt_climp}
    # #
    # fig, ax = plt.subplots()
    # ax.boxplot(dct.values())
    # # ax.set_xticks(ind)
    # ax.set_xticklabels(dct.keys())
    # plt.ylim([1.25, 2.6])
    # plt.title('Climp average degree variation for all movies across t=1 to t=100')
    # plt.xlabel('Timestamp')
    # plt.ylabel('average degree')
    #
    # # sns.boxplot(lt)
    # # sns.boxplot(lt_climp)
    # # plt.plot(lt_climp, label='mean Climp')
    # # plt.plot(lt_std_climp, label='std Climp')
    # # plt.legend()
    # plt.savefig('Climp average degree variation', bbox_inches='tight', pad_inches=1)
    #
    # # plt.show()
    # plt.close()

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

    for _ in range(100):
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
