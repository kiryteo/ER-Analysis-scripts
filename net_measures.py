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


max_degree_list = []


def get_nx_graph():
    graph_dict = {}
    for i in range(100):
        ske = imageio.imread(
            '/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A9/A9_decon_t0%s_ch00_skel.png' % f'{i:02d}')
        grph = sknw.build_sknw(ske)
        G = nx.Graph()

        # edges = grph.edges()
        G.add_nodes_from(grph.nodes)
        G.add_edges_from(grph.edges)
        graph_dict[i] = G

    return graph_dict


dt = get_nx_graph()
nodes_list = []
edges_list = []
avg_deg_list = []
conn_comp_list = []
bet_cen_list = []
for idx, graph in dt.items():
    # cl = nx.average_degree_connectivity(graph)
    # print(cl)
    # print(graph)
    # print(graph.nodes)
    # print(graph.edges)

    nodes_list.append(len(graph.nodes))
    edges_list.append(len(graph.edges))
    deg = graph.degree()
    sm = 0
    for (a, b) in deg:
        sm += b
    avg_deg_list.append(sm / len(deg))
    # print(sm / len(deg))
    # sum_edges = sum(cl.values())
    # print(sum_edges / len(graph.nodes))
    # break
    # conn_comp_list.append(nx.number_connected_components(graph))
    # bet_cen_list.append(sum(nx.betweenness_centrality(graph).values())/ len(nx.betweenness_centrality(graph)))
    # break

# plt.plot(edges_list)
# sns.distplot(edges_list)
# plt.scatter(avg_deg_list)
# # plt.hist(avg_deg_list)

# plt.plot(bet_cen_list)
plt.show()

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