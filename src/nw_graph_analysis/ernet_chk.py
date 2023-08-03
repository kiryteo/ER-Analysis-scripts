import sknw
import imageio
import networkx as nx
import pandas as pd


df = pd.DataFrame()

def get_props():
    nodes = []
    edges = []
    assort = []
    clust = []
    conn = []
    for i in range(1, 30):
        # img = imageio.imread(f'~/Documents/ref_skels/ref_skels_rtn/rtn{i}_proc_skel.png')
        img = imageio.imread(f'~/Documents/er-mean-proc/rtn/rtn{i}_er_mean_proc_enhance_skel.png')
        graph = sknw.build_sknw(img, multi=False)

        num_nodes = graph.nodes
        nodes.append(len(num_nodes))
        # print(len(num_nodes))

        num_edges = graph.edges
        edges.append(len(num_edges))
        # print(len(num_edges))

        assort_coeff = nx.degree_assortativity_coefficient(graph)
        # print(assort_coeff)
        assort.append(assort_coeff)

        clust_coeff = nx.average_clustering(graph)
        # print(clust_coeff)
        clust.append(clust_coeff)

        # cc = nx.clustering(graph)
        # print(cc)

        num_conn = nx.number_connected_components(graph)
        # print(num_conn)
        conn.append(num_conn)
    df['nodes'] = pd.Series(nodes)
    df['edges'] = pd.Series(edges)
    df['assort'] = pd.Series(assort)
    df['clust']= pd.Series(clust)
    df['conn'] = pd.Series(conn)

    return df

# df = get_props()
# # print(df)
# with open('rtn_df.csv', 'w') as fl:
#     df.to_csv(fl, index=False)

def get_group_data(group):
    data = pd.read_csv('~/Documents/output/20230801053946898_graph_metrics.csv')
    group_data = data[data['Filename'].str.contains(f'{group}')]
    group_data.to_csv(f'{group}_ernet.csv', index=False)


def sort_df(group):
    df = pd.read_csv(f'/localhome/asa420/Downloads/output/{group}_ernet.csv')

    df['Number'] = df['Filename'].str.extract(r'(\d+)').astype(int)

    df['Filename'] = df['Filename'].str.replace(r':\d+', '', regex=True)

    df_sorted = df.sort_values(by='Number').reset_index(drop=True)
    df_sorted = df_sorted.drop('Number', axis=1)
    df_sorted.to_csv(f'/localhome/asa420/Downloads/output/{group}_ernet_sorted.csv', index=False)


# sort_df('atl')
# sort_df('climp')
# sort_df('control')
# sort_df('rtn')


nodes_method = [204, 200, 110, 253, 196, 203, 178, 204, 218, 100, 175, 207, 219, 126, 220, 231, 221, 226, 147, 142, 232, 167, 149, 167, 157, 247]

nodes_ernet = [147, 109, 57, 87, 95, 102, 71, 84, 101, 57, 95, 73, 56, 94, 129, 65, 35, 85, 82, 61, 98, 71, 62, 61, 66, 88]

nodes_gt = [203, 198, 106, 237, 186, 198, 168, 199, 212, 95, 161, 178, 186, 124, 207, 214, 197, 214, 136, 128, 225, 156, 146, 150, 146, 221]


def get_abs_diff(d1, d2):
    return [abs(a - b) for a, b in zip(d1, d2)]


abs_diff = get_abs_diff(nodes_gt, nodes_ernet)
mean_abs_diff = sum(abs_diff) / len(abs_diff)

print(mean_abs_diff)

abs_diff = get_abs_diff(nodes_gt, nodes_method)
mean_abs_diff = sum(abs_diff) / len(abs_diff)

print(mean_abs_diff)


def get_rel_diff(d1, d2):
    return [abs(a - b) / a for a, b in zip(d1, d2)]

rel_diff = get_rel_diff(nodes_gt, nodes_ernet)
mean_rel_diff = sum(rel_diff) / len(rel_diff)

print(mean_rel_diff)
