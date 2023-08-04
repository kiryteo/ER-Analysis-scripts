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



# First Column (er)
er = [0.01163, 0, 0, 0.0119, 0.01157, 0.01773, 0.06494, 0.00833, 0.02899, 0.00809, 0.02688, 0.00958, 0, 0, 0.01344, 0.02381, 0.03922, 0.01613, 0.03824, 0.04286, 0, 0, 0, 0.01429, 0, 0.03947, 0.00766, 0, 0, 0, 0.03435, 0.00877]

# Second Column (mt)
mt = [0.028333333333333, 0.023474178403756, 0.012345679012346, 0, 0, 0.006887052341598, 0.037581699346405, 0.008620689655172, 0.013513513513514, 0.03968253968254, 0.022486772486773, 0.006720430107527, 0.008620689655172, 0.011111111111111, 0.011574074074074, 0, 0.029069767441861, 0.013157894736842, 0.046747967479675, 0.035294117647059, 0.015942028985507, 0.011494252873563, 0.011904761904762, 0.045918367346939, 0.008928571428571, 0.016548463356974, 0, 0.008680555555556, 0.014529914529915, 0.033879781420765, 0.00534188034188]

# Third Column (gt)
gt = [0.033333333333333, 0.029761904761905, 0.014492753623188, 0.011627906976744, 0.027491408934708, 0.020661157024793, 0.039518900343643, 0.018691588785047, 0.032407407407407, 0.058139534883721, 0.017094017094017, 0.013550135501355, 0.038461538461539, 0.042194092827004, 0.012437810945274, 0.049565217391304, 0.059925093632959, 0.013157894736842, 0.060077519379845, 0.048780487804878, 0.028908554572271, 0.012820512820513, 0.018954248366013, 0.041237113402062, 0.022935779816514, 0.050574712643678, 0.056603773584906, 0.028968253968254, 0.018214936247723, 0.062062615101289, 0.015315315315315]





def get_abs_diff(d1, d2):
    return [abs(a - b) for a, b in zip(d1, d2)]



# def get_rel_diff(d1, d2):
#     return [abs(a - b) / a for a, b in zip(d1, d2)]

def get_rel_diff(d1, d2):
    return [abs((a - b) / b) for a, b in zip(d1, d2)]

rel_diff_ernet = get_rel_diff(er, gt)

print(rel_diff_ernet)

rel_diff_method = get_rel_diff(mt, gt)

print(rel_diff_method)

improvement = [a - b for a, b in zip(rel_diff_ernet, rel_diff_method)]

print(improvement)

# mean_rel_diff = sum(rel_diff) / len(rel_diff)

# print(mean_rel_diff)
