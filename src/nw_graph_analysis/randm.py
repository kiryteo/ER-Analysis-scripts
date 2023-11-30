import random

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import statannot
import imageio


# def runner(writer):
#     pref = '/localhome/asa420/MIAL/data/confocal-data/ATL/A9_junc_viz/'
#     for frame in range(100):
#         filename = f'{pref}A9_junc_viz_t{frame:02d}.png'
#         image = imageio.imread(filename)
#         writer.append_data(image)

# def create_sequence():
#     with imageio.get_writer(f'atl9_junc_viz.gif', mode='I', duration=0.5) as writer:
#         runner(writer)


def std_data(data):
    return list((data - min(data)) / (max(data) - min(data)))


def get_graph_perf(group):
    # sted_analyzer = pkl.load(open('nw_graph_analysis/sted_analyzer.pkl', 'rb'))
    # sted_ernet = pkl.load(open('nw_graph_analysis/sted_ernet.pkl', 'rb'))
    # sted_ernet_v2 = pkl.load(open('nw_graph_analysis/sted_ernet_v2.pkl', 'rb'))
    # sted_nerdy = pkl.load(open('nw_graph_analysis/sted_nerdy.pkl', 'rb'))
    # sted_p4m = pkl.load(open('nw_graph_analysis/sted_p4m.pkl', 'rb'))

    # df = pd.DataFrame()

    # val_data = sted_analyzer + sted_ernet + sted_ernet_v2 + sted_nerdy + sted_p4m

    # l1 = [value for sublist in val_data for value in sublist]
    
    # df['Values'] = l1

    # metric_names = []
    # method_names = []
    sted_climp_analyzer = pkl.load(open(f'nw_graph_analysis/sted_{group}_analyzer_graph_err.pkl', 'rb'))
    sted_climp_ernet = pkl.load(open(f'nw_graph_analysis/sted_{group}_ernet_graph_err.pkl', 'rb'))
    sted_climp_ernet_v2 = pkl.load(open(f'nw_graph_analysis/sted_{group}_erv2_graph_err.pkl', 'rb'))
    sted_climp_nerdy = pkl.load(open(f'nw_graph_analysis/sted_{group}_nerdy_graph_err.pkl', 'rb'))
    sted_climp_nerdy_p4m = pkl.load(open(f'nw_graph_analysis/sted_{group}_p4m_graph_err.pkl', 'rb'))

    sted_climp_analyzer = np.array(sted_climp_analyzer).T
    sted_climp_ernet = np.array(sted_climp_ernet).T
    sted_climp_ernet_v2 = np.array(sted_climp_ernet_v2).T
    sted_climp_nerdy = np.array(sted_climp_nerdy).T
    sted_climp_nerdy_p4m = np.array(sted_climp_nerdy_p4m).T


    num_nodes_climp_analyzer = sted_climp_analyzer[0]
    num_nodes_climp_analyzer = std_data(num_nodes_climp_analyzer)
    num_nodes_climp_ernet = sted_climp_ernet[0]
    num_nodes_climp_ernet = std_data(num_nodes_climp_ernet)
    num_nodes_climp_ernet_v2 = sted_climp_ernet_v2[0]
    num_nodes_climp_ernet_v2 = std_data(num_nodes_climp_ernet_v2)
    num_nodes_climp_nerdy = sted_climp_nerdy[0]
    num_nodes_climp_nerdy = std_data(num_nodes_climp_nerdy)
    num_nodes_climp_nerdy_p4m = sted_climp_nerdy_p4m[0]
    num_nodes_climp_nerdy_p4m = std_data(num_nodes_climp_nerdy_p4m)

    num_edges_climp_analyzer = sted_climp_analyzer[1]
    num_edges_climp_analyzer = std_data(num_edges_climp_analyzer)
    num_edges_climp_ernet = sted_climp_ernet[1]
    num_edges_climp_ernet = std_data(num_edges_climp_ernet)
    num_edges_climp_ernet_v2 = sted_climp_ernet_v2[1]
    num_edges_climp_ernet_v2 = std_data(num_edges_climp_ernet_v2)
    num_edges_climp_nerdy = sted_climp_nerdy[1]
    num_edges_climp_nerdy = std_data(num_edges_climp_nerdy)
    num_edges_climp_nerdy_p4m = sted_climp_nerdy_p4m[1]
    num_edges_climp_nerdy_p4m = std_data(num_edges_climp_nerdy_p4m)

    assortativity_climp_analyzer = sted_climp_analyzer[2]
    assortativity_climp_analyzer = std_data(assortativity_climp_analyzer)
    assortativity_climp_ernet = sted_climp_ernet[2]
    assortativity_climp_ernet = std_data(assortativity_climp_ernet)
    assortativity_climp_ernet_v2 = sted_climp_ernet_v2[2]
    assortativity_climp_ernet_v2 = std_data(assortativity_climp_ernet_v2)
    assortativity_climp_nerdy = sted_climp_nerdy[2]
    assortativity_climp_nerdy = std_data(assortativity_climp_nerdy)
    assortativity_climp_nerdy_p4m = sted_climp_nerdy_p4m[2]
    assortativity_climp_nerdy_p4m = std_data(assortativity_climp_nerdy_p4m)

    clustering_climp_analyzer = sted_climp_analyzer[3]
    clustering_climp_analyzer = std_data(clustering_climp_analyzer)
    clustering_climp_ernet = sted_climp_ernet[3]
    clustering_climp_ernet = std_data(clustering_climp_ernet)
    clustering_climp_ernet_v2 = sted_climp_ernet_v2[3]
    clustering_climp_ernet_v2 = std_data(clustering_climp_ernet_v2)
    clustering_climp_nerdy = sted_climp_nerdy[3]
    clustering_climp_nerdy = std_data(clustering_climp_nerdy)
    clustering_climp_nerdy_p4m = sted_climp_nerdy_p4m[3]
    clustering_climp_nerdy_p4m = std_data(clustering_climp_nerdy_p4m)

    # num_components_climp_analyzer = sted_climp_analyzer[4]
    # num_components_climp_analyzer = std_data(num_components_climp_analyzer)
    # num_components_climp_ernet = sted_climp_ernet[4]
    # num_components_climp_ernet = std_data(num_components_climp_ernet)
    # num_components_climp_ernet_v2 = sted_climp_ernet_v2[4]
    # num_components_climp_ernet_v2 = std_data(num_components_climp_ernet_v2)
    # num_components_climp_nerdy = sted_climp_nerdy[4]
    # num_components_climp_nerdy = std_data(num_components_climp_nerdy)
    # num_components_climp_nerdy_p4m = sted_climp_nerdy_p4m[4]
    # num_components_climp_nerdy_p4m = std_data(num_components_climp_nerdy_p4m)

    # ratio_nodes_climp_analyzer = sted_climp_analyzer[5]
    # ratio_nodes_climp_analyzer = std_data(ratio_nodes_climp_analyzer)
    # ratio_nodes_climp_ernet = sted_climp_ernet[5]
    # ratio_nodes_climp_ernet = std_data(ratio_nodes_climp_ernet)
    # ratio_nodes_climp_ernet_v2 = sted_climp_ernet_v2[5]
    # ratio_nodes_climp_ernet_v2 = std_data(ratio_nodes_climp_ernet_v2)
    # ratio_nodes_climp_nerdy = sted_climp_nerdy[5]
    # ratio_nodes_climp_nerdy = std_data(ratio_nodes_climp_nerdy)
    # ratio_nodes_climp_nerdy_p4m = sted_climp_nerdy_p4m[5]
    # ratio_nodes_climp_nerdy_p4m = std_data(ratio_nodes_climp_nerdy_p4m)

    # ratio_edges_climp_analyzer = sted_climp_analyzer[6]
    # ratio_edges_climp_analyzer = std_data(ratio_edges_climp_analyzer)
    # ratio_edges_climp_ernet = sted_climp_ernet[6]
    # ratio_edges_climp_ernet = std_data(ratio_edges_climp_ernet)
    # ratio_edges_climp_ernet_v2 = sted_climp_ernet_v2[6]
    # ratio_edges_climp_ernet_v2 = std_data(ratio_edges_climp_ernet_v2)
    # ratio_edges_climp_nerdy = sted_climp_nerdy[6]
    # ratio_edges_climp_nerdy = std_data(ratio_edges_climp_nerdy)
    # ratio_edges_climp_nerdy_p4m = sted_climp_nerdy_p4m[6]
    # ratio_edges_climp_nerdy_p4m = std_data(ratio_edges_climp_nerdy_p4m)

    # global_efficiency_climp_analyzer = sted_climp_analyzer[7]
    # global_efficiency_climp_analyzer = std_data(global_efficiency_climp_analyzer)
    # global_efficiency_climp_ernet = sted_climp_ernet[7]
    # global_efficiency_climp_ernet = std_data(global_efficiency_climp_ernet)
    # global_efficiency_climp_ernet_v2 = sted_climp_ernet_v2[7]
    # global_efficiency_climp_ernet_v2 = std_data(global_efficiency_climp_ernet_v2)
    # global_efficiency_climp_nerdy = sted_climp_nerdy[7]
    # global_efficiency_climp_nerdy = std_data(global_efficiency_climp_nerdy)
    # global_efficiency_climp_nerdy_p4m = sted_climp_nerdy_p4m[7]
    # global_efficiency_climp_nerdy_p4m = std_data(global_efficiency_climp_nerdy_p4m)

    # density_climp_analyzer = sted_climp_analyzer[8]
    # density_climp_analyzer = std_data(density_climp_analyzer)
    # density_climp_ernet = sted_climp_ernet[8]
    # density_climp_ernet = std_data(density_climp_ernet)
    # density_climp_ernet_v2 = sted_climp_ernet_v2[8]
    # density_climp_ernet_v2 = std_data(density_climp_ernet_v2)
    # density_climp_nerdy = sted_climp_nerdy[8]
    # density_climp_nerdy = std_data(density_climp_nerdy)
    # density_climp_nerdy_p4m = sted_climp_nerdy_p4m[8]
    # density_climp_nerdy_p4m = std_data(density_climp_nerdy_p4m)

    df = pd.DataFrame()

    # val_data = assortativity_climp_analyzer + assortativity_climp_ernet + assortativity_climp_ernet_v2 + assortativity_climp_nerdy + assortativity_climp_nerdy_p4m +

    val_data = num_nodes_climp_analyzer + num_nodes_climp_ernet + num_nodes_climp_ernet_v2 + num_nodes_climp_nerdy + num_nodes_climp_nerdy_p4m + num_edges_climp_analyzer + num_edges_climp_ernet + num_edges_climp_ernet_v2 + num_edges_climp_nerdy + num_edges_climp_nerdy_p4m +   clustering_climp_analyzer + clustering_climp_ernet + clustering_climp_ernet_v2 + clustering_climp_nerdy + clustering_climp_nerdy_p4m + assortativity_climp_analyzer + assortativity_climp_ernet + assortativity_climp_ernet_v2 + assortativity_climp_nerdy + assortativity_climp_nerdy_p4m

    # val_data = num_components_climp_analyzer + num_components_climp_ernet + num_components_climp_ernet_v2 + num_components_climp_nerdy + num_components_climp_nerdy_p4m + ratio_nodes_climp_analyzer + ratio_nodes_climp_ernet + ratio_nodes_climp_ernet_v2 + ratio_nodes_climp_nerdy + ratio_nodes_climp_nerdy_p4m + ratio_edges_climp_analyzer + ratio_edges_climp_ernet + ratio_edges_climp_ernet_v2 + ratio_edges_climp_nerdy + ratio_edges_climp_nerdy_p4m +   global_efficiency_climp_analyzer + global_efficiency_climp_ernet + global_efficiency_climp_ernet_v2 + global_efficiency_climp_nerdy + global_efficiency_climp_nerdy_p4m + density_climp_analyzer + density_climp_ernet + density_climp_ernet_v2 + density_climp_nerdy + density_climp_nerdy_p4m

    # l1 = [value for sublist in val_data for value in sublist]

    df['Values'] = val_data

    metric_names = []

    metric_names.append(['Num_nodes']*len(num_nodes_climp_analyzer))
    metric_names.append(['Num_nodes']*len(num_nodes_climp_ernet))
    metric_names.append(['Num_nodes']*len(num_nodes_climp_ernet_v2))
    metric_names.append(['Num_nodes']*len(num_nodes_climp_nerdy))
    metric_names.append(['Num_nodes']*len(num_nodes_climp_nerdy_p4m))

    metric_names.append(['Num_edges']*len(num_edges_climp_analyzer))
    metric_names.append(['Num_edges']*len(num_edges_climp_ernet))
    metric_names.append(['Num_edges']*len(num_edges_climp_ernet_v2))
    metric_names.append(['Num_edges']*len(num_edges_climp_nerdy))
    metric_names.append(['Num_edges']*len(num_edges_climp_nerdy_p4m))

    metric_names.append(['Assortativity']*len(assortativity_climp_analyzer))
    metric_names.append(['Assortativity']*len(assortativity_climp_ernet))
    metric_names.append(['Assortativity']*len(assortativity_climp_ernet_v2))
    metric_names.append(['Assortativity']*len(assortativity_climp_nerdy))
    metric_names.append(['Assortativity']*len(assortativity_climp_nerdy_p4m))

    metric_names.append(['Clustering']*len(clustering_climp_analyzer))
    metric_names.append(['Clustering']*len(clustering_climp_ernet))
    metric_names.append(['Clustering']*len(clustering_climp_ernet_v2))
    metric_names.append(['Clustering']*len(clustering_climp_nerdy))
    metric_names.append(['Clustering']*len(clustering_climp_nerdy_p4m))

    # metric_names.append(['Num_components']*len(num_components_climp_analyzer))
    # metric_names.append(['Num_components']*len(num_components_climp_ernet))
    # metric_names.append(['Num_components']*len(num_components_climp_ernet_v2))
    # metric_names.append(['Num_components']*len(num_components_climp_nerdy))
    # metric_names.append(['Num_components']*len(num_components_climp_nerdy_p4m))

    # metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_analyzer))
    # metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_ernet))
    # metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_ernet_v2))
    # metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_nerdy))
    # metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_nerdy_p4m))

    # metric_names.append(['ratio_edges']*len(ratio_edges_climp_analyzer))
    # metric_names.append(['ratio_edges']*len(ratio_edges_climp_ernet))
    # metric_names.append(['ratio_edges']*len(ratio_edges_climp_ernet_v2))
    # metric_names.append(['ratio_edges']*len(ratio_edges_climp_nerdy))
    # metric_names.append(['ratio_edges']*len(ratio_edges_climp_nerdy_p4m))

    # metric_names.append(['Global_efficiency']*len(global_efficiency_climp_analyzer))
    # metric_names.append(['Global_efficiency']*len(global_efficiency_climp_ernet))
    # metric_names.append(['Global_efficiency']*len(global_efficiency_climp_ernet_v2))
    # metric_names.append(['Global_efficiency']*len(global_efficiency_climp_nerdy))
    # metric_names.append(['Global_efficiency']*len(global_efficiency_climp_nerdy_p4m))

    # metric_names.append(['Density']*len(density_climp_analyzer))
    # metric_names.append(['Density']*len(density_climp_ernet))
    # metric_names.append(['Density']*len(density_climp_ernet_v2))
    # metric_names.append(['Density']*len(density_climp_nerdy))
    # metric_names.append(['Density']*len(density_climp_nerdy_p4m))


    method_names = []    
    
    method_names.append(['AnalyzER']*len(sted_climp_analyzer[0]))
    method_names.append(['ERnet']*len(sted_climp_ernet[0]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[0]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[0]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[0]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[1]))
    method_names.append(['ERnet']*len(sted_climp_ernet[1]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[1]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[1]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[1]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[2]))
    method_names.append(['ERnet']*len(sted_climp_ernet[2]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[2]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[2]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[2]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[3]))
    method_names.append(['ERnet']*len(sted_climp_ernet[3]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[3]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[3]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[3]))

    # method_names.append(['AnalyzER']*len(sted_climp_analyzer[4]))
    # method_names.append(['ERnet']*len(sted_climp_ernet[4]))
    # method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[4]))
    # method_names.append(['nERdy']*len(sted_climp_nerdy[4]))
    # method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[4]))

    # method_names.append(['AnalyzER']*len(sted_climp_analyzer[5]))
    # method_names.append(['ERnet']*len(sted_climp_ernet[5]))
    # method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[5]))
    # method_names.append(['nERdy']*len(sted_climp_nerdy[5]))
    # method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[5]))

    # method_names.append(['AnalyzER']*len(sted_climp_analyzer[6]))
    # method_names.append(['ERnet']*len(sted_climp_ernet[6]))
    # method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[6]))
    # method_names.append(['nERdy']*len(sted_climp_nerdy[6]))
    # method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[6]))

    # method_names.append(['AnalyzER']*len(sted_climp_analyzer[7]))
    # method_names.append(['ERnet']*len(sted_climp_ernet[7]))
    # method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[7]))
    # method_names.append(['nERdy']*len(sted_climp_nerdy[7]))
    # method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[7]))

    # method_names.append(['AnalyzER']*len(sted_climp_analyzer[8]))
    # method_names.append(['ERnet']*len(sted_climp_ernet[8]))
    # method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[8]))
    # method_names.append(['nERdy']*len(sted_climp_nerdy[8]))
    # method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[8]))

    df['Metric'] = [value for sublist in metric_names for value in sublist]

    df['Method'] = [value for sublist in method_names for value in sublist]

    # sns.set(style="whitegrid")
    ax = sns.boxplot(x="Metric", y="Values", hue="Method", data=df, showfliers=False, width=0.8, palette="Set2")

    # plt.ylim(0, 0.8)
    # plt.xlim(-1, 4.0)

    yt = ax.get_yticks()
    yt = [f'{y:.1f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)#, rotation=45)
    # ax.set_xticklabels(['NC', 'RN', 'RE', 'GE', 'D'], fontsize=12)


    ax.grid(axis='y')#, linestyle='-', linewidth=0.5, color='white')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=5, fancybox=True, shadow=True, fontsize=10)
    plt.xlabel('Metric', fontsize=15)
    plt.ylabel('Error', fontsize=15)

    # plt.subplots_adjust(hspace=0.3)

    plt.gcf().set_size_inches(8, 10)
    # plt.gcf().set_size_inches(4, 6)

    # plt.show()
    plt.savefig(f'nw_graph_analysis/sted_{group}_graph_err_p1.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.close()

# get_graph_perf('control')
# exit()


def get_conf_graph_perf(group):
    # sted_analyzer = pkl.load(open('nw_graph_analysis/sted_analyzer.pkl', 'rb'))
    # sted_ernet = pkl.load(open('nw_graph_analysis/sted_ernet.pkl', 'rb'))
    # sted_ernet_v2 = pkl.load(open('nw_graph_analysis/sted_ernet_v2.pkl', 'rb'))
    # sted_nerdy = pkl.load(open('nw_graph_analysis/sted_nerdy.pkl', 'rb'))
    # sted_p4m = pkl.load(open('nw_graph_analysis/sted_p4m.pkl', 'rb'))

    # df = pd.DataFrame()

    # val_data = sted_analyzer + sted_ernet + sted_ernet_v2 + sted_nerdy + sted_p4m

    # l1 = [value for sublist in val_data for value in sublist]
    
    # df['Values'] = l1

    # metric_names = []
    # method_names = []
    climp_analyzer = pkl.load(open(f'nw_graph_analysis/{group}_analyzer_graph_err.pkl', 'rb'))
    climp_ernet = pkl.load(open(f'nw_graph_analysis/{group}_ernet_graph_err.pkl', 'rb'))
    climp_ernet_v2 = pkl.load(open(f'nw_graph_analysis/{group}_erv2_graph_err.pkl', 'rb'))
    climp_nerdy = pkl.load(open(f'nw_graph_analysis/{group}_nerdy_graph_err.pkl', 'rb'))
    climp_nerdy_p4m = pkl.load(open(f'nw_graph_analysis/{group}_p4m_graph_err.pkl', 'rb'))

    sted_climp_analyzer = np.array(climp_analyzer)
    sted_climp_ernet = np.array(climp_ernet)
    sted_climp_ernet_v2 = np.array(climp_ernet_v2)
    sted_climp_nerdy = np.array(climp_nerdy)
    sted_climp_nerdy_p4m = np.array(climp_nerdy_p4m)


    # num_nodes_climp_analyzer = sted_climp_analyzer[0]
    # num_nodes_climp_analyzer = std_data(num_nodes_climp_analyzer)
    # num_nodes_climp_ernet = sted_climp_ernet[0]
    # num_nodes_climp_ernet = std_data(num_nodes_climp_ernet)
    # num_nodes_climp_ernet_v2 = sted_climp_ernet_v2[0]
    # num_nodes_climp_ernet_v2 = std_data(num_nodes_climp_ernet_v2)
    # num_nodes_climp_nerdy = sted_climp_nerdy[0]
    # num_nodes_climp_nerdy = std_data(num_nodes_climp_nerdy)
    # num_nodes_climp_nerdy_p4m = sted_climp_nerdy_p4m[0]
    # num_nodes_climp_nerdy_p4m = std_data(num_nodes_climp_nerdy_p4m)

    # num_edges_climp_analyzer = sted_climp_analyzer[1]
    # num_edges_climp_analyzer = std_data(num_edges_climp_analyzer)
    # num_edges_climp_ernet = sted_climp_ernet[1]
    # num_edges_climp_ernet = std_data(num_edges_climp_ernet)
    # num_edges_climp_ernet_v2 = sted_climp_ernet_v2[1]
    # num_edges_climp_ernet_v2 = std_data(num_edges_climp_ernet_v2)
    # num_edges_climp_nerdy = sted_climp_nerdy[1]
    # num_edges_climp_nerdy = std_data(num_edges_climp_nerdy)
    # num_edges_climp_nerdy_p4m = sted_climp_nerdy_p4m[1]
    # num_edges_climp_nerdy_p4m = std_data(num_edges_climp_nerdy_p4m)

    # assortativity_climp_analyzer = sted_climp_analyzer[2]
    # assortativity_climp_analyzer = std_data(assortativity_climp_analyzer)
    # assortativity_climp_ernet = sted_climp_ernet[2]
    # assortativity_climp_ernet = std_data(assortativity_climp_ernet)
    # assortativity_climp_ernet_v2 = sted_climp_ernet_v2[2]
    # assortativity_climp_ernet_v2 = std_data(assortativity_climp_ernet_v2)
    # assortativity_climp_nerdy = sted_climp_nerdy[2]
    # assortativity_climp_nerdy = std_data(assortativity_climp_nerdy)
    # assortativity_climp_nerdy_p4m = sted_climp_nerdy_p4m[2]
    # assortativity_climp_nerdy_p4m = std_data(assortativity_climp_nerdy_p4m)

    # clustering_climp_analyzer = sted_climp_analyzer[3]
    # clustering_climp_analyzer = std_data(clustering_climp_analyzer)
    # clustering_climp_ernet = sted_climp_ernet[3]
    # clustering_climp_ernet = std_data(clustering_climp_ernet)
    # clustering_climp_ernet_v2 = sted_climp_ernet_v2[3]
    # clustering_climp_ernet_v2 = std_data(clustering_climp_ernet_v2)
    # clustering_climp_nerdy = sted_climp_nerdy[3]
    # clustering_climp_nerdy = std_data(clustering_climp_nerdy)
    # clustering_climp_nerdy_p4m = sted_climp_nerdy_p4m[3]
    # clustering_climp_nerdy_p4m = std_data(clustering_climp_nerdy_p4m)

    num_components_climp_analyzer = sted_climp_analyzer[4]
    num_components_climp_analyzer = std_data(num_components_climp_analyzer)
    num_components_climp_ernet = sted_climp_ernet[4]
    num_components_climp_ernet = std_data(num_components_climp_ernet)
    num_components_climp_ernet_v2 = sted_climp_ernet_v2[4]
    num_components_climp_ernet_v2 = std_data(num_components_climp_ernet_v2)
    num_components_climp_nerdy = sted_climp_nerdy[4]
    num_components_climp_nerdy = std_data(num_components_climp_nerdy)
    num_components_climp_nerdy_p4m = sted_climp_nerdy_p4m[4]
    num_components_climp_nerdy_p4m = std_data(num_components_climp_nerdy_p4m)

    ratio_nodes_climp_analyzer = sted_climp_analyzer[5]
    ratio_nodes_climp_analyzer = std_data(ratio_nodes_climp_analyzer)
    ratio_nodes_climp_ernet = sted_climp_ernet[5]
    ratio_nodes_climp_ernet = std_data(ratio_nodes_climp_ernet)
    ratio_nodes_climp_ernet_v2 = sted_climp_ernet_v2[5]
    ratio_nodes_climp_ernet_v2 = std_data(ratio_nodes_climp_ernet_v2)
    ratio_nodes_climp_nerdy = sted_climp_nerdy[5]
    ratio_nodes_climp_nerdy = std_data(ratio_nodes_climp_nerdy)
    ratio_nodes_climp_nerdy_p4m = sted_climp_nerdy_p4m[5]
    ratio_nodes_climp_nerdy_p4m = std_data(ratio_nodes_climp_nerdy_p4m)

    ratio_edges_climp_analyzer = sted_climp_analyzer[6]
    ratio_edges_climp_analyzer = std_data(ratio_edges_climp_analyzer)
    ratio_edges_climp_ernet = sted_climp_ernet[6]
    ratio_edges_climp_ernet = std_data(ratio_edges_climp_ernet)
    ratio_edges_climp_ernet_v2 = sted_climp_ernet_v2[6]
    ratio_edges_climp_ernet_v2 = std_data(ratio_edges_climp_ernet_v2)
    ratio_edges_climp_nerdy = sted_climp_nerdy[6]
    ratio_edges_climp_nerdy = std_data(ratio_edges_climp_nerdy)
    ratio_edges_climp_nerdy_p4m = sted_climp_nerdy_p4m[6]
    ratio_edges_climp_nerdy_p4m = std_data(ratio_edges_climp_nerdy_p4m)

    global_efficiency_climp_analyzer = sted_climp_analyzer[7]
    global_efficiency_climp_analyzer = std_data(global_efficiency_climp_analyzer)
    global_efficiency_climp_ernet = sted_climp_ernet[7]
    global_efficiency_climp_ernet = std_data(global_efficiency_climp_ernet)
    global_efficiency_climp_ernet_v2 = sted_climp_ernet_v2[7]
    global_efficiency_climp_ernet_v2 = std_data(global_efficiency_climp_ernet_v2)
    global_efficiency_climp_nerdy = sted_climp_nerdy[7]
    global_efficiency_climp_nerdy = std_data(global_efficiency_climp_nerdy)
    global_efficiency_climp_nerdy_p4m = sted_climp_nerdy_p4m[7]
    global_efficiency_climp_nerdy_p4m = std_data(global_efficiency_climp_nerdy_p4m)

    density_climp_analyzer = sted_climp_analyzer[8]
    density_climp_analyzer = std_data(density_climp_analyzer)
    density_climp_ernet = sted_climp_ernet[8]
    density_climp_ernet = std_data(density_climp_ernet)
    density_climp_ernet_v2 = sted_climp_ernet_v2[8]
    density_climp_ernet_v2 = std_data(density_climp_ernet_v2)
    density_climp_nerdy = sted_climp_nerdy[8]
    density_climp_nerdy = std_data(density_climp_nerdy)
    density_climp_nerdy_p4m = sted_climp_nerdy_p4m[8]
    density_climp_nerdy_p4m = std_data(density_climp_nerdy_p4m)

    df = pd.DataFrame()

    val_data = num_components_climp_analyzer + num_components_climp_ernet + num_components_climp_ernet_v2 + num_components_climp_nerdy + num_components_climp_nerdy_p4m + ratio_nodes_climp_analyzer + ratio_nodes_climp_ernet + ratio_nodes_climp_ernet_v2 + ratio_nodes_climp_nerdy + ratio_nodes_climp_nerdy_p4m + ratio_edges_climp_analyzer + ratio_edges_climp_ernet + ratio_edges_climp_ernet_v2 + ratio_edges_climp_nerdy + ratio_edges_climp_nerdy_p4m +   global_efficiency_climp_analyzer + global_efficiency_climp_ernet + global_efficiency_climp_ernet_v2 + global_efficiency_climp_nerdy + global_efficiency_climp_nerdy_p4m + density_climp_analyzer + density_climp_ernet + density_climp_ernet_v2 + density_climp_nerdy + density_climp_nerdy_p4m

    
    # val_data = num_nodes_climp_analyzer + num_nodes_climp_ernet + num_nodes_climp_ernet_v2 + num_nodes_climp_nerdy + num_nodes_climp_nerdy_p4m + num_edges_climp_analyzer + num_edges_climp_ernet + num_edges_climp_ernet_v2 + num_edges_climp_nerdy + num_edges_climp_nerdy_p4m + clustering_climp_analyzer + clustering_climp_ernet + clustering_climp_ernet_v2 + clustering_climp_nerdy + clustering_climp_nerdy_p4m + assortativity_climp_analyzer + assortativity_climp_ernet + assortativity_climp_ernet_v2 + assortativity_climp_nerdy + assortativity_climp_nerdy_p4m

    # l1 = [value for sublist in val_data for value in sublist]

    df['Values'] = val_data

    metric_names = []

    # metric_names.append(['Num_nodes']*len(num_nodes_climp_analyzer))
    # metric_names.append(['Num_nodes']*len(num_nodes_climp_ernet))
    # metric_names.append(['Num_nodes']*len(num_nodes_climp_ernet_v2))
    # metric_names.append(['Num_nodes']*len(num_nodes_climp_nerdy))
    # metric_names.append(['Num_nodes']*len(num_nodes_climp_nerdy_p4m))

    # metric_names.append(['Num_edges']*len(num_edges_climp_analyzer))
    # metric_names.append(['Num_edges']*len(num_edges_climp_ernet))
    # metric_names.append(['Num_edges']*len(num_edges_climp_ernet_v2))
    # metric_names.append(['Num_edges']*len(num_edges_climp_nerdy))
    # metric_names.append(['Num_edges']*len(num_edges_climp_nerdy_p4m))

    # metric_names.append(['Assortativity']*len(assortativity_climp_analyzer))
    # metric_names.append(['Assortativity']*len(assortativity_climp_ernet))
    # metric_names.append(['Assortativity']*len(assortativity_climp_ernet_v2))
    # metric_names.append(['Assortativity']*len(assortativity_climp_nerdy))
    # metric_names.append(['Assortativity']*len(assortativity_climp_nerdy_p4m))

    # metric_names.append(['Clustering']*len(clustering_climp_analyzer))
    # metric_names.append(['Clustering']*len(clustering_climp_ernet))
    # metric_names.append(['Clustering']*len(clustering_climp_ernet_v2))
    # metric_names.append(['Clustering']*len(clustering_climp_nerdy))
    # metric_names.append(['Clustering']*len(clustering_climp_nerdy_p4m))

    metric_names.append(['Num_components']*len(num_components_climp_analyzer))
    metric_names.append(['Num_components']*len(num_components_climp_ernet))
    metric_names.append(['Num_components']*len(num_components_climp_ernet_v2))
    metric_names.append(['Num_components']*len(num_components_climp_nerdy))
    metric_names.append(['Num_components']*len(num_components_climp_nerdy_p4m))

    metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_analyzer))
    metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_ernet))
    metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_ernet_v2))
    metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_nerdy))
    metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_nerdy_p4m))

    metric_names.append(['ratio_edges']*len(ratio_edges_climp_analyzer))
    metric_names.append(['ratio_edges']*len(ratio_edges_climp_ernet))
    metric_names.append(['ratio_edges']*len(ratio_edges_climp_ernet_v2))
    metric_names.append(['ratio_edges']*len(ratio_edges_climp_nerdy))
    metric_names.append(['ratio_edges']*len(ratio_edges_climp_nerdy_p4m))

    # metric_names.append(['Assortativity_coeff']*len(assortativity_climp_analyzer))
    # metric_names.append(['Assortativity_coeff']*len(assortativity_climp_ernet))
    # metric_names.append(['Assortativity_coeff']*len(assortativity_climp_ernet_v2))
    # metric_names.append(['Assortativity_coeff']*len(assortativity_climp_nerdy))
    # metric_names.append(['Assortativity_coeff']*len(assortativity_climp_nerdy_p4m))

    # metric_names.append(['Clustering_coeff']*len(clustering_climp_analyzer))
    # metric_names.append(['Clustering_coeff']*len(clustering_climp_ernet))
    # metric_names.append(['Clustering_coeff']*len(clustering_climp_ernet_v2))
    # metric_names.append(['Clustering_coeff']*len(clustering_climp_nerdy))
    # metric_names.append(['Clustering_coeff']*len(clustering_climp_nerdy_p4m))

    metric_names.append(['Global_efficiency']*len(global_efficiency_climp_analyzer))
    metric_names.append(['Global_efficiency']*len(global_efficiency_climp_ernet))
    metric_names.append(['Global_efficiency']*len(global_efficiency_climp_ernet_v2))
    metric_names.append(['Global_efficiency']*len(global_efficiency_climp_nerdy))
    metric_names.append(['Global_efficiency']*len(global_efficiency_climp_nerdy_p4m))

    metric_names.append(['Density']*len(density_climp_analyzer))
    metric_names.append(['Density']*len(density_climp_ernet))
    metric_names.append(['Density']*len(density_climp_ernet_v2))
    metric_names.append(['Density']*len(density_climp_nerdy))
    metric_names.append(['Density']*len(density_climp_nerdy_p4m))

    method_names = []    
    
    # method_names.append(['AnalyzER']*len(sted_climp_analyzer[0]))
    # method_names.append(['ERnet']*len(sted_climp_ernet[0]))
    # method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[0]))
    # method_names.append(['nERdy']*len(sted_climp_nerdy[0]))
    # method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[0]))

    # method_names.append(['AnalyzER']*len(sted_climp_analyzer[1]))
    # method_names.append(['ERnet']*len(sted_climp_ernet[1]))
    # method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[1]))
    # method_names.append(['nERdy']*len(sted_climp_nerdy[1]))
    # method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[1]))

    # method_names.append(['AnalyzER']*len(sted_climp_analyzer[2]))
    # method_names.append(['ERnet']*len(sted_climp_ernet[2]))
    # method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[2]))
    # method_names.append(['nERdy']*len(sted_climp_nerdy[2]))
    # method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[2]))

    # method_names.append(['AnalyzER']*len(sted_climp_analyzer[3]))
    # method_names.append(['ERnet']*len(sted_climp_ernet[3]))
    # method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[3]))
    # method_names.append(['nERdy']*len(sted_climp_nerdy[3]))
    # method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[3]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[4]))
    method_names.append(['ERnet']*len(sted_climp_ernet[4]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[4]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[4]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[4]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[5]))
    method_names.append(['ERnet']*len(sted_climp_ernet[5]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[5]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[5]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[5]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[6]))
    method_names.append(['ERnet']*len(sted_climp_ernet[6]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[6]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[6]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[6]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[7]))
    method_names.append(['ERnet']*len(sted_climp_ernet[7]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[7]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[7]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[7]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[8]))
    method_names.append(['ERnet']*len(sted_climp_ernet[8]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[8]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[8]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[8]))

    df['Metric'] = [value for sublist in metric_names for value in sublist]

    df['Method'] = [value for sublist in method_names for value in sublist]

    # sns.set(style="whitegrid")
    ax = sns.boxplot(x="Metric", y="Values", hue="Method", data=df, showfliers=False, width=0.8, palette="Set2")

    # plt.ylim(0, 0.8)
    # plt.xlim(-1, 4.0)

    yt = ax.get_yticks()
    yt = [f'{y:.1f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)#, rotation=45)
    # ax.set_xticklabels(['NC', 'RN', 'RE', 'GE', 'D'], fontsize=12)


    ax.grid(axis='y')#, linestyle='-', linewidth=0.5, color='white')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=5, fancybox=True, shadow=True, fontsize=10)
    plt.xlabel('Metric', fontsize=15)
    plt.ylabel('Error', fontsize=15)

    # plt.subplots_adjust(hspace=0.3)

    plt.gcf().set_size_inches(8, 10)
    # plt.gcf().set_size_inches(4, 6)

    # plt.show()
    plt.savefig(f'nw_graph_analysis/confocal_{group}_graph_err_p2.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.close()


# get_conf_graph_perf('atl')
# get_conf_graph_perf('climp')
# get_conf_graph_perf('rtn')
# get_conf_graph_perf('control')
# exit()


def get_group_seg_perf(group):
    conf_dice = pkl.load(open(f'nw_graph_analysis/conf_{group}_dice.pkl', 'rb'))
    conf_dice_p4m = pkl.load(open(f'nw_graph_analysis/conf_{group}_dice_p4m.pkl', 'rb'))
    conf_f1 = pkl.load(open(f'nw_graph_analysis/conf_{group}_f1.pkl', 'rb'))
    conf_f1_p4m = pkl.load(open(f'nw_graph_analysis/conf_{group}_f1_p4m.pkl', 'rb'))
    conf_jaccard = pkl.load(open(f'nw_graph_analysis/conf_{group}_jacc.pkl', 'rb'))
    conf_jaccard_p4m = pkl.load(open(f'nw_graph_analysis/conf_{group}_jacc_p4m.pkl', 'rb'))

    sted_dice = pkl.load(open(f'nw_graph_analysis/sted_{group}_dice.pkl', 'rb'))
    sted_f1 = pkl.load(open(f'nw_graph_analysis/sted_{group}_f1.pkl', 'rb'))
    sted_jaccard = pkl.load(open(f'nw_graph_analysis/sted_{group}_iou.pkl', 'rb'))

    df = pd.DataFrame()

    # val_data = conf_dice + [conf_dice_p4m] + conf_f1 + [conf_f1_p4m] + conf_jaccard + [conf_jaccard_p4m]

    val_data = sted_dice + sted_f1 + sted_jaccard

    l1 = [value for sublist in val_data for value in sublist]

    df['Values'] = l1

    metric_names = []
    method_names = []

    # for method in conf_dice:
    #     metric_names.append(['Dice score']*len(method))
    # metric_names.append(['Dice score']*len(conf_dice_p4m))

    # method_names.append(['ERnet']*len(conf_dice[0]))
    # method_names.append(['ERnet-v2']*len(conf_dice[1]))
    # method_names.append(['AnalyzER']*len(conf_dice[2]))
    # method_names.append(['nERdy']*len(conf_dice[3]))
    # method_names.append(['nERdy+']*len(conf_dice_p4m))

    # for method in conf_f1:
    #     metric_names.append(['F1 score']*len(method))
    # metric_names.append(['F1 score']*len(conf_f1_p4m))

    # method_names.append(['ERnet']*len(conf_f1[0]))
    # method_names.append(['ERnet-v2']*len(conf_f1[1]))
    # method_names.append(['AnalyzER']*len(conf_f1[2]))
    # method_names.append(['nERdy']*len(conf_f1[3]))
    # method_names.append(['nERdy+']*len(conf_f1_p4m))

    # for method in conf_jaccard:
    #     metric_names.append(['Jaccard Index']*len(method))
    # metric_names.append(['Jaccard Index']*len(conf_jaccard_p4m))

    # method_names.append(['ERnet']*len(conf_jaccard[0]))
    # method_names.append(['ERnet-v2']*len(conf_jaccard[1]))
    # method_names.append(['AnalyzER']*len(conf_jaccard[2]))
    # method_names.append(['nERdy']*len(conf_jaccard[3]))
    # method_names.append(['nERdy+']*len(conf_jaccard_p4m))

    #########################################

    for method in sted_dice:
        metric_names.append(['Dice score']*len(method))
    for method in sted_f1:
        metric_names.append(['F1 score']*len(method))
    for method in sted_jaccard:
        metric_names.append(['Jaccard Index']*len(method))
    
    method_names.append(['AnalyzER']*len(sted_dice[0]))
    method_names.append(['ERnet-v2']*len(sted_dice[1]))
    method_names.append(['ERnet']*len(sted_dice[2]))
    method_names.append(['nERdy']*len(sted_dice[3]))
    method_names.append(['nERdy+']*len(sted_dice[4]))

    method_names.append(['AnalyzER']*len(sted_f1[0]))
    method_names.append(['ERnet-v2']*len(sted_f1[1]))
    method_names.append(['ERnet']*len(sted_f1[2]))
    method_names.append(['nERdy']*len(sted_f1[3]))
    method_names.append(['nERdy+']*len(sted_f1[4]))

    method_names.append(['AnalyzER']*len(sted_jaccard[0]))
    method_names.append(['ERnet-v2']*len(sted_jaccard[1]))
    method_names.append(['ERnet']*len(sted_jaccard[2]))
    method_names.append(['nERdy']*len(sted_jaccard[3]))
    method_names.append(['nERdy+']*len(sted_jaccard[4]))

    l2 = [value for sublist in metric_names for value in sublist]

    df['Metric'] = l2

    df['Method'] = [value for sublist in method_names for value in sublist]

    # print(df)

    # sns.set(style="whitegrid")
    ax = sns.boxplot(x="Metric", y="Values", hue="Method", data=df, showfliers=False, width=0.9, palette="Set2")

    plt.ylim(0.1, 1.0)
    # plt.xlim(-1, 4.0)

    yt = ax.get_yticks()
    yt = [f'{y:.1f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)#, rotation=45)

    ax.grid(axis='y')#, linestyle='-', linewidth=0.5, color='white')
    plt.xlabel('Metric', fontsize=16)
    plt.ylabel('Value', fontsize=16)

    plt.gcf().set_size_inches(5, 8)

    # plt.show()
    plt.savefig(f'nw_graph_analysis/sted_{group}_segmentation.png', dpi=300, bbox_inches='tight')

    plt.close()
     
# get_group_seg_perf('climp')
# exit()

def get_segmentation_perf():
    conf_dice = pkl.load(open('nw_graph_analysis/conf_dice.pkl', 'rb'))
    conf_dice_p4m = pkl.load(open('nw_graph_analysis/conf_dice_p4m.pkl', 'rb'))
    conf_f1 = pkl.load(open('nw_graph_analysis/conf_f1.pkl', 'rb'))
    conf_f1_p4m = pkl.load(open('nw_graph_analysis/conf_f1_p4m.pkl', 'rb'))
    conf_jaccard = pkl.load(open('nw_graph_analysis/conf_jacc.pkl', 'rb'))
    conf_jaccard_p4m = pkl.load(open('nw_graph_analysis/conf_jacc_p4m.pkl', 'rb'))


    sted_dice = pkl.load(open('nw_graph_analysis/sted_dice.pkl', 'rb'))
    sted_f1 = pkl.load(open('nw_graph_analysis/sted_f1.pkl', 'rb'))
    sted_jaccard = pkl.load(open('nw_graph_analysis/sted_iou.pkl', 'rb'))


    df = pd.DataFrame()

    # val_data = conf_dice + [conf_dice_p4m] + conf_f1 + [conf_f1_p4m] + conf_jaccard + [conf_jaccard_p4m]

    # l1 = [value for sublist in val_data for value in sublist]


    sted_data = sted_dice + sted_f1 + sted_jaccard

    l1 = [value for sublist in sted_data for value in sublist]


    # df['Values'] = conf_dice + [conf_dice_p4m] + conf_f1 + [conf_f1_p4m] + conf_jaccard + [conf_jaccard_p4m]# + sted_dice + sted_f1 + sted_jaccard

    df['Values'] = l1

    # print(df['Values'])

    metric_names = []
    method_names = []


    # Conf

    # for method in conf_dice:
    #     metric_names.append(['Dice score']*len(method))
    # metric_names.append(['Dice score']*len(conf_dice_p4m))

    # method_names.append(['AnalyzER']*len(conf_dice[0]))
    # method_names.append(['ERnet']*len(conf_dice[1]))
    # method_names.append(['ERnet-v2']*len(conf_dice[2]))
    # method_names.append(['nERdy']*len(conf_dice[3]))
    # method_names.append(['nERdy+']*len(conf_dice_p4m))

    # for method in conf_f1:
    #     metric_names.append(['F1 score']*len(method))
    # metric_names.append(['F1 score']*len(conf_f1_p4m))

    # method_names.append(['AnalyzER']*len(conf_f1[0]))
    # method_names.append(['ERnet']*len(conf_f1[1]))
    # method_names.append(['ERnet-v2']*len(conf_f1[2]))
    # method_names.append(['nERdy']*len(conf_f1[3]))
    # method_names.append(['nERdy+']*len(conf_f1_p4m))

    # for method in conf_jaccard:
    #     metric_names.append(['Jaccard Index']*len(method))
    # metric_names.append(['Jaccard Index']*len(conf_jaccard_p4m))

    # method_names.append(['AnalyzER']*len(conf_jaccard[0]))
    # method_names.append(['ERnet']*len(conf_jaccard[1]))
    # method_names.append(['ERnet-v2']*len(conf_jaccard[2]))
    # method_names.append(['nERdy']*len(conf_jaccard[3]))
    # method_names.append(['nERdy+']*len(conf_jaccard_p4m))


    #########################################

    for method in sted_dice:
        metric_names.append(['Dice score']*len(method))
    for method in sted_f1:
        metric_names.append(['F1 score']*len(method))
    for method in sted_jaccard:
        metric_names.append(['Jaccard Index']*len(method))

    method_names.append(['AnalyzER']*len(sted_dice[0]))
    method_names.append(['ERnet']*len(sted_dice[1]))
    method_names.append(['ERnet-v2']*len(sted_dice[2]))
    method_names.append(['nERdy']*len(sted_dice[3]))
    method_names.append(['nERdy+']*len(sted_dice[4]))

    method_names.append(['AnalyzER']*len(sted_f1[0]))
    method_names.append(['ERnet']*len(sted_f1[1]))
    method_names.append(['ERnet-v2']*len(sted_f1[2]))
    method_names.append(['nERdy']*len(sted_f1[3]))
    method_names.append(['nERdy+']*len(sted_f1[4]))

    method_names.append(['AnalyzER']*len(sted_jaccard[0]))
    method_names.append(['ERnet']*len(sted_jaccard[1]))
    method_names.append(['ERnet-v2']*len(sted_jaccard[2]))
    method_names.append(['nERdy']*len(sted_jaccard[3]))
    method_names.append(['nERdy+']*len(sted_jaccard[4]))

    ###########################################

    l2 = [value for sublist in metric_names for value in sublist]

    df['Metric'] = l2

    df['Method'] = [value for sublist in method_names for value in sublist]



    # print(df)

    # sns.set(style="whitegrid")
    ax = sns.boxplot(x="Metric", y="Values", hue="Method", data=df, showfliers=False, width=0.9, palette="Set2")

    plt.ylim(0.2, 1.0)
    # plt.xlim(-1, 4.0)

    yt = ax.get_yticks()
    yt = [f'{y:.1f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)#, rotation=45)

    # statannot.add_stat_annotation(ax, data=df, x="Metric", y="Values", hue="Method",
    #                     box_pairs=[(("Dice score", "ERnet"), ("Dice score", "ERnet-v2")),
    #                                     (("Dice score", "ERnet"), ("Dice score", "AnalyzER")),
    #                                     (("Dice score", "ERnet"), ("Dice score", "nERdy")),
    #                                     (("Dice score", "ERnet"), ("Dice score", "nERdy+")),
    #                                     (("Dice score", "ERnet-v2"), ("Dice score", "AnalyzER")),
    #                                     (("Dice score", "ERnet-v2"), ("Dice score", "nERdy")),
    #                                     (("Dice score", "ERnet-v2"), ("Dice score", "nERdy+")),
    #                                     (("Dice score", "AnalyzER"), ("Dice score", "nERdy")),
    #                                     (("Dice score", "AnalyzER"), ("Dice score", "nERdy+")),
    #                                     (("Dice score", "nERdy"), ("Dice score", "nERdy+")),
    #                                     (("F1 score", "ERnet"), ("F1 score", "ERnet-v2")),
    #                                     (("F1 score", "ERnet"), ("F1 score", "AnalyzER")),
    #                                     (("F1 score", "ERnet"), ("F1 score", "nERdy")),
    #                                     (("F1 score", "ERnet"), ("F1 score", "nERdy+")),
    #                                     (("F1 score", "ERnet-v2"), ("F1 score", "AnalyzER")),
    #                                     (("F1 score", "ERnet-v2"), ("F1 score", "nERdy")),
    #                                     (("F1 score", "ERnet-v2"), ("F1 score", "nERdy+")),
    #                                     (("F1 score", "AnalyzER"), ("F1 score", "nERdy")),
    #                                     (("F1 score", "AnalyzER"), ("F1 score", "nERdy+")),
    #                                     (("F1 score", "nERdy"), ("F1 score", "nERdy+")),
    #                                     (("Jaccard Index", "ERnet"), ("Jaccard Index", "ERnet-v2")),
    #                                     (("Jaccard Index", "ERnet"), ("Jaccard Index", "AnalyzER")),
    #                                     (("Jaccard Index", "ERnet"), ("Jaccard Index", "nERdy")),
    #                                     (("Jaccard Index", "ERnet"), ("Jaccard Index", "nERdy+")),
    #                                     (("Jaccard Index", "ERnet-v2"), ("Jaccard Index", "AnalyzER")),
    #                                     (("Jaccard Index", "ERnet-v2"), ("Jaccard Index", "nERdy")),
    #                                     (("Jaccard Index", "ERnet-v2"), ("Jaccard Index", "nERdy+")),
    #                                     (("Jaccard Index", "AnalyzER"), ("Jaccard Index", "nERdy")),
    #                                     (("Jaccard Index", "AnalyzER"), ("Jaccard Index", "nERdy+")),
    #                                     (("Jaccard Index", "nERdy"), ("Jaccard Index", "nERdy+"))],
    #                     test='Mann-Whitney', text_format='star', loc='inside', verbose=2)


    ax.grid(axis='y')#, linestyle='-', linewidth=0.5, color='white')
    plt.xlabel('Metric', fontsize=16)
    plt.ylabel('Value', fontsize=16)

    plt.gcf().set_size_inches(5, 8)

    # plt.show()
    plt.savefig('nw_graph_analysis/sted_segmentation.png', dpi=300, bbox_inches='tight')

    plt.close()

