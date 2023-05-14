import sknw
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import data
from junction_analysis_modules import JunctionAnalysis as JA
import imageio
import pickle

junc_analysis = JA('/localhome/asa420/MIAL/data/confocal_movies/')


groups = {'ATL': 26, 'Climp': 31, 'RTN': 29, 'Control':31}
group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}

def get_temporal_graph(group, num):
    er_prefix = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/std_egfp/'
    skel_prefix = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/'
    graph_data = []
    for i in range(100):
        er_path = f'{er_prefix}{group_pref[group]}{num}_decon_t0{i:02d}_ch00_std.png'
        skel_path = f'{skel_prefix}{group_pref[group]}{num}/{group_pref[group]}{num}_decon_t0{i:02d}_ch00_skel.png'
        _, graph = junc_analysis.get_junctions(er_path, skel_path)
        graph_data.append(graph)
    
    return graph_data


def get_group_temporal_graphs(group):
    num_series = groups[group]
    return [get_temporal_graph(group, i) for i in range(1, num_series+1)]


def create_group_graph_pickles(group):
    group_graphs = get_group_temporal_graphs(group)
    with open(f'{group}_graphs.pkl', 'wb') as f:
        pickle.dump(group_graphs, f)


# create_group_graph_pickles('ATL')
create_group_graph_pickles('Climp')
create_group_graph_pickles('RTN')
create_group_graph_pickles('Control')

# atl_graphs = pickle.load(open('ATL_graphs.pkl', 'rb'))


# a1 = atl_graphs[0]
# for i in range(100):
#     print(len(a1[i].nodes()))
# exit()

# for i in range(26):
#     print(len(atl_graphs[i][0].nodes()))
# exit()


def get_mean_skel_graphs(group, num_series):

    data = []
    for num in range(1, num_series+1):
        skel_mean_path = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/er_mean_proc/{group.lower()}{num}_proc_skel.png'

        graph = sknw.build_sknw(imageio.imread(skel_mean_path), multi=True, iso=False)
        data.append(graph)

    with open(f'{group}_mean_graphs.pkl', 'wb') as f:
        pickle.dump(data, f)


# get_mean_skel_graphs('ATL', 26)
# get_mean_skel_graphs('Climp', 31)
# get_mean_skel_graphs('RTN', 29)
# get_mean_skel_graphs('Control', 31)